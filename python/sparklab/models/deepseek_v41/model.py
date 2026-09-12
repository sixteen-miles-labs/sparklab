"""Native V4.1 text decoder with bounded disk weights and request-local state.

Math follows DeepSeek's MIT-licensed inference/model.py at
df42c109f1defefcbfcedbe7d905718a12266e40 (see LICENSE.deepseek).
This research path batches prompt work layer-by-layer and evaluates decode
eagerly. It prioritizes bounded complete-model execution over prefix reuse.
"""
import torch
import torch.nn.functional as F

from sparklab.core import get_global_ctx
from sparklab.models.blocks import BaseLLMModel

from .config import CACHE_BYTES
from .engram import EngramLayout, NgramHashState
from .ops import (
    apply_rotary_emb, precompute_freqs_cis, hc_split_sinkhorn,
    fp8_roundtrip, fp4_roundtrip, dequant, linear, norm,
    sparse_attention, candidate_mask,
)
from .weight import DiskWeights
from .expert_cache import ExpertBank


class Decoder:
    def __init__(self, args, store, tokenizer=None):
        self.args, self.store = args, store
        self.device = store.device
        store.validate_geometry(args)
        self.layout = EngramLayout.from_args(args)
        # Hashing is CPU-only: transferring a handful of row IDs is bounded,
        # whereas copying either entire table to CPU/GPU would exceed GB10 RAM.
        with torch.device("cpu"):
            self.hash = NgramHashState(args, self.layout, tokenizer) if self.layout else None
        with torch.device(self.device):
            self.freqs = {
                compressed: precompute_freqs_cis(
                    args.rope_head_dim, args.max_seq_len,
                    args.original_seq_len if compressed else 0,
                    args.compress_rope_theta if compressed else args.rope_theta,
                    args.rope_factor, args.beta_fast, args.beta_slow,
                ) for compressed in (False, True)
            }
        packed_experts = "layers.0.ffn.experts.0.w1.scale" in store.metadata
        self.expert_bank = (
            ExpertBank(args, store) if self.device.type == "cuda" and packed_experts else None
        )
        self.dspark = None
        self._verify_carries = None
        self.reset()

    def reset(self):
        self.position = 0
        self._verify_carries = None
        self.windows = {}
        self.compressed = {}
        self.keys = {}
        self.carry = {}
        if self.hash is not None:
            self.hash.cache.zero_()
        if self.dspark is not None:
            self.dspark.reset()

    def enable_dspark(self, steps):
        from .dspark import DSparkDraft

        self.dspark = DSparkDraft(self, steps)

    @staticmethod
    def _clone_tree(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: Decoder._clone_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [Decoder._clone_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(Decoder._clone_tree(item) for item in value)
        return value

    def snapshot(self):
        return {
            "position": self.position,
            "windows": self._clone_tree(self.windows),
            "compressed": self._clone_tree(self.compressed),
            "keys": self._clone_tree(self.keys),
            "carry": self._clone_tree(self.carry),
            "hash": self.hash.cache.clone() if self.hash is not None else None,
            "dspark": self.dspark.snapshot() if self.dspark is not None else None,
        }

    def restore(self, snapshot):
        self.position = snapshot["position"]
        self.windows = snapshot["windows"]
        self.compressed = snapshot["compressed"]
        self.keys = snapshot["keys"]
        self.carry = snapshot["carry"]
        if self.hash is not None:
            self.hash.cache.copy_(snapshot["hash"])
        if self.dspark is not None:
            self.dspark.restore(snapshot["dspark"])

    def commit_prefix(self, snapshot, length):
        """Retain an accepted verification prefix without replaying the model."""
        start = snapshot["position"]
        current = self.snapshot()
        positions = torch.arange(start, start + length, device=self.device)
        self.restore(snapshot)
        for layer, window in self.windows.items():
            slots = positions % self.args.window_size
            window[slots] = current["windows"][layer][slots]
        for layer, values in self.compressed.items():
            ratio = self.args.compress_ratios[layer]
            old_count, new_count = start // ratio, (start + length) // ratio
            if new_count > old_count:
                values[old_count:new_count] = current["compressed"][layer][
                    old_count:new_count
                ]
                self.keys[layer][old_count:new_count] = current["keys"][layer][
                    old_count:new_count
                ]
        if self._verify_carries is not None:
            for layer, states in self._verify_carries.items():
                self.carry[layer] = self._clone_tree(states[length - 1])
        if self.hash is not None:
            self.hash.cache[:, start:start + length] = current["hash"][
                :, start:start + length
            ]
        if self.dspark is not None:
            self.dspark.commit_prefix(
                snapshot["dspark"], current["dspark"], start, length
            )
        self.position = start + length
        self._verify_carries = None

    def _linear(self, name, x):
        return linear(self.store, name, x)

    def _norm(self, name, x):
        return norm(self.store, name, x, self.args.norm_eps)

    def _rope(self, x, pos, compressed, inverse=False):
        rd = self.args.rope_head_dim
        apply_rotary_emb(x[..., -rd:], self.freqs[compressed][pos:pos + 1], inverse)
        return x

    def _compress(self, layer, x, pos):
        a = self.args
        name = f"layers.{layer}.attn.compressor"
        ratio = a.compress_ratios[layer]
        kv = self._linear(name + ".wkv", x.float() if ratio > 1 else x)
        if ratio > 1:
            score = self._linear(name + ".wgate", x.float())
            carry = self.carry.setdefault(layer, [])
            carry.append((kv, score))
            if len(carry) < ratio:
                return None
            values = torch.stack([item[0] for item in carry])
            scores = torch.stack([item[1] for item in carry])
            kv = (values * scores.softmax(0)).sum(0).to(x.dtype)
            carry.clear()
        return self._norm(name + ".norm", kv)

    def _attention(self, layer, x, pos, shared):
        a, store = self.args, self.store
        name = f"layers.{layer}.attn"
        ratio = a.compress_ratios[layer]
        qr = self._norm(name + ".q_norm", self._linear(name + ".wq_a", x))
        q = self._linear(name + ".wq_b", qr).view(1, 1, a.n_heads, a.head_dim)
        q = self._rope(q, pos, bool(ratio))[0, 0]
        kv = self._norm(name + ".kv_norm", self._linear(name + ".wkv", x))
        kv = fp8_roundtrip(self._rope(kv, pos, bool(ratio)))
        if layer not in self.windows:
            self.windows[layer] = torch.empty(a.window_size, a.head_dim, dtype=x.dtype, device=x.device)
        window = self.windows[layer]
        window[pos % a.window_size] = kv.reshape(-1)
        # Match the reference ring order (chronological, oldest first).
        if pos >= a.window_size - 1:
            slot = (pos + 1) % a.window_size
            visible = torch.cat((window[slot:], window[:slot]))
        else:
            visible = window[:pos + 1]
        if ratio:
            latent = self._compress(layer, x, pos) if layer in a.kv_source_layers else None
            length = (pos + 1) // ratio
            if layer in a.kv_source_layers:
                if layer not in self.compressed:
                    self.compressed[layer] = torch.empty(a.max_seq_len // ratio, a.head_dim, dtype=x.dtype, device=x.device)
                    self.keys[layer] = torch.empty(a.max_seq_len // ratio, a.index_head_dim, dtype=x.dtype, device=x.device)
                shared["kv"] = self.compressed[layer]
                shared["keys"] = self.keys[layer]
                if latent is not None:
                    key = self._norm(name + ".indexer.k_norm", self._linear(name + ".indexer.wk", latent))
                    shared["keys"][length - 1] = fp4_roundtrip(self._rope(key, pos + 1 - ratio, True)).reshape(-1)
                    shared["kv"][length - 1] = fp4_roundtrip(
                        self._rope(latent, pos + 1 - ratio, True), 16, e4m3_scale=True,
                    ).reshape(-1)
            if layer in a.index_source_layers:
                iq = self._linear(name + ".indexer.wq_b", qr).view(1, 1, a.index_n_heads, a.index_head_dim)
                iq = fp4_roundtrip(self._rope(iq, pos, True))[0, 0]
                weights = self._linear(name + ".indexer.weights_proj", x).reshape(-1)
                weights = weights * (a.index_head_dim**-.5 * a.index_n_heads**-.5)
                # Reference einsum materializes BF16 scores before the weighted sum.
                scores = (iq @ shared["keys"][:length].T).relu()
                scores = (scores * weights[:, None]).sum(0)
                if length and layer == a.candidate_source_layer:
                    shared["candidates"] = candidate_mask(scores, a.candidate_topk_blocks, a.candidate_block_size)
                elif length and 0 <= a.candidate_source_layer < layer:
                    scores = scores.masked_fill(~shared["candidates"], -torch.inf)
                shared["indices"] = scores.topk(min(a.index_topk, length), sorted=False).indices.sort().values
            visible = torch.cat((visible, shared["kv"][shared["indices"]]))
        output = sparse_attention(q, visible, store.get(name + ".attn_sink"), a.head_dim**-.5)
        output = self._rope(output.view(1, 1, a.n_heads, a.head_dim), pos, bool(ratio), inverse=True)
        weight = store.get(name + ".wo_a.weight")
        if name + ".wo_a.scale" in store.metadata:
            weight = dequant(weight, store.get(name + ".wo_a.scale")).to(torch.bfloat16)
        weight = weight.view(a.o_groups, a.o_lora_rank, -1)
        output = torch.einsum("bsgd,grd->bsgr", output.view(1, 1, a.o_groups, -1), weight.to(output.dtype))
        return self._linear(name + ".wo_b", output.flatten(2))

    def _expert(self, name, x, routing_weight=None):
        gate = self._linear(name + ".w1", x).float()
        up = self._linear(name + ".w3", x).float()
        limit = self.args.swiglu_limit
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        hidden = F.silu(gate) * up
        if routing_weight is not None:
            hidden = hidden * routing_weight
        return self._linear(name + ".w2", hidden.to(x.dtype))

    def _moe(
        self,
        layer,
        x,
        *,
        name=None,
        n_activated_experts=None,
    ):
        a, store = self.args, self.store
        name = name or f"layers.{layer}.ffn"
        n_activated_experts = n_activated_experts or a.n_activated_experts
        scores = F.linear(x.float(), store.get(name + ".gate.weight").float()) / a.gate_temp
        if a.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        elif a.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(-1)
        indices = (scores + store.get(name + ".gate.bias")).topk(
            n_activated_experts, dim=-1
        ).indices
        weights = scores.gather(-1, indices)
        if a.norm_topk_prob and n_activated_experts > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * a.route_scale
        if self.expert_bank is not None:
            order = indices.argsort(-1)
            sorted_indices = indices.gather(-1, order)
            sorted_weights = weights.gather(-1, order)
            slots = self.expert_bank.slots(layer, sorted_indices)
            if slots is not None:
                from sparklab.moe.fused_ds_fp4 import routed_experts_fp4

                flat = x.reshape(-1, x.shape[-1])
                output = routed_experts_fp4(
                    flat,
                    slots.reshape(flat.shape[0], -1),
                    sorted_weights.reshape(flat.shape[0], -1),
                    self.expert_bank.gate_up,
                    self.expert_bank.gate_up_scale,
                    self.expert_bank.down,
                    self.expert_bank.down_scale,
                    a.swiglu_limit,
                    activation_block=32,
                    sum_in_fp32=True,
                ).reshape_as(x)
                output += self._expert(name + ".shared_experts", x).float()
                return output.to(x.dtype)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_indices = indices.reshape(flat_x.shape[0], -1)
        flat_weights = weights.reshape_as(flat_indices)
        output = torch.zeros_like(flat_x, dtype=torch.float32)
        # Upstream accumulates in expert-ID order, with weighting before w2.
        for slot in flat_indices.flatten().argsort().tolist():
            row, route = divmod(slot, flat_indices.shape[1])
            expert = int(flat_indices[row, route])
            output[row:row + 1] += self._expert(
                f"{name}.experts.{expert}",
                flat_x[row:row + 1],
                flat_weights[row, route],
            )
        output += self._expert(name + ".shared_experts", flat_x)
        return output.reshape_as(x).to(x.dtype)

    def _engram(self, layer, h, hashes):
        a, store = self.args, self.store
        name = f"layers.{layer}.engram"
        ids = hashes[:, :, self.layout.layer_ids.index(layer), :]
        values = store.rows(name + ".embed.weight", ids).float().unflatten(-1, (-1, 32))
        scale = store.rows(name + ".embed.scale", ids).view(torch.uint8).float()
        values = (values * torch.exp2(scale - 127).unsqueeze(-1)).flatten(-2).to(h.dtype)
        kv = self._linear(name + ".wkv", values.flatten(-2))
        key, value = kv.split([a.hc_mult * a.dim, a.dim], dim=-1)
        key = key.float().unflatten(-1, (a.hc_mult, a.dim))
        weight = store.get(name + ".q_weight").float() * store.get(name + ".k_weight").float()
        xf = h.float()
        rstd = torch.rsqrt(xf.square().mean(-1) + a.norm_eps) * torch.rsqrt(key.square().mean(-1) + a.norm_eps)
        dot = (xf * weight * key).sum(-1) * rstd * a.dim**-.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (xf + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)

    def _mixes(self, layer, kind, h):
        a, store = self.args, self.store
        name = f"layers.{layer}.hc_{kind}"
        x = h.flatten(2).float()
        mixes = F.linear(x, store.get(name + "_fn").float()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + a.norm_eps)
        pre, post, comb = hc_split_sinkhorn(mixes.reshape(-1, mixes.shape[-1]), store.get(name + "_scale"), store.get(name + "_base"),
                                           a.hc_mult, a.hc_sinkhorn_iters, a.hc_eps)
        lead = mixes.shape[:-1]
        return (pre.reshape(*lead, a.hc_mult), post.reshape(*lead, a.hc_mult),
                comb.reshape(*lead, a.hc_mult, a.hc_mult))

    @staticmethod
    def _pre(h, pre):
        return (pre.unsqueeze(-1) * h.float()).sum(2).to(h.dtype)

    @staticmethod
    def _post(x, residual, post, comb):
        return (post.unsqueeze(-1) * x.unsqueeze(-2)
                + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(2)).to(x.dtype)

    @torch.inference_mode()
    def step(self, token):
        a, pos = self.args, self.position
        if pos >= a.max_seq_len:
            raise ValueError("DeepSeek V4.1 research context limit exceeded")
        if not 0 <= token < a.vocab_size or token == a.image_token_id:
            raise ValueError("DeepSeek V4.1 native research supports text token IDs only")
        ids = torch.tensor([[token]], dtype=torch.int64, device="cpu")
        hashes = self.hash(ids, pos) if self.hash is not None else None
        h = self.store.rows("embed.weight", ids).to(torch.bfloat16).unsqueeze(2).repeat(1, 1, a.hc_mult, 1)
        pre = torch.zeros((1, 1, a.hc_mult), device=self.device, dtype=torch.float32)
        pre[..., 0] = 1
        shared = {}
        aux = []
        for layer in range(a.n_layers):
            if self.layout and layer in self.layout.layer_ids:
                h = self._engram(layer, h, hashes)
            if self.dspark is not None and layer in a.dspark_target_layer_ids:
                aux.append(h.mean(2))
            attn_pre, post, comb = self._mixes(layer, "attn", h)
            x = self._norm(f"layers.{layer}.attn_norm", self._pre(h, pre))
            h = self._post(self._attention(layer, x, pos, shared), h, post, comb)
            pre, post, comb = self._mixes(layer, "ffn", h)
            x = self._norm(f"layers.{layer}.ffn_norm", self._pre(h, attn_pre))
            h = self._post(self._moe(layer, x), h, post, comb)
        h = self._norm("norm", self._pre(h, pre))
        head = self.store.get("head.weight")
        if h.is_cuda and head.is_cuda:
            from sparklab.kernels.triton.dsv4.skinny import bf16_skinny_linear

            logits = bf16_skinny_linear(h[:, -1], head, out_dtype=torch.float32)
        else:
            logits = F.linear(h[:, -1].float(), head.float())
        if self.dspark is not None:
            self.dspark.store_target(aux, pos)
        self.position += 1
        return logits

    @torch.inference_mode()
    def prefill(self, tokens, *, return_all_logits=False):
        """Evaluate a prompt layer-by-layer while retaining sequential state.

        Attention and its projections still advance each query causally, while
        Engram reads, hyper-connections, and MoE routing/compute operate on the
        whole prompt. Each selected expert is loaded at most once per layer.
        """
        a, start = self.args, self.position
        tokens = [int(token) for token in tokens]
        if not tokens:
            raise ValueError("DeepSeek V4.1 prefill requires at least one token")
        if start + len(tokens) > a.max_seq_len:
            raise ValueError("DeepSeek V4.1 research context limit exceeded")
        if any(not 0 <= token < a.vocab_size or token == a.image_token_id for token in tokens):
            raise ValueError("DeepSeek V4.1 native research supports text token IDs only")
        ids = torch.tensor([tokens], dtype=torch.int64, device="cpu")
        hashes = self.hash(ids, start) if self.hash is not None else None
        h = self.store.rows("embed.weight", ids).to(torch.bfloat16).unsqueeze(2).repeat(1, 1, a.hc_mult, 1)
        pre = torch.zeros((1, len(tokens), a.hc_mult), device=self.device, dtype=torch.float32)
        pre[..., 0] = 1
        shared = [{} for _ in tokens]
        self._verify_carries = {} if return_all_logits else None
        aux = []
        for layer in range(a.n_layers):
            if self.layout and layer in self.layout.layer_ids:
                h = self._engram(layer, h, hashes)
            if self.dspark is not None and layer in a.dspark_target_layer_ids:
                aux.append(h.mean(2))
            attn_pre, post, comb = self._mixes(layer, "attn", h)
            x = self._norm(f"layers.{layer}.attn_norm", self._pre(h, pre))
            pieces, carry_states = [], []
            for local in range(len(tokens)):
                pieces.append(
                    self._attention(
                        layer, x[:, local:local + 1], start + local, shared[local]
                    )
                )
                if return_all_logits and layer in a.kv_source_layers:
                    carry_states.append(self._clone_tree(self.carry.get(layer, [])))
            attention = torch.cat(pieces, dim=1)
            if carry_states:
                self._verify_carries[layer] = carry_states
            h = self._post(attention, h, post, comb)
            pre, post, comb = self._mixes(layer, "ffn", h)
            x = self._norm(f"layers.{layer}.ffn_norm", self._pre(h, attn_pre))
            h = self._post(self._moe(layer, x), h, post, comb)
        h = self._norm("norm", self._pre(h, pre))
        head = self.store.get("head.weight")
        output = h[0] if return_all_logits else h[:, -1]
        if output.is_cuda and head.is_cuda:
            from sparklab.kernels.triton.dsv4.skinny import bf16_skinny_linear

            logits = bf16_skinny_linear(output, head, out_dtype=torch.float32)
        else:
            logits = F.linear(output.float(), head.float())
        if self.dspark is not None:
            self.dspark.store_target(aux, start)
        self.position += len(tokens)
        return logits


class DeepseekV41ForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self.args = config.dsv41_args
        self.speculative_tokens = int(getattr(config, "speculative_tokens", 0) or 0)
        self.decoder = None
        self.uid = None

    def prepare_for_weight_load(self, model_path, dummy=False):
        if dummy:
            raise ValueError("DeepSeek V4.1 disk research path requires real checkpoint weights")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        store = DiskWeights(model_path, device=torch.device("cuda", torch.cuda.current_device()), cache_bytes=CACHE_BYTES)
        self.decoder = Decoder(self.args, store, tokenizer)
        if self.speculative_tokens:
            self.decoder.enable_dspark(self.speculative_tokens)

    def state_dict(self, **kwargs):
        return {}

    def load_state_dict(self, state_dict, **kwargs):
        if state_dict:
            raise ValueError("DeepSeek V4.1 reads source weights through its bounded disk store")

    def forward(self):
        batch = get_global_ctx().batch
        if len(batch.reqs) != 1 or batch.reqs[0].mm_embeds is not None:
            raise ValueError("DeepSeek V4.1 research requires one text request")
        req = batch.reqs[0]
        positions = batch.positions.cpu().tolist()
        if self.uid != req.uid or positions[0] == 0:
            self.decoder.reset()
            self.uid = req.uid
        if positions != list(range(self.decoder.position, self.decoder.position + len(positions))):
            raise ValueError("DeepSeek V4.1 state requires consecutive tokens without prefix reuse")
        tokens = batch.input_ids.cpu().tolist()
        if len(tokens) > 1:
            return self.decoder.prefill(
                tokens, return_all_logits=batch.return_all_logits
            )
        outputs = []
        for token in tokens:
            result = self.decoder.step(token)
            if batch.return_all_logits:
                outputs.append(result)
        return torch.cat(outputs) if outputs else result

    def propose_mtp(self, batch, next_token):
        if self.decoder.dspark is None or batch.size != 1:
            return None
        return self.decoder.dspark.propose(next_token)

    def take_speculative_probs(self):
        return None

    def snapshot_speculative_state(self):
        return self.decoder.snapshot()

    def restore_speculative_state(self, snapshot):
        self.decoder.restore(snapshot)

    def commit_speculative_prefix(self, snapshot, length):
        self.decoder.commit_prefix(snapshot, length)
