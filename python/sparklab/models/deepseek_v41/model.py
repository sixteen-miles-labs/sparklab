"""Native V4.1 text decoder with bounded disk weights and request-local state.

Math follows DeepSeek's MIT-licensed inference/model.py at
df42c109f1defefcbfcedbe7d905718a12266e40 (see LICENSE.deepseek).
This initial research path evaluates tokens eagerly, including prefill. It
prioritizes a bounded complete-model path over throughput or prefix reuse.
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
        self.reset()

    def reset(self):
        self.position = 0
        self.windows = {}
        self.compressed = {}
        self.keys = {}
        self.carry = {}

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

    def _moe(self, layer, x):
        a, store = self.args, self.store
        name = f"layers.{layer}.ffn"
        scores = F.linear(x.float(), store.get(name + ".gate.weight").float()) / a.gate_temp
        if a.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        elif a.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(-1)
        indices = (scores + store.get(name + ".gate.bias")).topk(a.n_activated_experts, dim=-1).indices
        weights = scores.gather(-1, indices)
        if a.norm_topk_prob and a.n_activated_experts > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * a.route_scale
        output = torch.zeros_like(x, dtype=torch.float32)
        # Upstream accumulates in expert-ID order, with weighting before w2.
        for slot in indices.reshape(-1).argsort().tolist():
            expert = int(indices.reshape(-1)[slot])
            output += self._expert(f"{name}.experts.{expert}", x, weights.reshape(-1)[slot])
        output += self._expert(name + ".shared_experts", x)
        return output.to(x.dtype)

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
        pre, post, comb = hc_split_sinkhorn(mixes.reshape(1, -1), store.get(name + "_scale"), store.get(name + "_base"),
                                           a.hc_mult, a.hc_sinkhorn_iters, a.hc_eps)
        return pre.view(1, 1, a.hc_mult), post.view(1, 1, a.hc_mult), comb.view(1, 1, a.hc_mult, a.hc_mult)

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
        for layer in range(a.n_layers):
            if self.layout and layer in self.layout.layer_ids:
                h = self._engram(layer, h, hashes)
            attn_pre, post, comb = self._mixes(layer, "attn", h)
            x = self._norm(f"layers.{layer}.attn_norm", self._pre(h, pre))
            h = self._post(self._attention(layer, x, pos, shared), h, post, comb)
            pre, post, comb = self._mixes(layer, "ffn", h)
            x = self._norm(f"layers.{layer}.ffn_norm", self._pre(h, attn_pre))
            h = self._post(self._moe(layer, x), h, post, comb)
        h = self._norm("norm", self._pre(h, pre))
        logits = F.linear(h[:, -1].float(), self.store.get("head.weight").float())
        self.position += 1
        return logits


class DeepseekV41ForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self.args = config.dsv41_args
        self.decoder = None
        self.uid = None

    def prepare_for_weight_load(self, model_path, dummy=False):
        if dummy:
            raise ValueError("DeepSeek V4.1 disk research path requires real checkpoint weights")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        store = DiskWeights(model_path, device=torch.device("cuda", torch.cuda.current_device()), cache_bytes=CACHE_BYTES)
        self.decoder = Decoder(self.args, store, tokenizer)

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
        outputs = []
        for token in batch.input_ids.cpu().tolist():
            result = self.decoder.step(token)
            if batch.return_all_logits:
                outputs.append(result)
        return torch.cat(outputs) if outputs else result
