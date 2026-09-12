"""Checkpoint-native DSpark drafter for the V4.1 disk decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ops import apply_rotary_emb, dequant, fp8_roundtrip, hc_split_sinkhorn


class DSparkDraft:
    """Three-stage, parallel five-token drafter stored under ``mtp.*``."""

    def __init__(self, decoder, steps: int):
        self.decoder = decoder
        self.args = decoder.args
        self.store = decoder.store
        self.steps = min(int(steps), self.args.dspark_block_size)
        self.windows = {}
        self.last_position = -1
        self.last_confidence = None

    def reset(self):
        self.windows = {}
        self.last_position = -1
        self.last_confidence = None

    def snapshot(self):
        return {
            "windows": {stage: value.clone() for stage, value in self.windows.items()},
            "last_position": self.last_position,
        }

    def restore(self, snapshot):
        self.windows = snapshot["windows"]
        self.last_position = snapshot["last_position"]
        self.last_confidence = None

    def commit_prefix(self, snapshot, current, start: int, length: int):
        self.restore(snapshot)
        device = next(iter(current["windows"].values())).device
        positions = torch.arange(
            start, start + length, dtype=torch.long, device=device
        )
        for stage, window in self.windows.items():
            window[positions % self.args.window_size] = current["windows"][stage][
                positions % self.args.window_size
            ]
        self.last_position = start + length - 1

    def _linear(self, name, x):
        return self.decoder._linear(name, x)

    def _norm(self, name, x):
        return self.decoder._norm(name, x)

    def _rope(self, x, positions, inverse=False):
        rd = self.args.rope_head_dim
        apply_rotary_emb(
            x[..., -rd:], self.decoder.freqs[False][positions], inverse
        )
        return x

    def _mixes(self, name, kind, h):
        a = self.args
        root = f"{name}.hc_{kind}"
        x = h.flatten(2).float()
        mixes = F.linear(x, self.store.get(root + "_fn").float())
        mixes *= torch.rsqrt(x.square().mean(-1, keepdim=True) + a.norm_eps)
        pre, post, comb = hc_split_sinkhorn(
            mixes.reshape(-1, mixes.shape[-1]),
            self.store.get(root + "_scale"),
            self.store.get(root + "_base"),
            a.hc_mult,
            a.hc_sinkhorn_iters,
            a.hc_eps,
        )
        lead = mixes.shape[:-1]
        return (
            pre.reshape(*lead, a.hc_mult),
            post.reshape(*lead, a.hc_mult),
            comb.reshape(*lead, a.hc_mult, a.hc_mult),
        )

    def store_target(self, aux, start_pos: int):
        if len(aux) != len(self.args.dspark_target_layer_ids):
            raise RuntimeError("V4.1 DSpark target features are incomplete")
        main = self._norm(
            "mtp.0.main_norm",
            self._linear("mtp.0.main_proj", torch.cat(aux, dim=-1)),
        )
        positions = torch.arange(
            start_pos,
            start_pos + main.shape[1],
            dtype=torch.long,
            device=main.device,
        )
        for stage in range(self.args.n_mtp_layers):
            name = f"mtp.{stage}.attn"
            kv = self._norm(name + ".kv_norm", self._linear(name + ".wkv", main))
            kv = fp8_roundtrip(self._rope(kv, positions)).reshape(-1, self.args.head_dim)
            window = self.windows.get(stage)
            if window is None:
                window = self.windows[stage] = torch.empty(
                    self.args.window_size,
                    self.args.head_dim,
                    dtype=kv.dtype,
                    device=kv.device,
                )
            window[positions % self.args.window_size] = kv
        self.last_position = start_pos + main.shape[1] - 1

    def _visible(self, stage):
        window = self.windows[stage]
        pos, width = self.last_position, self.args.window_size
        if pos < width - 1:
            return window[: pos + 1]
        slot = (pos + 1) % width
        return torch.cat((window[slot:], window[:slot]))

    def _attention(self, stage, x, positions):
        a = self.args
        name = f"mtp.{stage}.attn"
        qr = self._norm(name + ".q_norm", self._linear(name + ".wq_a", x))
        q = self._linear(name + ".wq_b", qr).view(
            1, self.steps, a.n_heads, a.head_dim
        )
        q = self._rope(q, positions)[0]
        kv = self._norm(name + ".kv_norm", self._linear(name + ".wkv", x))
        kv = fp8_roundtrip(self._rope(kv, positions)).reshape(self.steps, a.head_dim)
        visible = torch.cat((self._visible(stage), kv))
        scores = torch.einsum("shd,td->sht", q.float(), visible.float())
        scores *= a.head_dim**-0.5
        sink = self.store.get(name + ".attn_sink").float()
        scores = torch.cat((scores, sink.view(1, -1, 1).expand(self.steps, -1, -1)), -1)
        weights = scores.softmax(-1)[..., :-1]
        output = torch.einsum("sht,td->shd", weights, visible.float()).to(x.dtype)
        output = self._rope(output.unsqueeze(0), positions, inverse=True)
        weight = self.store.get(name + ".wo_a.weight")
        if name + ".wo_a.scale" in self.store.metadata:
            weight = dequant(weight, self.store.get(name + ".wo_a.scale")).to(torch.bfloat16)
        weight = weight.view(a.o_groups, a.o_lora_rank, -1)
        output = torch.einsum(
            "bsgd,grd->bsgr",
            output.view(1, self.steps, a.o_groups, -1),
            weight.to(output.dtype),
        )
        return self._linear(name + ".wo_b", output.flatten(2))

    @torch.inference_mode()
    def propose(self, anchor: torch.Tensor):
        if self.last_position < 0 or self.steps <= 0:
            return None
        a = self.args
        anchor = anchor.reshape(1).long()
        ids = torch.cat((
            anchor.cpu(),
            torch.full(
                (self.steps - 1,), a.dspark_noise_token_id, dtype=torch.long
            ),
        )).view(1, -1)
        h = self.store.rows("embed.weight", ids).to(torch.bfloat16)
        h = h.unsqueeze(2).repeat(1, 1, a.hc_mult, 1)
        pre = torch.zeros(
            (1, self.steps, a.hc_mult), device=h.device, dtype=torch.float32
        )
        pre[..., 0] = 1
        positions = torch.arange(
            self.last_position + 1,
            self.last_position + 1 + self.steps,
            dtype=torch.long,
            device=h.device,
        )
        for stage in range(a.n_mtp_layers):
            name = f"mtp.{stage}"
            residual = h
            attn_pre, post, comb = self._mixes(name, "attn", h)
            x = self._norm(name + ".attn_norm", self.decoder._pre(h, pre))
            h = self.decoder._post(
                self._attention(stage, x, positions), residual, post, comb
            )
            residual = h
            pre, post, comb = self._mixes(name, "ffn", h)
            x = self._norm(name + ".ffn_norm", self.decoder._pre(h, attn_pre))
            output = self.decoder._moe(
                a.n_layers + stage,
                x,
                name=name + ".ffn",
                n_activated_experts=a.dspark_n_activated_experts,
            )
            h = self.decoder._post(output, residual, post, comb)

        # A block returns its FFN pre-mix for the following sublayer.  The
        # terminal head consumes that value directly; there is no hc_head.
        hidden = self.decoder._pre(h, pre)
        hidden = self._norm(f"mtp.{a.n_mtp_layers - 1}.norm", hidden)[0]
        from sparklab.kernels.triton.dsv4.skinny import bf16_skinny_linear

        base_logits = bf16_skinny_linear(
            hidden, self.store.get("head.weight"), out_dtype=torch.float32
        )
        final = a.n_mtp_layers - 1
        markov_embed = self.store.get(f"mtp.{final}.markov_head.embed.weight")
        markov_head = self.store.get(f"mtp.{final}.markov_head.head.weight")
        previous = anchor.to(h.device)
        drafts = []
        markov_features = []
        for row in range(self.steps):
            markov = F.embedding(previous, markov_embed)
            markov_features.append(markov.squeeze(0))
            bias = bf16_skinny_linear(markov, markov_head, out_dtype=torch.float32)
            token = (base_logits[row : row + 1] + bias).argmax(-1)
            drafts.append(token.squeeze(0).to(torch.int32))
            previous = token
        features = torch.cat((hidden.float(), torch.stack(markov_features).float()), -1)
        confidence = F.linear(
            features,
            self.store.get(f"mtp.{final}.confidence_head.proj.weight").float(),
        ).squeeze(-1)
        self.last_confidence = confidence
        return torch.stack(drafts)


__all__ = ["DSparkDraft"]
