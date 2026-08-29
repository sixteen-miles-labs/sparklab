"""NoPE latent MLA and learned KPool indexer for GLM-5.3-Flash."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sparklab.core import get_global_ctx
from sparklab.layers import BaseOP, LinearReplicated, RMSNorm

from .mlp import _linear


class _LayerNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6):
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class Glm5NextIndexer(BaseOP):
    def __init__(self, config):
        args = config.glm5_next_args
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.wq_b = LinearReplicated(
            args.q_lora_rank, self.n_heads * self.head_dim, has_bias=False
        )
        self.wk = LinearReplicated(args.hidden_size, self.head_dim, has_bias=False)
        self.k_norm = _LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = LinearReplicated(args.hidden_size, self.n_heads, has_bias=False)
        self.index_kpool_compress_ape = torch.empty(args.index_kpool, self.head_dim)
        self.index_kpool_compress_gate = torch.empty(self.head_dim, args.hidden_size)

    def compute(
        self, x: torch.Tensor, q_resid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = x.shape[0]
        q = self.wq_b.forward(q_resid).view(t, self.n_heads, self.head_dim)
        key = self.k_norm.forward(self.wk.forward(x))
        gate = F.linear(x, self.index_kpool_compress_gate)
        packed = torch.cat((key, gate), dim=-1)
        weights = self.weights_proj.forward(x).float() * (self.n_heads**-0.5)
        return q, packed, weights, self.index_kpool_compress_ape


class Glm5NextMLAAttention(BaseOP):
    def __init__(self, config, layer_id: int):
        args = config.glm5_next_args
        assert args is not None
        self.layer_id = layer_id
        self.indexer = Glm5NextIndexer(config)
        self.num_heads = args.num_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank

        self.q_a_proj = _linear(config.attn_quant, args.hidden_size, args.q_lora_rank)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=args.norm_eps)
        self.q_b_proj = _linear(
            config.attn_quant, args.q_lora_rank, self.num_heads * self.qk_head_dim
        )
        self.kv_a_proj_with_mqa = _linear(
            config.attn_quant, args.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)
        # kv_b is an absorbed BMM operand and is intentionally BF16 in the release.
        self.kv_b_proj = LinearReplicated(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            has_bias=False,
        )
        self.o_proj = _linear(
            config.attn_quant, self.num_heads * self.v_head_dim, args.hidden_size
        )
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w_uk is None:
            weight = self.kv_b_proj.weight.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            self._w_uk = weight[:, : self.qk_nope_head_dim].contiguous()
            self._w_uv = weight[:, self.qk_nope_head_dim :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        t = x.shape[0]
        w_uk, w_uv = self._kv_b()
        q_resid = self.q_a_layernorm.forward(self.q_a_proj.forward(x))
        q = self.q_b_proj.forward(q_resid).view(t, self.num_heads, self.qk_head_dim)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]
        compressed = self.kv_a_proj_with_mqa.forward(x)
        c_kv = self.kv_a_layernorm.forward(compressed[..., : self.kv_lora_rank])
        k_rope = compressed[..., self.kv_lora_rank :]
        q_absorbed = torch.bmm(
            q_nope.transpose(0, 1).contiguous(), w_uk
        ).transpose(0, 1)
        indexer_state = (
            self.indexer.compute(x, q_resid)
            if getattr(ctx.attn_backend, "dsa_enabled", False)
            else None
        )
        latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(),
            q_pe.contiguous(),
            c_kv.contiguous(),
            k_rope.contiguous(),
            self.layer_id,
            ctx.batch,
            indexer_qkw=indexer_state,
        )
        output = torch.bmm(
            latent.transpose(0, 1).contiguous(), w_uv
        ).transpose(0, 1)
        return self.o_proj.forward(output.reshape(t, self.num_heads * self.v_head_dim))


__all__ = ["Glm5NextIndexer", "Glm5NextMLAAttention"]
