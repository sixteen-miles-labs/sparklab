"""Kimi K3 gated NoPE MLA with latent-KV cache weight absorption."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sparklab.core import get_global_ctx
from sparklab.layers import BaseOP, LinearReplicated, RMSNorm

from .ops import kimi_linear

if TYPE_CHECKING:
    from sparklab.models.config import ModelConfig


class KimiMLAAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.kimi_k3_args
        assert args is not None
        self.layer_id = layer_id
        self.num_heads = config.num_qo_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.scaling = self.q_head_dim**-0.5

        self.q_a_proj = kimi_linear(config.attn_quant, config.hidden_size, self.q_lora_rank)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = kimi_linear(
            config.attn_quant, self.q_lora_rank, self.num_heads * self.q_head_dim
        )
        self.kv_a_proj_with_mqa = kimi_linear(
            config.attn_quant,
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = LinearReplicated(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            has_bias=False,
        )
        self.g_proj = (
            kimi_linear(config.attn_quant, config.hidden_size, self.num_heads * self.v_head_dim)
            if args.mla_output_gate
            else None
        )
        self.o_proj = kimi_linear(
            config.attn_quant, self.num_heads * self.v_head_dim, config.hidden_size
        )
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            self._w_uk = w[:, : self.qk_nope_head_dim].contiguous()
            self._w_uv = w[:, self.qk_nope_head_dim :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        tokens = x.shape[0]
        w_uk, w_uv = self._kv_b()
        q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.view(tokens, self.num_heads, self.q_head_dim)
        q_nope, q_rope = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv, k_rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm.forward(c_kv)

        # K3's ``mla_use_nope`` is literal: these 64 dimensions are retained in
        # the latent cache geometry but are not passed through RoPE.
        q_rope = q_rope.contiguous()
        q_absorbed = torch.bmm(q_nope.transpose(0, 1).contiguous(), w_uk).transpose(0, 1)
        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(),
            q_rope.contiguous(),
            c_kv.contiguous(),
            k_rope.contiguous(),
            self.layer_id,
            ctx.batch,
        )
        out = torch.bmm(o_latent.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)
        out = out.reshape(tokens, self.num_heads * self.v_head_dim)
        if self.g_proj is not None:
            out = out * self.g_proj.forward(x).sigmoid()
        return self.o_proj.forward(out)


__all__ = ["KimiMLAAttention"]
