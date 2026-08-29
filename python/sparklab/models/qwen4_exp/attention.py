from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sparklab.core import get_global_ctx
from sparklab.runtime.distributed import get_tp_info
from sparklab.layers import (
    BaseOP,
    GemmaPlusOneRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
)
from sparklab.layers.rotary import get_rope
from sparklab.utils import nvtx_annotate

if TYPE_CHECKING:
    from sparklab.models.config import ModelConfig


class Qwen4ExpAttention(BaseOP):
    """Gated GQA plus Qwen Sparse Attention's pooled-key indexer."""

    def __init__(self, config: ModelConfig, layer_id: int):
        if get_tp_info().size != 1:
            raise NotImplementedError("Qwen4-Exp currently supports TP=1")
        args = config.qwen4_exp_args
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_dim = self.num_q * self.head_dim
        self.kv_dim = self.num_kv * self.head_dim
        self._split = [2 * self.q_dim, self.kv_dim, self.kv_dim]
        self.qkv_proj = LinearColParallelMerged(config.hidden_size, self._split, has_bias=False)
        self.q_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = LinearReplicated(self.q_dim, config.hidden_size, has_bias=False)

        self.index_n_heads = args.index_n_heads
        self.index_dim = args.index_head_dim
        self.index_q_dim = self.index_n_heads * self.index_dim
        self.index_qk_proj = LinearReplicated(
            config.hidden_size, self.index_q_dim + self.index_dim, has_bias=False
        )
        self.index_q_norm = GemmaPlusOneRMSNorm(self.index_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaPlusOneRMSNorm(self.index_dim, eps=config.rms_norm_eps)
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
        )
        self.index_rotary = get_rope(
            head_dim=self.index_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
        )

    @nvtx_annotate("QSA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qg, k, v = self.qkv_proj.forward(x).split(self._split, dim=-1)
        qg = qg.view(-1, self.num_q, 2 * self.head_dim)
        q, gate = qg.chunk(2, dim=-1)
        q, gate = q.contiguous(), gate.contiguous().view(-1, self.q_dim)
        k = k.contiguous().view(-1, self.num_kv, self.head_dim)
        v = v.contiguous()
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)
        q_flat, k_flat = self.rotary.forward(
            ctx.batch.positions, q.view(-1, self.q_dim), k.view(-1, self.kv_dim)
        )

        iq, raw_ik = self.index_qk_proj.forward(x).split(
            [self.index_q_dim, self.index_dim], dim=-1
        )
        iq = iq.contiguous().view(-1, self.index_n_heads, self.index_dim)
        raw_ik = raw_ik.contiguous()
        self.index_q_norm.forward_inplace(iq)
        # Only queries are normalized/rotated here. QSA first averages raw key
        # groups, then normalizes and rotates the pooled key at the group start.
        iq_flat, _ = self.index_rotary.forward(
            ctx.batch.positions,
            iq.view(-1, self.index_q_dim),
            raw_ik.clone(),
        )
        out = ctx.attn_backend.qsa_forward(
            q_flat.view(-1, self.num_q, self.head_dim),
            k_flat, v,
            iq_flat.view(-1, self.index_n_heads, self.index_dim),
            raw_ik,
            self.index_k_norm.weight,
            self.index_rotary,
            self.layer_id,
            ctx.batch,
        )
        out = out.reshape(-1, self.q_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(out)


__all__ = ["Qwen4ExpAttention"]
