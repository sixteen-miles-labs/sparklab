"""MiniMax-M3 attention: GQA with per-head Gemma q/k norms, partial NeoX RoPE, and
(on the trailing layers) the block-sparse lightning-indexer branch.

The indexer's weights live HERE (``index_qk_proj`` + per-head norms); scoring,
top-k block selection and the sparse attend live in the ``m3_sparse`` backend
(``attention/m3_sparse.py``), the same model/backend split as GLM's DSA. Sparse
layers hand the backend this token's (index_q, index_k) via ``bsa_forward``; the
backend caches the keys in the pool's index slab and selects per-KV-head top-k
128-token blocks. The leading dense layers go through the generic
``forward(q, k, v)`` contract, which the backend serves with a wrapped FULL
backend -- and the SPARKLAB_M3_SPARSE=0 ablation builds no indexer at all, so
every layer takes that path under a plain FULL backend.

Index q/k get the SAME per-head Gemma (1+w) norm + partial NeoX rope as the main
q/k (vLLM reference: ``index_rotary_emb = self.rotary_emb``); norms run before
rope, per head, in fp32 (the fused reference kernel's order).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sparklab.core import get_global_ctx
from sparklab.runtime.distributed import get_tp_info
from sparklab.layers import BaseOP, GemmaPlusOneRMSNorm
from sparklab.layers.rotary import get_rope
from sparklab.utils import nvtx_annotate

from .mlp import make_proj

if TYPE_CHECKING:
    import torch

    from sparklab.models.config import ModelConfig


class MiniMaxM3Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.m3_args
        assert get_tp_info().size == 1, "MiniMax-M3 currently supports TP=1 only"
        self.layer_id = layer_id
        self.num_qo_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        self.head_dim = args.head_dim
        self.qo_attn_dim = self.num_qo_heads * self.head_dim
        self.kv_attn_dim = self.num_kv_heads * self.head_dim

        quant = config.attn_quant
        self.qkv_proj = make_proj(
            quant, args.hidden_size, self.qo_attn_dim + 2 * self.kv_attn_dim
        )
        self.o_proj = make_proj(quant, self.qo_attn_dim, args.hidden_size)

        # Per-head Gemma (1+w) q/k norms (qk_norm_type == "per_head").
        self.q_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=args.norm_eps)

        # Partial NeoX rope (rotary_dim 64 of 128); one shared instance (get_rope
        # caches) serves the main q/k AND the indexer q/k, like the reference.
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=args.rotary_dim,
            max_position=args.max_position,
            base=args.rope_theta,
        )

        self.is_sparse = args.is_sparse_layer(layer_id)
        if self.is_sparse:
            self.num_index_heads = args.num_index_heads
            self.index_dim = args.index_dim
            self.index_q_dim = self.num_index_heads * self.index_dim
            # Merged [index_q | index_k] projection (one shared index KEY head).
            self.index_qk_proj = make_proj(
                quant, args.hidden_size, self.index_q_dim + self.index_dim
            )
            self.index_q_norm = GemmaPlusOneRMSNorm(self.index_dim, eps=args.norm_eps)
            self.index_k_norm = GemmaPlusOneRMSNorm(self.index_dim, eps=args.norm_eps)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        # split() returns strided views into the merged qkv buffer; make each part
        # contiguous (the norm kernel then runs in place on the copy) -- v must be
        # contiguous for the KV-cache store kernel regardless.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)

        if self.is_sparse:
            iqk = self.index_qk_proj.forward(x)
            del x
            iq, ik = iqk.split([self.index_q_dim, self.index_dim], dim=-1)
            iq = iq.contiguous()
            ik = ik.contiguous()
            self.index_q_norm.forward_inplace(
                iq.view(-1, self.num_index_heads, self.index_dim)
            )
            self.index_k_norm.forward_inplace(ik.view(-1, 1, self.index_dim))
            iq, ik = self.rotary.forward(ctx.batch.positions, iq, ik)
            o = ctx.attn_backend.bsa_forward(
                q.view(-1, self.num_qo_heads, self.head_dim),
                k, v,
                iq.view(-1, self.num_index_heads, self.index_dim),
                ik,
                self.layer_id,
                ctx.batch,
            )
        else:
            del x
            o = ctx.attn_backend.forward(
                q.view(-1, self.num_qo_heads, self.head_dim), k, v,
                self.layer_id, ctx.batch,
            )
        return self.o_proj.forward(o.reshape(-1, self.qo_attn_dim))


__all__ = ["MiniMaxM3Attention"]
