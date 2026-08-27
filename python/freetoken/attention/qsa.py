"""Correctness-first Qwen4 QSA backend.

QSA compresses complete four-token groups only for *selection*: the main GQA
attention still reads every selected token's original K/V. Selection is kept
separate from MiniMax BSA because its pooling, scoring, grouping and budget are
all different. The selected physical rows are handed to FreeToken's paged
attention kernel, so only the index scoring/top-k path remains eager here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata


@dataclass
class QSAMetadata(BaseAttnMetadata):
    qo_indptr: tuple[int, ...]
    last_indices: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


def _plus_one_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (y * (1.0 + weight.float())).to(x.dtype)


class QSAAttnBackend(BaseAttnBackend):
    def __init__(self, config) -> None:
        from freetoken.kvcache.bsa_pool import BSAKVCache

        args = config.qwen4_exp_args
        if args is None or not args.qsa_layer_ids:
            raise ValueError("qsa backend requires Qwen4ExpArgs with QSA layers")
        self.config = config
        self.args = args
        self.kvcache = get_global_ctx().kv_cache
        if not isinstance(self.kvcache, BSAKVCache):
            raise TypeError(f"qsa backend needs indexed GQA storage, got {type(self.kvcache).__name__}")
        self.device = self.kvcache.device
        self._idx_slot = {lid: args.qsa_slot(lid) for lid in args.qsa_layer_ids}
        self.sm_scale = config.attn_sm_scale or config.head_dim**-0.5

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        lengths = [r.extend_len for r in reqs]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        batch.attn_metadata = QSAMetadata(
            qo_indptr=tuple(offsets),
            last_indices=torch.tensor(
                [x - 1 for x in offsets[1:]], dtype=torch.int32, device=self.device
            ),
        )

    def forward(
        self, q, k, v, layer_id: int, batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ):
        raise RuntimeError("Qwen4 QSA layers must call qsa_forward with index projections")

    def _selected_rows(
        self, index_q: torch.Tensor, raw_keys: torch.Tensor | None, physical_rows: torch.Tensor,
        visible: int, k_norm_weight: torch.Tensor, rotary,
    ) -> torch.Tensor:
        ratio = self.args.index_compress_ratio
        complete = visible // ratio
        # Until there are more complete blocks than the budget, QSA selects every
        # complete block and the uncompressed tail. Scoring/top-k can only permute
        # that same set, and attention is permutation invariant. This covers the
        # ordinary <=2051-token agent path and avoids index-K gather, pooling,
        # norm, RoPE, matmul and top-k entirely.
        if complete <= self.args.index_block_topk:
            return physical_rows[:visible]
        if raw_keys is None:
            raise RuntimeError("QSA raw index keys are required beyond the dense budget")
        if complete:
            pooled = raw_keys[: complete * ratio].view(complete, ratio, -1).float().mean(1)
            pooled = _plus_one_rms(pooled.to(raw_keys.dtype), k_norm_weight, self.config.rms_norm_eps)
            starts = torch.arange(
                0, complete * ratio, ratio, dtype=torch.int64, device=raw_keys.device
            )
            # RotaryEmbedding mutates both arguments. A clone for the unused query
            # avoids aliasing pooled into two in-place operands.
            _, pooled = rotary.forward(starts, pooled.clone(), pooled)
            score = torch.relu(index_q.float() @ pooled.float().T).sum(0)
            score = score / math.sqrt(self.args.index_head_dim)
            take = min(self.args.index_block_topk, complete)
            blocks = torch.topk(score, take, sorted=False).indices
            offsets = torch.arange(ratio, device=blocks.device)
            logical = (blocks[:, None] * ratio + offsets).flatten()
        else:
            logical = torch.empty(0, dtype=torch.long, device=raw_keys.device)
        tail = torch.arange(complete * ratio, visible, device=raw_keys.device)
        # HF applies the selected mask to the original chronological K/V axis.
        # Sort the packed equivalent so floating-point reduction order follows
        # that reference too (topk itself does not promise score order).
        logical = torch.sort(torch.cat((logical, tail)).to(torch.long)).values
        return physical_rows.index_select(0, logical)

    def qsa_forward(
        self, q, k, v, index_q, raw_index_k, k_norm_weight, rotary,
        layer_id: int, batch: Batch,
    ) -> torch.Tensor:
        md = batch.attn_metadata
        if not isinstance(md, QSAMetadata):
            raise TypeError(f"QSA metadata expected, got {type(md).__name__}")
        slot = self._idx_slot[layer_id]
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        self.kvcache.store_index_k(raw_index_k, batch.out_loc, slot)

        ctx = get_global_ctx()
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        selected: list[torch.Tensor] = []
        for req_idx, req in enumerate(reqs):
            q0, q1 = md.qo_indptr[req_idx : req_idx + 2]
            physical = ctx.page_table[req.table_idx, : req.device_len].to(torch.int32)
            largest_visible = req.cached_len + (q1 - q0)
            sparse = (
                largest_visible // self.args.index_compress_ratio
                > self.args.index_block_topk
            )
            raw = (
                self.kvcache.index_k_cache(slot).index_select(0, physical.to(torch.long))
                if sparse else None
            )
            for local, iq in enumerate(index_q[q0:q1]):
                selected.append(self._selected_rows(
                    iq, raw, physical, req.cached_len + local + 1,
                    k_norm_weight, rotary,
                ))

        from freetoken.kernel.triton.attention import paged_attention

        counts = torch.tensor(
            [x.numel() for x in selected], dtype=torch.int32, device=self.device
        )
        indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        indices = torch.cat(selected) if selected else counts.new_empty(0)
        q_to_req = torch.arange(q.shape[0], dtype=torch.int32, device=self.device)
        q_positions = counts.to(torch.int64) - 1
        k_cache = self.kvcache.k_cache(layer_id).view(-1, self.config.num_kv_heads, self.config.head_dim)
        v_cache = self.kvcache.v_cache(layer_id).view_as(k_cache)
        return paged_attention(
            q=q, k_cache=k_cache, v_cache=v_cache,
            indptr=indptr, indices=indices, q_to_req=q_to_req,
            q_positions=q_positions, sm_scale=self.sm_scale,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        if bs_list:
            raise RuntimeError(
                "Qwen4 QSA decode is currently eager; serve with --cuda-graph-max-bs 0"
            )

    def prepare_for_capture(self, batch: Batch) -> None:
        raise RuntimeError("Qwen4 QSA CUDA graph capture is not implemented")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise RuntimeError("Qwen4 QSA CUDA graph replay is not implemented")


__all__ = ["QSAAttnBackend", "QSAMetadata"]
