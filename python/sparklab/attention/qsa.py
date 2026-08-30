"""Correctness-first Qwen4 QSA backend.

QSA compresses complete four-token groups only for *selection*: the main GQA
attention still reads every selected token's original K/V. Selection is kept
separate from MiniMax BSA because its pooling, scoring, grouping and budget are
all different. The selected physical rows are handed to SparkLab's paged
attention kernel, so only the index scoring/top-k path remains eager here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List

import torch
from sparklab.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata


@dataclass
class QSAMetadata(BaseAttnMetadata):
    qo_indptr: tuple[int, ...]
    last_indices: torch.Tensor
    q_to_req: torch.Tensor
    dense_indptr: torch.Tensor | None = None
    dense_indices: torch.Tensor | None = None
    dense_q_positions: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


def _plus_one_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (y * (1.0 + weight.float())).to(x.dtype)


class QSAAttnBackend(BaseAttnBackend):
    def __init__(self, config) -> None:
        from sparklab.runtime.kvcache.bsa_pool import BSAKVCache

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
        self._fused_selection = os.getenv(
            "SPARKLAB_DISABLE_QSA_FUSED_SELECTION", "0"
        ).lower() not in {"1", "true", "yes"}

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        lengths = [r.extend_len for r in reqs]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        # QSA packs a distinct selected-KV segment for every query token below,
        # rather than one segment per request. ``paged_attention`` uses this tensor
        # to index ``indptr``, so it must map query N to segment N even when several
        # queries belong to the same request.
        q_to_req = torch.arange(
            offsets[-1], dtype=torch.int32, device=self.device
        )
        dense = all(
            (req.cached_len + req.extend_len) // self.args.index_compress_ratio
            <= self.args.index_block_topk
            for req in reqs
        )
        dense_indptr = dense_indices = dense_q_positions = None
        if dense:
            # Before the sparse threshold QSA selection is layer-independent: every
            # visible physical row is selected. Build this packed metadata once per
            # batch rather than repeating Python list construction and GPU tensor ops
            # in all twelve QSA layers.
            ctx = get_global_ctx()
            selected = []
            for req in reqs:
                physical = ctx.page_table[req.table_idx, : req.device_len].to(torch.int32)
                selected.extend(
                    physical[: req.cached_len + local + 1]
                    for local in range(req.extend_len)
                )
            counts = torch.tensor(
                [rows.numel() for rows in selected], dtype=torch.int32, device=self.device
            )
            dense_indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            dense_indices = torch.cat(selected) if selected else counts.new_empty(0)
            dense_q_positions = counts.to(torch.int64) - 1

        batch.attn_metadata = QSAMetadata(
            qo_indptr=tuple(offsets),
            last_indices=torch.tensor(
                [x - 1 for x in offsets[1:]], dtype=torch.int32, device=self.device
            ),
            q_to_req=q_to_req,
            dense_indptr=dense_indptr,
            dense_indices=dense_indices,
            dense_q_positions=dense_q_positions,
        )

    @staticmethod
    def needs_index_query(batch: Batch) -> bool:
        md = batch.attn_metadata
        if not isinstance(md, QSAMetadata):
            raise TypeError(f"QSA metadata expected, got {type(md).__name__}")
        return md.dense_indices is None

    def forward(
        self, q, k, v, layer_id: int, batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ):
        raise RuntimeError("Qwen4 QSA layers must call qsa_forward with index projections")

    def _selected_rows(
        self, index_q: torch.Tensor, pooled_keys: torch.Tensor | None,
        physical_rows: torch.Tensor, visible: int,
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
        if pooled_keys is None:
            raise RuntimeError("QSA pooled index keys are required beyond the dense budget")
        if complete:
            pooled = pooled_keys[:complete]
            if index_q.is_cuda and self._fused_selection:
                from sparklab.kernels.triton.qwen4 import qsa_index_scores

                score = qsa_index_scores(index_q.contiguous(), pooled.contiguous())
            else:
                score = torch.relu(index_q.float() @ pooled.float().T).sum(0)
                score = score / math.sqrt(self.args.index_head_dim)
            take = min(self.args.index_block_topk, complete)
            blocks = torch.topk(score, take, sorted=False).indices
            if (
                blocks.is_cuda
                and take == self.args.index_block_topk
                and self._fused_selection
            ):
                from sparklab.kernels.triton.qwen4 import qsa_expand_selected_rows

                return qsa_expand_selected_rows(
                    blocks.contiguous(),
                    physical_rows.contiguous(),
                    ratio=ratio,
                    visible=visible,
                )
            offsets = getattr(self, "_pool_offsets", None)
            if offsets is None or offsets.device != blocks.device:
                offsets = torch.arange(ratio, device=blocks.device)
                self._pool_offsets = offsets
            logical = (blocks[:, None] * ratio + offsets).flatten()
        else:
            logical = torch.empty(0, dtype=torch.long, device=pooled_keys.device)
        tail = torch.arange(complete * ratio, visible, device=pooled_keys.device)
        # HF applies the selected mask to the original chronological K/V axis.
        # Sort the packed equivalent so floating-point reduction order follows
        # that reference too (topk itself does not promise score order).
        logical = torch.sort(torch.cat((logical, tail)).to(torch.long)).values
        return physical_rows.index_select(0, logical)

    def _pool_completed_keys(self, slot: int, reqs, k_norm_weight, rotary) -> None:
        """Replace each completed four-token group's last raw key with its fixed pool.

        Selection never needs the group's individual raw index keys after completion.
        Reusing the last physical row therefore adds no KV memory while avoiding an
        O(context) gather, mean, normalization, and rotary transform on every later
        decode token. The last row is also the request-owned side of a prefix-cache
        boundary, so two requests completing the same partial shared group cannot
        overwrite raw prefix state needed by one another.
        """
        ratio = self.args.index_compress_ratio
        ctx = get_global_ctx()
        cache = None
        offsets = getattr(self, "_pool_offsets", None)
        for req in reqs:
            first_group = req.cached_len // ratio
            complete_groups = (req.cached_len + req.extend_len) // ratio
            if first_group >= complete_groups:
                continue
            if cache is None:
                cache = self.kvcache.index_k_cache(slot)
                if offsets is None:
                    offsets = torch.arange(
                        ratio, dtype=torch.long, device=self.device
                    )
                    self._pool_offsets = offsets
            starts = torch.arange(
                first_group,
                complete_groups,
                dtype=torch.long,
                device=self.device,
            ) * ratio
            # Gather only the newly completed groups. Converting the complete
            # page-table prefix to int64 here made this once-per-four-token path
            # grow with context length even though it consumes four rows/group.
            logical = starts[:, None] + offsets[None, :]
            rows = ctx.page_table[req.table_idx, logical].to(torch.long)
            pooled = cache.index_select(0, rows.flatten()).view(
                starts.numel(), ratio, -1
            ).float().mean(1)
            pooled = _plus_one_rms(
                pooled.to(cache.dtype), k_norm_weight, self.config.rms_norm_eps
            )
            # RotaryEmbedding mutates both arguments. The query clone is unused but
            # keeps its in-place write from aliasing the pooled key.
            _, pooled = rotary.forward(starts, pooled.clone(), pooled)
            cache[rows[:, -1]] = pooled

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
        self._pool_completed_keys(slot, reqs, k_norm_weight, rotary)

        from sparklab.kernels.triton.attention import paged_attention

        k_cache = self.kvcache.k_cache(layer_id).view(
            -1, self.config.num_kv_heads, self.config.head_dim
        )
        v_cache = self.kvcache.v_cache(layer_id).view_as(k_cache)
        if md.dense_indices is not None:
            assert md.dense_indptr is not None and md.dense_q_positions is not None
            return paged_attention(
                q=q, k_cache=k_cache, v_cache=v_cache,
                indptr=md.dense_indptr, indices=md.dense_indices,
                q_to_req=md.q_to_req, q_positions=md.dense_q_positions,
                sm_scale=self.sm_scale,
            )
        if index_q is None:
            raise RuntimeError("QSA index queries are required beyond the dense budget")

        selected: list[torch.Tensor] = []
        for req_idx, req in enumerate(reqs):
            q0, q1 = md.qo_indptr[req_idx : req_idx + 2]
            physical = ctx.page_table[req.table_idx, : req.device_len].to(torch.int32)
            largest_visible = req.cached_len + (q1 - q0)
            sparse = (
                largest_visible // self.args.index_compress_ratio
                > self.args.index_block_topk
            )
            pooled_end = (
                largest_visible // self.args.index_compress_ratio
                * self.args.index_compress_ratio
            )
            pooled = (
                self.kvcache.index_k_cache(slot).index_select(
                    0,
                    physical[
                        self.args.index_compress_ratio - 1
                        :pooled_end
                        :self.args.index_compress_ratio
                    ].to(torch.long),
                )
                if sparse else None
            )
            for local, iq in enumerate(index_q[q0:q1]):
                selected.append(self._selected_rows(
                    iq, pooled, physical, req.cached_len + local + 1,
                ))

        counts = torch.tensor(
            [x.numel() for x in selected], dtype=torch.int32, device=self.device
        )
        indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        indices = torch.cat(selected) if selected else counts.new_empty(0)
        q_positions = counts.to(torch.int64) - 1
        return paged_attention(
            q=q, k_cache=k_cache, v_cache=v_cache,
            indptr=indptr, indices=indices, q_to_req=md.q_to_req,
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
