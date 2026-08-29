"""GLM-5.2 DSA (IndexShare lightning indexer) addressing, mixed into the backend.

Same contract as ``dsv4_indexer.py``: the indexer's weights/projections stay in the
MODEL (it hands per-token q/k/weights in); this mixin owns the ADDRESSING -- scoring
against the paged index-key slab, causal top-k selection, and the map from selected
positions to physical rows. ``-1`` is a gather-only sentinel (the sparse-attention
kernel masks it, it never reaches a store), and device-side counts bound every kernel
so CUDA-graph replays track the live position, not the staged width.

Selection SEMANTICS are shared with DSV4 outright: GLM's token-granular top-k is
exactly ``dsv4_indexer.IndexerBackendMixin.indexer_select_{prefill,decode}`` at
``ratio=1, offset=0`` (a "block" of one token), so this mixin inherits them rather
than reimplementing. Scoring differs (token keys off the row snapshot, DSV4 scores
compressed blocks), so that part is GLM's own fused kernel.
"""

from __future__ import annotations

import torch

from .dsv4_indexer import IndexerBackendMixin


class DSAIndexerMixin(IndexerBackendMixin):
    def dsa_decode_scores(
        self, q_idx: torch.Tensor, w: torch.Tensor, slot: int,
        rows: torch.Tensor, kvlen: torch.Tensor,
    ) -> torch.Tensor:
        """Fused head-reduced logits ``[bs, W]`` fp32 for a decode step: keys gathered
        off the row snapshot inside the kernel, live length read from device memory,
        ``-inf`` past it (so the shared select's -inf ordering holds)."""
        from sparklab.kernels.triton.glm_dsa_sparse import glm_dsa_decode_logits

        return glm_dsa_decode_logits(
            q_idx, w * self.index_scale, self.kvcache.index_k_cache(slot), rows, kvlen
        )

    def dsa_prefill_logits(
        self, q_idx: torch.Tensor, k_all: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Head-reduced logits ``[m, kv_len]`` fp32 over a dense key slab (dsv4's
        fused ``indexer_logits`` -- no per-head transient)."""
        from sparklab.kernels.triton.dsv4.indexer import indexer_logits

        return indexer_logits(
            q_idx.unsqueeze(0), k_all.unsqueeze(0),
            (w * self.index_scale).to(torch.float32).unsqueeze(0),
        )[0]

    @staticmethod
    def dsa_map_rows(picks: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Selected POSITIONS -> physical rows, ``-1`` sentinel passed through.

        ``picks`` [..., K] int (from the shared select fns), ``rows`` broadcastable
        position-ordered physical rows. Returns int32."""
        sel = rows.gather(-1, picks.clamp_min(0).long()).to(torch.int32)
        return torch.where(picks < 0, sel.new_full((), -1), sel)


__all__ = ["DSAIndexerMixin"]
