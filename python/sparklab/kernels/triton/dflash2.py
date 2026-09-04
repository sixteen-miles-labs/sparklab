"""Small fused kernels for native DFlash2 decoding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _greedy_selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    tokens_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """Walk one dependent K-way lattice without a kernel launch per slot."""
    offsets = tl.arange(0, top_k)
    previous = 0
    for slot in range(slots):
        base = slot * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        best = tl.max(scores, axis=0)
        # Match torch.argmax's first-index tie break.
        index = tl.min(tl.where(scores == best, offsets, top_k), axis=0)
        tl.store(tokens_ptr + slot, tl.load(candidate_ptr + base + index))
        previous = index


def greedy_selector_walk(
    candidate_ids: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """Return the greedy token path for ``scores[slot, predecessor, candidate]``."""
    slots, top_k = candidate_ids.shape
    if top_k <= 0 or top_k & (top_k - 1):
        raise ValueError("DFlash2 selector top_k must be a power of two")
    tokens = torch.empty(slots, dtype=candidate_ids.dtype, device=scores.device)
    _greedy_selector_walk_kernel[(1,)](
        scores.contiguous(),
        candidate_ids.contiguous(),
        tokens,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
    return tokens


__all__ = ["greedy_selector_walk"]
