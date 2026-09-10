"""Batched causal QSA selection, preserving the single-query reduction and ties."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .qwen4 import _qsa_index_scores_kernel


@triton.jit
def _dense_rows(physical, indptr, output, cached_len, block: tl.constexpr):
    query = tl.program_id(0)
    offsets = tl.arange(0, block)
    visible = cached_len + query + 1
    start = tl.load(indptr + query)
    rows = tl.load(physical + offsets, offsets < visible, other=0)
    tl.store(output + start + offsets, rows, offsets < visible)


@triton.jit
def _sparse_rows(
    blocks, physical, indptr, output, query_start, cached_len,
    stride_blocks, topk: tl.constexpr, ratio: tl.constexpr,
):
    local = tl.program_id(0)
    query = query_start + local
    visible = cached_len + query + 1
    start = tl.load(indptr + query)
    offsets = tl.arange(0, topk)
    selected = tl.load(blocks + local * stride_blocks + offsets).to(tl.int32)
    selected = tl.sort(selected, descending=False)
    token = tl.arange(0, ratio)
    logical = selected[:, None] * ratio + token[None, :]
    rows = tl.load(physical + logical)
    tl.store(output + start + offsets[:, None] * ratio + token[None, :], rows)
    tail_start = visible // ratio * ratio
    tail = tl.load(physical + tail_start + token, token < visible % ratio, other=0)
    tl.store(output + start + topk * ratio + token, tail, token < visible % ratio)


def qsa_prefill_indices(
    query: torch.Tensor,
    pooled_keys: torch.Tensor,
    physical_rows: torch.Tensor,
    *,
    cached_len: int,
    ratio: int,
    topk: int,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, list[int]]:
    """Pack chronological physical rows for a contiguous span of query tokens.

    Scores use the decode kernel's FP32 reduction, with a causal mask per query.
    Boundary ties fall back to the original one-dimensional top-k: padding scores
    with -inf can otherwise change which equally scored blocks PyTorch selects.
    Chunking bounds the number of query rows in score/top-k scratch memory.
    """
    if not (query.is_cuda and pooled_keys.is_cuda and physical_rows.is_cuda):
        raise ValueError("QSA prefill selection requires CUDA tensors")
    if query.ndim != 3 or pooled_keys.ndim != 2 or physical_rows.ndim != 1:
        raise ValueError("expected query [tokens, heads, dim], keys [blocks, dim], rows [tokens]")
    if query.device != pooled_keys.device or query.device != physical_rows.device:
        raise ValueError("QSA inputs must share a CUDA device")
    if query.shape[-1] != pooled_keys.shape[-1]:
        raise ValueError("QSA query/key dimensions must match")
    if not all(x.is_contiguous() for x in (query, pooled_keys, physical_rows)):
        raise ValueError("QSA prefill selection requires contiguous inputs")
    if cached_len < 0 or chunk_size <= 0:
        raise ValueError("cached_len must be nonnegative and chunk_size positive")
    if any(x <= 0 or x & (x - 1) for x in (ratio, topk)):
        raise ValueError("QSA ratio and topk must be positive powers of two")
    tokens, heads, dim = query.shape
    if cached_len + tokens > physical_rows.numel():
        raise ValueError("physical rows do not cover the query span")
    if (cached_len + tokens) // ratio > pooled_keys.shape[0]:
        raise ValueError("pooled keys do not cover the query span")
    counts = [min(v // ratio, topk) * ratio + v % ratio
              for v in range(cached_len + 1, cached_len + tokens + 1)]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    indptr = torch.tensor(offsets, dtype=torch.int64, device=query.device)
    output = torch.empty(offsets[-1], dtype=physical_rows.dtype, device=query.device)
    dense = min(tokens, max(0, (topk + 1) * ratio - 1 - cached_len))
    if dense:
        _dense_rows[(dense,)](
            physical_rows, indptr, output, cached_len,
            block=triton.next_power_of_2((topk + 1) * ratio - 1),
        )
    for start in range(dense, tokens, chunk_size):
        stop = min(tokens, start + chunk_size)
        # Do not score keys that are invisible to every query in this chunk.
        key_count = (cached_len + stop) // ratio
        scores = torch.empty((stop - start, key_count), dtype=torch.float32, device=query.device)
        _qsa_index_scores_kernel[(key_count, stop - start)](
            query[start:stop], pooled_keys, scores,
            query.stride(1), pooled_keys.stride(0), key_count,
            heads=heads, dim=dim, block_d=triton.next_power_of_2(dim),
            scale=dim ** -0.5, stride_qq=query.stride(0),
            cached_len=cached_len + start, ratio=ratio, batched=True, num_warps=4,
        )
        best = torch.topk(scores, topk + 1, dim=-1, sorted=True)
        blocks = best.indices[:, :topk].contiguous()
        tied = (best.values[:, topk - 1] == best.values[:, topk]).nonzero().flatten().tolist()
        for local in tied:
            complete = (cached_len + start + local + 1) // ratio
            blocks[local] = torch.topk(scores[local, :complete], topk, sorted=False).indices
        _sparse_rows[(stop - start,)](
            blocks, physical_rows, indptr, output, start, cached_len,
            blocks.stride(0), topk=topk, ratio=ratio, num_warps=8,
        )
    return output, counts
