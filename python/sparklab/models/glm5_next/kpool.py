"""Pure KPool compression and token-selection primitives for GLM-5.3."""

from __future__ import annotations

import torch


def pool_index_states(
    packed: torch.Tensor, ape: torch.Tensor, pool_size: int
) -> torch.Tensor:
    """Build learned keys for every complete pool in ``packed``.

    ``packed`` is ``[tokens, key_dim + gate_dim]`` and both dimensions equal
    ``ape.shape[-1]``. Incomplete trailing tokens never form a candidate pool;
    they are appended verbatim by :func:`select_kpool_tokens`.
    """
    dim = ape.shape[-1]
    if packed.shape[-1] != 2 * dim:
        raise ValueError(f"packed KPool width must be {2 * dim}, got {packed.shape[-1]}")
    complete = packed.shape[0] // pool_size
    if complete == 0:
        return packed.new_empty((0, dim))
    state = packed[: complete * pool_size].view(complete, pool_size, 2 * dim)
    keys, gates = state.split(dim, dim=-1)
    logits = gates.float() + ape.float().view(1, pool_size, dim)
    probability = torch.softmax(logits, dim=1).to(keys.dtype)
    return (probability * keys).sum(dim=1)


def _index_scores(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if q.is_cuda:
        from sparklab.kernels.triton.dsv4.indexer import indexer_logits

        return indexer_logits(q.unsqueeze(0), keys.unsqueeze(0), weights.unsqueeze(0))[0]
    per_head = torch.einsum("mhd,pd->mhp", q.float(), keys.float()).relu_()
    return (per_head * weights.float().unsqueeze(-1)).sum(dim=1)


def select_kpool_tokens(
    q: torch.Tensor,
    weights: torch.Tensor,
    packed: torch.Tensor,
    ape: torch.Tensor,
    visible: torch.Tensor,
    *,
    token_topk: int,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select logical token positions for each query.

    The first ``token_topk`` columns contain expanded complete pools and the final
    ``pool_size-1`` columns reserve the always-visible incomplete tail.  Entries
    after each row's returned count are ``-1``. Short rows use chronological
    identity selection, which is exactly equivalent to selecting all pools.
    """
    m = q.shape[0]
    width = token_topk + pool_size - 1
    out = torch.full((m, width), -1, dtype=torch.int64, device=q.device)
    visible = visible.to(device=q.device, dtype=torch.int64)
    dense = visible <= token_topk
    if bool(dense.any()):
        sequence = torch.arange(width, device=q.device).view(1, width)
        dense_rows = dense.nonzero(as_tuple=False).flatten()
        dense_values = sequence.expand(dense_rows.numel(), -1)
        dense_visible = visible.index_select(0, dense_rows).view(-1, 1)
        out[dense_rows] = torch.where(dense_values < dense_visible, dense_values, -1)

    sparse_rows = (~dense).nonzero(as_tuple=False).flatten()
    if sparse_rows.numel():
        keys = pool_index_states(packed, ape, pool_size)
        pool_budget = token_topk // pool_size
        choose = min(pool_budget, keys.shape[0])
        if choose:
            qs = q.index_select(0, sparse_rows)
            ws = weights.index_select(0, sparse_rows)
            scores = _index_scores(qs, keys, ws)
            live_pools = visible.index_select(0, sparse_rows).div(pool_size, rounding_mode="floor")
            pool_ids = torch.arange(keys.shape[0], device=q.device).view(1, -1)
            scores.masked_fill_(pool_ids >= live_pools.view(-1, 1), torch.finfo(scores.dtype).min)
            picks = scores.topk(choose, dim=-1).indices
            valid = picks < live_pools.view(-1, 1)
            raw = picks.unsqueeze(-1) * pool_size + torch.arange(pool_size, device=q.device)
            raw = raw.masked_fill(~valid.unsqueeze(-1), -1).flatten(1)
            out[sparse_rows, : raw.shape[1]] = raw

        full_selected = torch.minimum(
            visible.index_select(0, sparse_rows).div(pool_size, rounding_mode="floor"),
            torch.tensor(pool_budget, device=q.device),
        )
        tail_count = visible.index_select(0, sparse_rows).remainder(pool_size)
        tail_start = visible.index_select(0, sparse_rows) - tail_count
        offsets = torch.arange(pool_size - 1, device=q.device)
        tail = tail_start[:, None] + offsets[None, :]
        tail = tail.masked_fill(offsets[None, :] >= tail_count[:, None], -1)
        columns = full_selected[:, None] * pool_size + offsets[None, :]
        row_grid = sparse_rows[:, None].expand_as(columns)
        valid_tail = offsets[None, :] < tail_count[:, None]
        out[row_grid[valid_tail], columns[valid_tail]] = tail[valid_tail]

    counts = torch.where(
        dense,
        visible,
        torch.minimum(
            visible.div(pool_size, rounding_mode="floor"),
            torch.tensor(token_topk // pool_size, device=q.device),
        )
        * pool_size
        + visible.remainder(pool_size),
    )
    return out.to(torch.int32), counts.to(torch.int32)


__all__ = ["pool_index_states", "select_kpool_tokens"]
