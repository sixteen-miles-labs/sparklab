"""P0 — global-paged SWA pool (Option A) allocator semantics.

CPU-only. Covers the pieces that must match sglang's SWATokenToKVPoolAllocator:
the dense full->swa mapping with the slot-0 sentinel, alloc_swa + mapping-based
translate, free_swa idempotence over the sentinel (no double-free), exhaustion,
and the rebuild reset (mapping + free-list re-sized atomically with the buffer).
"""
from __future__ import annotations

import pytest
import torch


def _patch_tp(monkeypatch) -> None:
    from sparklab.runtime.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "sparklab.runtime.kvcache.hybrid_swa_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _specs():
    from sparklab.models.config import KVCacheGroupSpec

    return (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8, sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8, sliding_window=4),
    )


def _paged_pool(num_full=16, num_swa=8):
    from sparklab.runtime.kvcache.hybrid_swa_pool import HybridSWAKVCache

    return HybridSWAKVCache(
        groups=_specs(),
        num_layers=2,
        num_full_pages=num_full,  # page_size=1 -> full_num_tokens == num_full
        page_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        num_swa_tokens=num_swa,
    )


def _allocated_mask(pool) -> torch.Tensor:
    """full slots whose mapping is a live swa slot (> 0)."""
    return pool.full_to_swa_index_mapping[: pool.full_num_tokens] > 0


def test_construction_mapping_and_freelist(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)

    assert pool.swa_paged
    # mapping: full_num_tokens + page_size zeros, trailing -1 sentinel for last_loc == -1.
    assert pool.full_to_swa_index_mapping.numel() == 16 + 1 + 1
    assert pool.full_to_swa_index_mapping.dtype == torch.int64
    assert int(pool.full_to_swa_index_mapping[-1]) == -1
    assert torch.all(pool.full_to_swa_index_mapping[:-1] == 0)
    # slot 0 reserved as the "no live swa slot" sentinel -> 7 allocatable of 8.
    assert pool.swa_available_size() == 7


def test_alloc_swa_writes_mapping_and_translate_reads_it(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)

    full = torch.tensor([0, 3, 5], dtype=torch.int32)
    pool.alloc_swa(full)
    assert pool.swa_available_size() == 7 - 3

    swa = pool.translate_loc_from_full_to_swa(full)
    assert swa.dtype == torch.int32
    # every mapped slot is a distinct, in-range, non-sentinel swa slot.
    assert torch.all(swa >= 1) and torch.all(swa < 8)
    assert len(set(swa.tolist())) == 3
    # unallocated full slots translate to the 0 sentinel.
    assert int(pool.translate_loc_from_full_to_swa(torch.tensor([7], dtype=torch.int32))[0]) == 0

    # slot conservation: free + live-mapped == total allocatable, always.
    assert pool.swa_available_size() + int(_allocated_mask(pool).sum()) == 7


def test_free_swa_is_idempotent_over_sentinel(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)

    full = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    pool.alloc_swa(full)
    assert pool.swa_available_size() == 3

    pool.free_swa(torch.tensor([0, 1], dtype=torch.int32))
    assert pool.swa_available_size() == 5  # 2 returned
    assert int(pool.full_to_swa_index_mapping[0]) == 0
    assert int(pool.full_to_swa_index_mapping[1]) == 0

    # freeing the same (now-sentinel) slots again must be a no-op: filter > 0, no double-free.
    pool.free_swa(torch.tensor([0, 1], dtype=torch.int32))
    assert pool.swa_available_size() == 5
    assert pool.swa_available_size() + int(_allocated_mask(pool).sum()) == 7


def test_free_then_realloc_reuses_slots_no_leak(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)

    pool.alloc_swa(torch.arange(7, dtype=torch.int32))  # exhaust all 7 allocatable
    assert pool.swa_available_size() == 0
    pool.free_swa(torch.arange(7, dtype=torch.int32))
    assert pool.swa_available_size() == 7
    assert torch.all(pool.full_to_swa_index_mapping[:-1] == 0)
    # can fully re-allocate after freeing -> no leak.
    pool.alloc_swa(torch.tensor([10, 11, 12, 13, 14, 15, 9], dtype=torch.int32))
    assert pool.swa_available_size() == 0


def test_alloc_swa_exhaustion_raises(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)
    with pytest.raises(RuntimeError, match="SWA pool exhausted"):
        pool.alloc_swa(torch.arange(8, dtype=torch.int32))  # only 7 allocatable


def test_rebuild_resets_mapping_and_freelist(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)
    pool.alloc_swa(torch.tensor([0, 1, 2], dtype=torch.int32))
    assert pool.swa_available_size() == 4

    pool.rebuild(num_full_pages=32, num_swa_tokens=12)

    assert pool.full_num_tokens == 32
    assert pool.swa_num_tokens == 12
    assert pool.full_to_swa_index_mapping.numel() == 32 + 1 + 1
    assert int(pool.full_to_swa_index_mapping[-1]) == -1
    assert torch.all(pool.full_to_swa_index_mapping[:-1] == 0)  # all stale mappings cleared
    assert pool.swa_available_size() == 11  # fresh free-list at new size, no leak
    # allocator still works at the new geometry.
    pool.alloc_swa(torch.tensor([30, 31], dtype=torch.int32))
    assert pool.swa_available_size() == 9
