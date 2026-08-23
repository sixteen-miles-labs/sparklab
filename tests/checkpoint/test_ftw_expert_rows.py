from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.checkpoint.ftw import (
    FTWDiskExpertSource,
    FTWReader,
    FTWWriter,
    layer_bank_entry_name,
)


def _write_ftw(path, tensors):
    writer = FTWWriter(str(path), shard_limit=4096 * 3)
    for name, tensor in tensors:
        writer.add_tensor(name, tensor, kind="experts_bank")
    writer.finalize({"expert_bank_num_layers": 2, "quant_format": "test"})


def test_writer_syncs_and_evicts_each_completed_shard(tmp_path, monkeypatch):
    calls = []
    real_fsync = os.fsync
    real_fadvise = getattr(os, "posix_fadvise", None)

    def record_fsync(fd):
        calls.append(("fsync", fd))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    if real_fadvise is not None:
        def record_fadvise(fd, offset, length, advice):
            calls.append(("fadvise", fd, offset, length, advice))
            return real_fadvise(fd, offset, length, advice)

        monkeypatch.setattr(os, "posix_fadvise", record_fadvise)

    # Each aligned 4096-byte tensor fills one shard, so the second add rolls the first
    # and finalize completes the second.
    writer = FTWWriter(str(tmp_path), shard_limit=4096)
    writer.add_tensor("a", torch.zeros(4096, dtype=torch.uint8))
    writer.add_tensor("b", torch.ones(4096, dtype=torch.uint8))
    writer.finalize({})

    assert len([call for call in calls if call[0] == "fsync"]) == 2
    if real_fadvise is not None and hasattr(os, "POSIX_FADV_DONTNEED"):
        advice = [call for call in calls if call[0] == "fadvise"]
        assert len(advice) == 2
        assert all(call[2:] == (0, 0, os.POSIX_FADV_DONTNEED) for call in advice)


def test_per_layer_expert_rows_round_trip_across_shards(tmp_path):
    layers = []
    tensors = []
    for layer_id in range(2):
        value = torch.arange(5 * 777, dtype=torch.int32).view(5, 777) + layer_id * 100_000
        layers.append(value)
        tensors.append((layer_bank_entry_name("packed", layer_id), value))
    tensors.append(("gate_up_alpha", torch.ones(10, dtype=torch.float32)))
    _write_ftw(tmp_path, tensors)

    reader = FTWReader(str(tmp_path))
    try:
        index = reader.expert_row_descriptors(num_layers=2)
        assert len(index) == 2 * 5
        assert all(key[2] == "packed" for key in index)
        for layer_id in range(2):
            for expert_id in range(5):
                desc = index[(layer_id, expert_id, "packed")]
                assert desc.read_off % 4096 == 0
                assert desc.read_nbytes % 4096 == 0
                assert 0 <= desc.head_pad < 4096
                assert torch.equal(reader.read_expert_row(desc), layers[layer_id][expert_id])
    finally:
        reader.close()


def test_flat_expert_rows_round_trip(tmp_path):
    flat = torch.arange(2 * 3 * 513, dtype=torch.int16).view(6, 513)
    _write_ftw(tmp_path, [("weight", flat)])

    reader = FTWReader(str(tmp_path))
    try:
        index = reader.expert_row_descriptors(num_layers=2)
        assert len(index) == 6
        for layer_id in range(2):
            for expert_id in range(3):
                got = reader.read_expert_row(index[(layer_id, expert_id, "weight")])
                assert torch.equal(got, flat[layer_id * 3 + expert_id])
    finally:
        reader.close()


def test_expert_row_index_rejects_missing_layer(tmp_path):
    value = torch.zeros(2, 16, dtype=torch.uint8)
    _write_ftw(tmp_path, [(layer_bank_entry_name("packed", 0), value)])

    reader = FTWReader(str(tmp_path))
    try:
        with pytest.raises(ValueError, match=r"expected \[0, 1\]"):
            reader.expert_row_descriptors(num_layers=2)
    finally:
        reader.close()


def test_disk_expert_source_uses_bounded_lru(tmp_path):
    values = torch.arange(2 * 3 * 513, dtype=torch.int16).view(6, 513)
    _write_ftw(tmp_path, [("weight", values)])
    reader = FTWReader(str(tmp_path))
    staging = {"weight": SimpleNamespace(tensor=torch.empty(3, 513, dtype=torch.int16))}
    cache = {"weight": SimpleNamespace(tensor=torch.empty(2, 513, dtype=torch.int16))}
    source = FTWDiskExpertSource(
        reader,
        reader.expert_row_descriptors(num_layers=2),
        staging,
        cache,
    )
    try:
        source.stage(0, [0, 1])
        assert source.read_ops == 2
        assert torch.equal(staging["weight"].tensor[0], values[0])

        # Refresh (0, 0), then insert (1, 0): (0, 1) must be the LRU victim.
        source.stage(0, [0])
        source.stage(1, [0])
        source.stage(0, [1])
        stats = source.stats()
        assert stats["cache_capacity_entries"] == 2
        assert stats["cache_occupancy_entries"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 4
        assert stats["cache_evictions"] == 2
        assert source.read_ops == 4
        assert torch.equal(staging["weight"].tensor[1], values[1])
    finally:
        reader.close()


def test_disk_expert_source_coalesces_concurrent_request(tmp_path):
    values = torch.arange(2 * 3 * 513, dtype=torch.int16).view(6, 513)
    _write_ftw(tmp_path, [("weight", values)])
    reader = FTWReader(str(tmp_path))
    staging = {"weight": SimpleNamespace(tensor=torch.empty(3, 513, dtype=torch.int16))}
    cache = {"weight": SimpleNamespace(tensor=torch.empty(1, 513, dtype=torch.int16))}
    source = FTWDiskExpertSource(
        reader,
        reader.expert_row_descriptors(num_layers=2),
        staging,
        cache,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: source.stage(0, [2]), range(2)))
        assert source.read_ops == 1
        assert source.cache_misses == 1
        assert source.cache_hits == 1
        assert torch.equal(staging["weight"].tensor[2], values[2])
    finally:
        reader.close()


def test_prefill_bypass_preserves_decode_lru(tmp_path):
    values = torch.arange(2 * 3 * 513, dtype=torch.int16).view(6, 513)
    _write_ftw(tmp_path, [("weight", values)])
    reader = FTWReader(str(tmp_path))
    staging = {"weight": SimpleNamespace(tensor=torch.empty(3, 513, dtype=torch.int16))}
    cache = {"weight": SimpleNamespace(tensor=torch.empty(2, 513, dtype=torch.int16))}
    source = FTWDiskExpertSource(
        reader,
        reader.expert_row_descriptors(num_layers=2),
        staging,
        cache,
    )
    try:
        source.stage(0, [0, 1])  # decode admissions
        source.stage(1, [0, 1, 2], admit=False)  # dense prefill scan
        assert source.cache_bypasses == 3
        assert source.cache_evictions == 0
        assert source.stats()["cache_occupancy_entries"] == 2

        reads = source.read_ops
        source.stage(0, [0, 1])
        assert source.read_ops == reads
        assert source.cache_hits == 2
    finally:
        reader.close()
