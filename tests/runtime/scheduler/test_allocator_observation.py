"""Allocator lifetime evidence is separate from whole-model or UMA accounting."""

import copy

import pytest
import torch

from sparklab.runtime.scheduler.scheduler import Scheduler
from sparklab.runtime.scheduler.telemetry import allocator_snapshot
from tests.runtime.scheduler.test_runtime_export import shell


def test_disabled_and_cpu_export_do_not_query_the_cuda_allocator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unavailable or disabled observation touched CUDA")

    monkeypatch.setattr(torch.cuda, "memory_stats", forbidden)
    owner = shell()
    owner.device = torch.device("cuda")
    owner._runtime_timing = False
    assert Scheduler._runtime_snapshot(owner, 1.0, {}) is None
    owner._runtime_timing = True
    owner.device = torch.device("cpu")
    snapshot = Scheduler._runtime_snapshot(owner, 1.0, {})
    assert snapshot["resources"] == {
        "gpu_allocated_bytes": None,
        "gpu_reserved_bytes": None,
    }


def test_allocator_pair_is_rate_limited_independent_and_does_not_mutate_input(
    monkeypatch,
):
    values = {
        "allocated_bytes.all.current": 17,
        "reserved_bytes.all.current": 64,
        "unexported_tensor_count": 1000,
    }
    reads = []

    def observe(device):
        reads.append(device)
        return values.copy()

    monkeypatch.setattr(torch.cuda, "memory_stats", observe)
    owner = shell()
    owner.device = torch.device("cuda")
    resources = {"gpu_reserved_bytes": 900, "running_requests": 1}
    first = Scheduler._runtime_snapshot(owner, 1.0, resources)
    assert first["resources"] == {
        "gpu_reserved_bytes": 64,
        "gpu_allocated_bytes": 17,
        "running_requests": 1,
    }
    assert resources == {"gpu_reserved_bytes": 900, "running_requests": 1}
    values["allocated_bytes.all.current"] = 3
    assert Scheduler._runtime_snapshot(owner, 1.999, resources) is None
    assert len(reads) == 1
    second = Scheduler._runtime_snapshot(owner, 2.0, resources)
    assert len(reads) == 2 and second["resources"]["gpu_allocated_bytes"] == 3
    assert first["resources"]["gpu_allocated_bytes"] == 17


def test_missing_allocator_statistics_remain_unknown(monkeypatch):
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: {})
    assert allocator_snapshot(torch.device("cuda")) == {
        "gpu_allocated_bytes": None,
        "gpu_reserved_bytes": None,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA allocator lifetime gate"
)
def test_real_cuda_allocated_bytes_follow_lifetime_without_device_waits(monkeypatch):
    owner = shell()
    owner.device = torch.device("cuda")
    size = 1 << 20
    before = torch.cuda.memory_allocated(owner.device)
    buffer = torch.empty(size, dtype=torch.uint8, device=owner.device)
    expected = torch.cuda.memory_allocated(owner.device)
    assert expected == before + size
    reserved = torch.cuda.memory_reserved(owner.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("allocator observation synchronized or copied GPU data")

    with monkeypatch.context() as guard:
        guard.setattr(torch.cuda, "synchronize", forbidden)
        guard.setattr(torch.cuda, "mem_get_info", forbidden)
        guard.setattr(torch.Tensor, "cpu", forbidden)
        guard.setattr(torch.Tensor, "item", forbidden)
        guard.setattr(torch.Tensor, "tolist", forbidden)
        first = Scheduler._runtime_snapshot(owner, 1.0, {})
        prior = copy.deepcopy(first["resources"])
        assert prior["gpu_allocated_bytes"] == expected
        assert prior["gpu_reserved_bytes"] == reserved
        del buffer
        second = Scheduler._runtime_snapshot(owner, 2.0, {})
        assert second["resources"]["gpu_allocated_bytes"] == before
        assert second["resources"]["gpu_reserved_bytes"] >= before
        assert first["resources"] == prior
