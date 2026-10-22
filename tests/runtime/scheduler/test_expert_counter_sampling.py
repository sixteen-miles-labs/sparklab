"""Fixed tensor oracles and real CUDA ordering for live expert observations."""

import copy
from types import SimpleNamespace

import pytest
import torch

from sparklab.runtime.scheduler.scheduler import Scheduler
from sparklab.runtime.scheduler.telemetry import ExpertCounterSampler
from tests.runtime.scheduler.test_runtime_export import shell


def test_live_observation_never_replaces_a_newer_terminal_observation():
    owner = shell()
    owner.engine.stream = object()
    observed = {"observed_at": 1.0, "counters": {"active": 10}}
    owner._runtime_expert_sampler = SimpleNamespace(
        poll=lambda *args: observed,
        device=torch.empty(4, dtype=torch.int64),
        host=torch.empty(4, dtype=torch.int64),
        pending=None,
    )
    first = Scheduler._runtime_snapshot(owner, 2.0, {}, {"active": 20})
    assert first["expert_decode"] == {"observed_at": 2.0, "counters": {"active": 20}}
    second = Scheduler._runtime_snapshot(owner, 3.0, {})
    assert second["expert_decode"] == first["expert_decode"]
    observed = {"observed_at": 3.1, "counters": {"active": 30}}
    third = Scheduler._runtime_snapshot(owner, 4.0, {})
    assert third["expert_decode"] == observed
    assert first["expert_decode"]["counters"]["active"] == 20


def test_disabled_runtime_observation_does_not_poll_sampler():
    owner = shell()
    owner._runtime_timing = False

    def forbidden(*args):
        raise AssertionError("disabled telemetry polled a CUDA sample")

    owner._runtime_expert_sampler = SimpleNamespace(poll=forbidden)
    assert Scheduler._runtime_snapshot(owner, 1.0, {}) is None


def counter_cache(hybrid, empty=False):
    cache = SimpleNamespace(
        device=torch.device("cuda"),
        decode_target="hybrid" if hybrid else "gpu",
        collect_stats=True,
        lru_stats=torch.tensor(
            [[3, 1, 2], [7, 2, 4]], dtype=torch.int64, device="cuda"
        ),
        stat_active=torch.tensor(13, dtype=torch.int64, device="cuda"),
        stat_missing=torch.tensor(5, dtype=torch.int64, device="cuda"),
        stat_calls=torch.tensor(7, dtype=torch.int64, device="cuda"),
        stat_fetched=torch.tensor(3 if hybrid else 2, dtype=torch.int64, device="cuda"),
        prefill_hit_rows=11,
        prefill_total_rows=17,
        sparse_prefill_layers=2,
    )
    if empty:
        for name in (
            "lru_stats",
            "stat_active",
            "stat_missing",
            "stat_calls",
            "stat_fetched",
        ):
            getattr(cache, name).zero_()
    return cache


def guarded_poll(monkeypatch, sampler, cache, stream, now):
    original = torch.Tensor.tolist

    def host_only(tensor):
        assert tensor.device.type == "cpu", "sample converted a live GPU tensor"
        return original(tensor)

    def forbidden(*args, **kwargs):
        raise AssertionError("sample synchronized the device")

    with monkeypatch.context() as guards:
        guards.setattr(torch.cuda, "synchronize", forbidden)
        guards.setattr(torch.cuda.Event, "synchronize", forbidden)
        guards.setattr(torch.cuda.Stream, "synchronize", forbidden)
        guards.setattr(torch.Tensor, "item", forbidden)
        guards.setattr(torch.Tensor, "cpu", forbidden)
        guards.setattr(torch.Tensor, "tolist", host_only)
        return sampler.poll(cache, stream, now)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="live expert CUDA counter gate"
)
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_real_cuda_scalars_are_rate_limited_and_observations_remain_immutable(
    monkeypatch,
    hybrid,
    empty,
):
    cache = counter_cache(hybrid, empty)
    sampler = ExpertCounterSampler(cache)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    buffers = (sampler.device.data_ptr(), sampler.host.data_ptr(), id(sampler.event))
    assert sampler.device.numel() == sampler.host.numel() == 4
    assert sampler.host.is_pinned()
    assert guarded_poll(monkeypatch, sampler, cache, stream, 10.0) is None
    sampler.event.synchronize()  # test oracle only; production poll may never wait
    first = guarded_poll(monkeypatch, sampler, cache, stream, 10.1)
    active, missing, calls, fetched = (13, 5, 7, 3) if hybrid else (10, 3, 6, 2)
    if empty:
        active = missing = calls = fetched = 0
    assert first == {
        "observed_at": 10.0,
        "counters": {
            "active": active,
            "missing": missing,
            "calls": calls,
            "layer_calls": calls,
            "fetched": fetched,
            "miss_rate": missing / active if active else 0.0,
            "prefill_hit_rows": 11,
            "prefill_rows": 17,
            "sparse_prefill_layers": 2,
        },
    }
    saved = copy.deepcopy(first)
    with torch.cuda.stream(stream):
        for name in (
            "lru_stats",
            "stat_active",
            "stat_missing",
            "stat_calls",
            "stat_fetched",
        ):
            getattr(cache, name).mul_(2)
    assert guarded_poll(monkeypatch, sampler, cache, stream, 10.9) == first
    assert sampler.pending is None and sampler.last_submit == 10.0
    cache.prefill_total_rows = 99
    assert guarded_poll(monkeypatch, sampler, cache, stream, 11.0) == first
    cache.prefill_total_rows = 1000
    sampler.event.synchronize()
    second = guarded_poll(monkeypatch, sampler, cache, stream, 11.1)
    assert second["observed_at"] == 11.0
    assert second["counters"]["active"] == 2 * active
    assert second["counters"]["missing"] == 2 * missing
    assert second["counters"]["calls"] == 2 * calls
    assert second["counters"]["fetched"] == 2 * fetched
    assert second["counters"]["prefill_rows"] == 99
    assert first == saved and sampler.pending is None
    assert buffers == (
        sampler.device.data_ptr(),
        sampler.host.data_ptr(),
        id(sampler.event),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="live expert pending-copy gate"
)
def test_pending_real_copy_does_not_reuse_window_or_wait_for_busy_engine(monkeypatch):
    from tests.runtime.checkpoint.test_transfer_fence import HostGate

    cache = counter_cache(False)
    sampler = ExpertCounterSampler(cache)
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    progress = torch.ones(1, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache.lru_stats.mul_(1)
    with torch.cuda.stream(other):
        progress.add_(1)
    other.synchronize()
    sampler.poll(cache, stream, 0.0)
    sampler.event.synchronize()
    first = sampler.poll(cache, stream, 0.1)
    gate = HostGate()
    try:
        gate.wait(stream)
        with torch.cuda.stream(stream):
            cache.lru_stats.mul_(2)
        assert guarded_poll(monkeypatch, sampler, cache, stream, 1.0) == first
        assert not sampler.event.query()
        for now in (1.1, 2.0, 100.0):
            assert guarded_poll(monkeypatch, sampler, cache, stream, now) == first
            assert sampler.pending[0] == sampler.last_submit == 1.0
            assert not sampler.event.query()
        with torch.cuda.stream(other):
            progress.add_(1)
        other.synchronize()
        assert progress.item() == 3 and not sampler.event.query()
        assert not gate.expired.is_set()
        gate.release()
        sampler.event.synchronize()
        latest = guarded_poll(monkeypatch, sampler, cache, stream, 101.0)
        assert latest["observed_at"] == 1.0 and latest["counters"]["active"] == 20
        assert latest["counters"]["missing"] == 6 and latest["counters"]["calls"] == 12
        assert first["counters"]["active"] == 10
        assert sampler.pending[0] == 101.0
        sampler.event.synchronize()
        assert (
            guarded_poll(monkeypatch, sampler, cache, stream, 101.1)["observed_at"]
            == 101.0
        )
        assert sampler.pending is None
    finally:
        gate.release()
        stream.synchronize()
        gate.close()
    assert not gate.expired.is_set()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="live expert export integration gate"
)
def test_actual_scheduler_export_refreshes_between_terminal_replies():
    owner = shell()
    cache = counter_cache(False)
    cache.disk_source = None
    owner.engine.moe_offload_cache = cache
    owner.engine.stream = torch.cuda.Stream()
    owner.engine.stream.wait_stream(torch.cuda.current_stream())
    sampler = owner._runtime_expert_sampler = ExpertCounterSampler(cache)
    initial = Scheduler._runtime_snapshot(owner, 5.0, {})
    assert initial["expert_decode"] is None
    assert initial["retained_workspaces"]["expert_counter_sampling"] == {
        "device_bytes": 32,
        "pinned_host_bytes": 32,
        "pending": True,
    }
    sampler.event.synchronize()
    assert Scheduler._runtime_snapshot(owner, 5.2, {}) is None
    assert owner._runtime_expert_decode["observed_at"] == 5.0
    exported = Scheduler._runtime_snapshot(owner, 6.0, {})
    assert exported["expert_decode"]["counters"]["active"] == 10
    assert exported["expert_decode"]["counters"]["missing"] == 3
    sampler.event.synchronize()
