"""Real admission and selection, with CPU preparation and an independent clock."""

from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sparklab.core import SamplingParams
from sparklab.message import UserMsg
from sparklab.runtime.scheduler.policy import SchedulingPolicy
from sparklab.runtime.scheduler.scheduler import Scheduler
from sparklab.runtime.scheduler.telemetry import RuntimeTelemetry
from tests.runtime.scheduler.test_abort_inflight_prefill import _setup, _launch_req


@contextmanager
def selection_case(monkeypatch, *, mode="baseline", active=True, pending=True):
    pool, cm, tm, dm, pm, _, owner = _setup()
    cm.rebuild(6, cm.page_table)
    clock = [10.0]
    monkeypatch.setattr(
        "sparklab.runtime.scheduler.scheduler.time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    owner.config.max_running_req = 4
    owner.prefill_budget = 512
    owner.scheduling_policy = SchedulingPolicy(mode)
    owner.runtime_metrics = RuntimeTelemetry(trace_capacity=8, clock=lambda: clock[0])
    owner._runtime_events = deque()
    owner._poll_runtime_timings = lambda: None
    owner._report_prompt_admissions = lambda batch: Scheduler._report_prompt_admissions(
        owner, batch
    )
    owner._schedule_decode_batch = lambda budget: Scheduler._schedule_decode_batch(
        owner, budget
    )
    owner.engine = SimpleNamespace(adaptive_dflash=None)
    prepared, owned = [], []
    if active:
        req = _launch_req(
            pool, cm, tm, torch.tensor([1, 2], dtype=torch.int32), output_len=4
        )
        req.first_dispatch_at = 8.0
        req.last_output_ready_at = 9.0
        owned.append(req)
        dm.filter_reqs([req])
    if pending:
        pm.add_one_req(
            UserMsg(
                2,
                torch.tensor([10, 11, 12], dtype=torch.int32),
                SamplingParams(max_tokens=2),
            )
        )
        pm.pending_list[0].queued_at = 0.0 if active else 8.0

    def prepare(batch):
        cm.allocate_paged(batch.reqs)
        batch.padded_reqs = batch.reqs
        prepared.extend(batch.reqs)
        clock[0] += 0.5
        return SimpleNamespace(batch=batch)

    owner._prepare_batch = prepare
    try:
        yield owner, clock, owned
    finally:
        pm.pending_list.clear()
        dm.running_reqs.clear()
        for req in {id(r): r for r in owned + prepared}.values():
            req.aborted = True
            owner._free_req_resources(req)
        cm.check_integrity()


@pytest.mark.parametrize("mode", ["baseline", "latency"])
def test_decode_fallback_reports_unavailable_prefill_without_changing_admission(
    monkeypatch, mode
):
    with selection_case(monkeypatch, mode=mode) as (owner, clock, owned):
        result = Scheduler._schedule_next_batch(owner)
        assert result.batch.phase == "decode" and result.batch.reqs == owned
        assert len(owner.prefill_manager.pending_list) == 1
        row = list(owner.runtime_metrics.trace)[-1]
        assert row["reason"] == "prefill_unavailable"
        assert row["policy_reason"] == (
            "baseline" if mode == "baseline" else "prefill_service"
        )
        assert row["queued_before"] == row["queued"] == 1
        assert row["selection_at"] == 10.0 and row["time"] == 10.5
        assert row["active_output_age_seconds"] == 1.0
        assert row["oldest_queue_age_seconds"] == 10.0
        assert row["tokens"] == row["requests"] == row["active_requests"] == 1
        assert owner.runtime_metrics.counters["prefill_unavailable_dispatches"] == 1


@pytest.mark.parametrize("mode", ["baseline", "latency"])
def test_admitted_prompt_does_not_erase_the_queue_peak_or_selection_age(
    monkeypatch, mode
):
    with selection_case(monkeypatch, mode=mode, active=False) as (owner, clock, owned):
        result = Scheduler._schedule_next_batch(owner)
        assert (
            result.batch.phase == "prefill" and not owner.prefill_manager.pending_list
        )
        row = list(owner.runtime_metrics.trace)[-1]
        assert row["queued_before"] == 1 and row["queued"] == 0
        assert row["oldest_queue_age_seconds"] == 2.0
        assert row["active_output_age_seconds"] is None
        assert row["tokens"] == 3 and row["token_budget"] == 512
        assert owner.runtime_metrics.counters["queue_depth_peak"] == 1
        assert sum(owner.runtime_metrics.histograms["active_output_age"]) == 0
        assert sum(owner.runtime_metrics.histograms["oldest_queue_age"]) == 1


@pytest.mark.parametrize(
    "ready,first,age",
    [(0.0, 5.0, 10.0), (8.0, 0.0, 2.0), (None, 0.0, 10.0), (None, None, None)],
)
def test_output_age_preserves_zero_timestamps_and_unknown_observations(
    monkeypatch, ready, first, age
):
    with selection_case(monkeypatch, pending=False) as (owner, clock, owned):
        owned[0].last_output_ready_at = ready
        owned[0].first_dispatch_at = first
        result = Scheduler._schedule_next_batch(owner)
        assert result.batch.phase == "decode"
        row = list(owner.runtime_metrics.trace)[-1]
        assert row["reason"] == "decode_only"
        assert row["active_output_age_seconds"] == age
        assert row["oldest_queue_age_seconds"] is None
        assert sum(owner.runtime_metrics.histograms["active_output_age"]) == int(
            age is not None
        )
