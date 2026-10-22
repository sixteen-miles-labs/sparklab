"""Actual scheduler output drain must retire and train request-local rounds."""

from collections import deque
from types import SimpleNamespace
import time

import pytest
import torch

from sparklab.core import Batch
from sparklab.runtime.scheduler.scheduler import Scheduler
from sparklab.runtime.scheduler.telemetry import RuntimeTelemetry
from tests.runtime.engine.test_adaptive_dflash import engine_with_proposer
from tests.runtime.scheduler.test_abort_inflight_prefill import _setup, _launch_req


@pytest.mark.parametrize("depth", [0, 3])
@pytest.mark.parametrize("ending", ["continuing", "terminal", "aborted"])
@pytest.mark.parametrize("tracing", [False, True])
def test_scheduler_drain_retires_round_and_trains_after_handoff(depth, ending, tracing):
    pool, cm, tm, dm, _, sent, owner = _setup()
    count = 1 if depth == 0 else 2
    req = _launch_req(
        pool,
        cm,
        tm,
        torch.tensor([1, 2], dtype=torch.int32),
        output_len=20,
    )
    req.sampling_params.temperature = 0
    if ending == "terminal":
        owner.eos_token_ids = {20 + count - 1}
    engine, _ = engine_with_proposer(depth)
    owner.engine = engine
    owner._runtime_timing = tracing
    owner.runtime_metrics = RuntimeTelemetry(trace_capacity=16) if tracing else None
    owner._runtime_events = deque()
    dm.filter_reqs([req])
    current = engine.prepare_adaptive_round(req, torch.tensor([19], dtype=torch.int32))
    assert current.depth == depth
    current.started = time.monotonic() - 1
    current.accepted = count - 1
    policy = req.adaptive_policy
    # Proposal consumption and outputs stand in for the GPU forward only; all
    # scheduler drain, lifecycle, policy observation and next preparation are real.
    req.speculative_drafts = None
    req.aborted = ending == "aborted"
    events = []
    original_observe = policy.observe

    def observe(**kwargs):
        assert events == ["copy-fence", "reply-handoff"]
        return original_observe(**kwargs)

    policy.observe = observe

    def send(messages):
        assert events == ["copy-fence"]
        events.append("reply-handoff")
        sent.extend(messages)

    owner.send_result = send
    output = SimpleNamespace(
        next_tokens_cpu=torch.arange(20, 20 + count, dtype=torch.int32),
        token_counts=(count,),
        copy_done_event=SimpleNamespace(
            synchronize=lambda: events.append("copy-fence")
        ),
    )
    Scheduler._process_last_data(
        owner, (SimpleNamespace(batch=Batch(reqs=[req], phase="decode")), output)
    )
    assert req.adaptive_round is None
    assert policy.rounds == (0 if ending == "aborted" else 1)
    assert len(sent) == (0 if ending == "aborted" else count)
    if ending == "aborted":
        assert not any(policy.costs.values()) and not any(policy.acceptance)
    else:
        assert len(policy.costs[depth]) == 1 and policy.costs[depth][0] > 0
        if depth:
            assert [list(v) for v in policy.acceptance[:3]] == [[True], [False], []]
    if tracing:
        counters = owner.runtime_metrics.counters
        assert counters["adaptive_rounds"] == (0 if ending == "aborted" else 1)
        assert counters["adaptive_proposed"] == (0 if ending == "aborted" else depth)
        assert counters["adaptive_accepted"] == (
            0 if ending == "aborted" else count - 1
        )
        assert counters["adaptive_target_only_rounds"] == int(
            depth == 0 and ending != "aborted"
        )
        assert counters["adaptive_rejected_rounds"] == int(
            depth > 0 and ending != "aborted"
        )
        assert counters["adaptive_rejection_position_1"] == int(
            depth > 0 and ending != "aborted"
        )
    if ending == "continuing":
        # This previously failed with "previous adaptive round has not drained".
        following = engine.prepare_adaptive_round(
            req, torch.tensor([29], dtype=torch.int32)
        )
        assert following is not current and following.policy is policy
        assert following.depth == depth and policy.rounds == 1
        req.adaptive_round = None
        req.speculative_drafts = None
        owner._free_req_resources(req)
    cm.check_integrity()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="adaptive round CUDA timing gate"
)
def test_scheduler_drains_actual_round_events_before_observing_gpu_cost():
    pool, cm, tm, dm, _, _, owner = _setup()
    req = _launch_req(
        pool, cm, tm, torch.tensor([1, 2], dtype=torch.int32), output_len=20
    )
    req.sampling_params.temperature = 0
    engine, _ = engine_with_proposer(3)
    engine.device = torch.device("cuda")
    owner.engine = engine
    owner._runtime_timing = True
    owner.runtime_metrics = RuntimeTelemetry(trace_capacity=16)
    owner._runtime_events = deque()
    dm.filter_reqs([req])
    current = engine.prepare_adaptive_round(req, torch.tensor([19], dtype=torch.int32))
    # This is actual CUDA work and event ownership, not full model execution.
    work = torch.ones(8, device="cuda")
    work.add_(1)
    current.gpu_end.record()
    ready = torch.cuda.Event()
    ready.record()
    current.accepted = 1
    req.speculative_drafts = None
    output = SimpleNamespace(
        next_tokens_cpu=torch.tensor([20, 21], dtype=torch.int32),
        token_counts=(2,),
        copy_done_event=ready,
    )
    Scheduler._process_last_data(
        owner, (SimpleNamespace(batch=Batch(reqs=[req], phase="decode")), output)
    )
    assert req.adaptive_round is None and req.adaptive_policy.rounds == 1
    assert current.gpu_end.query()
    measured = list(owner.runtime_metrics.trace)[-1]
    assert measured["phase"] == "adaptive"
    assert 0 <= measured["proposal_gpu_seconds"] <= measured["round_gpu_seconds"]
    assert sum(owner.runtime_metrics.histograms["speculative_round_gpu"]) == 1
    assert (
        engine.prepare_adaptive_round(req, torch.tensor([29], dtype=torch.int32))
        is not current
    )
    req.adaptive_round = None
    req.speculative_drafts = None
    owner._free_req_resources(req)
    cm.check_integrity()
