"""Real Gloo agreement under deliberately different per-rank clocks/costs."""

from datetime import timedelta
import json
import multiprocessing
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from sparklab.runtime.distributed.info import DistributedInfo
from sparklab.runtime.scheduler.policy import Decision, SchedulingPolicy


def inputs(**changes):
    result = dict(
        now=10.0,
        pending=True,
        decoders=True,
        oldest_queue_age=0.0,
        output_age=0.0,
        prefill_budget=512,
        alignment=64,
        prefill_seconds=0.01,
        decode_seconds=0.005,
    )
    result.update(changes)
    return result


def forbidden(**kwargs):
    raise AssertionError("peer must not make a local policy choice")


@pytest.mark.parametrize("mode,size", [("baseline", 2), ("latency", 1)])
def test_local_path_has_no_collectives(mode, size):
    policy = SchedulingPolicy(mode)
    decision, service = policy.choose_coordinated(
        tp_info=DistributedInfo(0, size), cpu_group=object(), **inputs()
    )
    assert decision.phase == "prefill"
    assert service == (0.1 if mode == "latency" else None)


def test_missing_cpu_group_fails():
    with pytest.raises(RuntimeError, match="CPU process group"):
        SchedulingPolicy("latency").choose_coordinated(
            tp_info=DistributedInfo(0, 2), **inputs()
        )


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    reports = []
    try:
        cases = [
            (inputs(), inputs(output_age=0.09, prefill_seconds=0.2), "prefill", 512),
            (inputs(output_age=0.09, prefill_seconds=0.2), inputs(), "decode", 0),
            (
                inputs(oldest_queue_age=2.0, output_age=0.09, prefill_seconds=0.25),
                inputs(),
                "prefill",
                64,
            ),
            (
                inputs(pending=False, decoders=False),
                inputs(pending=False, decoders=False),
                "idle",
                0,
            ),
            (inputs(pending=False), inputs(pending=False), "decode", 0),
        ]
        for primary, peer, phase, budget in cases:
            policy = SchedulingPolicy("latency")
            if rank:
                policy.choose = forbidden
            decision, service = policy.choose_coordinated(
                tp_info=DistributedInfo(rank, 2),
                cpu_group=dist.group.WORLD,
                **(primary if rank == 0 else peer),
            )
            assert decision.phase == phase and decision.token_budget == budget
            assert service == max(0.001, 0.1 - primary["output_age"])
            reports.append(
                [decision.phase, decision.token_budget, decision.reason, service]
            )
        # Every rank receives source failure instead of hanging in broadcast.
        policy = SchedulingPolicy("latency")
        policy.choose = forbidden
        with pytest.raises(RuntimeError, match="Primary TP scheduler"):
            policy.choose_coordinated(
                tp_info=DistributedInfo(rank, 2), cpu_group=dist.group.WORLD, **inputs()
            )
        # A valid source choice incompatible with a peer fails on both ranks.
        for peer in (inputs(prefill_budget=128), inputs(pending=False)):
            policy = SchedulingPolicy("latency")
            with pytest.raises(RuntimeError, match="runnable envelope"):
                policy.choose_coordinated(
                    tp_info=DistributedInfo(rank, 2),
                    cpu_group=dist.group.WORLD,
                    **(inputs() if rank == 0 else peer),
                )
        # Both rejection paths leave the CPU group usable.
        policy = SchedulingPolicy("latency")
        decision, _ = policy.choose_coordinated(
            tp_info=DistributedInfo(rank, 2), cpu_group=dist.group.WORLD, **inputs()
        )
        assert decision.phase == "prefill"
        Path(output, f"rank-{rank}.json").write_text(json.dumps(reports))
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_agreement(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    children = [
        ctx.Process(
            target=_worker, args=(rank, f"file://{tmp_path}/init", str(tmp_path))
        )
        for rank in range(2)
    ]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(timeout=35)
        assert [child.exitcode for child in children] == [0, 0]
        assert (tmp_path / "rank-0.json").read_text() == (
            tmp_path / "rank-1.json"
        ).read_text()
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)


def test_scheduler_uses_coordinated_choice(monkeypatch):
    from tests.runtime.scheduler.test_dispatch_observations import selection_case
    from sparklab.runtime.scheduler.scheduler import Scheduler

    with selection_case(monkeypatch, mode="latency") as (owner, _, owned):
        received = []

        def coordinated(**kwargs):
            received.append(kwargs)
            return Decision("decode", 0, "output_age"), 0.02

        owner.scheduling_policy.choose = forbidden
        owner.scheduling_policy.choose_coordinated = coordinated
        result = Scheduler._schedule_next_batch(owner)
        assert result.batch.phase == "decode" and result.batch.reqs == owned
        assert len(received) == 1 and received[0]["pending"]
        assert list(owner.runtime_metrics.trace)[-1]["reason"] == "output_age"
