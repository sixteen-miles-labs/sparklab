"""Legal preparation, deferred proposal ownership, and complete round timing."""

from types import SimpleNamespace
import pytest
import torch

from sparklab.core import Req, SamplingParams
from sparklab.runtime.engine.adaptive_dflash import (
    AdaptiveDFlash,
    GRAPH_RESERVE_BYTES,
)
from sparklab.runtime.engine.engine import Engine
from sparklab.runtime.engine.graph import (
    DFlashVerificationGraphs,
    MTPVerificationGraphRunner,
)
from sparklab.runtime.checkpoint.config import runtime_reserve_bytes


def configuration(**changes):
    return SimpleNamespace(
        **{
            "speculative_tokens": 12,
            "speculative_method": "dflash2",
            "tp_info": SimpleNamespace(size=1),
            "max_running_req": 1,
            "attention_backend": "triton",
            "model_config": SimpleNamespace(mtp_cuda_graph=True, model_type="qwen3_5"),
            **changes,
        }
    )


def test_configuration_requires_finite_prepared_shape_and_physical_reserve(monkeypatch):
    monkeypatch.setenv("SPARKLAB_DFLASH_POLICY", "fixed")
    assert AdaptiveDFlash.from_env(configuration()) is None
    monkeypatch.setenv("SPARKLAB_EXACT_PREFIX", "0")
    monkeypatch.delenv("SPARKLAB_PREFIX_STORE", raising=False)
    monkeypatch.setenv("SPARKLAB_DFLASH_POLICY", "cost")
    prepared = AdaptiveDFlash.from_env(configuration())
    assert prepared.depths == (0, 3, 7, 11)
    assert runtime_reserve_bytes() == GRAPH_RESERVE_BYTES
    for changes in [
        {"max_running_req": 2},
        {"attention_backend": "flashinfer"},
        {"speculative_tokens": 3},
        {"speculative_method": "mtp"},
        {"tp_info": SimpleNamespace(size=2)},
    ]:
        with pytest.raises(ValueError):
            AdaptiveDFlash.from_env(configuration(**changes))
    for depths in ["0,2,11", "3,7,11", "0,3,7", "0,3,4,7,11"]:
        monkeypatch.setenv("SPARKLAB_DFLASH_DEPTHS", depths)
        with pytest.raises(ValueError):
            AdaptiveDFlash.from_env(configuration())


def request():
    req = Req(
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        0,
        3,
        20,
        1,
        SamplingParams(temperature=0),
        None,
    )
    return req


def engine_with_proposer(depth=3):
    engine = Engine.__new__(Engine)
    engine.adaptive_dflash = AdaptiveDFlash((0, 3, 7, 11), 11, depth)
    engine.max_seq_len = 128
    calls = []

    def propose(batch, anchor):
        req = batch.reqs[0]
        calls.append((req.device_len, req.dflash_depth, anchor.clone()))
        return torch.arange(req.dflash_depth, dtype=torch.int32)

    engine.model = SimpleNamespace(propose_mtp=propose)
    return engine, calls


def test_request_local_proposal_uses_consumed_prefix_and_starts_own_round():
    engine, calls = engine_with_proposer()
    a, b = request(), request()
    anchor = torch.tensor([99], dtype=torch.int32)
    round_data = engine.prepare_adaptive_round(a, anchor)
    assert calls[0][:2] == (3, 3)  # consumed prefix, not sampled anchor length
    assert torch.equal(calls[0][2], anchor)
    assert a.device_len == 4 and a.speculative_drafts.tolist() == [0, 1, 2]
    assert a.adaptive_round is round_data
    assert b.adaptive_policy is None and b.adaptive_round is None
    with pytest.raises(AssertionError, match="drained"):
        engine.prepare_adaptive_round(a, anchor)
    round_data.accepted = 1
    assert round_data.finish(clock=lambda: round_data.started + 0.5) == 0.5
    assert a.adaptive_policy.costs[3] == __import__("collections").deque(
        [0.5], maxlen=64
    )
    assert [list(x) for x in a.adaptive_policy.acceptance[:3]] == [[True], [False], []]


def test_target_only_has_no_proposal_and_transition_restores_depth():
    engine, calls = engine_with_proposer(0)
    req = request()
    anchor = torch.tensor([9], dtype=torch.int32)
    first = engine.prepare_adaptive_round(req, anchor)
    assert first.depth == 0 and not calls and req.speculative_drafts is None
    first.finish(clock=lambda: first.started + 0.1)
    req.adaptive_round = None
    engine.adaptive_dflash = AdaptiveDFlash((0, 3, 7, 11), 11, 7)
    second = engine.prepare_adaptive_round(req, anchor)
    assert second.depth == 7 and calls[0][1] == 7
    assert len(req.adaptive_policy.costs[0]) == 1


def test_proposal_failure_does_not_leave_temporary_device_length():
    engine, _ = engine_with_proposer()
    req = request()

    def fail(*args):
        raise RuntimeError("injected proposal failure")

    engine.model.propose_mtp = fail
    with pytest.raises(RuntimeError, match="proposal failure"):
        engine.prepare_adaptive_round(req, torch.tensor([9]))
    assert req.device_len == 4
    assert req.adaptive_policy.rounds == 0


def test_non_greedy_and_short_tail_do_not_execute_illegal_proposal():
    engine, calls = engine_with_proposer(11)
    req = request()
    req.sampling_params.temperature = 1.0
    assert engine.prepare_adaptive_round(req, torch.tensor([9])).depth == 0
    req = request()
    req.max_device_len = req.cached_len + 2
    assert engine.prepare_adaptive_round(req, torch.tensor([9])).depth == 0
    assert not calls


def test_graph_set_dispatches_only_the_exact_prepared_width():
    graphs = DFlashVerificationGraphs.__new__(DFlashVerificationGraphs)
    runners = {}
    for rows in (4, 8, 12):
        runner = MTPVerificationGraphRunner.__new__(MTPVerificationGraphRunner)
        runner.rows, runner.dflash = rows, True
        runner.replay = lambda batch, n=rows: n
        runners[rows] = runner
    graphs.runners = runners
    for rows in (4, 8, 12):
        batch = SimpleNamespace(is_verify=True, size=1, input_ids=torch.zeros(rows))
        assert graphs.can_use_cuda_graph(batch)
        assert graphs.replay(batch) == rows
    batch = SimpleNamespace(is_verify=True, size=1, input_ids=torch.zeros(6))
    assert not graphs.can_use_cuda_graph(batch)
    with pytest.raises(RuntimeError, match="unprepared"):
        graphs.replay(batch)
