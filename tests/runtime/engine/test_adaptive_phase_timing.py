"""Real engine/drain boundaries with synthetic target logits and bounded events."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sparklab.runtime.engine.adaptive_dflash import AdaptiveRound
from sparklab.runtime.engine.sample import BatchSamplingArgs
from sparklab.runtime.scheduler.scheduler import Scheduler, ForwardInput
from sparklab.runtime.scheduler.telemetry import RuntimeTelemetry
from tests.runtime.engine.test_adaptive_dflash import engine_with_proposer
from tests.runtime.scheduler.test_abort_inflight_prefill import _setup, _launch_req


def fake_events(monkeypatch, clock):
    created = []

    class Event:
        at = None

        def __init__(self, **kwargs):
            created.append(self)

        def record(self, stream=None):
            self.at = clock[0]

        def query(self):
            return self.at is not None

        def synchronize(self):
            assert self.query()

        def elapsed_time(self, other):
            assert self.query() and other.query()
            return other.at - self.at

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: None)
    return created


def test_phase_events_are_optional_and_require_ordered_completed_markers(
    monkeypatch,
):
    clock = [0.0]
    created = fake_events(monkeypatch, clock)
    policy = engine_with_proposer()[0].adaptive_dflash.policy()
    plain = AdaptiveRound(policy, 3, 0.0)
    plain.begin_gpu(torch.device("cuda"))
    assert len(created) == 3 and plain.gpu_verification_end is None
    current = AdaptiveRound(policy, 3, 0.0)
    current.begin_gpu(torch.device("cuda"), phase_timing=True)
    assert len(created) == 8
    with pytest.raises(AssertionError, match="out of order"):
        current.commit_ready(torch.device("cuda"))
    current.proposal_ready(torch.device("cuda"))
    current.verification_ready(torch.device("cuda"))
    with pytest.raises(AssertionError, match="twice"):
        current.verification_ready(torch.device("cuda"))
    with pytest.raises(AssertionError, match="output fence"):
        current.gpu_timings()
    current.gpu_end.record()
    with pytest.raises(AssertionError, match="lacks completed"):
        current.gpu_timings()
    assert policy.rounds == 0


@pytest.mark.parametrize("depth,accepted", [(0, 0), (3, 0), (3, 1), (3, 3)])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA adaptive phase boundary gate",
            ),
        ),
    ],
)
def test_actual_engine_and_scheduler_measure_phase_boundaries(
    monkeypatch, request, depth, accepted, device
):
    clock = [0.0]
    if device == "cpu":
        fake_events(monkeypatch, clock)
    _, cm, tm, dm, _, sent, owner = _setup()

    def release_test_owners():
        # The shared scheduler shell has a self-capturing cleanup callback.
        # Break it so its tiny GPU buffers cannot survive until a later test's
        # unrelated allocator observation triggers cyclic collection.
        owner._free_req_resources = None
        owner.engine = None
        owner.token_pool = None

    request.addfinalizer(release_test_owners)
    req = _launch_req(
        cm.linear_state_pool,
        cm,
        tm,
        torch.tensor([1, 2], dtype=torch.int32),
        output_len=20,
    )
    req.sampling_params.temperature = 0
    engine, _ = engine_with_proposer(depth)
    engine.device = torch.device("cuda")  # CPU control substitutes events only.
    engine.stream = torch.cuda.current_stream()
    owner.device = torch.device(device)
    owner.engine = engine
    owner._runtime_timing = True
    owner.runtime_metrics = RuntimeTelemetry(trace_capacity=16)
    owner._runtime_events = deque()
    owner.token_pool = torch.zeros(
        tm.token_pool.shape, dtype=torch.int32, device=device
    )
    owner.token_pool[req.table_idx, req.cached_len] = 19
    work = torch.zeros(8, device=device)

    def advance(ms):
        clock[0] += ms
        work.add_(ms)

    def propose(batch, anchor):
        advance(2)
        return torch.arange(depth, dtype=torch.int32, device=device)

    engine.model.propose_mtp = propose
    engine.model.begin_external_inputs = lambda batch: None
    engine.config = SimpleNamespace(
        speculative_tokens=4,
        speculative_method="dflash2",
        model_config=SimpleNamespace(glm_dsa_args=None),
    )
    engine.ctx = SimpleNamespace(forward_batch=lambda _: nullcontext())
    engine.cpu_moe_executor = None
    engine.mtp_stats = dict(target_forwards=0, drafted=0, accepted=0, outputs=0)
    dm.filter_reqs([req])
    batch = Scheduler._schedule_decode_batch(owner)
    current = req.adaptive_round
    assert (
        current.gpu_verification_end is not None and current.gpu_commit_end is not None
    )
    if depth:
        req.speculative_drafts = None
        owner.token_pool[req.table_idx, req.device_len : req.device_len + depth] = (
            torch.arange(depth, device=device)
        )
        batch.verify_cached_lens = (req.cached_len,)
        req.device_len += depth
        batch.phase = "verify"
    batch.padded_reqs = batch.reqs
    cm.allocate_paged(batch.reqs)
    rows = depth + 1
    logits = torch.zeros(rows, 100, device=device)
    truth = list(range(depth)) + [99]
    truth[accepted] = 99
    logits[torch.arange(rows), torch.tensor(truth)] = 1

    def target():
        advance(5)
        assert not current._verification_recorded and not current._commit_recorded
        return logits

    def snapshot(*args):
        advance(3)

    def commit(*args):
        assert current._verification_recorded and not current._commit_recorded
        advance(7)

    engine.linear_state_pool = SimpleNamespace(
        num_slots=16,
        cached_initial_state_step=None,
        verify_steps=4,
        copy_from=snapshot,
        commit_verify_prefix=commit,
        _verify_aux_inputs={},
    )
    engine.mtp_graph_runner = SimpleNamespace(
        can_use_cuda_graph=lambda _: True, replay=lambda _: target()
    )
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda _: False)
    engine.model.forward = target

    def sample(values, args):
        advance(11)
        return values.argmax(-1)

    engine.sampler = SimpleNamespace(sample=sample)
    import sparklab.runtime.engine.engine as engine_module

    original_verify = engine_module._verify_speculative_tokens

    def verify(*args):
        result = original_verify(*args)
        advance(11)
        return result

    monkeypatch.setattr(engine_module, "_verify_speculative_tokens", verify)
    start = req.cached_len
    indices = (torch.full((rows,), req.table_idx), torch.arange(start, start + rows))
    output_copies = []
    original_to = torch.Tensor.to

    def observed_to(tensor, *args, **kwargs):
        if (
            args
            and isinstance(args[0], str)
            and args[0] == "cpu"
            and kwargs.get("non_blocking")
        ):
            assert current._commit_recorded
            output_copies.append(True)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observed_to)
    output = Scheduler._forward(
        owner, ForwardInput(batch, BatchSamplingArgs(None), indices, indices)
    )
    assert output_copies == [True]
    assert current._verification_recorded and current._commit_recorded
    Scheduler._process_last_data(owner, (SimpleNamespace(batch=batch), output))
    assert req.adaptive_round is None and req.adaptive_policy.rounds == 1
    measured = list(owner.runtime_metrics.trace)[-1]
    if device == "cpu":
        assert measured["proposal_gpu_seconds"] == (0.002 if depth else 0.0)
        assert measured["verification_gpu_seconds"] == (0.019 if depth else 0.016)
        assert measured["commit_gpu_seconds"] == (0.007 if depth else 0.0)
        assert measured["handoff_gpu_seconds"] == 0.0
    phases = [
        measured[name + "_gpu_seconds"]
        for name in ("proposal", "verification", "commit", "handoff")
    ]
    assert all(value >= 0 for value in phases)
    assert sum(phases) == pytest.approx(measured["round_gpu_seconds"], abs=1e-5)
    for name in ("proposal", "verification", "commit", "handoff"):
        assert (
            sum(owner.runtime_metrics.histograms["speculative_" + name + "_gpu"]) == 1
        )
    assert len(sent) == accepted + 1 and sent[-1].next_token == 99
    owner._free_req_resources(req)
    cm.check_integrity()
