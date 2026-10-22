"""Stage ownership through actual scheduling, chunk inheritance and burst drain."""

from types import SimpleNamespace

import torch

from sparklab.core import Batch, SamplingParams
from sparklab.message import UserMsg
from sparklab.runtime.scheduler.scheduler import Scheduler, ForwardInput
from tests.runtime.scheduler.test_abort_inflight_prefill import _setup, _launch_req


def test_chunked_admission_preserves_original_stamps_and_first_dispatch(monkeypatch):
    _, cm, _, dm, pm, sent, scheduler = _setup()
    now = [10.0]
    monkeypatch.setattr(
        "sparklab.runtime.scheduler.scheduler.time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )
    scheduler.engine = SimpleNamespace(max_seq_len=64)
    msg = UserMsg(
        17,
        torch.arange(5, dtype=torch.int32),
        SamplingParams(max_tokens=2),
        timing={
            "http_ingress": 1.0,
            "frontend_submit": 2.0,
            "tokenize_start": 3.0,
            "tokenize_end": 4.0,
        },
    )
    Scheduler._process_one_msg(scheduler, msg)
    assert pm.pending_list[0].timing["scheduler_received"] == 10.0
    first = pm.schedule_next_batch(2)
    cm.allocate_paged(first.reqs)
    now[0] = 11.0
    Scheduler._report_prompt_admissions(scheduler, first)
    admission = sent.pop()
    assert admission.timing["admitted"] == 11.0
    assert "first_dispatch" not in admission.timing
    scheduler.token_pool = torch.zeros(4, 64, dtype=torch.int32)
    dm.filter_reqs = lambda _: None

    def forward(batch, sample_args):
        for req in batch.reqs:
            req.complete_one()
        return SimpleNamespace(
            managed_writes=False, next_tokens_gpu=torch.tensor([42], dtype=torch.int32)
        )

    scheduler.engine.forward_batch = forward
    scheduler.engine.stream = None
    index = (torch.tensor([0]), torch.tensor([0]))
    now[0] = 12.0
    Scheduler._forward(scheduler, ForwardInput(first, None, index, index))
    assert first.reqs[0].timing["first_dispatch"] == 12.0
    # The already-sent admission image must not be mutated by later stages.
    assert "first_dispatch" not in admission.timing
    second = pm.schedule_next_batch(2)
    cm.allocate_paged(second.reqs)
    now[0] = 13.0
    Scheduler._forward(scheduler, ForwardInput(second, None, index, index))
    assert second.reqs[0].timing["first_dispatch"] == 12.0
    assert second.reqs[0].timing["admitted"] == 11.0
    assert len(sent) == 0


def test_speculative_burst_reports_one_first_generated_stamp_after_existing_fence(
    monkeypatch,
):
    pool, cm, tm, _, _, sent, scheduler = _setup()
    req = _launch_req(
        pool, cm, tm, torch.tensor([1, 2], dtype=torch.int32), output_len=6
    )
    req.timing = {"http_ingress": 1.0, "admitted": 2.0, "first_dispatch": 3.0}
    now = [4.0]
    monkeypatch.setattr(
        "sparklab.runtime.scheduler.scheduler.time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )

    def synchronize():
        now[0] = 5.0  # observation must occur after the existing output fence

    output = SimpleNamespace(
        next_tokens_cpu=torch.tensor([20, 21, 22], dtype=torch.int32),
        token_counts=(3,),
        copy_done_event=SimpleNamespace(synchronize=synchronize),
    )
    batch = Batch(reqs=[req], phase="decode")
    Scheduler._process_last_data(scheduler, (SimpleNamespace(batch=batch), output))
    assert len(sent) == 3
    assert [msg.next_token for msg in sent] == [20, 21, 22]
    assert sent[0].timing["first_generated"] == 5.0
    assert sent[1].timing is sent[2].timing is None
    assert req.timing is None
    # Later rounds contribute usage/delivery gaps, never another first output.
    Scheduler._process_last_data(scheduler, (SimpleNamespace(batch=batch), output))
    assert len(sent) == 6
    assert all(msg.timing is None for msg in sent[3:])
