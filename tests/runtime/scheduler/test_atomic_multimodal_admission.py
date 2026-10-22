"""Mixed-batch token pressure must defer a fitting atomic image prompt."""

import pytest
import torch

from sparklab.core import SamplingParams
from sparklab.message import UserMsg
from sparklab.runtime.scheduler.prefill import PrefillAdder
from tests.runtime.scheduler.test_abort_inflight_prefill import _setup


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA atomic image admission ownership",
            ),
        ),
    ],
)
@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("first_is_image", [False, True])
def test_image_prompt_defers_when_only_the_remaining_batch_budget_is_short(
    monkeypatch, page_size, first_is_image, device
):
    pool, cm, tm, _, pm, _, owner = _setup(page_size=page_size, device=device)
    admitted = []
    original = PrefillAdder._add_one_req

    def record(self, *args, **kwargs):
        req = original(self, *args, **kwargs)
        if req is not None:
            admitted.append(req)
        return req

    monkeypatch.setattr(PrefillAdder, "_add_one_req", record)
    embeddings = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    positions = torch.tensor([1, 2], dtype=torch.int32)
    initial_tables = tm.available_size
    initial_slots = pool.num_free_slots
    pm.add_one_req(
        UserMsg(
            1,
            torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            SamplingParams(max_tokens=2),
            mm_embeds=embeddings if first_is_image else None,
            mm_positions=positions if first_is_image else None,
        )
    )
    pm.add_one_req(
        UserMsg(
            2,
            torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32),
            SamplingParams(max_tokens=2),
            mm_embeds=embeddings + 100,
            mm_positions=positions,
        )
    )
    try:
        first = pm.schedule_next_batch(8)
        assert first is not None and [r.uid for r in first.reqs] == [1]
        assert [p.uid for p in pm.pending_list] == [2]
        assert pm.pending_list[0].chunked_req is None
        assert tm.available_size == initial_tables - 1
        assert pool.num_free_slots == initial_slots - 3
        assert len(cm.free_slots) == cm.num_pages
        # The deferred attempt released its table, root pin and recurrent slots.
        second = pm.schedule_next_batch(8)
        assert second is not None and [r.uid for r in second.reqs] == [2]
        assert not pm.pending_list
        req = second.reqs[0]
        assert req.extend_len == 6 and req.cached_len == 0
        assert torch.equal(req.mm_embeds, embeddings + 100)
        assert torch.equal(req.mm_positions, positions)
        assert req.table_idx != first.reqs[0].table_idx
        assert pool.num_free_slots == initial_slots - 6
        for batch in (first, second):
            cm.allocate_paged(batch.reqs)
        expected_pages = (4 + page_size - 1) // page_size + (
            6 + page_size - 1
        ) // page_size
        assert len(cm.free_slots) == cm.num_pages - expected_pages
        cm.prefix_cache.check_integrity()
    finally:
        pm.pending_list.clear()
        for req in admitted:
            req.aborted = True
            owner._free_req_resources(req)
        owner._free_req_resources = None
        if device == "cuda":
            torch.cuda.synchronize()  # test-only completion oracle before pool checks
    cm.check_integrity()
    assert tm.available_size == initial_tables
    assert pool.num_free_slots == initial_slots
    assert len(cm.free_slots) == cm.num_pages


def test_individually_oversized_image_still_raises_before_copy_or_budget_mutation():
    from types import SimpleNamespace
    from sparklab.runtime.scheduler.utils import PendingReq

    req = PendingReq(
        1,
        torch.arange(9, dtype=torch.int32),
        SamplingParams(max_tokens=2),
        mm_embeds=torch.ones(1, 6),
    )
    token_pool = torch.full((1, 16), -1, dtype=torch.int32)
    adder = PrefillAdder(
        8,
        0,
        SimpleNamespace(swa_paged=False, exact_endpoints=None),
        SimpleNamespace(token_pool=token_pool),
    )
    with pytest.raises(NotImplementedError, match="single prefill chunk"):
        adder._add_one_req(req, SimpleNamespace(), 0, 0)
    assert adder.token_budget == 8 and adder.reserved_size == 0
    assert torch.all(token_pool == -1)
