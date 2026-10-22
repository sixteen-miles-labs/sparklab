"""Physical-page oracles for immutable tails and recomputation under pressure."""

import pytest
import torch

from sparklab.core import SamplingParams
from sparklab.message import UserMsg
from sparklab.runtime.scheduler.utils import PendingReq
from sparklab.runtime.checkpoint.transfer import GPUStateSink
from sparklab.runtime.scheduler.prefill import ChunkedReq
from tests.runtime.checkpoint.test_exact import consume, finish, setup
from tests.runtime.checkpoint.test_service import seeded, drain, CPUCopySink


def allocate_declared_work(cm, linear, kv, req):
    cm.allocate_paged([req])
    consume(cm, linear, kv, req)
    # Seat the full consumed budget (the last sampled token is unconsumed).
    extra = req.output_len - 1
    if extra:
        req.input_ids = torch.cat(
            [req.input_ids, torch.arange(100, 100 + extra, dtype=torch.int32)]
        )
        req.device_len = len(req.input_ids)
        cm.allocate_paged([req])
        consume(cm, linear, kv, req)


@pytest.mark.parametrize("prefix", [3, 7, 11])
def test_restore_cannot_spend_the_page_needed_by_its_live_private_tail(
    tmp_path, monkeypatch, prefix
):
    # P=4, immutable snapshot uses ceil(prefix/4) pages. A branch at P-1
    # needs two private pages for one suffix token and two output tokens.
    snapshot_pages = (prefix + 3) // 4
    pages = snapshot_pages + 1  # one page short for snapshot + live branch
    service, cm, tm, pm, linear, kv, tokens, _, _ = seeded(
        tmp_path,
        monkeypatch,
        token_count=prefix,
        num_pages=pages,
    )
    req = None
    before = len(cm.free_slots), linear.num_free_slots
    try:
        query = torch.tensor(tokens + (91,), dtype=torch.int32)
        assert not service.maybe_restore(
            PendingReq(1, query, SamplingParams(max_tokens=2))
        )
        assert (len(cm.free_slots), linear.num_free_slots) == before
        assert service.outstanding == 0
        pm.add_one_req(UserMsg(2, query, SamplingParams(max_tokens=2)))
        for chunk in range(3):
            batch = pm.schedule_next_batch(512)
            assert batch is not None and len(batch.reqs) == 1
            req = batch.reqs[0]
            if chunk == 0:
                assert req.cached_len == 0
            cm.allocate_paged([req])
            if isinstance(req, ChunkedReq):
                consume(cm, linear, kv, req)
                continue
            allocate_declared_work(cm, linear, kv, req)
            break
        else:
            pytest.fail("cold checkpoint chunks did not finish")
    finally:
        if req is not None:
            req.aborted = True
            finish(cm, tm, req)
        assert service.close()
    cm.check_integrity()
    assert (len(cm.free_slots), linear.num_free_slots) == before


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA private-tail admission gate",
            ),
        ),
    ],
)
def test_memory_tail_pressure_evicts_unread_snapshot_and_admits_cold_work(
    tmp_path,
    monkeypatch,
    pinned,
    device,
):
    # Five pages can restore prefix 11 plus a short branch. A later six-output
    # branch needs three live pages; cold recomputation fits in five total pages.
    service, cm, tm, pm, linear, kv, tokens, _, _ = seeded(
        tmp_path,
        monkeypatch,
        token_count=11,
        num_pages=5,
        device=device,
        sink_factory=GPUStateSink if device == "cuda" else CPUCopySink,
    )
    req = None
    lease = None
    try:
        assert service.maybe_restore(
            PendingReq(
                1,
                torch.tensor(tokens + (91,), dtype=torch.int32),
                SamplingParams(max_tokens=1),
            )
        )
        drain(service)
        if device == "cuda":
            torch.cuda.synchronize()  # demand a ready source; test oracle only
        assert tokens in cm.exact_endpoints.entries and len(cm.free_slots) == 2
        pm.add_one_req(
            UserMsg(
                2,
                torch.tensor(tokens + (92,), dtype=torch.int32),
                SamplingParams(max_tokens=6),
            )
        )
        if pinned:
            lease = cm.match_req(pm.pending_list[0]).cuda_handle
            cm.lock(lease)
            before = len(cm.free_slots), linear.num_free_slots, tm.available_size
            blocked = pm.schedule_next_batch(512)
            if blocked is not None:
                req = blocked.reqs[0]  # preserve failure cleanup on prior code
            assert blocked is None, "a live source lease must remain protected"
            assert (
                len(cm.free_slots),
                linear.num_free_slots,
                tm.available_size,
            ) == before
            assert tokens in cm.exact_endpoints.entries
            cm.unlock(lease)
            lease = None
        batch = pm.schedule_next_batch(512)
        assert batch is not None, (
            "reusable state must not starve a request that fits cold"
        )
        req = batch.reqs[0]
        assert req.cached_len == 0, "tail copy plus declared live pages do not fit"
        allocate_declared_work(cm, linear, kv, req)
    finally:
        if lease is not None:
            cm.unlock(lease)
        if req is not None:
            req.aborted = True
            finish(cm, tm, req)
        assert service.close()
    cm.exact_endpoints.clear()
    evicted = cm.prefix_cache.evict_full(1 << 20)
    cm._free(evicted.kv_indices)
    linear.free(evicted.mamba_slots)
    cm.check_integrity()
    assert len(cm.free_slots) == 5 and linear.num_free_slots == 31


def test_two_cold_admissions_cannot_share_a_fractional_page_reservation(
    tmp_path, monkeypatch
):
    service, cm, tm, pm, linear, kv, _, _, _ = seeded(
        tmp_path, monkeypatch, num_pages=3
    )
    live = []
    completed = []
    try:
        for uid in (1, 2):
            pm.add_one_req(
                UserMsg(
                    uid,
                    torch.tensor([uid, 21, 22, 23], dtype=torch.int32),
                    SamplingParams(max_tokens=2),
                )
            )
        for expected_uid in (1, 2):
            for _ in range(3):
                batch = pm.schedule_next_batch(512)
                assert batch is not None
                live = batch.reqs
                cm.allocate_paged(live)
                assert len(live) == 1, "each declared request needs two private pages"
                req = live[0]
                assert req.uid == expected_uid
                if isinstance(req, ChunkedReq):
                    consume(cm, linear, kv, req)
                    continue
                allocate_declared_work(cm, linear, kv, req)
                completed.append(req.uid)
                req.aborted = True
                finish(cm, tm, req)
                live = []
                break
            else:
                pytest.fail("message/prompt checkpoint chunks did not complete")
        assert completed == [1, 2] and not pm.pending_list
    finally:
        for req in live:
            req.aborted = True
            finish(cm, tm, req)
        assert service.close()
    cm.check_integrity()
    assert len(cm.free_slots) == 3 and linear.num_free_slots == 31


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA published-tail reclamation gate",
            ),
        ),
    ],
)
def test_publication_preserves_live_decode_headroom(monkeypatch, device):
    cm, tm, pm, linear, kv = setup(monkeypatch, num_pages=3, device=device)
    pm.add_one_req(
        UserMsg(1, torch.arange(1, 9, dtype=torch.int32), SamplingParams(max_tokens=2))
    )
    req = None
    try:
        for _ in range(3):
            batch = pm.schedule_next_batch(512)
            assert batch is not None
            req = batch.reqs[0]
            cm.allocate_paged([req])
            consume(cm, linear, kv, req)
            if isinstance(req, ChunkedReq):
                assert cm.exact_future_pages == 1
                assert not cm.exact_endpoints.publish(req)
                continue
            cm.cache_req(req, finished=False)
            break
        else:
            pytest.fail("prefill checkpoint chunks did not finish")
        if device == "cuda":
            torch.cuda.synchronize()  # completed producer fence, as in native drain
        assert len(cm.free_slots) == 1
        assert tuple(range(1, 8)) not in cm.exact_endpoints.entries
        parent = cm.exact_endpoints.entries[tuple(range(1, 9))]
        before = (
            kv._kv_buffer.flatten(2, 3)
            .index_select(2, parent.anchor.kv_indices.long())
            .clone()
        )
        req.append_host(torch.tensor([91], dtype=torch.int32))
        cm.allocate_paged([req])
        consume(cm, linear, kv, req)
        assert tuple(range(1, 8)) not in cm.exact_endpoints.entries
        after = kv._kv_buffer.flatten(2, 3).index_select(
            2, parent.anchor.kv_indices.long()
        )
        assert torch.equal(before, after)
    finally:
        if req is not None:
            req.aborted = True
            finish(cm, tm, req)
    cm.exact_endpoints.clear()
    evicted = cm.prefix_cache.evict_full(1 << 20)
    cm._free(evicted.kv_indices)
    linear.free(evicted.mamba_slots)
    cm.check_integrity()
    assert len(cm.free_slots) == 3 and linear.num_free_slots == 31
    assert cm.exact_future_pages == 0


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unread-tail allocation gate",
            ),
        ),
    ],
)
def test_allocator_reclaims_only_ready_unread_exact_storage(monkeypatch, device):
    cm, tm, pm, linear, kv = setup(monkeypatch, num_pages=3, device=device)
    pm.add_one_req(
        UserMsg(1, torch.arange(1, 8, dtype=torch.int32), SamplingParams(max_tokens=1))
    )
    req = None
    allocated = []
    try:
        for _ in range(3):
            batch = pm.schedule_next_batch(512)
            assert batch is not None
            req = batch.reqs[0]
            cm.allocate_paged([req])
            consume(cm, linear, kv, req)
            if isinstance(req, ChunkedReq):
                cm.exact_endpoints.publish(req)
                continue
            finish(cm, tm, req)
            req = None
            break
        else:
            pytest.fail("checkpoint publication did not complete")
        if device == "cuda":
            torch.cuda.synchronize()
        assert tuple(range(1, 8)) in cm.exact_endpoints.entries
        # Allocate standalone private owners, as the inbound service does.
        allocated.append(cm._page_to_token(cm._allocate(1)))
        allocated.append(cm._page_to_token(cm._allocate(1)))
        assert tuple(range(1, 8)) not in cm.exact_endpoints.entries
        assert allocated[0].numel() == allocated[1].numel() == 4
    finally:
        if req is not None:
            req.aborted = True
            finish(cm, tm, req)
        for indices in allocated:
            cm._free(indices)
    cm.exact_endpoints.clear()
    evicted = cm.prefix_cache.evict_full(1 << 20)
    cm._free(evicted.kv_indices)
    linear.free(evicted.mamba_slots)
    cm.check_integrity()
    assert len(cm.free_slots) == 3 and linear.num_free_slots == 31
    assert cm.exact_future_pages == 0


def test_intermediate_checkpoint_preserves_the_whole_pending_prompt_budget(monkeypatch):
    cm, tm, pm, linear, kv = setup(monkeypatch, num_pages=5)
    pm.add_one_req(
        UserMsg(
            1,
            torch.arange(1, 17, dtype=torch.int32),
            SamplingParams(max_tokens=2),
            checkpoint_endpoints=(3,),
        )
    )
    req = pm.schedule_next_batch(512).reqs[0]
    try:
        assert isinstance(req, ChunkedReq) and req.device_len == 3
        cm.allocate_paged([req])
        consume(cm, linear, kv, req)
        assert req.kv_reservation_end == 18 and cm.exact_future_pages == 4
        assert len(cm.free_slots) == 4
        assert not cm.exact_endpoints.publish(req)
    finally:
        assert pm.abort_req(req.uid) is req
        req.aborted = True
        finish(cm, tm, req)
    assert cm.exact_future_pages == 0 and len(cm.free_slots) == 5
    cm.check_integrity()


def test_publication_keeps_headroom_for_every_active_request(monkeypatch):
    cm, tm, pm, linear, kv = setup(monkeypatch, num_pages=4)
    for uid in (1, 2):
        pm.add_one_req(
            UserMsg(
                uid,
                torch.tensor([uid, 21, 22, 23], dtype=torch.int32),
                SamplingParams(max_tokens=2),
            )
        )
    batch = pm.schedule_next_batch(512)
    assert len(batch.reqs) == 2
    cm.allocate_paged(batch.reqs)
    try:
        assert len(cm.free_slots) == cm.exact_future_pages == 2
        for req in batch.reqs:
            consume(cm, linear, kv, req)
            assert not cm.exact_endpoints.publish(req)
    finally:
        for req in batch.reqs:
            assert pm.abort_req(req.uid) is req
            req.aborted = True
            cm.cache_req(req, finished=True)
            tm.free(req.table_idx)
            req.table_idx = -1
    assert cm.exact_future_pages == 0 and len(cm.free_slots) == 4
    assert linear.num_free_slots == 31
    cm.check_integrity()


@pytest.mark.parametrize("prefix", [3, 7, 11])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA parallel private-tail admission gate",
            ),
        ),
    ],
)
@pytest.mark.parametrize("continuation", [False, True])
def test_parallel_private_tails_are_charged_once(
    tmp_path, monkeypatch, prefix, device, continuation
):
    # P=4: the immutable source takes ceil(prefix/4) pages and each branch
    # needs two private pages. Exactly four free pages must seat both branches.
    pages = (prefix + 3) // 4 + 4
    service, cm, tm, pm, linear, kv, tokens, expected, _ = seeded(
        tmp_path,
        monkeypatch,
        token_count=prefix,
        num_pages=pages,
        device=device,
        sink_factory=GPUStateSink if device == "cuda" else CPUCopySink,
    )
    live = []
    try:
        assert service.maybe_restore(
            PendingReq(
                9,
                torch.tensor(tokens + (91,), dtype=torch.int32),
                SamplingParams(max_tokens=1),
            )
        )
        drain(service)
        if device == "cuda":
            torch.cuda.synchronize()  # demand a ready source; test oracle only
        assert len(cm.free_slots) == 4
        parent = cm.exact_endpoints.entries[tokens]
        if continuation:
            pm.add_one_req(
                UserMsg(
                    1,
                    torch.tensor(tokens + (91, 92, 93), dtype=torch.int32),
                    SamplingParams(max_tokens=2),
                )
            )
            live = pm.schedule_next_batch(2).reqs
            assert len(live) == 1 and isinstance(live[0], ChunkedReq)
            cm.allocate_paged(live)
            consume(cm, linear, kv, live[0])
        for uid in (2,) if continuation else (1, 2):
            pm.add_one_req(
                UserMsg(
                    uid,
                    torch.tensor(tokens + (90 + uid,), dtype=torch.int32),
                    SamplingParams(max_tokens=4),
                )
            )
        batch = pm.schedule_next_batch(512)
        assert batch is not None
        live = batch.reqs
        assert len(live) == 2, (
            "a copied tail must not be charged again as future allocation"
        )
        assert len(cm.free_slots) == 2 - int(continuation) and parent.readers == 2
        assert [req.cached_len for req in live] == [
            prefix + 2 if continuation else prefix,
            prefix,
        ]
        cm.allocate_paged(live)
        assert cm.exact_future_pages == 2 - int(continuation)
        for req in live:
            allocate_declared_work(cm, linear, kv, req)
        assert len(cm.free_slots) == cm.exact_future_pages == 0
        full = prefix // 4 * 4
        first, second = [
            set(cm.page_table[r.table_idx, full : full + 8].tolist()) for r in live
        ]
        assert first.isdisjoint(second)
        assert first.isdisjoint(parent.tail.tolist()) and second.isdisjoint(
            parent.tail.tolist()
        )
        mapping = torch.cat([parent.anchor.kv_indices, parent.tail[:3]])
        unchanged = cm.exact_endpoints.adapter.snapshot(
            expected.compatibility,
            tokens,
            mapping,
            parent.state_slot,
            max_bytes=1 << 20,
        )
        for name, value in unchanged.inspect().items():
            assert torch.equal(value.cpu(), expected.inspect()[name].cpu()), name
    finally:
        for uid in (1, 2):
            pm.abort_req(uid)
        for req in live:
            req.aborted = True
            cm.cache_req(req, finished=True)
            tm.free(req.table_idx)
            req.table_idx = -1
        assert service.close()
    cm.exact_endpoints.clear()
    evicted = cm.prefix_cache.evict_full(1 << 20)
    cm._free(evicted.kv_indices)
    linear.free(evicted.mamba_slots)
    cm.check_integrity()
    assert len(cm.free_slots) == pages and linear.num_free_slots == 31
