"""Drive real page, radix, recurrent, prefill and exact owners against cold state."""

from types import SimpleNamespace
import pytest
import torch
from sparklab.core import Context, SamplingParams, get_global_ctx, set_global_ctx
from sparklab.message import UserMsg
from sparklab.models.config import LinearGatedDeltaGroupConfig
from sparklab.runtime.kvcache.mha_pool import MHAKVCache
from sparklab.runtime.kvcache.linear_state_pool import LinearStatePool
from sparklab.runtime.scheduler.cache import CacheManager
from sparklab.runtime.scheduler.table import TableManager
from sparklab.runtime.scheduler.prefill import PrefillManager, ChunkedReq
from sparklab.runtime.scheduler.decode import DecodeManager
from sparklab.runtime.checkpoint.exact import ExactEndpointCache


def setup(monkeypatch, page_size=4, records=8, device="cpu", num_pages=128):
    device = torch.device(device)
    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=page_size))
    monkeypatch.setattr(
        "sparklab.runtime.kvcache.mha_pool.get_tp_info", lambda: SimpleNamespace(size=1)
    )
    kv = MHAKVCache(1, 2, 4, num_pages + 1, page_size, torch.bfloat16, device)
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 1),
        num_key_heads=1,
        num_value_heads=1,
        key_head_dim=4,
        value_head_dim=4,
        conv_kernel_dim=3,
        output_gate=True,
    )
    linear = LinearStatePool(group, 32, torch.bfloat16, device, tp_size=1)
    linear.ensure_aux_state("ple", (3,), torch.int64, fill=-1)
    table = torch.zeros(3, 512, dtype=torch.int32, device=device)
    cm = CacheManager(num_pages, page_size, table, "hybrid_radix", linear_state_pool=linear)
    cm.exact_endpoints = ExactEndpointCache(cm, kv, max_records=records)
    tm = TableManager(2, table)
    pm = PrefillManager(cm, tm, DecodeManager(page_size))
    return cm, tm, pm, linear, kv


def consume(cm, linear, kv, req):
    # Independent cold-state oracle: state is a function of the exact token IDs.
    tokens = req.input_ids.tolist()
    n = len(tokens)
    for i in range(req.cached_len, n):
        physical = int(cm.page_table[req.table_idx, i])
        kv._kv_buffer.flatten(2, 3)[:, :, physical].fill_(tokens[i])
    linear.conv_states[:, req.linear_slot_idx].fill_(sum(tokens[-2:]))
    linear.recurrent_states[:, req.linear_slot_idx].fill_(sum(tokens))
    linear._aux_states["ple"][req.linear_slot_idx] = torch.tensor(
        ([-1] * 3 + tokens)[-3:]
    )
    req.complete_one()


def finish(cm, tm, req):
    cm.cache_req(req, finished=True)
    tm.free(req.table_idx)
    req.table_idx = -1
    cm.check_integrity()


@pytest.mark.parametrize("page_size", [4, 16])
@pytest.mark.parametrize("n", [1, 3, 4, 5, 7, 15, 16, 17, 63, 64, 65])
def test_private_partial_tail_and_complete_state_survive_two_branches(
    monkeypatch, n, page_size
):
    cm, tm, pm, linear, kv = setup(monkeypatch, page_size)
    tokens = list(range(1, n + 1))
    pm.add_one_req(
        UserMsg(
            1, torch.tensor(tokens, dtype=torch.int32), SamplingParams(max_tokens=4)
        )
    )
    while pm.runnable:
        batch = pm.schedule_next_batch(512)
        req = batch.reqs[0]
        cm.allocate_paged(batch.reqs)
        consume(cm, linear, kv, req)
        if isinstance(req, ChunkedReq):
            cm.exact_endpoints.publish(req)
        else:
            cm.cache_req(req, finished=False)
    finish(cm, tm, req)
    parent = cm.exact_endpoints.entries[tuple(tokens)]
    state = linear.recurrent_states[:, parent.state_slot].clone()
    tail = (
        kv._kv_buffer.flatten(2, 3).index_select(2, parent.tail.long()).clone()
        if parent.tail is not None
        else None
    )
    for suffix in [91, 92]:
        branch = tokens + [suffix]
        pm.add_one_req(
            UserMsg(
                suffix,
                torch.tensor(branch, dtype=torch.int32),
                SamplingParams(max_tokens=4),
            )
        )
        batch = pm.schedule_next_batch(512)
        req = batch.reqs[0]
        assert req.cached_len == n
        assert req.exact_source is parent and parent.readers == 1
        cm.allocate_paged(batch.reqs)
        linear.copy_from(req.mamba_restore_src, req.linear_slot_idx)
        req.mamba_restore_src = None
        assert linear.recurrent_states[:, req.linear_slot_idx].eq(sum(tokens)).all()
        assert (
            linear._aux_states["ple"][req.linear_slot_idx].tolist()
            == ([-1] * 3 + tokens)[-3:]
        )
        for i, token in enumerate(tokens):
            physical = int(cm.page_table[req.table_idx, i])
            assert kv._kv_buffer.flatten(2, 3)[:, :, physical].eq(token).all()
        if n % cm.page_size:
            assert int(cm.page_table[req.table_idx, n - 1]) != int(
                parent.tail[n % cm.page_size - 1]
            )
        consume(cm, linear, kv, req)
        finish(cm, tm, req)
        assert parent.readers == 0
        assert torch.equal(linear.recurrent_states[:, parent.state_slot], state)
        if tail is not None:
            actual = kv._kv_buffer.flatten(2, 3).index_select(2, parent.tail.long())
            # Only initialized consumed tail rows are part of the checkpoint.
            count = n % cm.page_size
            assert torch.equal(actual[:, :, :count], tail[:, :, :count])
    cm.exact_endpoints.clear()
    # Eviction after records drain must restore both physical resource currencies.
    ev = cm.prefix_cache.evict_full(1 << 20)
    cm._free(ev.kv_indices)
    linear.free(ev.mamba_slots)
    assert len(cm.free_slots) == cm.num_pages
    assert linear.num_free_slots == linear.num_slots - 1


def test_pinned_record_cannot_be_evicted_and_rebuild_discards_old_ids(monkeypatch):
    cm, tm, pm, linear, kv = setup(monkeypatch, records=2)
    pm.add_one_req(
        UserMsg(
            1, torch.tensor([1, 2, 3], dtype=torch.int32), SamplingParams(max_tokens=2)
        )
    )
    while pm.runnable:
        batch = pm.schedule_next_batch(512)
        req = batch.reqs[0]
        cm.allocate_paged(batch.reqs)
        consume(cm, linear, kv, req)
        cm.cache_req(req, finished=False)
    finish(cm, tm, req)
    handle = cm.exact_endpoints.match([1, 2, 3])
    cm.lock(handle)
    while cm.exact_endpoints.evict_one():
        pass
    assert tuple([1, 2, 3]) in cm.exact_endpoints.entries
    with pytest.raises(AssertionError, match="readers"):
        cm.exact_endpoints.invalidate_rebuild()
    cm.unlock(handle)
    cm.rebuild(cm.num_pages, cm.page_table)
    cm.check_integrity()
    assert not cm.exact_endpoints.entries
    assert linear.num_free_slots == linear.num_slots - 1


def test_pool_pressure_evicts_exact_records_before_admission_can_starve(monkeypatch):
    cm, tm, pm, linear, kv = setup(monkeypatch, records=8)
    linear.rebuild(8)
    for trial in range(30):
        tokens = [trial * 10 + i for i in range(1, 6)]
        pm.add_one_req(
            UserMsg(
                trial,
                torch.tensor(tokens, dtype=torch.int32),
                SamplingParams(max_tokens=2),
            )
        )
        while pm.runnable:
            batch = pm.schedule_next_batch(512)
            assert batch is not None, (
                "retained snapshots must not permanently block admission"
            )
            req = batch.reqs[0]
            cm.allocate_paged(batch.reqs)
            consume(cm, linear, kv, req)
            cm.cache_req(req, finished=False)
        finish(cm, tm, req)
        assert len(cm.exact_endpoints.entries) <= 8
        assert (
            linear.num_free_slots + len(cm.exact_endpoints.entries)
            == linear.num_slots - 1
        )


def test_cancel_after_match_releases_read_pin_and_never_donates_unrestored_state(
    monkeypatch,
):
    cm, tm, pm, linear, kv = setup(monkeypatch)
    pm.add_one_req(
        UserMsg(
            1,
            torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            SamplingParams(max_tokens=2),
        )
    )
    while pm.runnable:
        batch = pm.schedule_next_batch(512)
        req = batch.reqs[0]
        cm.allocate_paged(batch.reqs)
        consume(cm, linear, kv, req)
        cm.cache_req(req, finished=False)
    finish(cm, tm, req)
    parent = cm.exact_endpoints.entries[(1, 2, 3, 4)]
    pm.add_one_req(
        UserMsg(
            2,
            torch.tensor([1, 2, 3, 4, 9], dtype=torch.int32),
            SamplingParams(max_tokens=2),
        )
    )
    req = pm.schedule_next_batch(512).reqs[0]
    assert parent.readers == 1
    req.aborted = True
    finish(cm, tm, req)
    assert parent.readers == 0
    assert not any(
        e.tokens == (1, 2, 3, 4, 9) for e in cm.exact_endpoints.entries.values()
    )


def test_unaligned_restore_reports_exact_cached_tokens_not_only_shared_pages(
    monkeypatch,
):
    cm, tm, pm, linear, kv = setup(monkeypatch, page_size=16)
    pm.add_one_req(
        UserMsg(1, torch.arange(1, 16, dtype=torch.int32), SamplingParams(max_tokens=1))
    )
    while pm.runnable:
        batch = pm.schedule_next_batch(512)
        req = batch.reqs[0]
        cm.allocate_paged(batch.reqs)
        consume(cm, linear, kv, req)
        cm.cache_req(req, finished=False)
    finish(cm, tm, req)
    pm.add_one_req(
        UserMsg(2, torch.arange(1, 17, dtype=torch.int32), SamplingParams(max_tokens=1))
    )
    batch = pm.schedule_next_batch(512)
    assert batch.log_cached_tokens == 15 and batch.log_new_tokens == 1
    assert batch.prompt_admissions == [(2, 16, 15)]


def test_retained_full_pages_are_evicted_before_kv_admission_stalls(monkeypatch):
    cm, tm, pm, linear, kv = setup(monkeypatch, page_size=4, records=8)
    cm.rebuild(8, cm.page_table)
    for trial in range(10):
        tokens = [trial * 100 + i for i in range(1, 18)]
        pm.add_one_req(
            UserMsg(
                trial,
                torch.tensor(tokens, dtype=torch.int32),
                SamplingParams(max_tokens=1),
            )
        )
        while pm.runnable:
            batch = pm.schedule_next_batch(512)
            assert batch is not None
            req = batch.reqs[0]
            cm.allocate_paged(batch.reqs)
            consume(cm, linear, kv, req)
            cm.cache_req(req, finished=False)
        finish(cm, tm, req)


def test_unfinished_producer_cannot_be_matched_evicted_or_cleared(monkeypatch):
    cm, tm, pm, linear, kv = setup(monkeypatch)
    pm.add_one_req(
        UserMsg(1, torch.tensor([1], dtype=torch.int32), SamplingParams(max_tokens=1))
    )
    req = pm.schedule_next_batch(512).reqs[0]
    cm.allocate_paged([req])
    consume(cm, linear, kv, req)
    cm.cache_req(req, finished=False)
    finish(cm, tm, req)
    entry = cm.exact_endpoints.entries[(1,)]

    class Fence:
        done = False

        def query(self):
            return self.done

    entry.ready = Fence()
    assert cm.exact_endpoints.match([1]) is None
    assert not cm.exact_endpoints.evict_one()
    with pytest.raises(RuntimeError, match="users"):
        cm.exact_endpoints.clear()
    entry.ready.done = True
    assert cm.exact_endpoints.match([1]) is not None
    assert cm.exact_endpoints.evict_one()
    cm.check_integrity()
