"""Real-file persistence driven through production admission and radix owners.

CPU copy transport stands in for CUDA fences only. The separate CUDA transfer
suite covers staging; these cases cover scheduling, publication and conservation.
"""

from dataclasses import replace
import threading
import time

import torch

from sparklab.core import SamplingParams
from sparklab.message import UserMsg
from sparklab.runtime.checkpoint.config import DiskSettings
from sparklab.runtime.checkpoint.record import TensorSource, RecordLimits
from sparklab.runtime.checkpoint.service import PrefixRetention
from sparklab.runtime.checkpoint.state import StateImage
from sparklab.runtime.checkpoint.store import PrefixStore
from sparklab.runtime.scheduler.utils import PendingReq
from tests.runtime.checkpoint.test_exact import setup
from tests.runtime.checkpoint.test_state import identity


class CPUCopySink:
    def __init__(self, adapter, compatibility, tokens, indices, state_slot, **kwargs):
        self.adapter, self.compatibility, self.tokens = adapter, compatibility, tokens
        self.indices, self.slot = indices, state_slot
        self.validated = False
        self.done = False
        self.fence = True

    def begin(self, header, layout):
        self.layout = layout
        assert header["tokens"] == list(self.tokens)
        self.parts = {
            name: torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]))
            for name, shape, dtype in layout.components
        }

    def chunk(self, name, offset, data):
        view = self.parts[name].view(torch.uint8).flatten()
        view[offset : offset + len(data)].copy_(
            torch.frombuffer(data, dtype=torch.uint8)
        )

    def validated_record(self, header, layout):
        image = StateImage(self.compatibility, self.tokens, layout, self.parts)
        self.adapter.restore(
            image, self.compatibility, self.indices[: len(self.tokens)], self.slot
        )
        self.validated = True

    def io_finished(self):
        self.done = True

    def transfer_complete(self):
        return self.done and self.fence


def seeded(
    tmp_path,
    monkeypatch,
    sink_factory=CPUCopySink,
    *,
    token_count=9,
    num_pages=128,
    device="cpu",
    **changes,
):
    cm, tm, pm, linear, kv = setup(monkeypatch, num_pages=num_pages, device=device)
    # This is deliberately a pre-existing disk record, independent of controller
    # publication. It exercises recovery after no in-memory ownership survives.
    tokens = tuple(range(1, token_count + 1))
    indices = cm._page_to_token(
        cm._allocate((token_count + cm.page_size - 1) // cm.page_size)
    )
    slot = linear.alloc(1)[0]
    kv._kv_buffer.fill_(17)
    linear.conv_states[:, slot].fill_(29)
    linear.recurrent_states[:, slot].fill_(41)
    linear._aux_states["ple"][slot].copy_(torch.tensor([7, 8, 9]))
    expected = cm.exact_endpoints.adapter.snapshot(
        identity(), tokens, indices[:token_count], slot, max_bytes=1 << 20
    )
    with_store = PrefixStore(
        tmp_path, identity(), quota_bytes=1 << 20, limits=RecordLimits(1 << 20, 4096)
    )
    portable = StateImage(
        expected.compatibility,
        expected.tokens,
        expected.layout,
        {
            name: tensor.cpu().contiguous()
            for name, tensor in expected.inspect().items()
        },
    )
    entry = with_store.write(TensorSource(portable))
    with_store.close()
    cm._free(indices)
    linear.free(slot)
    kv._kv_buffer.fill_(-71)
    settings = DiskSettings(
        str(tmp_path), 1 << 20, 16, 2, 4096, "always", 10.0, "single-instance-test"
    )
    settings = replace(settings, **changes)
    service = PrefixRetention(
        cm.exact_endpoints,
        identity(),
        settings,
        prefill_cost=lambda n: 2.0,
        sink_factory=sink_factory,
    )
    assert service.store.ready.wait(5) and service.store.failure is None
    pm.prefix_retention = service
    return service, cm, tm, pm, linear, kv, tokens, expected, entry


def pending(uid, tokens):
    return PendingReq(
        uid, torch.tensor(tokens, dtype=torch.int32), SamplingParams(max_tokens=4)
    )


def drain(service):
    deadline = time.monotonic() + 5
    while service.outstanding and time.monotonic() < deadline:
        service.poll()
        time.sleep(0.001)
    assert service.outstanding == 0


def test_restart_restore_coalesces_and_publishes_complete_private_state(
    tmp_path, monkeypatch
):
    service, cm, tm, pm, linear, kv, tokens, expected, entry = seeded(
        tmp_path, monkeypatch
    )
    try:
        a, b = pending(1, tokens + (91,)), pending(2, tokens + (92,))
        assert service.maybe_restore(a)
        assert service.maybe_restore(b)
        assert service.stats["restores_coalesced"] == 1
        assert cm.exact_endpoints.restoring_pages == 3
        assert not cm.exact_endpoints.entries
        drain(service)
        assert not a.restore_waiting and not b.restore_waiting
        endpoint = cm.exact_endpoints.entries[tokens]
        mapping = torch.cat([endpoint.anchor.kv_indices, endpoint.tail[:1]])
        restored = cm.exact_endpoints.adapter.snapshot(
            identity(), tokens, mapping, endpoint.state_slot, max_bytes=1 << 20
        )
        for name, tensor in expected.inspect().items():
            assert torch.equal(tensor, restored.inspect()[name]), name
        cm.check_integrity()
        assert (
            cm.exact_endpoints.restoring_pages
            == cm.exact_endpoints.restoring_states
            == 0
        )
        assert len(service.store.descriptors()) == 1
    finally:
        assert service.close()


def test_cancel_and_pending_gpu_fence_retain_private_resources(tmp_path, monkeypatch):
    started, proceed = threading.Event(), threading.Event()

    class Delayed(CPUCopySink):
        def begin(self, header, layout):
            started.set()
            assert proceed.wait(5)
            super().begin(header, layout)

    service, cm, _, _, linear, _, tokens, _, _ = seeded(
        tmp_path, monkeypatch, sink_factory=Delayed
    )
    pages_before, slots_before = len(cm.free_slots), linear.num_free_slots
    try:
        req = pending(1, tokens + (91,))
        assert service.maybe_restore(req)
        assert started.wait(5)
        ticket = service.restore.ticket
        ticket.payload.fence = False
        service.detach(1)
        service.poll()
        assert service.outstanding == 1 and not req.restore_waiting
        assert len(cm.free_slots) == pages_before - 3
        proceed.set()
        assert ticket.finished.wait(5)
        service.poll()
        assert service.outstanding == 1 and not cm.exact_endpoints.entries
        assert linear.num_free_slots == slots_before - 1
        ticket.payload.fence = True
        drain(service)
        assert len(cm.free_slots) == pages_before
        assert linear.num_free_slots == slots_before
        cm.check_integrity()
    finally:
        proceed.set()
        assert service.close()


def test_disk_waiter_does_not_block_ready_request_and_abort_detaches(
    tmp_path, monkeypatch
):
    service, cm, tm, pm, linear, _, tokens, _, _ = seeded(tmp_path, monkeypatch)
    try:
        pm.add_one_req(
            UserMsg(
                1,
                torch.tensor(tokens + (91,), dtype=torch.int32),
                SamplingParams(max_tokens=4),
            )
        )
        pm.add_one_req(
            UserMsg(
                2,
                torch.tensor([77, 78], dtype=torch.int32),
                SamplingParams(max_tokens=4),
            )
        )
        batch = pm.schedule_next_batch(128)
        assert [r.uid for r in batch.reqs] == [2]
        assert [r.uid for r in pm.pending_list] == [2, 1]
        assert pm.abort_req(1) is None
        req = batch.reqs[0]
        assert pm.abort_req(2) is req
        cm.allocate_paged([req])
        req.aborted = True
        cm.cache_req(req, finished=True)
        tm.free(req.table_idx)
        drain(service)
        assert not cm.exact_endpoints.entries
        cm.check_integrity()
    finally:
        assert service.close()


def test_corrupt_payload_never_publishes_and_returns_all_allocations(
    tmp_path, monkeypatch
):
    service, cm, _, _, linear, _, tokens, _, entry = seeded(tmp_path, monkeypatch)
    before = len(cm.free_slots), linear.num_free_slots
    try:
        # Corrupt after recovery/index validation. Header remains valid so private
        # writes may occur, but no shared endpoint may escape the checksum gate.
        path = service.store.store._path(entry.key)
        with path.open("r+b") as record:
            record.seek(-1, 2)
            byte = record.read(1)
            record.seek(-1, 2)
            record.write(bytes([byte[0] ^ 1]))
        req = pending(1, tokens + (91,))
        assert service.maybe_restore(req)
        drain(service)
        assert not req.restore_waiting and not cm.exact_endpoints.entries
        assert service.stats["read_failures"] == 1
        assert before == (len(cm.free_slots), linear.num_free_slots)
        cm.check_integrity()
    finally:
        assert service.close()


def test_unknown_measured_cost_recomputes_and_multimodal_never_reads(
    tmp_path, monkeypatch
):
    service, cm, _, _, _, _, tokens, _, _ = seeded(
        tmp_path, monkeypatch, policy="measured"
    )
    try:
        assert not service.maybe_restore(pending(1, tokens + (91,)))
        service._record_cost("read", len(tokens), 0.1)
        assert service.maybe_restore(pending(2, tokens + (92,)))
        multimodal = pending(3, tokens + (93,))
        multimodal.mm_embeds = torch.ones(1)
        assert not service.maybe_restore(multimodal)
        service.detach(2)
        drain(service)
        assert not cm.exact_endpoints.entries
    finally:
        assert service.close()


def test_timeout_detaches_without_reusing_private_storage(tmp_path, monkeypatch):
    service, cm, _, _, linear, _, tokens, _, _ = seeded(tmp_path, monkeypatch)
    now = [0.0]
    service.clock = lambda: now[0]
    before = len(cm.free_slots), linear.num_free_slots
    try:
        req = pending(1, tokens + (91,))
        assert service.maybe_restore(req)
        ticket = service.restore.ticket
        ticket.payload.fence = False
        now[0] = 11.0
        service.poll()
        assert not req.restore_waiting and service.outstanding == 1
        assert not cm.exact_endpoints.entries
        assert ticket.finished.wait(5)
        ticket.payload.fence = True
        drain(service)
        assert before == (len(cm.free_slots), linear.num_free_slots)
        cm.check_integrity()
    finally:
        assert service.close()


def test_unavailable_store_recomputes_through_normal_admission(tmp_path, monkeypatch):
    from tests.runtime.checkpoint.test_exact import consume, finish
    from sparklab.runtime.scheduler.prefill import ChunkedReq

    service, cm, tm, pm, linear, kv, tokens, _, _ = seeded(tmp_path, monkeypatch)
    try:
        service.store.store.failed = True
        service.poll()
        assert service.disabled and service.stats["worker_unavailable"] == 1
        pm.add_one_req(
            UserMsg(
                7,
                torch.tensor(tokens + (91,), dtype=torch.int32),
                SamplingParams(max_tokens=4),
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
        assert service.outstanding == 0
        cm.check_integrity()
    finally:
        assert service.close()
