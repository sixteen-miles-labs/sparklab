"""Real staged CUDA reads retain private owners across cancellation and retry.

The CPU record oracle contains synthetic state, not model inference. Gates pause
the actual I/O worker without substituting transport, checksums or CUDA copies.
"""

import hashlib
import json
import threading
import time

import pytest
import torch

from sparklab.core import SamplingParams
from sparklab.runtime.checkpoint.config import DiskSettings
from sparklab.runtime.checkpoint.record import RecordLimits, TensorSource
from sparklab.runtime.checkpoint.service import PrefixRetention
from sparklab.runtime.checkpoint.store import PrefixStore
from sparklab.runtime.checkpoint.transfer import GPUStateSink
from sparklab.runtime.scheduler.utils import PendingReq
from tests.runtime.checkpoint.test_exact import setup
from tests.runtime.checkpoint.test_state import identity

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA cancellation/transfer gate"
)


def seed_record(root, monkeypatch):
    cm, _, _, linear, kv = setup(monkeypatch)
    tokens = tuple(range(1, 10))
    indices = cm._page_to_token(cm._allocate(3))
    slot = linear.alloc(1)[0]
    # Independent, distinguishable values for every physical state component.
    kv._kv_buffer.copy_(
        torch.arange(kv._kv_buffer.numel()).reshape(kv._kv_buffer.shape)
    )
    linear.conv_states[:, slot].fill_(29)
    linear.recurrent_states[:, slot].fill_(41)
    linear._aux_states["ple"][slot].copy_(torch.tensor([7, 8, 9]))
    expected = cm.exact_endpoints.adapter.snapshot(
        identity(), tokens, indices[:9], slot, max_bytes=1 << 20
    )
    store = PrefixStore(
        root, identity(), quota_bytes=1 << 20, limits=RecordLimits(1 << 20, 4096)
    )
    try:
        entry = store.write(TensorSource(expected))
    finally:
        store.close()
    return tokens, expected, entry


@pytest.mark.parametrize("phase", ["before-copy", "after-copy", "validated"])
@pytest.mark.parametrize("cancel_all", [True, False], ids=["cancel-all", "keep-waiter"])
@torch.inference_mode()
def test_real_gpu_cancel_coalesced_restore_and_poisoned_retry(
    tmp_path, monkeypatch, phase, cancel_all
):
    tokens, expected, entry = seed_record(tmp_path / "prefixes", monkeypatch)
    cm, _, _, linear, kv = setup(monkeypatch, device="cuda")
    entered, proceed = threading.Event(), threading.Event()
    instances = []

    class PausedSink(GPUStateSink):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.first = not instances
            self.paused = False
            instances.append(self)

        def pause(self, at):
            if self.first and not self.paused and phase == at:
                self.paused = True
                entered.set()
                if not proceed.wait(10):
                    raise TimeoutError("test did not release the real read worker")

        def begin(self, header, layout):
            super().begin(header, layout)
            self.pause("before-copy")

        def chunk(self, name, offset, data):
            super().chunk(name, offset, data)
            self.pause("after-copy")

        def validated_record(self, header, layout):
            super().validated_record(header, layout)
            self.pause("validated")

    service = PrefixRetention(
        cm.exact_endpoints,
        identity(),
        DiskSettings(
            str(tmp_path / "prefixes"),
            1 << 20,
            16,
            2,
            4096,
            "always",
            30.0,
            "single-instance-test",
        ),
        prefill_cost=lambda n: None,
        sink_factory=PausedSink,
    )
    report = {
        "synthetic_state": True,
        "model_execution": False,
        "phase": phase,
        "cancel_all": cancel_all,
        "observations": [],
        "passed": False,
    }

    def observe(name):
        report["observations"].append({"stage": name, **service.snapshot()})

    def pending(uid):
        return PendingReq(
            uid,
            torch.tensor(tokens + (91,), dtype=torch.int32),
            SamplingParams(max_tokens=4),
        )

    def drain():
        deadline = time.monotonic() + 10
        while service.outstanding and time.monotonic() < deadline:
            service.poll()
            time.sleep(0.001)
        assert service.outstanding == 0

    def poison():
        kv._kv_buffer.fill_(-113)
        linear.conv_states.fill_(-127)
        linear.recurrent_states.fill_(-131)
        linear._aux_states["ple"].fill_(-137)

    try:
        assert service.store.ready.wait(5) and service.store.failure is None
        pages, slots = len(cm.free_slots), linear.num_free_slots
        poison()
        a, b = pending(11), pending(12)
        assert service.maybe_restore(a)
        assert entered.wait(5)
        assert service.maybe_restore(b)
        assert service.stats["restores_coalesced"] == 1
        ticket, sink = service.restore.ticket, instances[0]
        assert not ticket.finished.is_set() and not sink.transfer_complete()
        assert sink.validated == (phase == "validated")
        assert (sink.transferred_bytes == 0) == (phase == "before-copy")
        # The queued ticket and the executing synchronous read each own a pin.
        assert service.store.store.pins[entry.key] == 2
        observe("paused")
        service.detach(a.uid)
        if cancel_all:
            service.detach(b.uid)
        service.poll()
        assert not a.restore_waiting and b.restore_waiting == (not cancel_all)
        assert ticket.cancelled == cancel_all
        assert service.outstanding == 1 and not cm.exact_endpoints.entries
        assert (len(cm.free_slots), linear.num_free_slots) == (pages - 3, slots - 1)
        assert cm.exact_endpoints.restoring_pages == 3
        assert cm.exact_endpoints.restoring_states == 1
        cm.check_integrity()
        observe("detached")
        proceed.set()
        assert ticket.finished.wait(10)
        drain()
        assert not b.restore_waiting and service.restore is None
        assert sink.transfer_complete()
        assert not service.store.store.pins[entry.key]
        observe("drained")
        if cancel_all:
            assert not cm.exact_endpoints.entries
            assert (len(cm.free_slots), linear.num_free_slots) == (pages, slots)
            assert service.stats["restores_completed"] == 0
            # Recycled allocations must not retain the abandoned partial image.
            poison()
            retry = pending(13)
            assert service.maybe_restore(retry)
            drain()
            assert not retry.restore_waiting and len(instances) == 2
            assert instances[-1].validated
        endpoint = cm.exact_endpoints.entries[tokens]
        mapping = torch.cat([endpoint.anchor.kv_indices, endpoint.tail[:1]])
        restored = cm.exact_endpoints.adapter.snapshot(
            identity(), tokens, mapping, endpoint.state_slot, max_bytes=1 << 20
        )
        hashes = {}
        for name, tensor in expected.inspect().items():
            actual = restored.inspect()[name].cpu().contiguous().view(torch.uint8)
            wanted = tensor.contiguous().view(torch.uint8)
            assert torch.equal(actual, wanted), name
            hashes[name] = hashlib.sha256(actual.numpy().tobytes()).hexdigest()
        report["restored_component_sha256"] = hashes
        assert service.stats["restores_completed"] == 1
        assert not cm.exact_endpoints.restoring_pages
        assert not cm.exact_endpoints.restoring_states
        observe("verified")
        cm.check_integrity()
        cm.exact_endpoints.clear()
        evicted = cm.prefix_cache.evict_full(1 << 20)
        cm._free(evicted.kv_indices)
        linear.free(evicted.mamba_slots)
        cm.check_integrity()
        assert (len(cm.free_slots), linear.num_free_slots) == (pages, slots)
        report["released_pool_counts"] = {"pages": pages, "slots": slots}
        report["passed"] = True
    finally:
        proceed.set()
        report["closed"] = service.close(timeout=10)
        (tmp_path / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
        assert report["closed"]
