"""Completed file reads keep private state alive through an actual pending DMA.

The test adds an idempotent final 24-byte auxiliary-state DMA to a real restore.
A mapped host semaphore holds that DMA, without replacing CUDA event queries or
spinning a kernel. This is synthetic state, not full-model/API qualification.
"""

import ctypes
import json
import threading
import time

import pytest
import torch

from sparklab.core import SamplingParams
from sparklab.runtime.checkpoint.config import DiskSettings
from sparklab.runtime.checkpoint.service import PrefixRetention
from sparklab.runtime.checkpoint.transfer import GPUStateSink
from sparklab.runtime.scheduler.utils import PendingReq
from tests.runtime.checkpoint.test_exact import setup
from tests.runtime.checkpoint.test_state import identity
from tests.runtime.checkpoint.test_transfer_cancel import seed_record

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real pending CUDA fence gate"
)


class HostGate:
    """CUDA driver wait on mapped pinned host memory; bounded host rescue."""

    def __init__(self):
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.host = ctypes.c_void_p()
        self.device = ctypes.c_uint64()
        self.streams = []
        self.timer = None
        self.expired = threading.Event()
        self.call(
            "cuMemHostAlloc",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
            ctypes.byref(self.host),
            4,
            2,
        )
        try:
            self.call(
                "cuMemHostGetDevicePointer_v2",
                [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint],
                ctypes.byref(self.device),
                self.host,
                0,
            )
        except BaseException:
            self.call("cuMemFreeHost", [ctypes.c_void_p], self.host)
            raise
        self.word = ctypes.c_uint32.from_address(self.host.value)
        # Probe optional driver support on the test thread, so unavailable
        # primitives skip here rather than raising inside the I/O worker.
        self.word.value = 1
        try:
            stream = torch.cuda.current_stream()
            self.call(
                "cuStreamWaitValue32_v2",
                [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint],
                ctypes.c_void_p(stream.cuda_stream),
                self.device,
                1,
                1,
            )
            stream.synchronize()
        except BaseException:
            self.call("cuMemFreeHost", [ctypes.c_void_p], self.host)
            raise
        self.word.value = 0

    def call(self, name, types, *args):
        function = getattr(self.driver, name)
        function.argtypes, function.restype = types, ctypes.c_int
        code = function(*args)
        if code == 801:
            pytest.skip(f"CUDA driver does not support {name}")
        if code:
            raise RuntimeError(f"{name} failed with CUDA driver error {code}")

    def wait(self, stream):
        self.call(
            "cuStreamWaitValue32_v2",
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint],
            ctypes.c_void_p(stream.cuda_stream),
            self.device,
            1,
            1,
        )
        self.streams.append(stream)

        def rescue():
            self.expired.set()
            self.release()

        self.timer = threading.Timer(10, rescue)
        self.timer.start()

    def release(self):
        self.word.value = 1

    def close(self):
        self.release()
        for stream in self.streams:
            stream.synchronize()
        if self.timer is not None:
            self.timer.cancel()
            self.timer.join()
        self.call("cuMemFreeHost", [ctypes.c_void_p], self.host)


@pytest.mark.parametrize("action", ["cancel-all", "keep-waiter", "timeout", "shutdown"])
@torch.inference_mode()
def test_completed_read_retains_real_pending_dma_owners(tmp_path, monkeypatch, action):
    tokens, expected, entry = seed_record(tmp_path / "prefixes", monkeypatch)
    cm, _, _, linear, kv = setup(monkeypatch, device="cuda")
    other = torch.cuda.Stream()
    progress = torch.zeros(1, device="cuda")
    # Prime resources/kernels before holding a stream: lazy initialization must
    # not accidentally synchronize the context behind the semaphore.
    with torch.cuda.stream(other):
        progress.add_(1)
    other.synchronize()
    gate = HostGate()
    instances = []

    class FinalDMA(GPUStateSink):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.final_aux = expected.inspect()["aux.ple"].pin_memory()
            instances.append(self)

        @torch.inference_mode()
        def io_finished(self):
            # Original record validation and all its staged copies have run.
            # Add one bounded, idempotent write to the same private allocation.
            # The real stream/event stays pending after I/O completion is queued.
            try:
                assert self.validated
                gate.wait(self._stream)
                with torch.cuda.stream(self._stream):
                    self._parts["aux.ple"].copy_(self.final_aux, non_blocking=True)
                    self._event.record(self._stream)
            finally:
                super().io_finished()

    now = [time.monotonic()]
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
        sink_factory=FinalDMA,
        clock=lambda: now[0],
    )
    report = {
        "action": action,
        "model_execution": False,
        "final_dma_bytes": 24,
        "mapped_host_gate_bytes": 4,
        "observations": [],
        "passed": False,
    }

    def observe(name):
        report["observations"].append(
            {
                "stage": name,
                **service.snapshot(),
                "cuda_event_complete": instances[0]._event.query(),
                "cuda_stream_complete": instances[0]._stream.query(),
                "io_finished": instances[0]._io_done,
                "ticket_finished": ticket.finished.is_set(),
                "watchdog_fired": gate.expired.is_set(),
            }
        )

    try:
        assert service.store.ready.wait(5) and service.store.failure is None
        pages, slots = len(cm.free_slots), linear.num_free_slots
        a, b = [
            PendingReq(
                uid,
                torch.tensor(tokens + (91,), dtype=torch.int32),
                SamplingParams(max_tokens=4),
            )
            for uid in (11, 12)
        ]
        assert service.maybe_restore(a)
        ticket = service.restore.ticket
        assert ticket.finished.wait(5)
        sink = instances[0]
        assert ticket.error is None and sink.validated and sink._io_done
        assert not gate.expired.is_set() and not sink._event.query()
        assert not sink._stream.query() and not sink.transfer_complete()
        assert service.store.store.pins[entry.key] == 1
        assert service.maybe_restore(b)
        service.poll()
        assert service.completed == [ticket]
        assert not cm.exact_endpoints.entries
        observe("file-complete-dma-pending")
        with torch.cuda.stream(other):
            progress.add_(1)
        other.synchronize()
        assert progress.item() == 2 and not sink._event.query()
        report["independent_stream_completed_while_dma_pending"] = True
        if action in ("cancel-all", "keep-waiter"):
            service.detach(a.uid)
            if action == "cancel-all":
                service.detach(b.uid)
            service.poll()
        elif action == "timeout":
            now[0] += 31
            service.poll()
            assert service.stats["restore_timeout"] == 1
        else:
            assert not service.close(timeout=0.01)
        assert not a.restore_waiting
        assert b.restore_waiting == (action == "keep-waiter")
        assert not sink._event.query() and not gate.expired.is_set()
        assert service.outstanding == 1 and service.completed == [ticket]
        assert (len(cm.free_slots), linear.num_free_slots) == (pages - 3, slots - 1)
        assert not cm.exact_endpoints.entries
        assert service.store.store.pins[entry.key] == 1
        # The cache integrity checker is idle-only and performs CUDA reductions.
        # Inspect host ownership here, then run its full check after the DMA drains.
        observe("action-dma-still-pending")
        assert not gate.expired.is_set() and not sink._event.query()
        gate.release()
        sink._event.synchronize()
        deadline = time.monotonic() + 5
        while service.outstanding and time.monotonic() < deadline:
            service.poll()
            time.sleep(0.001)
        assert not service.outstanding and not service.completed
        assert not service.store.store.pins[entry.key]
        assert not a.restore_waiting and not b.restore_waiting
        if action == "keep-waiter":
            endpoint = cm.exact_endpoints.entries[tokens]
            mapping = torch.cat([endpoint.anchor.kv_indices, endpoint.tail[:1]])
            restored = cm.exact_endpoints.adapter.snapshot(
                identity(), tokens, mapping, endpoint.state_slot, max_bytes=1 << 20
            )
            for name, tensor in expected.inspect().items():
                actual = restored.inspect()[name].cpu().contiguous().view(torch.uint8)
                assert torch.equal(actual, tensor.contiguous().view(torch.uint8)), name
            assert service.stats["restores_completed"] == 1
        else:
            assert not cm.exact_endpoints.entries
            assert service.stats["restores_completed"] == 0
        cm.exact_endpoints.clear()
        evicted = cm.prefix_cache.evict_full(1 << 20)
        cm._free(evicted.kv_indices)
        linear.free(evicted.mamba_slots)
        cm.check_integrity()
        assert (len(cm.free_slots), linear.num_free_slots) == (pages, slots)
        report["released_pool_counts"] = {"pages": pages, "slots": slots}
        observe("released")
        assert not gate.expired.is_set()
        report["passed"] = True
    finally:
        gate.release()
        try:
            report["closed"] = service.close(timeout=10)
        finally:
            report["watchdog_fired"] = gate.expired.is_set()
            gate.close()
            (tmp_path / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
        assert report["closed"] and not report["watchdog_fired"]
