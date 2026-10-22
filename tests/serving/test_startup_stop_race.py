"""A stop at startup-error reporting cannot skip a constructed owner's drain."""

import gc
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from sparklab.serving import launch


@pytest.mark.parametrize("constructed", [False, True])
@pytest.mark.parametrize("cleanup_error", [False, True])
def test_reporting_interrupt_preserves_cause_and_cleans_constructed_owner(
    monkeypatch, constructed, cleanup_error
):
    import sparklab.runtime.scheduler as module

    original = ValueError("startup synchronization failed")
    cleaned = []

    class Worker:
        def __init__(self, args):
            if not constructed:
                raise original

        def sync_all_ranks(self):
            raise original

        def shutdown(self, *, failed=False):
            cleaned.append(failed)
            if cleanup_error:
                raise OSError("drain failed")

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "Scheduler", Worker)
    monkeypatch.setattr(launch, "_detach_process_group", lambda: None)
    monkeypatch.setattr(launch, "_install_worker_stop_handlers", lambda: lambda: None)
    monkeypatch.setattr(launch, "_report_startup_error", interrupt)
    args = SimpleNamespace(tp_info=SimpleNamespace(is_primary=lambda: False))
    queue = SimpleNamespace(close=lambda: None, join_thread=lambda: None)
    with pytest.raises(ValueError) as failure:
        launch._run_scheduler(args, queue)
    assert failure.value is original
    assert cleaned == ([True] if constructed else [])


def _race_worker(root, queue, stop_signal):
    import torch
    import sparklab.runtime.scheduler as module
    from sparklab.runtime.checkpoint.async_store import AsyncPrefixStore
    from sparklab.runtime.checkpoint.record import TensorSource, RecordLimits
    from tests.runtime.checkpoint.test_records import identity, image

    directory = Path(root)

    class Worker:
        def __init__(self, args):
            self.locks = [multiprocessing.Lock()]
            self.gate = threading.Event()
            entered = threading.Event()
            (directory / "worker.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "cuda_initialized": torch.cuda.is_initialized(),
                        "semaphores": [lock._semlock.name for lock in self.locks],
                    }
                )
            )
            plain = TensorSource(image())
            gate = self.gate

            class Paused:
                compatibility, tokens, layout = (
                    plain.compatibility,
                    plain.tokens,
                    plain.layout,
                )

                def chunks(self, name, size):
                    entered.set()
                    assert gate.wait(5.0)
                    yield from plain.chunks(name, size)

            self.store = AsyncPrefixStore(
                directory / "prefix",
                identity(),
                quota_bytes=5000,
                limits=RecordLimits(4096, 4096),
            )
            assert self.store.ready.wait(5.0) and self.store.failure is None
            self.ticket = self.store.submit_write(Paused())
            assert entered.wait(5.0)

        def sync_all_ranks(self):
            raise ValueError("constructed synchronization failed")

        def shutdown(self, *, failed=False):
            assert failed
            self.gate.set()  # only cleanup can unblock the real file writer
            self.store.close(timeout=0)
            deadline = time.monotonic() + 5.0
            while self.store.outstanding and time.monotonic() < deadline:
                for ticket in self.store.poll():
                    assert ticket.error is None
                    self.store.release(ticket)
                time.sleep(0.001)
            assert self.store.outstanding == 0 and self.store.close()
            self.locks.clear()
            (directory / "drained").touch()

    original_report = launch._report_startup_error

    def report_then_stop(queue, exc):
        original_report(queue, exc)
        (directory / "reported").touch()
        os.kill(os.getpid(), stop_signal)

    module.Scheduler = Worker
    launch._report_startup_error = report_then_stop
    args = SimpleNamespace(tp_info=SimpleNamespace(is_primary=lambda: False))
    launch._run_scheduler(args, queue)


@pytest.mark.skipif(
    sys.platform != "linux", reason="POSIX signals, locks and real files"
)
@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_real_reporting_stop_drains_prefix_write_and_exits_with_original_error(
    tmp_path, stop_signal
):
    from sparklab.runtime.checkpoint.store import PrefixStore
    from sparklab.runtime.checkpoint.record import RecordLimits
    from tests.runtime.checkpoint.test_records import identity

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    semaphores = [
        queue._rlock._semlock.name,
        queue._wlock._semlock.name,
        queue._sem._semlock.name,
    ]
    child = ctx.Process(target=_race_worker, args=(str(tmp_path), queue, stop_signal))
    try:
        child.start()
        child.join(timeout=15.0)
        assert child.exitcode == 1  # the real startup error remains fatal
        assert queue.get(timeout=2.0) == (
            "error",
            "ValueError: constructed synchronization failed",
        )
        assert (tmp_path / "reported").exists() and (tmp_path / "drained").exists()
        record = json.loads((tmp_path / "worker.json").read_text())
        assert record["cuda_initialized"] is False
        semaphores.extend(record["semaphores"])
        store = PrefixStore(
            tmp_path / "prefix",
            identity(),
            quota_bytes=5000,
            limits=RecordLimits(4096, 4096),
        )
        try:
            entries = store.descriptors()
            assert len(entries) == 1
            assert store.read(entries[0].key)[0]["tokens"] == [1, 2, 3, 4]
        finally:
            store.close()
        (tmp_path / "race-report.json").write_text(
            json.dumps(
                {
                    "signal": int(stop_signal),
                    "exitcode": child.exitcode,
                    "startup_error": "ValueError",
                    "validated_records": len(entries),
                    "cuda_initialized": False,
                }
            )
        )
    finally:
        if child.is_alive():
            child.kill()
        child.join(timeout=3.0)
        queue.close()
        queue.join_thread()
        queue = None
        gc.collect()
    assert not Path(f"/proc/{child.pid}").exists()
    assert all(
        not (Path("/dev/shm") / ("sem." + name.lstrip("/"))).exists()
        for name in semaphores
    )
