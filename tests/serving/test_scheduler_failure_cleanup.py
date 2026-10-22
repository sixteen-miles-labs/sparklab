"""Failure cleanup preserves the fatal cause and drains local owners only."""

from types import SimpleNamespace

import pytest

from sparklab.serving import launch
from sparklab.runtime.scheduler.scheduler import Scheduler
from sparklab.runtime.scheduler.io import SchedulerIOMixin


@pytest.mark.parametrize("stage", ["run", "sync", "system_exit"])
@pytest.mark.parametrize("cleanup_error", [False, True])
def test_worker_failure_drains_once_and_preserves_original(
    monkeypatch, stage, cleanup_error
):
    events = []
    original = (
        SystemExit(7)
        if stage == "system_exit"
        else RuntimeError("original worker failure")
    )

    class Worker:
        def __init__(self, args):
            pass

        def sync_all_ranks(self):
            if stage == "sync":
                raise original

        def run_forever(self):
            raise original

        def shutdown(self, *, failed=False):
            events.append(("shutdown", failed))
            if cleanup_error:
                raise OSError("local drain failed")

    import sparklab.runtime.scheduler as module

    monkeypatch.setattr(module, "Scheduler", Worker)
    monkeypatch.setattr(launch, "_detach_process_group", lambda: None)
    monkeypatch.setattr(launch, "_install_worker_stop_handlers", lambda: lambda: None)
    queue = SimpleNamespace(
        put=lambda value: None,
        close=lambda: events.append("queue-close"),
        join_thread=lambda: events.append("queue-join"),
    )
    args = SimpleNamespace(
        tp_info=SimpleNamespace(is_primary=lambda: False), silent_output=False
    )
    with pytest.raises(type(original)) as failure:
        launch._run_scheduler(args, queue)
    assert failure.value is original
    startup_ack = ["queue-close", "queue-join"] if stage == "sync" else []
    assert events == startup_ack + [("shutdown", True), "queue-close", "queue-join"]


@pytest.mark.parametrize("drained", [False, True])
def test_failed_scheduler_drains_prefix_owners_before_ipc_without_peer_wait(
    monkeypatch, drained
):
    events = []

    def close(timeout):
        assert timeout == 5.0
        events.append("prefix-drain")
        return drained

    def forbidden(*args, **kwargs):
        raise AssertionError("failed rank must not enter normal device/peer teardown")

    owner = SimpleNamespace(
        prefix_retention=SimpleNamespace(close=close),
        sync_all_ranks=forbidden,
        engine=SimpleNamespace(shutdown=forbidden),
    )
    monkeypatch.setattr("torch.cuda.synchronize", forbidden)
    monkeypatch.setattr(
        SchedulerIOMixin, "close_io", lambda self: events.append("ipc-close")
    )
    if drained:
        Scheduler.shutdown(owner, failed=True)
    else:
        with pytest.raises(RuntimeError, match="transfers still own runtime pools"):
            Scheduler.shutdown(owner, failed=True)
    assert events == ["prefix-drain", "ipc-close"]


def test_real_worker_inference_failure_drains_active_prefix_file(tmp_path):
    import json
    import os
    from pathlib import Path
    import signal
    import socket
    import subprocess
    import sys
    import sparklab
    from tests.serving.test_worker_lifecycle import (
        _health,
        _wait,
        _semaphores,
        _worker_finished,
    )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (
        (tmp_path / "stdout.log").open("w") as stdout,
        (tmp_path / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.serving.worker_lifecycle_probe",
                "--root",
                str(tmp_path),
                "--port",
                str(port),
                "--prefix-io",
            ],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": str(Path(sparklab.__file__).resolve().parent.parent),
            },
        )
        try:
            _wait(lambda: (_health(port) or {}).get("status") == "ok", process)
            _wait(lambda: (tmp_path / "worker-0.io-started").exists(), process)
            owned = _semaphores(tmp_path)
            assert owned and all(path.exists() for path in owned)
            (tmp_path / "fail-runtime").touch()
            _wait(lambda: (_health(port) or {}).get("status") == "error", process)
            _wait(lambda: _worker_finished(tmp_path), process, timeout=5.0)
            assert (tmp_path / "worker-0.failure-cleanup").exists()
            assert (tmp_path / "worker-0.drained").exists()
            # The I/O gate opens only from shutdown. Decode the actual committed
            # record independently; a cleanup marker alone cannot prove a drain.
            from tests.runtime.checkpoint.test_records import identity
            from sparklab.runtime.checkpoint.store import PrefixStore
            from sparklab.runtime.checkpoint.record import RecordLimits

            with_store = PrefixStore(
                tmp_path / "prefixes-0",
                identity(),
                quota_bytes=5000,
                limits=RecordLimits(4096, 4096),
            )
            try:
                entries = with_store.descriptors()
                assert len(entries) == 1
                header, _ = with_store.read(entries[0].key)
                assert header["tokens"] == [1, 2, 3, 4]
            finally:
                with_store.close()
            process.terminate()
            assert process.wait(timeout=10.0) == 143
            errors = (tmp_path / "stderr.log").read_text()
            assert "probe inference failure with prefix I/O" in errors
            assert "failure cleanup did not complete" not in errors
            assert not any(path.exists() for path in owned)
            parent = json.loads((tmp_path / "parent.json").read_text())
            assert all(not Path(f"/proc/{pid}").exists() for pid in parent["workers"])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5.0)
