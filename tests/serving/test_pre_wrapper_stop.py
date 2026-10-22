"""Real parent stop while the child has not imported Torch or installed handlers."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest
import sparklab

from tests.serving.test_worker_lifecycle import _health, _wait, _semaphores


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX spawn/stop ownership")
@pytest.mark.parametrize("phase", ["partial-launch", "returned-handle"])
def test_parent_stop_reaps_pre_wrapper_worker_and_releases_queue(tmp_path, phase):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    extra = ["--workers", "3", "--pause-launch"] if phase == "partial-launch" else []
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.serving.worker_lifecycle_probe",
            "--root",
            str(tmp_path),
            "--port",
            str(port),
            "--hold-entry",
            *extra,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(Path(sparklab.__file__).resolve().parent.parent),
        },
    )
    workers = []
    try:
        _wait(lambda: (tmp_path / "worker-0.entry.json").exists(), process)
        if phase == "partial-launch":
            _wait(lambda: (tmp_path / "launch-paused").exists(), process)
        else:
            _wait(lambda: (_health(port) or {}).get("status") == "loading", process)
        entry = json.loads((tmp_path / "worker-0.entry.json").read_text())
        assert entry["torch_imported"] is False and entry["default_sigterm"] is True
        assert not (tmp_path / "worker-0.initializing").exists()
        parent = json.loads((tmp_path / "parent.json").read_text())
        workers = parent["workers"]
        assert workers == [entry["pid"]]
        owned = _semaphores(tmp_path)
        assert len(owned) == 3 and all(path.exists() for path in owned)
        started = time.monotonic()
        process.terminate()
        stdout, stderr = process.communicate(timeout=15.0)
        (tmp_path / "stdout.log").write_bytes(stdout)
        (tmp_path / "stderr.log").write_bytes(stderr)
        assert process.returncode == 143
        elapsed = time.monotonic() - started
        assert elapsed < 12.0
        finalized = json.loads((tmp_path / "parent.finalized").read_text())
        assert finalized["exitcodes"] in ([1], [-signal.SIGINT])
        assert not any(Path(f"/proc/{pid}").exists() for pid in workers)
        assert not any(path.exists() for path in owned)
        assert b"exceeded shutdown grace" not in stdout + stderr
        assert b"resource_tracker:" not in stderr and b"leaked semaphore" not in stderr
        assert b"Backend supervisor:" not in stdout + stderr
        (tmp_path / "pre-wrapper-report.json").write_text(
            json.dumps(
                {
                    "phase": phase,
                    "seconds": elapsed,
                    "parent_exitcode": process.returncode,
                    "worker_exitcodes": finalized["exitcodes"],
                    "workers_reaped": len(workers),
                    "queue_semaphores_released": len(owned),
                    "torch_imported": False,
                    "scope": "Controlled first worker entry before Torch and stop-handler setup",
                }
            )
        )
    finally:
        # The gate prevents this owned child from reaching process-group detach.
        if process.poll() is None or any(
            Path(f"/proc/{pid}").exists() for pid in workers
        ):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5.0)
