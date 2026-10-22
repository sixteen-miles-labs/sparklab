"""Actual HTTP/IPC cancellation and drain; synthetic model replies only."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import urllib.request

import pytest
import sparklab

from tests.serving.test_worker_lifecycle import _health, _wait, _semaphores

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="POSIX process groups")


@pytest.mark.parametrize(
    "scenario", ["complete", "disconnect", "parent-term", "group-int", "group-term"]
)
def test_http_stream_owns_request_until_completion_or_disconnect(tmp_path, scenario):
    if (root := os.getenv("SPARKLAB_HTTP_LIFECYCLE_EVIDENCE_ROOT")) is not None:
        tmp_path = Path(root) / scenario
        tmp_path.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
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
                "--http-stream",
            ],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(Path(sparklab.__file__).resolve().parent.parent),
            },
        )
        first, proceed = threading.Event(), threading.Event()
        lines, errors = [], []

        def consume():
            try:
                request = urllib.request.Request(
                    base + "/generate",
                    data=json.dumps({"prompt": "probe", "max_tokens": 2}).encode(),
                    headers={"content-type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=10.0) as response:
                    for raw in response:
                        line = raw.decode().strip()
                        if line:
                            lines.append(line)
                        if "first" in line:
                            first.set()
                            assert proceed.wait(10.0)
                            if scenario == "disconnect":
                                break
            except BaseException as exc:
                errors.append(repr(exc))

        reader = threading.Thread(target=consume, daemon=True)
        try:
            _wait(lambda: (_health(port) or {}).get("status") == "ok", process)
            reader.start()
            _wait(first.is_set, process)
            with urllib.request.urlopen(base + "/probe-state") as response:
                assert json.load(response) == {"acks": 1, "events": 1}
            parent = json.loads((tmp_path / "parent.json").read_text())
            if scenario.startswith("group-"):
                assert all(
                    os.getpgid(pid) != os.getpgid(process.pid)
                    for pid in parent["workers"]
                )
            if scenario == "parent-term":
                process.terminate()
            elif scenario.startswith("group-"):
                os.killpg(
                    process.pid,
                    signal.SIGINT if scenario == "group-int" else signal.SIGTERM,
                )
            if scenario in ("parent-term", "group-int", "group-term"):
                _wait(
                    lambda: "Waiting for connections to close"
                    in (tmp_path / "stderr.log").read_text(),
                    process,
                )
                assert all(Path(f"/proc/{pid}").exists() for pid in parent["workers"])
            proceed.set()
            if scenario == "disconnect":
                reader.join(3.0)
                _wait(
                    lambda: (tmp_path / "request-0.aborted").exists(),
                    process,
                    timeout=3.0,
                )
                # A retry uses a new uid and completes independently after the
                # disconnected request has been removed from the worker queue.
                (tmp_path / "release-http").touch()
                retry = urllib.request.Request(
                    base + "/generate",
                    data=json.dumps({"prompt": "retry", "max_tokens": 2}).encode(),
                    headers={"content-type": "application/json"},
                )
                with urllib.request.urlopen(retry, timeout=5.0) as response:
                    assert (
                        response.read().decode()
                        == 'data: {"text": "first"}\n\ndata: {"text": "last"}\n\ndata: [DONE]\n\n'
                    )
            else:
                (tmp_path / "release-http").touch()
                reader.join(5.0)
                assert not reader.is_alive(), (
                    "stream stranded after stop with unfinished worker output"
                )
                assert not errors
                assert lines == [
                    'data: {"text": "first"}',
                    'data: {"text": "last"}',
                    "data: [DONE]",
                ]
            if scenario in ("complete", "disconnect"):

                def drained():
                    with urllib.request.urlopen(base + "/probe-state") as response:
                        return json.load(response) == {"acks": 0, "events": 0}

                _wait(drained, process, timeout=3.0)
                process.terminate()
            assert process.wait(timeout=8.0) == (0 if scenario == "group-int" else 143)
            parent = json.loads((tmp_path / "parent.json").read_text())
            assert all(not Path(f"/proc/{pid}").exists() for pid in parent["workers"])
            assert not any(p.exists() for p in _semaphores(tmp_path))
            assert (tmp_path / "worker-0.drained").exists()
            assert "resource_tracker:" not in (tmp_path / "stderr.log").read_text()
            assert "Backend supervisor:" not in (tmp_path / "stdout.log").read_text()
            (tmp_path / "http-result.json").write_text(
                json.dumps(
                    {
                        "lines": lines,
                        "errors": errors,
                        "returncode": process.returncode,
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            proceed.set()
            if process.poll() is None:
                process.kill()  # preserve the resource tracker for queue cleanup
                process.wait(timeout=5.0)
            # Backend workers have their own groups after the frontend takes
            # ownership of terminal signals. Clean up only the PIDs we created.
            if (parent_path := tmp_path / "parent.json").exists():
                for pid in json.loads(parent_path.read_text())["workers"]:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if reader.ident is not None:
                reader.join(timeout=1.0)
