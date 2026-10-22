"""Real process/signal/IPC ownership; no weights or CUDA initialization."""

import json
import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import queue
import time
from types import SimpleNamespace
import urllib.request

import pytest
import sparklab

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="native POSIX worker lifecycle"
)


def _wait(predicate, process, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        if process.poll() is not None:
            raise AssertionError(f"probe exited before condition: {process.returncode}")
        time.sleep(0.01)
    raise TimeoutError("probe did not reach lifecycle condition")


def _health(port):
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=0.3
        ) as response:
            return json.load(response)
    except OSError:
        return None


def _semaphores(root):
    paths = []
    for file in root.glob("*.json"):
        for name in json.loads(file.read_text()).get("semaphores", []):
            if name:
                paths.append(Path("/dev/shm") / ("sem." + name.lstrip("/")))
    return paths


def _worker_finished(root):
    pid = json.loads((root / "worker-0.json").read_text())["pid"]
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return True
    return stat[stat.rfind(")") + 2 :].split()[0] == "Z"


@pytest.mark.parametrize(
    "scenario",
    [
        "ready",
        "repeat",
        "loading",
        "startup-error",
        "error-stop-race",
        "forced",
        "prefix-io",
        "partial-launch",
        "tokenizer",
        "tokenizer-group",
    ],
)
def test_real_serve_stop_reaps_workers_and_releases_owned_semaphores(
    tmp_path, scenario
):
    if (evidence_root := os.getenv("SPARKLAB_LIFECYCLE_EVIDENCE_ROOT")) is not None:
        tmp_path = Path(evidence_root) / scenario
        tmp_path.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    extra = {
        "loading": ["--load-delay", "30"],
        "startup-error": ["--fail-startup"],
        "error-stop-race": ["--fail-startup"],
        "forced": ["--workers", "3", "--locks", "0", "--ignore-stop"],
        "prefix-io": ["--prefix-io"],
        "partial-launch": ["--workers", "3", "--fail-launch"],
        "tokenizer": ["--tokenizer"],
        "tokenizer-group": ["--tokenizer"],
    }.get(scenario, [])
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.serving.worker_lifecycle_probe",
            "--root",
            str(tmp_path),
            "--port",
            str(port),
            *extra,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(sparklab.__file__).resolve().parent.parent),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    try:
        if scenario == "partial-launch":
            stdout, stderr = process.communicate(timeout=20.0)
            (tmp_path / "stdout.log").write_bytes(stdout)
            (tmp_path / "stderr.log").write_bytes(stderr)
            assert process.returncode == 1
            assert b"probe partial launch failure" in stderr
        else:
            if scenario == "loading":
                _wait(lambda: (tmp_path / "worker-0.initializing").exists(), process)
                _wait(lambda: (_health(port) or {}).get("status") == "loading", process)
            elif scenario in ("startup-error", "error-stop-race"):
                failure = _wait(
                    lambda: (
                        h
                        if (h := _health(port)) and h.get("status") == "error"
                        else None
                    ),
                    process,
                )
                assert "probe startup failure" in failure["message"]
                # The error ack precedes process exit. Observe that exit before
                # testing a subsequent stop, rather than racing the finalizer.
                if scenario == "startup-error":
                    _wait(lambda: _worker_finished(tmp_path), process, timeout=5.0)
            else:
                _wait(lambda: (_health(port) or {}).get("status") == "ok", process)
                count = 3 if scenario == "forced" else 1
                _wait(
                    lambda: all(
                        (tmp_path / f"worker-{i}.running").exists()
                        for i in range(count)
                    ),
                    process,
                )
            if scenario == "prefix-io":
                _wait(lambda: (tmp_path / "worker-0.io-started").exists(), process)
            owned = _semaphores(tmp_path)
            assert owned
            if scenario not in ("startup-error", "error-stop-race"):
                assert all(path.exists() for path in owned)
            started = time.monotonic()
            if scenario == "tokenizer-group":
                worker_pid = json.loads((tmp_path / "worker-0.json").read_text())["pid"]
                assert os.getpgid(worker_pid) != os.getpgid(process.pid)
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            if scenario == "repeat":
                _wait(
                    lambda: (tmp_path / "worker-0.draining").exists(),
                    process,
                    timeout=3.0,
                )
                pid = json.loads((tmp_path / "worker-0.json").read_text())["pid"]
                os.kill(pid, signal.SIGTERM)
                process.terminate()
            stdout, stderr = process.communicate(timeout=20.0)
            (tmp_path / "stdout.log").write_bytes(stdout)
            (tmp_path / "stderr.log").write_bytes(stderr)
            assert process.returncode == 143
            if scenario == "forced":
                assert (
                    time.monotonic() - started < 15.0
                )  # shared 10s grace, not 3 x 10s
                assert b"exceeded shutdown grace" in stdout + stderr
        (tmp_path / "stdout.log").write_bytes(stdout)
        (tmp_path / "stderr.log").write_bytes(stderr)
        finalized = json.loads((tmp_path / "parent.finalized").read_text())
        if scenario in ("ready", "repeat", "prefix-io", "tokenizer", "tokenizer-group"):
            assert (tmp_path / "worker-0.drained").exists()
            assert finalized["exitcodes"] == [0]
        elif scenario == "loading":
            assert finalized["exitcodes"] == [0]
        elif scenario == "startup-error":
            assert finalized["exitcodes"] == [1]
        elif scenario == "error-stop-race":
            # A completed failure or an explicit interrupt while the interpreter
            # exits is legitimate; successful startup is not. The original error
            # was checked through /health before the stop and remains in logs.
            assert finalized["exitcodes"][0] in (1, -signal.SIGINT)
            assert b"probe startup failure" in stdout + stderr
        elif scenario == "forced":
            assert finalized["exitcodes"] == [-signal.SIGKILL] * 3
        if scenario != "forced":
            assert (
                b"leaked semaphore" not in stderr and b"resource_tracker:" not in stderr
            )
        assert not any(path.exists() for path in _semaphores(tmp_path))
        parent = json.loads((tmp_path / "parent.json").read_text())
        assert all(not Path(f"/proc/{pid}").exists() for pid in parent["workers"])
        for file in tmp_path.glob("worker-*.json"):
            assert json.loads(file.read_text())["cuda_initialized"] is False
        if scenario == "prefix-io":
            assert len(list(tmp_path.glob("prefixes-0/*/*.pxc"))) == 1
            assert not list(tmp_path.glob("prefixes-0/*/.tmp-*"))
    finally:
        parent_path = tmp_path / "parent.json"
        workers = (
            json.loads(parent_path.read_text())["workers"]
            if parent_path.exists()
            else []
        )
        if process.poll() is None or any(
            Path(f"/proc/{pid}").exists() for pid in workers
        ):
            # Workers detach from the terminal group. A failed test must also
            # stop its recorded children rather than leave them behind.
            for pid in workers:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.kill()  # leave the resource tracker to unlink owned queues
            except ProcessLookupError:
                pass
            process.communicate(timeout=5.0)


def test_shell_stop_unwinds_default_hangup_without_changing_ctrl_c(monkeypatch):
    from sparklab.serving import api_server

    handlers = {}
    monkeypatch.setattr(api_server, "_SHUTTING_DOWN", threading.Event())
    monkeypatch.setattr(
        api_server, "_GLOBAL_STATE", SimpleNamespace(backend_processes=[])
    )
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(
        signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler)
    )
    api_server._install_shell_stop_handlers()
    assert signal.SIGINT not in handlers
    with pytest.raises(SystemExit) as stopped:
        handlers[signal.SIGHUP](signal.SIGHUP, None)
    assert stopped.value.code == 128 + signal.SIGHUP
    assert api_server._SHUTTING_DOWN.is_set()


def test_explicit_embedding_sigterm_handler_remains_authoritative(monkeypatch):
    from sparklab.serving.api_server import _python_exit_on_default_sigterm

    def original(signum, frame):
        pass

    monkeypatch.setattr(signal, "getsignal", lambda sig: original)

    def forbidden(*args):
        raise AssertionError("replaced an explicit embedding signal handler")

    monkeypatch.setattr(signal, "signal", forbidden)
    with _python_exit_on_default_sigterm():
        assert signal.getsignal(signal.SIGTERM) is original


def test_shell_preserves_explicitly_ignored_stop_signals(monkeypatch):
    from sparklab.serving import api_server

    handlers = {}
    monkeypatch.setattr(api_server, "_SHUTTING_DOWN", threading.Event())
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_IGN)
    monkeypatch.setattr(
        signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler)
    )

    def forbidden(*args):
        raise AssertionError("ignored signal stopped backend workers")

    monkeypatch.setattr(api_server, "_terminate_backend_workers", forbidden)
    api_server._install_shell_stop_handlers()
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    handlers[signal.SIGHUP](signal.SIGHUP, None)
    assert not api_server._SHUTTING_DOWN.is_set()


@pytest.mark.parametrize("kind", ["ZmqPushQueue", "ZmqAsyncPushQueue"])
def test_owned_transport_can_close_without_waiting_for_an_absent_peer(tmp_path, kind):
    code = """
import asyncio, sys
from sparklab import utils

async def main():
    queue = getattr(utils, sys.argv[1])(sys.argv[2], create=False, encoder=lambda msg: msg)
    if sys.argv[1] == 'ZmqAsyncPushQueue':
        await queue.put({'queued': True})
    else:
        queue.put({'queued': True})
    queue.stop(linger=0)

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", code, kind, f"ipc://{tmp_path}/absent"],
        capture_output=True,
        timeout=10.0,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(sparklab.__file__).resolve().parent.parent),
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    assert result.returncode == 0, result.stderr.decode()


def test_scheduler_attempts_every_owned_link_close_even_if_one_fails():
    from sparklab.runtime.scheduler.io import SchedulerIOMixin

    closed = []

    def stop(name, *, linger):
        assert linger == 0
        closed.append(name)
        if name == "bad":
            raise OSError("injected close failure")

    owner = SimpleNamespace(
        _recv_from_tokenizer=SimpleNamespace(stop=lambda **kw: stop("reader", **kw)),
        _send_into_tokenizer=SimpleNamespace(stop=lambda **kw: stop("bad", **kw)),
        _send_into_ranks=SimpleNamespace(stop=lambda **kw: stop("ranks", **kw)),
    )
    with pytest.raises(OSError, match="injected close failure"):
        SchedulerIOMixin.close_io(owner)
    assert sorted(closed) == ["bad", "ranks", "reader"]


def test_frontend_closes_async_ipc_on_its_owner_loop_from_another_thread(
    tmp_path, monkeypatch
):
    from sparklab.serving import api_server
    from sparklab.utils import ZmqAsyncPushQueue, ZmqAsyncPullQueue

    monkeypatch.setattr(api_server, "_SHUTTING_DOWN", threading.Event())
    ready, closed = queue.Queue(), threading.Event()
    errors = []

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        links = []

        async def setup():
            owner = threading.get_ident()
            links.extend(
                [
                    ZmqAsyncPushQueue(
                        f"ipc://{tmp_path}/send", create=False, encoder=lambda x: x
                    ),
                    ZmqAsyncPullQueue(
                        f"ipc://{tmp_path}/recv", create=True, decoder=lambda x: x
                    ),
                ]
            )
            count = []
            for link in links:
                original = link.stop

                def stop(*, linger, original=original):
                    try:
                        assert threading.get_ident() == owner
                        original(linger=linger)
                        count.append(1)
                        if len(count) == 2:
                            closed.set()
                    except BaseException as exc:
                        errors.append(exc)
                        raise

                link.stop = stop
            state = api_server.FrontendManager(SimpleNamespace(), links[0], links[1])
            state._loop = loop
            return state

        try:
            state = loop.run_until_complete(setup())
            loop.call_soon(ready.put, (state, loop))
            loop.run_forever()
        finally:
            for link in links:
                link.stop(linger=0)
            loop.close()

    thread = threading.Thread(target=run)
    thread.start()
    state, loop = ready.get(timeout=5.0)
    try:
        state.shutdown()
        assert closed.wait(2.0) and not errors
        assert state._shutdown_started and state.maintenance_state == "stopping"
        state.shutdown()  # idempotent after transport close
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        assert not thread.is_alive()
