"""A stop after OS spawn must wait until the Process handle is registered."""

import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import pytest
import sparklab

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="POSIX spawn signal ownership"
)


def _cleanup_children():
    # This isolated controller is a Linux subreaper, so counterfactual orphan
    # children remain ours to terminate and reap instead of stranding under PID 1.
    children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    for child in children.read_text().split():
        try:
            os.kill(int(child), signal.SIGKILL)
        except ProcessLookupError:
            pass
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break


def _probe(root, stop_signal):
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER; controller only
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (
        (directory / "stdout.log").open("w") as stdout,
        (directory / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.serving.worker_lifecycle_probe",
                "--root",
                str(directory),
                "--port",
                str(port),
                "--hold-entry",
                "--interrupt-spawn",
                str(stop_signal),
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
            code = process.wait(timeout=25.0)
            observed_gap = json.loads((directory / "spawn-gap.json").read_text())
            (directory / "observed-exit.json").write_text(
                json.dumps(
                    {
                        "code": code,
                        "worker_alive_before_controller_cleanup": Path(
                            f"/proc/{observed_gap['worker']}"
                        ).exists(),
                        "finalized": json.loads(
                            (directory / "parent.finalized").read_text()
                        ),
                    }
                )
            )
            # CPython can surface an uncaught KeyboardInterrupt as either an
            # exception exit or a SIGINT exit (including on 3.11 patch releases).
            # Both must still satisfy every worker/semaphore ownership check.
            expected_codes = (143,) if stop_signal == signal.SIGTERM else (1, -signal.SIGINT)
            assert code in expected_codes, (
                code,
                (directory / "stderr.log").read_text(),
            )
            gap = json.loads((directory / "spawn-gap.json").read_text())
            assert gap["handle_assigned"] is False
            entry = json.loads((directory / "worker-0.entry.json").read_text())
            assert entry["torch_imported"] is False and entry["default_sigterm"] is True
            finalized = json.loads((directory / "parent.finalized").read_text())
            assert finalized["exitcodes"] in ([1], [-signal.SIGINT])
            assert not Path(f"/proc/{gap['worker']}").exists()
            assert all(
                not (Path("/dev/shm") / ("sem." + name.lstrip("/"))).exists()
                for name in gap["semaphores"]
            )
            errors = (directory / "stderr.log").read_text()
            assert (
                "exceeded shutdown grace" not in errors
                and "resource_tracker:" not in errors
            )
            (directory / "spawn-gap-report.json").write_text(
                json.dumps(
                    {
                        "signal": stop_signal,
                        "parent_exitcode": process.returncode,
                        "worker_exitcodes": finalized["exitcodes"],
                        "worker_reaped": True,
                        "handle_assigned_at_injection": False,
                        "torch_imported": False,
                        "scope": "Actual signal after Popen launch, before Process registration",
                    }
                )
            )
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5.0)
            _cleanup_children()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and POSIX spawn")
@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_stop_inside_process_start_keeps_child_visible_to_cleanup(
    tmp_path, stop_signal
):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.serving.test_spawn_gap_stop",
            str(tmp_path),
            str(int(stop_signal)),
        ],
        capture_output=True,
        timeout=35.0,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    (tmp_path / "controller.stdout.log").write_bytes(result.stdout)
    (tmp_path / "controller.stderr.log").write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode()


class StopRequested(BaseException):
    pass


def _fake_handlers(monkeypatch, callback):
    handlers = {
        signal.SIGINT: callback,
        signal.SIGTERM: callback,
        signal.SIGHUP: signal.SIG_DFL,
    }
    monkeypatch.setattr(signal, "getsignal", handlers.__getitem__)
    monkeypatch.setattr(signal, "signal", lambda sig, fn: handlers.__setitem__(sig, fn))
    monkeypatch.setattr(signal, "pthread_sigmask", lambda how, values: set())
    monkeypatch.setattr(signal, "raise_signal", lambda sig: handlers[sig](sig, None))
    return handlers


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_callable_stop_replays_after_start_returns_and_handlers_restore(
    monkeypatch, stop_signal
):
    from sparklab.serving import launch

    seen = []

    class Process:
        visible = False

        def start(self):
            handlers[stop_signal](stop_signal, None)
            self.visible = True

    process = Process()

    def stop(sig, frame):
        seen.append(process.visible)
        raise StopRequested

    handlers = _fake_handlers(monkeypatch, stop)
    before = dict(handlers)
    with pytest.raises(StopRequested):
        getattr(launch, "_start_owned_process", lambda p: p.start())(process)
    assert process.visible and seen == [True]
    assert handlers == before


def test_returning_handlers_are_coalesced_and_keep_authority(monkeypatch):
    from sparklab.serving import launch

    seen = []

    class Process:
        visible = False

        def start(self):
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGTERM):
                handlers[sig](sig, None)
            self.visible = True

    process = Process()
    handlers = _fake_handlers(
        monkeypatch, lambda sig, frame: seen.append((sig, process.visible))
    )
    before = dict(handlers)
    getattr(launch, "_start_owned_process", lambda p: p.start())(process)
    assert seen == [(signal.SIGTERM, True), (signal.SIGINT, True)]
    assert handlers == before


def test_start_error_remains_original_even_with_pending_stop(monkeypatch):
    from sparklab.serving import launch

    original = ValueError("start failed")
    seen = []

    def stop(sig, frame):
        seen.append(sig)
        raise StopRequested

    handlers = _fake_handlers(monkeypatch, stop)
    before = dict(handlers)

    class Process:
        def start(self):
            handlers[signal.SIGINT](signal.SIGINT, None)
            raise original

    with pytest.raises(ValueError) as failure:
        getattr(launch, "_start_owned_process", lambda p: p.start())(Process())
    assert failure.value is original and seen == [signal.SIGINT] and handlers == before


def test_stop_during_handler_restoration_drops_deferred_frames(monkeypatch):
    import weakref
    from sparklab.serving import launch

    refs = []
    handlers = _fake_handlers(monkeypatch, lambda *args: None)
    before = dict(handlers)

    class Frame:
        pass

    class Process:
        def start(self):
            frame = Frame()
            refs.append(weakref.ref(frame))
            handlers[signal.SIGTERM](signal.SIGTERM, frame)

    def unmask(how, values):
        if how == signal.SIG_SETMASK:
            raise StopRequested
        return set()

    monkeypatch.setattr(signal, "pthread_sigmask", unmask)
    with pytest.raises(StopRequested):
        getattr(launch, "_start_owned_process", lambda p: p.start())(Process())
    assert refs[0]() is None and handlers == before


def test_ignored_and_default_handlers_remain_unchanged(monkeypatch):
    from sparklab.serving import launch

    handlers = _fake_handlers(monkeypatch, lambda *args: None)
    handlers[signal.SIGTERM] = signal.SIG_IGN
    before = dict(handlers)

    class Process:
        def start(self):
            assert handlers[signal.SIGTERM] == signal.SIG_IGN
            assert handlers[signal.SIGHUP] == signal.SIG_DFL

    getattr(launch, "_start_owned_process", lambda p: p.start())(Process())
    assert handlers == before


def test_non_main_thread_never_changes_signal_handlers(monkeypatch):
    import threading
    from sparklab.serving import launch

    seen, errors = [], []

    def forbidden(*args):
        raise AssertionError("background start changed signal handlers")

    monkeypatch.setattr(signal, "signal", forbidden)
    process = type("Process", (), {"start": lambda self: seen.append("started")})()

    def run():
        try:
            getattr(launch, "_start_owned_process", lambda p: p.start())(process)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=3.0)
    assert not thread.is_alive() and not errors and seen == ["started"]


if __name__ == "__main__":
    _probe(sys.argv[1], int(sys.argv[2]))
