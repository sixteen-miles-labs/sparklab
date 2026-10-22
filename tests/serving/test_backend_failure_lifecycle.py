"""Real HTTP errors on worker failure, with synthetic model execution over IPC."""

import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest
import sparklab

from tests.serving.test_worker_lifecycle import _health, _wait, _semaphores

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="POSIX worker ownership"
)


def test_worker_failure_terminates_all_protocol_waiters_without_sealing_accounting(
    tmp_path,
):
    if (root := os.getenv("SPARKLAB_FAILURE_EVIDENCE_ROOT")) is not None:
        tmp_path = Path(root)
        tmp_path.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    payloads = {
        "/v1/completions": {"prompt": "probe", "max_tokens": 2},
        "/v1/chat/completions": {
            "messages": [{"role": "user", "content": "probe"}],
            "max_tokens": 2,
            "stream_options": {"include_usage": True},
        },
        "/v1/responses": {"input": "probe", "max_output_tokens": 2},
        "/v1/messages": {
            "messages": [{"role": "user", "content": "probe"}],
            "max_tokens": 2,
        },
    }
    cases = [
        (path, stream, payload)
        for path, payload in payloads.items()
        for stream in (False, True)
    ]
    cases.append(("/generate", True, {"prompt": "probe", "max_tokens": 2}))
    results, errors = {}, []

    def consume(index, path, stream, payload):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(
                {"model": "lifecycle-probe", "stream": stream, **payload}
            ).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            try:
                response = urllib.request.urlopen(request, timeout=12.0)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                body = response.read().decode()
                results[index] = {
                    "path": path,
                    "stream": stream,
                    "status": response.status,
                    "body": body,
                }
        except BaseException as exc:
            errors.append(repr(exc))

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
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": str(Path(sparklab.__file__).resolve().parent.parent),
            },
        )
        readers = []
        try:
            _wait(lambda: (_health(port) or {}).get("status") == "ok", process)
            for i, (path, stream, payload) in enumerate(cases):
                reader = threading.Thread(
                    target=consume, args=(i, path, stream, payload), daemon=True
                )
                readers.append(reader)
                reader.start()
            _wait(
                lambda: len(list(tmp_path.glob("request-*.admitted"))) == len(cases),
                process,
            )
            (tmp_path / "fail-http").touch()
            _wait(lambda: (_health(port) or {}).get("status") == "error", process)
            # One shared wait bound for all nine simultaneous requests.
            import time

            deadline = time.monotonic() + 5.0
            for reader in readers:
                reader.join(max(0.0, deadline - time.monotonic()))
            (tmp_path / "waiters.json").write_text(
                json.dumps(
                    {
                        "admitted": len(cases),
                        "pending": sum(reader.is_alive() for reader in readers),
                        "completed_http": len(results),
                        "transport_errors": errors,
                    },
                    indent=2,
                )
                + "\n"
            )
            assert not any(reader.is_alive() for reader in readers), (
                "backend death stranded admitted HTTP requests"
            )
            assert not errors
            assert len(results) == len(cases)
            for item in results.values():
                assert (
                    "/private/model/path" not in item["body"]
                    and "probe backend failure" not in item["body"]
                )
                if item["stream"]:
                    assert item["status"] == 200
                    assert "first" in item["body"]
                    frames = [
                        json.loads(line[5:])
                        for line in item["body"].splitlines()
                        if line.startswith("data:") and line != "data: [DONE]"
                    ]
                    if item["path"] == "/v1/responses":
                        failed = [
                            f for f in frames if f.get("type") == "response.failed"
                        ]
                        assert (
                            len(failed) == 1
                            and failed[0]["response"]["error"]["code"]
                            == "backend_unavailable"
                        )
                        assert failed[0]["response"].get("usage") is None
                        assert not any(
                            f.get("type")
                            in ("response.completed", "response.incomplete")
                            for f in frames
                        )
                    else:
                        failures = [f["error"] for f in frames if f.get("error")]
                        assert len(failures) == 1
                        assert failures[0]["type"] == (
                            "internal_error"
                            if item["path"] == "/v1/messages"
                            else "server_error"
                        )
                        if item["path"] != "/v1/messages":
                            assert failures[0]["code"] == "backend_unavailable"
                        assert not any(f.get("type") == "message_stop" for f in frames)
                        assert not any(
                            choice.get("finish_reason")
                            for f in frames
                            for choice in f.get("choices", [])
                        )
                        assert not any(
                            f.get("usage") for f in frames if f.get("choices") == []
                        )
                else:
                    assert item["status"] == 503, item
                    assert json.loads(item["body"]).get("error")
            with urllib.request.urlopen(base + "/probe-state") as response:
                assert json.load(response) == {"acks": 0, "events": 0}
            with urllib.request.urlopen(base + "/probe-accounting") as response:
                accounting = json.load(response)
            assert accounting == {
                "active": len(cases),
                "completed": 0,
                "prompt": len(cases),
                "completion": len(cases),
            }
            barrier = urllib.request.Request(
                base + "/v1/admin/prepare-stop",
                data=json.dumps({"drain_timeout_s": 0, "abort_timeout_s": 0}).encode(),
                headers={"content-type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as failure:
                urllib.request.urlopen(barrier, timeout=3.0)
            assert failure.value.code == 503
            barrier_result = json.load(failure.value)
            assert (
                barrier_result["drain_complete"] is False
                and barrier_result["engine_preserved"] is True
            )
            process.terminate()
            assert process.wait(timeout=8.0) == 143
            parent = json.loads((tmp_path / "parent.json").read_text())
            assert all(not Path(f"/proc/{pid}").exists() for pid in parent["workers"])
            assert not any(p.exists() for p in _semaphores(tmp_path))
            assert "resource_tracker:" not in (tmp_path / "stderr.log").read_text()
            (tmp_path / "result.json").write_text(
                json.dumps(
                    {
                        "requests": results,
                        "accounting": accounting,
                        "barrier": barrier_result,
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            if (parent := tmp_path / "parent.json").exists():
                for pid in json.loads(parent.read_text())["workers"]:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if process.poll() is None:
                # Keep the resource tracker alive to unlink the parent's queue
                # primitives after the recorded workers and frontend exit.
                process.kill()
                process.wait(timeout=5.0)
            for reader in readers:
                reader.join(timeout=0.1)


def _frontend():
    from sparklab.serving.api_server import FrontendManager

    incoming = asyncio.Queue()

    class DeadPeer:
        async def put(self, msg):
            raise AssertionError("sent work toward a known dead backend")

    state = FrontendManager(SimpleNamespace(), DeadPeer(), incoming)
    state.maintenance_state = "serving"
    return state, incoming


def test_threaded_failure_preserves_genuine_terminal_and_observes_late_accounting():
    from sparklab.message import UserReply

    async def run():
        state, incoming = _frontend()
        genuine, failed = state.new_user(), state.new_user()
        state._create_listener_once()
        await incoming.put(
            UserReply(
                genuine,
                "complete",
                True,
                prompt_tokens_delta=3,
                completion_tokens_delta=2,
            )
        )
        await asyncio.wait_for(state.event_map[genuine].wait(), 2.0)
        state.fatal_error = "private backend traceback"
        caller = threading.Thread(
            target=lambda: (
                state.fail_pending_generations(),
                state.fail_pending_generations(),
            )
        )
        caller.start()
        caller.join()
        await asyncio.wait_for(state.event_map[failed].wait(), 2.0)
        complete = [reply async for reply in state.wait_for_ack(genuine)]
        assert (
            len(complete) == 1
            and complete[0].incremental_output == "complete"
            and complete[0].error is None
        )
        errors = [reply async for reply in state.wait_for_ack(failed)]
        assert len(errors) == 1 and errors[0].error_code == "backend_unavailable"
        assert state.stats.active == 1 and state.stats.completed == 1
        # A genuine scheduler terminal ack buffered behind the failure still
        # counts. It must not resurrect the already-terminated HTTP request.
        await incoming.put(UserReply(failed, "late", True, completion_tokens_delta=4))

        async def observed():
            while state.stats.active:
                await asyncio.sleep(0)

        await asyncio.wait_for(observed(), 2.0)
        assert state.stats.completed == 2 and state.stats.completion_tokens_total == 6
        assert state.ack_map == state.event_map == {}

    asyncio.run(run())


def test_failure_racing_first_send_and_abort_cannot_forge_accounting_completion():
    from sparklab.core import SamplingParams
    from sparklab.message import TokenizeMsg
    from sparklab.serving.accounting import (
        AdmissionClosedError,
        AccountingDrainError,
        prepare_stop_accounting,
    )

    async def run():
        state, _ = _frontend()
        uid = state.new_user()
        assert state._loop is None
        state.fatal_error = "failure before listener setup"
        with pytest.raises(AdmissionClosedError):
            state.new_user()
        await state.send_one(
            TokenizeMsg(uid=uid, text="probe", sampling_params=SamplingParams())
        )
        replies = [reply async for reply in state.wait_for_ack(uid)]
        assert len(replies) == 1 and replies[0].error_code == "backend_unavailable"
        assert state.stats.active == 1 and state.stats.completed == 0
        with pytest.raises(AccountingDrainError, match="abort barrier"):
            await prepare_stop_accounting(state, drain_timeout_s=0, abort_timeout_s=0)
        assert (
            state.stats.active == 1
            and getattr(state, "_sealed_accounting", None) is None
        )

    asyncio.run(run())
