"""Actual TCP/ASGI backpressure with finite synthetic model output over IPC."""

import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import urllib.request

import pytest
import sparklab

from tests.serving.test_worker_lifecycle import _health, _wait, _semaphores

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="POSIX process lifecycle"
)


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=2.0) as response:
        return json.load(response)


def _request(protocol, marker, budget):
    payload = {"model": "lifecycle-probe", "stream": True}
    if protocol in ("generate", "completions"):
        payload.update(prompt=marker, max_tokens=budget)
    elif protocol == "responses":
        payload.update(input=marker, max_output_tokens=budget)
    else:
        payload.update(
            messages=[{"role": "user", "content": marker}], max_tokens=budget
        )
    if protocol in ("completions", "chat"):
        payload["stream_options"] = {"include_usage": True}
    return payload


def _decode(protocol, body):
    lines = [
        line[5:].strip()
        for line in body.decode().splitlines()
        if line.startswith("data:")
    ]
    if protocol in ("generate", "completions", "chat"):
        assert lines[-1] == "[DONE]"
        lines = lines[:-1]
    frames = [json.loads(line) for line in lines]
    assert not any(frame.get("error") for frame in frames)
    usage = None
    if protocol == "generate":
        pieces = [frame["text"] for frame in frames]
    elif protocol in ("completions", "chat"):
        pieces = []
        for frame in frames:
            for choice in frame["choices"]:
                pieces.append(
                    choice.get("text", "")
                    if protocol == "completions"
                    else choice.get("delta", {}).get("content", "")
                )
            if frame.get("usage"):
                usage = frame["usage"]
        assert any(
            choice.get("finish_reason") == "stop"
            for frame in frames
            for choice in frame["choices"]
        )
    elif protocol == "responses":
        pieces = [
            frame["delta"]
            for frame in frames
            if frame["type"] == "response.output_text.delta"
        ]
        assert frames[-1]["type"] == "response.completed"
        usage = frames[-1]["response"]["usage"]
    else:
        pieces = [
            frame["delta"]["text"]
            for frame in frames
            if frame["type"] == "content_block_delta"
            and frame["delta"].get("type") == "text_delta"
        ]
        assert frames[-1]["type"] == "message_stop"
        usage = next(
            frame["usage"]
            for frame in reversed(frames)
            if frame["type"] == "message_delta"
        )
    return pieces, usage


@pytest.mark.parametrize(
    "protocol", ["generate", "completions", "chat", "responses", "anthropic"]
)
@pytest.mark.parametrize("action", ["resume", "disconnect"])
def test_backpressured_stream_keeps_other_http_work_live_and_retires_request(
    tmp_path, action, protocol
):
    if (root := os.getenv("SPARKLAB_PRESSURE_EVIDENCE_ROOT")) is not None:
        tmp_path = Path(root) / protocol / action
        tmp_path.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    endpoint = {
        "generate": "/generate",
        "completions": "/v1/completions",
        "chat": "/v1/chat/completions",
        "responses": "/v1/responses",
        "anthropic": "/v1/messages",
    }[protocol]
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
        slow = response = None
        try:
            _wait(lambda: (_health(port) or {}).get("status") == "ok", process)
            slow = socket.create_connection(("127.0.0.1", port), timeout=10.0)
            slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32768)
            payload = json.dumps(_request(protocol, "pressure", 64)).encode()
            slow.sendall(
                f"POST {endpoint} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\nX-Probe-Pressure: 1\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            response = http.client.HTTPResponse(slow)
            response.begin()
            assert response.status == 200

            def blocked():
                observation = _get(base, "/probe-pressure")
                if (
                    observation["entered"] - observation["completed"] == 1
                    and observation["queued_characters"] >= 128 << 10
                ):
                    return observation
                return None

            observation = _wait(blocked, process, timeout=8.0)
            # Reading no response body has blocked a real ASGI send, not a
            # monkeypatched sleep. A separate request must still finish.
            fast = urllib.request.Request(
                base + endpoint,
                data=json.dumps(_request(protocol, "fast", 2)).encode(),
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(fast, timeout=3.0) as reply:
                pieces, fast_usage = _decode(protocol, reply.read())
                assert "".join(pieces) == "firstlast"
                if protocol in ("completions", "chat"):
                    assert (
                        fast_usage["prompt_tokens"] == 1
                        and fast_usage["completion_tokens"] == 2
                    )
                elif protocol in ("responses", "anthropic"):
                    assert (
                        fast_usage["input_tokens"] == 1
                        and fast_usage["output_tokens"] == 2
                    )
            assert blocked()
            result = {
                "paused": observation,
                "unrelated_request_completed": True,
                "action": action,
                "protocol": protocol,
                "fast_usage": fast_usage,
            }
            if action == "resume":
                body = response.read()
                pieces, usage = _decode(protocol, body)
                assert sum(map(len, pieces)) == 8 << 20
                assert all(piece == "x" * len(piece) for piece in pieces)
                if protocol in ("completions", "chat"):
                    assert (
                        usage["prompt_tokens"] == 1 and usage["completion_tokens"] == 64
                    )
                elif protocol in ("responses", "anthropic"):
                    assert usage["input_tokens"] == 1 and usage["output_tokens"] == 64
                result["slow_usage"] = usage
                result.update(
                    {
                        "delivered_characters": sum(map(len, pieces)),
                        "body_sha256": hashlib.sha256(body).hexdigest(),
                    }
                )
            else:
                response.close()
                slow.close()
                _wait(
                    lambda: (tmp_path / "request-0.aborted").exists(),
                    process,
                    timeout=3.0,
                )
            _wait(
                lambda: _get(base, "/probe-state") == {"acks": 0, "events": 0},
                process,
                timeout=3.0,
            )
            _wait(
                lambda: _get(base, "/probe-accounting")["active"] == 0,
                process,
                timeout=3.0,
            )
            accounting = _get(base, "/probe-accounting")
            assert accounting["completed"] == (2 if action == "resume" else 1)
            assert accounting["prompt"] == 2
            if action == "resume":
                assert accounting["completion"] == 66
            else:
                assert 2 < accounting["completion"] < 66
            result["accounting"] = accounting
            result["final_pressure"] = _get(base, "/probe-pressure")
            assert result["final_pressure"]["queued_characters"] == 0
            if action == "resume":
                assert (
                    result["final_pressure"]["entered"]
                    == result["final_pressure"]["completed"]
                )
            process.terminate()
            assert process.wait(timeout=8.0) == 143
            parent = json.loads((tmp_path / "parent.json").read_text())
            assert all(not Path(f"/proc/{pid}").exists() for pid in parent["workers"])
            assert not any(path.exists() for path in _semaphores(tmp_path))
            error_log = (tmp_path / "stderr.log").read_text()
            assert "resource_tracker:" not in error_log
            assert "Traceback" not in error_log and "Task exception" not in error_log
            (tmp_path / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        finally:
            if response is not None:
                response.close()
            if slow is not None:
                slow.close()
            if process.poll() is None:
                if (path := tmp_path / "parent.json").exists():
                    for pid in json.loads(path.read_text())["workers"]:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                process.kill()
                process.wait(timeout=5.0)
