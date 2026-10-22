"""Real protocol/worker stage plumbing; small correctness smoke, not overhead proof."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import pytest
import torch

from tests.e2e.test_cache_rebuild import (
    _free_port,
    _free_gpu_gib,
    _get,
    _post,
    _wait_until_serving,
)

pytestmark = pytest.mark.needs_weights


def test_real_protocol_stages_reconcile_with_completed_http_requests(tmp_path):
    model = os.getenv("SPARKLAB_TIMING_TEST_MODEL")
    if not model or not Path(model).is_dir() or not torch.cuda.is_available():
        pytest.skip("requires CUDA and SPARKLAB_TIMING_TEST_MODEL complete checkpoint")
    if _free_gpu_gib() < float(os.getenv("SPARKLAB_TIMING_MIN_FREE_GIB", "30")):
        pytest.skip("insufficient free physical GPU memory")
    args = json.loads(os.getenv("SPARKLAB_TIMING_SERVER_ARGS", "[]"))
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("SPARKLAB_TIMING_SERVER_ARGS must be a JSON string array")
    root = Path(os.getenv("SPARKLAB_TIMING_EVIDENCE_ROOT", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{_free_port()}"
    cases = [
        ("/v1/completions", {"prompt": "Say ready.", "max_tokens": 16}, "[DONE]"),
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Say ready."}],
                "max_tokens": 16,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "[DONE]",
        ),
        (
            "/v1/responses",
            {"input": "Say ready.", "max_output_tokens": 16},
            ("response.completed", "response.incomplete"),
        ),
        (
            "/v1/messages",
            {"messages": [{"role": "user", "content": "Say ready."}], "max_tokens": 16},
            "message_stop",
        ),
    ]
    observations = []
    command = [
        sys.executable,
        "-m",
        "sparklab",
        "serve",
        "--model-path",
        model,
        "--served-model-name",
        "timing-test",
        "--host",
        "127.0.0.1",
        "--port",
        base.rsplit(":", 1)[1],
        *args,
    ]
    env = {
        **os.environ,
        "PYTHONPATH": os.getenv("SPARKLAB_TIMING_PACKAGE_PATH", "python"),
        "SPARKLAB_RUNTIME_TIMING": "1",
        "SPARKLAB_RUNTIME_TRACE_CAPACITY": "128",
    }
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            _wait_until_serving(base, proc, time.monotonic() + 300.0)
            for endpoint, payload, terminal in cases:
                started = time.monotonic()
                request = urllib.request.Request(
                    base + endpoint,
                    data=json.dumps(
                        {
                            "model": "timing-test",
                            "stream": True,
                            "temperature": 0.0,
                            **payload,
                        }
                    ).encode(),
                    headers={"content-type": "application/json"},
                )
                lines = []
                with urllib.request.urlopen(request, timeout=180.0) as response:
                    for line in response:
                        if line.strip():
                            lines.append(
                                {
                                    "time": time.monotonic(),
                                    "line": line.decode().strip(),
                                }
                            )
                markers = (terminal,) if isinstance(terminal, str) else terminal
                assert lines and any(
                    any(marker in item["line"] for marker in markers) for item in lines
                ), lines
                for item in lines:
                    if (
                        item["line"].startswith("data:")
                        and item["line"] != "data: [DONE]"
                    ):
                        data = json.loads(item["line"][5:])
                        assert not data.get("error"), data
                        assert data.get("type") not in ("error", "response.failed"), (
                            data
                        )
                observations.append(
                    {
                        "endpoint": endpoint,
                        "start": started,
                        "end": time.monotonic(),
                        "events": lines,
                    }
                )
            code, _ = _post(
                base,
                "/v1/chat/completions",
                {
                    "model": "timing-test",
                    "messages": [{"role": "user", "content": "bad budget"}],
                    "max_tokens": 0,
                },
            )
            assert code in (400, 422)
            # Final-body telemetry is committed after the ASGI send returns.
            deadline = time.monotonic() + 5.0
            while True:
                snapshot = _get(base, "/v1/stats")
                timing = snapshot["runtime_timing"]
                if (
                    timing["counters"].get("http_samples", 0) >= 5
                    or time.monotonic() >= deadline
                ):
                    break
                time.sleep(0.01)
            counts = timing["counters"]
            assert counts["http_samples"] == 5
            assert counts["admitted_samples"] == counts["generated_samples"] == 4
            assert counts["http_complete"] == 4 and counts["http_error"] == 1
            assert counts.get("invalid_intervals", 0) == 0
            for name in (
                "frontend_work",
                "tokenization_queue",
                "tokenization",
                "backend_transport",
                "admission_queue",
                "dispatch_wait",
                "first_generation",
                "ingress_first_generated",
                "detokenization_queue",
                "detokenization",
                "frontend_transport",
                "semantic_ready",
            ):
                assert sum(timing["histograms"][name]) == 4, name
            assert sum(timing["histograms"]["http_complete"]) == 5
            assert snapshot["requests"]["active"] == 0
            assert snapshot["requests"]["completed"] == 4
            native = snapshot["native_runtime"]
            assert native["schema_version"] == 1 and native["age_seconds"] >= 0
            assert native["resources"]["kv_total_pages"] > 0
            assert native["scheduler"]["trace"] == []
            assert len(native["scheduler"]["gpu_costs"]) <= 64
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30.0)
            (root / "observations.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "requests": observations,
                        "stats": locals().get("snapshot"),
                        "exit_code": proc.returncode,
                        "note": "Four-stream stage/accounting smoke; no quality/performance/soak claim.",
                    },
                    indent=2,
                )
                + "\n"
            )
