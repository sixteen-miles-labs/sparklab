"""Independent wall-clock fixtures and real HTTP streams for mixed-load evidence."""

import importlib.util
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

spec = importlib.util.spec_from_file_location(
    "runtime_mixed", Path(__file__).parents[1] / "benchmarks/bench_runtime_mixed.py"
)
mixed = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mixed
spec.loader.exec_module(mixed)


def test_delivery_latency_and_successful_work_use_independent_timeline():
    a = mixed.Observation(
        "a", 0, 0.25, ended=4, deliveries=[1, 1.5, 3], completion_tokens=12, done=True
    )
    b = mixed.Observation("b", 1, 2, ended=6, error="timeout", completion_tokens=100)
    summary = mixed.summarize([a, b])
    assert (
        summary["successful_work_tokens_per_second"] == 2
    )  # 12 completed / six seconds including timeout
    assert summary["success_rate"] == 0.5
    assert summary["ttft_p50_seconds"] == 0.75
    assert summary["delivery_gap_p95_seconds"] == pytest.approx(1.45)
    assert summary["client_queue_delay_p95_seconds"] == pytest.approx(0.9625)
    assert summary["delivery_gap_count"] == 2
    assert summary["delivery_gap_p99_seconds"] is None


def test_speculative_burst_is_one_delivery_not_fabricated_token_arrivals():
    result = mixed.Observation("burst", 0, 0)
    mixed.observe_event(result, {"choices": [{"text": "many tokens in one burst"}]}, 2)
    mixed.observe_event(result, {"choices": [{"text": "next burst"}]}, 5)
    mixed.observe_event(
        result,
        {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 16}},
        6,
    )
    result.done, result.ended = True, 6
    summary = mixed.summarize([result])
    assert summary["delivery_gap_count"] == 1
    assert summary["delivery_gap_p50_seconds"] == 3
    assert summary["completed_output_tokens"] == 16


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                {"choices": [{"text": "hello", "finish_reason": None}]},
                {"choices": [{"text": " world", "finish_reason": "length"}]},
                {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
            ]
            if payload.get("bad_usage"):
                events[-1]["usage"].pop("completion_tokens")
            try:
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                if not payload.get("missing_done"):
                    self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    thread.join()
    httpd.server_close()


def test_real_stream_reconciles_usage_and_output_and_counts_failures(server):
    import time

    case = {
        "id": "ok",
        "expected_prompt_tokens": 3,
        "request": {"prompt": [1, 2, 3], "max_tokens": 2},
    }
    result = mixed.run_case(server, "fixture", case, time.monotonic(), 5)
    assert result.success and result.output == "hello world"
    assert len(result.deliveries) == 2 and result.completion_tokens == 2
    for mutation in [{"bad_usage": True}, {"missing_done": True}, {"max_tokens": 3}]:
        altered = dict(case, request={**case["request"], **mutation})
        failed = mixed.run_case(server, "fixture", altered, time.monotonic(), 5)
        assert failed.error is not None and not failed.success
    cancelled = mixed.run_case(
        server, "fixture", dict(case, cancel_after_events=1), time.monotonic(), 5
    )
    assert cancelled.cancelled and not cancelled.success


def test_comparison_rejects_mismatched_and_incomplete_work():
    report = {
        "identity": {"artifact": "a"},
        "workload_sha256": "w",
        "summary": {
            "attempted": 1,
            "completed": 1,
            "successful_work_tokens_per_second": 10,
            "delivery_gap_p95_seconds": 2,
            "ttft_p95_seconds": 4,
        },
    }
    candidate = json.loads(json.dumps(report))
    candidate["summary"]["delivery_gap_p95_seconds"] = 1
    assert (
        mixed.compare(report, candidate)["delivery_gap_p95_seconds"]["change_percent"]
        == -50
    )
    for mutate in [
        lambda c: c["identity"].update(artifact="b"),
        lambda c: c.update(workload_sha256="other"),
        lambda c: c["summary"].update(completed=0),
    ]:
        c = json.loads(json.dumps(report))
        mutate(c)
        with pytest.raises(ValueError):
            mixed.compare(report, c)


@pytest.mark.parametrize("at", [float("nan"), float("inf"), -1])
def test_nonfinite_arrivals_are_rejected(at):
    with pytest.raises(ValueError):
        mixed.validate_cases([{"id": "x", "at": at, "request": {"max_tokens": 1}}])
