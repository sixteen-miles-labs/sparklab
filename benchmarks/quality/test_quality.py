from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run_quality", HERE / "run_quality.py")
run_quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_quality
SPEC.loader.exec_module(run_quality)


def test_select_indices_and_extract_answer():
    assert run_quality.select_indices("0,2-3,2", 5) == [0, 2, 3]
    assert run_quality.extract_answer(r"work \\boxed{17}") == "17"
    assert run_quality.extract_answer("The final answer is 23.") == "23"


def test_validate_scenario_enforces_paper_turns_and_evaluator():
    scenario = {
        "steps": [{"argv": ["true"]}] * 3,
        "evaluator": {"argv": ["true"]},
    }
    run_quality.validate_scenario("W2", scenario, False)
    try:
        run_quality.validate_scenario("W4", scenario, False)
    except ValueError as exc:
        assert "exactly 13" in str(exc)
    else:
        raise AssertionError("W4 accepted fewer than thirteen turns")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            payload = {"data": [{"id": "test-model"}]}
        elif self.path == "/health":
            payload = {"status": "ok", "model": "test-model", "version": "test"}
        else:
            self.send_error(404)
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        json.loads(self.rfile.read(length))
        events = [
            {"choices": [{"delta": {"reasoning_content": "2+3=5"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": r"\\boxed{5}"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def test_w1_smoke_scores_visible_final_channel(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = tmp_path / "aime.jsonl"
    dataset.write_text(json.dumps({"problem": "What is 2+3?", "answer": "5"}) + "\n")
    args = SimpleNamespace(
        aime=dataset,
        problems="0",
        mode="smoke",
        max_tokens=32,
        temperature=None,
        top_p=None,
        seed=None,
        include_output=False,
        request_timeout=10,
        base_url=f"http://127.0.0.1:{server.server_port}",
        workload="W1",
        model="test-model",
        weight_format="BF16",
    )
    try:
        result = run_quality.run_w1(args, "test-model", None)
    finally:
        server.shutdown()
        thread.join()
    assert result["quality"] == {
        "passed": True,
        "correct": 1,
        "total": 1,
        "accuracy": 1,
        "gate": "final extracted answer equals the AIME reference answer",
    }
    assert result["requests"][0]["completion_tokens"] == 4
