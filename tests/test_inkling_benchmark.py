import importlib.util
import io
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "bench_inkling", Path(__file__).parents[1] / "benchmarks/frameworks/bench_inkling.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def response(events, done=True):
    lines = ["data: " + json.dumps(event) + "\n\n" for event in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return io.BytesIO("".join(lines).encode())


def test_stream_uses_exact_usage_and_preserves_reasoning(monkeypatch):
    ticks = iter([0.0, 1.0, 3.0, 4.0])
    monkeypatch.setattr(MODULE.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(MODULE, "request", lambda *a: response([
        {"choices": [{"delta": {"reasoning": "thinking"}}]},
        {"choices": [{"delta": {"content": "many tokens in one delta"}, "finish_reason": "length"}]},
        {"usage": {"completion_tokens": 21, "prompt_tokens": 10}},
    ]))
    trial = MODULE.stream("http://test", 256)
    assert trial["decode_tokens_per_second"] == 10.0
    assert trial["completion_tokens_per_second"] == 5.25
    assert trial["ttft_seconds"] == 1.0
    assert trial["reasoning"] == "thinking"
    assert trial["content"] == "many tokens in one delta"


@pytest.mark.parametrize("events,done", [
    ([{"choices": [{"delta": {"content": "x"}}]}], True),
    ([{"choices": [{"delta": {"content": "x"}}]}, {"usage": {"completion_tokens": 4}}], False),
    ([{"error": {"message": "OOM"}}], True),
])
def test_invalid_stream_cannot_produce_performance_claim(monkeypatch, events, done):
    monkeypatch.setattr(MODULE, "request", lambda *a: response(events, done))
    with pytest.raises(RuntimeError):
        MODULE.stream("http://test", 256)
