import importlib.util
import io
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "benchmarks/frameworks/bench_diffusion.py"
SPEC = importlib.util.spec_from_file_location("bench_diffusion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def response(events, done=True):
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return io.BytesIO("".join(lines).encode())


def test_single_block_throughput_includes_denoising_latency():
    ticks = iter([0.0, 2.0, 2.5])
    def opener(request, **kwargs):
        body = json.loads(request.data)
        assert "temperature" not in body and "seed" not in body
        return response([
            {"choices": [{"delta": {"content": "a whole block"}}]},
            {"usage": {"completion_tokens": 256}},
        ])
    result = MODULE.stream_once("http://test", "model", 256, False,
        clock=lambda: next(ticks), opener=opener)
    assert result["ttft_seconds"] == 2.0
    assert result["content_deltas"] == 1
    assert result["completion_tokens_per_second"] == pytest.approx(102.4)


@pytest.mark.parametrize("events,done,error", [
    ([{"choices": [{"delta": {"content": "x"}}]}], True, "usage"),
    ([{"error": {"message": "OOM"}}], True, "server stream error"),
    ([{"choices": [{"delta": {"content": "x"}}]}], False, "incomplete"),
])
def test_benchmark_fails_closed(events, done, error):
    with pytest.raises(RuntimeError, match=error):
        MODULE.stream_once("http://test", "model", 256, False,
            opener=lambda *a, **kw: response(events, done))


def test_reasoning_and_content_in_same_delta_are_preserved():
    result = MODULE.stream_once("http://test", "model", 256, True,
        opener=lambda *a, **kw: response([
            {"choices": [{"delta": {"reasoning": "think", "content": "answer"}}]},
            {"usage": {"completion_tokens": 2}},
        ]))
    assert result["reasoning"] == "think"
    assert result["content"] == "answer"
