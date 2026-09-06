"""Keep the optimization quality protocol deterministic and its grading honest."""

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "benchmarks/quality/qwen4_regression.py"
    spec = importlib.util.spec_from_file_location("qwen4_regression_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_smoke_cases_and_mgsm_subset():
    module = _module()
    cases = module.cases()
    assert len(cases) == len({row[0] for row in cases}) == 52
    assert module.INDICES == list(range(2, 250, 5))
    assert len(module.INDICES) == 50
    assert len(module.CODE_CASES) == 5


def test_smoke_distinguishes_correct_answer_from_strict_format(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(
        module,
        "cases",
        lambda: [
            ("exact", "unused", "42", "final"),
            ("format", "unused", "42", "final"),
            ("wrong", "unused", "42", "final"),
            ("object", "unused", {"sum": 42}, "json"),
        ],
    )
    answers = iter(
        ["FINAL=42", "The answer is FINAL=42", "FINAL=42 then FINAL=41", '{"sum":42}']
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": next(answers)}}]}
            ).encode()

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_, **__: Response())
    path = tmp_path / "report.json"
    module.run_smoke("http://localhost", "fixture", path)
    report = json.loads(path.read_text())
    assert report["status"] == "complete"
    assert report["expected_total"] == report["total"] == 4
    assert report["passed"] == 2
    assert [row["answer_correct"] for row in report["cases"]] == [
        True,
        True,
        False,
        True,
    ]


def test_lifecycle_protocol_checks_boundaries_and_closes_streams(monkeypatch, tmp_path):
    module = _module()
    current_label = ""

    def request(_origin, _model, prompt, *, tokens=1024, extra=None, **_):
        nonlocal current_label
        finish = "stop"
        if prompt.startswith("List"):
            answer, finish = "1", "length"
        elif prompt.startswith("Copy"):
            answer = "START-ALPHA "
        elif prompt.startswith("Reply exactly"):
            answer = prompt.removeprefix("Reply exactly with ")
        else:
            last = extra["messages"][-1]["content"]
            if "Replace any previous label" in last:
                current_label = last.split(" is ")[1].split(".")[0]
                answer = "ACK"
            else:
                answer = current_label
        return {
            "usage": {"completion_tokens": tokens},
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": answer},
                }
            ],
        }

    closed = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            closed.append(True)

        def __iter__(self):
            for _ in range(3):
                yield b'data: {"choices":[{"delta":{"content":"part"}}]}\n'

    monkeypatch.setattr(module, "request", request)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_, **__: Stream())
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    path = tmp_path / "lifecycle.json"
    module.run_lifecycle("http://localhost", "fixture", path)
    result = json.loads(path.read_text())
    assert result["status"] == "complete"
    assert result["passed"] == result["expected_total"] == result["total"] == 15
    assert len(closed) == 3
    assert [row["observed_chunks"] for row in result["cases"][-3:]] == [1, 2, 3]


def test_interrupted_lifecycle_report_is_not_complete(monkeypatch, tmp_path):
    module = _module()
    calls = 0

    def request(*_, tokens, **__):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("server failed")
        return {
            "usage": {"completion_tokens": tokens},
            "choices": [{"finish_reason": "length"}],
        }

    monkeypatch.setattr(module, "request", request)
    path = tmp_path / "partial.json"
    with pytest.raises(RuntimeError, match="server failed"):
        module.run_lifecycle("http://localhost", "fixture", path)
    result = json.loads(path.read_text())
    assert result["status"] == "running"
    assert result["total"] == 3
    assert result["expected_total"] == 15
