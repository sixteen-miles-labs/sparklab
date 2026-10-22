"""Independent record facts and actual HTTP controls for the session validator."""

import copy
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading

import pytest

spec = importlib.util.spec_from_file_location(
    "runtime_sessions",
    Path(__file__).parents[1] / "benchmarks/validate_runtime_sessions.py",
)
sessions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sessions)
EXPECTED = {
    "k0": {"fact": "k0", "value": "AX-4317", "source": "P001"},
    "k1": {"fact": "k1", "value": "2031-06-19", "source": "P500"},
    "k2": {"fact": "k2", "value": 7341, "source": "P999"},
}
# Handwritten case answers are independent of the runner's answer constants.
CASE_SUMMARY = {
    "case_id": "SYN-1042",
    "original_deadline": {"value": "2031-04-30", "source": "L001"},
    "positions": [
        {
            "party": "Oak Ltd",
            "statement": "headers were missing",
            "status": "unverified",
            "source": "L002",
        },
        {
            "party": "Juniper Labs",
            "statement": "export delivered on 2031-04-29",
            "status": "unverified",
            "source": "L003",
        },
    ],
    "remedy": {
        "action": "corrected export",
        "deadline": "2031-05-08",
        "source": "L004",
    },
    "court_finding": None,
}
CASE_CHRONOLOGY = {
    "documents": [
        {"date": "2031-04-02", "source": "L001"},
        {"date": "2031-05-02", "source": "L002"},
        {"date": "2031-05-03", "source": "L003"},
        {"date": "2031-05-04", "source": "L004"},
    ]
}


def render(payload):
    # This HTTP fixture has no model. Its oracle defines one token per space-
    # separated word; full-model runs use TokenizeManager instead.
    text = json.dumps(payload["messages"], sort_keys=True)
    return {
        "prompt_tokens": len(text.split()),
        "token_ids_sha256": sessions.digest(text),
    }


def test_reference_tokenizer_uses_actual_wire_order_for_nested_tool_schema(tmp_path, monkeypatch):
    import torch
    import sparklab.tokenizer.tokenize as tokenizer_module
    import sparklab.utils

    seen = []

    class Manager:
        def __init__(self, _tokenizer):
            pass

        def tokenize(self, messages):
            seen.append(json.dumps(messages[0].tools))
            return [torch.tensor([1, 2])]

    monkeypatch.setattr(tokenizer_module, "TokenizeManager", Manager)
    monkeypatch.setattr(sparklab.utils, "load_tokenizer", lambda _: object())
    reference = sessions.tokenizer_renderer(tmp_path)
    payload = sessions.request_payload("fixture", [{"role":"user","content":"lookup fact k0"}], 256, tool=True)
    payload["tool_choice"] = "auto"
    # The first payload has schema properties/type in construction order; the
    # second is what the HTTP JSON decoder actually receives.
    reference(payload)
    reference(json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"))))
    assert seen[0] == seen[1]


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        seen = []
        ids = {}
        fault = None

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.seen.append(payload)
            last = payload["messages"][-1]
            text = last.get("content", "")
            calls = None
            finish = "stop"
            if payload.get("tools"):
                key = re.search(r"fact (k[012])", text)[1]
                call_id = f"call-{len(self.seen)}"
                self.ids[call_id] = key
                calls = [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "lookup_fact",
                            "arguments": json.dumps({"fact": key}),
                        },
                    }
                ]
                content = None
                finish = "tool_calls"
                if self.fault == "tool":
                    calls[0]["function"]["arguments"] = '{"fact":"wrong"}'
            elif last["role"] == "tool":
                assistant = payload["messages"][-2]
                cid = last["tool_call_id"]
                assert cid == assistant["tool_calls"][0]["id"] and cid in self.ids
                content = text
            elif text.startswith("LEGAL SUMMARY:"):
                answer = copy.deepcopy(CASE_SUMMARY)
                if self.fault == "legal-status":
                    answer["positions"][0]["status"] = "established"
                elif self.fault == "legal-source":
                    answer["positions"][1]["source"] = "L001"
                elif self.fault == "legal-deadline":
                    answer["original_deadline"]["value"] = "2031-05-08"
                elif self.fault == "legal-omission":
                    answer["positions"].pop()
                elif self.fault == "legal-ruling":
                    answer["court_finding"] = "supplier liable"
                elif self.fault == "legal-extra":
                    answer["damages"] = 1800
                elif self.fault == "legal-type":
                    answer["court_finding"] = False
                content = json.dumps(answer)
            elif text.startswith("LEGAL CHRONOLOGY:"):
                answer = copy.deepcopy(CASE_CHRONOLOGY)
                if self.fault == "legal-order":
                    answer["documents"].reverse()
                content = json.dumps(answer)
            elif text.startswith("SUMMARY:"):
                content = json.dumps({"facts": list(EXPECTED.values())})
            elif "AMENDMENT-A" in text:
                content = '{"fact":"k0","value":"AX-8879","source":"AMENDMENT-A"}'
            else:
                key = re.search(r"fact (k[012])", text)[1]
                # A branch amendment must not leak into the parent request.
                wrong = key == "k0" and "AMENDMENT-A" in json.dumps(payload["messages"])
                content = json.dumps(
                    {**EXPECTED[key], "value": "AX-8879"} if wrong else EXPECTED[key]
                )
            if self.fault == "fact" and len(self.seen) == 1:
                content = '{"fact":"k0","value":"incorrect","source":"P001"}'
            if self.fault == "source" and len(self.seen) == 1:
                content = '{"fact":"k0","value":"AX-4317","source":"P999"}'
            if self.fault == "json" and len(self.seen) == 1:
                content = "not JSON"
            pt = render(payload)["prompt_tokens"]
            body = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                            **({"tool_calls": calls} if calls else {}),
                        },
                        "finish_reason": finish,
                    }
                ],
                "usage": {
                    "prompt_tokens": pt,
                    "completion_tokens": 8,
                    "total_tokens": pt + 8,
                },
            }
            if self.fault == "usage":
                body["usage"]["completion_tokens"] = None
            if self.fault == "length":
                body["choices"][0]["finish_reason"] = "length"
            if self.fault == "render":
                body["usage"]["prompt_tokens"] += 1
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", Handler
    httpd.shutdown()
    thread.join()
    httpd.server_close()


def test_twenty_turns_execute_tools_and_isolate_branches_and_summary(server):
    base, handler = server
    result = sessions.SessionRun(base, "fixture", {"artifact": "fixed"}, render).run()
    assert result["summary"] == {
        "planned": 28,
        "attempted": 28,
        "passed": 28,
        "failed": 0,
        "not_run": 0,
        "complete": True,
    }
    assert len([p for p in handler.seen if p.get("tools")]) == 5
    assert len([p for p in handler.seen if p["messages"][-1]["role"] == "tool"]) == 5
    parent = next(
        r["request"]["messages"]
        for r in result["observations"]
        if r["case"] == "branch-parent"
    )
    assert "AMENDMENT-A" not in json.dumps(parent)
    assert sessions.compare(result, copy.deepcopy(result))["passed"]


@pytest.mark.parametrize(
    "fault", ["fact", "source", "json", "tool", "usage", "length", "render"]
)
def test_failure_never_turns_into_passed_or_silently_dropped_work(server, fault):
    base, handler = server
    handler.fault = fault
    result = sessions.SessionRun(base, "fixture", {"artifact": "fixed"}, render).run()
    assert result["summary"]["failed"] > 0 and not result["summary"]["complete"]
    assert len(result["observations"]) == len(result["planned_cases"]) == 28
    assert result["summary"]["attempted"] + result["summary"]["not_run"] == 28
    with pytest.raises(ValueError):
        sessions.compare(result, result)


@pytest.mark.parametrize(
    "mutation", ["identity", "duplicate", "rendered", "output", "counts", "error"]
)
def test_comparison_rejects_changed_inputs_outputs_work_and_incomplete_records(
    server, mutation
):
    base, _ = server
    good = sessions.SessionRun(base, "fixture", {"artifact": "fixed"}, render).run()
    altered = copy.deepcopy(good)
    if mutation == "identity":
        altered["identity"]["artifact"] = "other"
    elif mutation == "duplicate":
        altered["observations"].append(altered["observations"][0])
    elif mutation == "rendered":
        altered["observations"][0]["rendered"]["token_ids_sha256"] = "different"
    elif mutation == "output":
        altered["observations"][0]["output_signature"]["content"] = "different"
    elif mutation == "counts":
        altered["observations"][0]["response"]["usage"]["completion_tokens"] += 1
    else:
        altered["observations"][0]["error"] = "hidden failure"
    try:
        comparison = sessions.compare(good, altered)
    except ValueError:
        pass
    else:
        assert not comparison["passed"]


def test_exact_recall_builder_measures_rendered_length_for_all_locations():
    for length in (8192, 32768):
        for key in ("k0", "k1", "k2"):
            messages = sessions.exact_recall_messages(
                render, "fixture", 32, key, length
            )
            assert (
                render(sessions.request_payload("fixture", messages, 32))[
                    "prompt_tokens"
                ]
                == length
            )
            text = messages[1]["content"]
            assert text.index("P001") < text.index("P500") < text.index("P999")


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', "NaN", '{"value":Infinity}'])
def test_ambiguous_or_nonfinite_json_is_not_accepted(text):
    with pytest.raises(ValueError):
        sessions.strict_json(text)


def test_interrupted_step_is_saved_as_incomplete_and_never_counted_success(monkeypatch):
    run = sessions.SessionRun("unused", "fixture", {"artifact": "fixed"}, render)
    saved = []
    run.progress = lambda report: saved.append(copy.deepcopy(report))

    def interrupted(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(sessions, "http_chat", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run.run()
    report = run.report()
    assert (
        not report["finalized"]
        and report["summary"]["passed"] == 0
        and report["summary"]["failed"] == 1
    )
    assert saved and not saved[-1]["summary"]["complete"]


@pytest.mark.parametrize("turns", [19, 51, True])
def test_requested_session_envelope_is_not_silently_reduced(turns):
    with pytest.raises(ValueError):
        sessions.SessionRun(
            "unused", "fixture", {"artifact": "fixed"}, render, turns=turns
        )


@pytest.mark.parametrize("legal_case", [False, True])
def test_cli_writes_paired_evidence_and_cannot_overwrite_its_control(
    server, tmp_path, monkeypatch, legal_case
):
    import sys

    base, handler = server
    identity = tmp_path / "identity.json"
    identity.write_text('{"artifact":"fixed"}')
    monkeypatch.setattr(sessions, "tokenizer_renderer", lambda _: render)

    def invoke(output, baseline=None):
        args = [
            "sessions",
            "--base-url",
            base,
            "--model",
            "fixture",
            "--tokenizer",
            str(tmp_path),
            "--identity",
            str(identity),
            "--output",
            str(output),
            "--label",
            "fixture",
        ]
        if baseline is not None:
            args.extend(["--baseline", str(baseline)])
        if legal_case:
            args.append("--legal-case")
        monkeypatch.setattr(sys, "argv", args)
        sessions.main()

    control = tmp_path / "baseline.json"
    invoke(control)
    handler.seen.clear()
    handler.ids.clear()
    candidate = tmp_path / "candidate.json"
    invoke(candidate, control)
    report = json.loads(candidate.read_text())
    assert (
        report["finalized"]
        and report["summary"]["complete"]
        and report["comparison"]["passed"]
    )
    assert report["provenance"]["harness_sha256"]
    assert report["summary"]["planned"] == (30 if legal_case else 28)
    assert ("legal_case" in report["workload"]) == legal_case
    # With no fixture reset, generated IDs enter this fixture's token stream.
    # Semantic answers still pass, but unmatched inputs must fail the CLI.
    drift = tmp_path / "different-inputs.json"
    with pytest.raises(SystemExit) as failure:
        invoke(drift, control)
    assert failure.value.code == 1
    assert not json.loads(drift.read_text())["comparison"]["passed"]
    prior = control.read_bytes()
    with pytest.raises(SystemExit):
        invoke(control, control)
    assert control.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_keeps_interrupted_request_in_its_partial_evidence(tmp_path, monkeypatch):
    import sys

    identity = tmp_path / "identity.json"
    identity.write_text('{"artifact":"fixed"}')
    output = tmp_path / "partial.json"
    monkeypatch.setattr(sessions, "tokenizer_renderer", lambda _: render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sessions",
            "--base-url",
            "http://unused",
            "--model",
            "fixture",
            "--tokenizer",
            str(tmp_path),
            "--identity",
            str(identity),
            "--output",
            str(output),
            "--label",
            "interrupted",
        ],
    )

    def interrupted(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(sessions, "http_chat", interrupted)
    with pytest.raises(KeyboardInterrupt):
        sessions.main()
    report = json.loads(output.read_text())
    assert (
        not report["finalized"]
        and report["summary"]["passed"] == 0
        and report["summary"]["failed"] == 1
    )
    assert (
        report["observations"][0]["request"]["messages"]
        and report["observations"][0]["rendered"]
    )


def test_fifty_turn_run_keeps_all_planned_tool_and_branch_work(server):
    base, handler = server
    report = sessions.SessionRun(
        base, "fixture", {"artifact": "fixed"}, render, turns=50
    ).run()
    assert report["summary"]["complete"] and report["summary"]["passed"] == 65
    assert (
        len(handler.seen) == 65
        and sum(bool(p.get("tools")) for p in handler.seen) == 12
    )


def test_optional_case_executes_attributed_summary_and_date_order_against_fixed_oracle(
    server,
):
    base, handler = server
    report = sessions.SessionRun(
        base, "fixture", {"artifact": "fixed"}, render, legal_case=True
    ).run()
    assert report["summary"]["passed"] == report["summary"]["planned"] == 30
    assert report["summary"]["complete"]
    cases = {
        r["case"]: r for r in report["observations"] if r["case"].startswith("legal-")
    }
    for name, expected in [
        ("legal-summary", CASE_SUMMARY),
        ("legal-chronology", CASE_CHRONOLOGY),
    ]:
        record = cases[name]
        assert (
            json.loads(record["response"]["choices"][0]["message"]["content"])
            == expected
        )
        assert record["request"]["messages"][1]["content"].index("L003") < record[
            "request"
        ]["messages"][1]["content"].index("L001")
        assert record["rendered"] == render(record["request"])
    assert len(handler.seen) == 30
    assert sessions.compare(report, copy.deepcopy(report))["passed"]


@pytest.mark.parametrize(
    "fault",
    [
        "legal-status",
        "legal-source",
        "legal-deadline",
        "legal-omission",
        "legal-ruling",
        "legal-extra",
        "legal-type",
        "legal-order",
    ],
)
def test_legal_case_incomplete_misattributed_or_invented_answers_fail(server, fault):
    base, handler = server
    handler.fault = fault
    report = sessions.SessionRun(
        base, "fixture", {"artifact": "fixed"}, render, legal_case=True
    ).run()
    assert report["summary"]["planned"] == report["summary"]["attempted"] == 30
    assert report["summary"]["failed"] == 1 and report["summary"]["passed"] == 29
    assert not report["summary"]["complete"]
    failure = next(r for r in report["observations"] if r["error"])
    assert failure["case"] == (
        "legal-chronology" if fault == "legal-order" else "legal-summary"
    )
    assert "oracle mismatch" in failure["error"]
    with pytest.raises(ValueError, match="fully completed"):
        sessions.compare(report, report)
