"""Real HTTP checker controls with synthetic replies, not model endurance evidence."""

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error

import pytest

from benchmarks import validate_runtime_soak as soak
from benchmarks.bench_runtime_mixed import run_case


@pytest.fixture
def service():
    state = {
        "output": "ok",
        "rebuilds": 0,
        "geometry": {"num_pages": 64},
        "fail": False,
        "instance_reads": 0,
        "restart": False,
        "wrong_geometry": False,
        "busy_rebuilds": 0,
        "busy_rebuild_code": 200,
        "rebuild_error": None,
        "generation_error": None,
        "padding": "",
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value, code=200):
            payload = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/v1/stats":
                state["instance_reads"] += 1
                instance = (
                    "changed"
                    if state["restart"] and state["instance_reads"] > 1
                    else "original"
                )
                self.reply(
                    {
                        "instance_id": instance,
                        "requests": {"active": 0},
                        "native_runtime": {
                            "age_seconds": 8,
                            "resources": {"running_requests": 1},
                        },
                        "padding": state["padding"],
                    }
                )
            elif self.path == "/v1/cache/status":
                geometry = (
                    {"num_pages": 12}
                    if state["wrong_geometry"] and state["rebuilds"]
                    else state["geometry"]
                )
                self.reply({"geometry": geometry})
            else:
                self.reply({}, 404)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path == "/v1/cache/rebuild":
                assert payload == {"num_pages": 64}
                if state["rebuild_error"] is not None:
                    body, code = state["rebuild_error"]
                    self.reply(body, code)
                    return
                if state["busy_rebuilds"]:
                    state["busy_rebuilds"] -= 1
                    self.reply({"status": "busy"}, state["busy_rebuild_code"])
                    return
                state["rebuilds"] += 1
                self.reply({"status": "ok"})
                return
            if state["generation_error"] is not None:
                body, code = state["generation_error"]
                self.reply(body, code)
                return
            if state["fail"]:
                self.reply({"error": "injected"}, 500)
                return
            events = [{"choices": [{"text": c}]} for c in state["output"]] + [
                {
                    "choices": [{"finish_reason": "length"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            ]
            raw = (
                "".join("data: " + json.dumps(e) + "\n\n" for e in events)
                + "data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(raw.encode())))
            self.end_headers()
            self.wfile.write(raw.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    workload = {
        "cases": [
            {
                "id": "complete",
                "request": {"prompt": "test", "max_tokens": 2, "temperature": 0},
            },
            {
                "id": "cancel",
                "request": {"prompt": "test", "max_tokens": 2, "temperature": 0},
                "cancel_after_events": 1,
            },
        ]
    }
    identity = {"served_model": "test", "checkpoint": "synthetic-control"}
    epoch = time.monotonic()
    observations = [
        asdict(run_case(base, "test", c, epoch, 2.0)) for c in workload["cases"]
    ]
    baseline = {
        "identity": identity,
        "workload_sha256": soak.digest(workload),
        "observations": observations,
    }
    try:
        yield (
            dict(
                base=base,
                model="test",
                workload=workload,
                identity=identity,
                baseline=baseline,
                duration=0.03,
                timeout=2.0,
                sample_interval=0.01,
            ),
            state,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


def test_short_real_http_collection_keeps_work_and_never_claims_qualification(
    service, tmp_path
):
    arguments, state = service
    report = soak.run_soak(**arguments, output=tmp_path / "run", rebuild_every=1)
    assert report["status"] == "collected" and report["duration_satisfied"]
    assert report["cycles"] >= 1 and report["warmup_cycles"] == 1
    assert (
        report["completed_requests"]
        == report["cancelled_requests"]
        == report["cycles"] + 1
    )
    assert report["rebuilds"] == state["rebuilds"] == report["cycles"]
    assert not report["qualification"]
    cycles = [
        json.loads(line)
        for line in (tmp_path / "run/cycles.jsonl").read_text().splitlines()
    ]
    assert len(cycles) == report["cycles"] + 1 and cycles[0]["warmup"]
    assert all(not c["warmup"] for c in cycles[1:])
    resources = [
        json.loads(line)
        for line in (tmp_path / "run/resources.jsonl").read_text().splitlines()
    ]
    # Deliberately stale worker state remains visible, never reinterpreted as idle.
    assert resources[-1]["stats"]["native_runtime"]["age_seconds"] == 8


@pytest.mark.parametrize(
    "fault,match",
    [
        ("output", "strict baseline output"),
        ("fail", "request failed"),
        ("restart", "server restarted"),
        ("wrong_geometry", "geometry changed"),
    ],
)
def test_faults_preserve_failed_partial_evidence(service, tmp_path, fault, match):
    arguments, state = service
    state[fault] = "no" if fault == "output" else True
    with pytest.raises(ValueError, match=match):
        soak.run_soak(**arguments, output=tmp_path / "failed", rebuild_every=1)
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "failed" and not report["qualification"]
    assert match in report["error"]
    if fault == "restart":
        assert report["attempted_requests"] == report["unobserved_requests"] == 2


def test_changed_baseline_is_rejected_before_writing_or_sending_requests(
    service, tmp_path
):
    arguments, state = service
    arguments["baseline"]["identity"] = {"checkpoint": "wrong"}
    with pytest.raises(ValueError, match="identity differs"):
        soak.run_soak(**arguments, output=tmp_path / "absent")
    assert not (tmp_path / "absent").exists() and state["instance_reads"] == 0


@pytest.mark.parametrize("code", [200, 409, 503])
def test_busy_transfer_can_drain_before_idle_rebuild(service, tmp_path, code):
    arguments, state = service
    state["busy_rebuilds"] = 2
    state["busy_rebuild_code"] = code
    report = soak.run_soak(**arguments, output=tmp_path / "busy", rebuild_every=1)
    assert state["busy_rebuilds"] == 0 and report["rebuilds"] >= 1
    assert report["status"] == "collected" and not report["qualification"]


@pytest.mark.parametrize("body,code", [
    ({"status": "failed"}, 503),
    ({"status": "loading"}, 503),
    ({"status": "rejected"}, 409),
    ({"status": "busy"}, 500),
    (["busy"], 503),
])
def test_other_rebuild_http_errors_remain_fatal(service, body, code):
    arguments, state = service
    state["rebuild_error"] = (body, code)
    with pytest.raises(urllib.error.HTTPError) as error:
        soak.get_json(arguments["base"], "/v1/cache/rebuild", {"num_pages": 64})
    assert error.value.code == code


def test_generation_http_503_is_not_a_rebuild_retry(service):
    arguments, state = service
    state["generation_error"] = ({"status": "busy"}, 503)
    with pytest.raises(urllib.error.HTTPError) as error:
        soak.get_json(arguments["base"], "/v1/completions", {})
    assert error.value.code == 503


def test_evidence_budget_fails_explicitly_without_losing_prior_samples(
    service, tmp_path
):
    arguments, state = service
    state["padding"] = "x" * 700000
    with pytest.raises(ValueError, match="evidence budget exhausted"):
        soak.run_soak(**arguments, output=tmp_path / "bounded", max_evidence_mib=1)
    report = json.loads((tmp_path / "bounded/report.json").read_text())
    actual = sum(
        (tmp_path / "bounded" / name).stat().st_size
        for name in ("cycles.jsonl", "resources.jsonl")
    )
    assert 0 < actual == report["evidence_bytes"] <= 1 << 20
    assert report["status"] == "failed" and not report["qualification"]


def test_existing_evidence_directory_is_preserved(service, tmp_path):
    arguments, _ = service
    output = tmp_path / "previous"
    output.mkdir()
    (output / "report.json").write_text("previous evidence")
    with pytest.raises(FileExistsError):
        soak.run_soak(**arguments, output=output)
    assert (output / "report.json").read_text() == "previous evidence"


def test_standalone_command_collects_actual_http_evidence(service, tmp_path):
    arguments, _ = service
    for name in ("workload", "identity", "baseline"):
        (tmp_path / f"{name}.json").write_text(json.dumps(arguments[name]))
    command = [
        sys.executable,
        str(Path(soak.__file__)),
        "--base-url",
        arguments["base"],
        "--model",
        "test",
        "--workload",
        str(tmp_path / "workload.json"),
        "--identity",
        str(tmp_path / "identity.json"),
        "--baseline",
        str(tmp_path / "baseline.json"),
        "--output",
        str(tmp_path / "cli"),
        "--duration-seconds",
        ".03",
        "--sample-interval",
        ".01",
        "--timeout",
        "2",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "collected" and report["duration_satisfied"]
    assert report["warmup_cycles"] == 1 and report["cycles"] >= 1
    assert not report["qualification"] and report["attempted_requests"] >= 4


def test_early_arrival_is_not_queued_behind_a_later_sleeping_request(service, tmp_path):
    arguments, _ = service
    # The input intentionally puts the later request first. One client worker
    # must still send the immediate cancellation before the later full output.
    arguments["workload"]["cases"][0]["at"] = 0.03
    arguments["baseline"]["workload_sha256"] = soak.digest(arguments["workload"])
    soak.run_soak(**{**arguments, "concurrency": 1}, output=tmp_path / "arrival")
    first = json.loads((tmp_path / "arrival/cycles.jsonl").read_text().splitlines()[0])
    immediate, later = first["observations"]
    assert immediate["case"] == "cancel" and later["case"] == "complete"
    assert immediate["started"] < later["started"]
