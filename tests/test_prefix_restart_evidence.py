"""Independent dispatch timelines; a restore counter alone cannot qualify restart."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prefix_restart",
    Path(__file__).parents[1] / "benchmarks/validate_prefix_restart.py",
)
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def valid():
    report = {
        "prefix_tokens": 65,
        "query_tokens": 66,
        "output_tokens": 4,
        "prefix_ids_sha256": "a" * 64,
        "query_prefix_ids_sha256": "a" * 64,
        "query_ids_sha256": "b" * 64,
        "runs": {},
        "records_before_rollback": {"namespace/key.pxc": [123, 456]},
        "records_after_rollback": {"namespace/key.pxc": [123, 456]},
    }
    for mode in ("cold", "publish", "restore", "rollback"):
        enabled = mode in ("publish", "restore")
        work = 65 if mode == "publish" else 1 if mode == "restore" else 66
        output = 1 if mode == "publish" else 4
        prefix = (
            {
                "records": 1,
                "disabled": False,
                "counters": {
                    "writes_completed": 1,
                    "restores_completed": int(mode == "restore"),
                },
                **{
                    key: 0
                    for key in (
                        "unreleased_tickets",
                        "completed_pending_fence",
                        "staging_bytes",
                        "restore_waiters",
                        "private_restore_pages",
                        "private_restore_slots",
                    )
                },
            }
            if enabled
            else None
        )
        report["runs"][mode] = {
            "mode": mode,
            "identity": {"artifact": "fixed"},
            "geometry": {"num_pages": 512, "page_size": 16},
            "exact_enabled": enabled,
            "disk_enabled": enabled,
            "exit_code": 143,
            "forced_stop": False,
            "semaphore_warning": False,
            "startup_metrics": {"counters": {}, "trace": []},
            "startup_prefix": {"records": 1} if mode == "restore" else None,
            "prefix_ids_sha256": "a" * 64,
            "query_ids_sha256": "b" * 64,
            "metrics": {
                "counters": {
                    "prefill_computed_tokens": work,
                    "prefix_reused_tokens": 65 if mode == "restore" else 0,
                    "committed_output_tokens": output,
                },
                "trace": [
                    {"phase": "prefill", "requests": 1, "tokens": work},
                    *(
                        [{"phase": "decode", "requests": 1, "tokens": 1}] * 3
                        if output == 4
                        else []
                    ),
                ],
            },
            "reply": {
                "choices": [
                    {
                        "text": " result" if mode != "publish" else " seed",
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 65 if mode == "publish" else 66,
                    "completion_tokens": output,
                    "total_tokens": (65 if mode == "publish" else 66) + output,
                    "prompt_tokens_details": {
                        "cached_tokens": 65 if mode == "restore" else 0
                    },
                },
            },
            "prefix": prefix,
        }
    return report


def test_independent_cold_suffix_only_and_disabled_dispatch_oracle():
    result = proof.verify(valid())
    assert result["computed_prefill_tokens"] == {
        "cold": 66,
        "publish": 65,
        "restore": 1,
        "rollback": 66,
    }
    assert result["strict_output_equal"]


@pytest.mark.parametrize(
    "fault",
    [
        "fake_hit",
        "truncated",
        "state_output",
        "cached_usage",
        "tickets",
        "fence",
        "staging",
        "waiters",
        "pages",
        "slots",
        "cold_warm",
        "geometry",
        "identity",
        "wire_ids",
        "startup_warm",
        "missing_startup",
        "forced",
        "semaphore",
        "disable",
        "deleted_records",
        "unfinished_output",
        "usage",
        "missing_trace",
    ],
)
def test_counter_or_success_response_never_masks_missing_restart_proof(fault):
    report = valid()
    run = report["runs"]["restore"]
    if fault == "fake_hit":
        run["metrics"]["trace"][0]["tokens"] = 66
        run["metrics"]["counters"]["prefill_computed_tokens"] = 66
    elif fault == "truncated":
        run["metrics"]["trace"].pop(0)
    elif fault == "state_output":
        run["reply"]["choices"][0]["text"] = " wrong"
    elif fault == "cached_usage":
        run["reply"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 64
    elif fault in ("tickets", "fence", "staging", "waiters", "pages", "slots"):
        key = {
            "tickets": "unreleased_tickets",
            "fence": "completed_pending_fence",
            "staging": "staging_bytes",
            "waiters": "restore_waiters",
            "pages": "private_restore_pages",
            "slots": "private_restore_slots",
        }[fault]
        run["prefix"][key] = 1
    elif fault == "cold_warm":
        report["runs"]["cold"]["metrics"]["counters"]["prefix_reused_tokens"] = 16
    elif fault == "geometry":
        run["geometry"]["num_pages"] = 1024
    elif fault == "identity":
        run["identity"]["artifact"] = "other"
    elif fault == "wire_ids":
        run["query_ids_sha256"] = "c" * 64
    elif fault == "startup_warm":
        run["startup_metrics"]["trace"] = [
            {"phase": "prefill", "tokens": 1, "requests": 1}
        ]
    elif fault == "missing_startup":
        run["startup_prefix"] = None
    elif fault == "forced":
        run["forced_stop"] = True
    elif fault == "semaphore":
        run["semaphore_warning"] = True
    elif fault == "disable":
        report["runs"]["rollback"]["disk_enabled"] = True
    elif fault == "deleted_records":
        report["records_after_rollback"] = {}
    elif fault == "unfinished_output":
        run["metrics"]["counters"]["committed_output_tokens"] = 3
    elif fault == "usage":
        run["reply"]["usage"]["completion_tokens"] = 3
    else:
        run["metrics"]["trace"] = []
    with pytest.raises(ValueError):
        proof.verify(report)


def test_parser_preserves_last_complete_raw_snapshot_and_retries_partial_tail(tmp_path):
    path = tmp_path / "server.log"
    metric = valid()["runs"]["restore"]["metrics"]
    prefix = valid()["runs"]["restore"]["prefix"]
    path.write_text(
        "loading...\n\x1b[32mRuntime measurements: "
        + json.dumps(metric)
        + "\x1b[0m\nPrefix measurements: "
        + json.dumps(prefix)
        + '\nRuntime measurements: {"counters":'
    )
    result = proof.read_measurements(path)
    assert result["metrics"] == metric and result["prefix"] == prefix
    assert result["snapshot_counts"] == {"metrics": 1, "prefix": 1}
    path.write_text("Runtime measurements: broken\n")
    with pytest.raises(ValueError, match="malformed"):
        proof.read_measurements(path)


def test_inventory_bounds_owned_records_and_detects_rollback_mutation(tmp_path):
    namespace = tmp_path / "namespace"
    namespace.mkdir()
    path = namespace / "record.pxc"
    path.write_bytes(b"checkpoint")
    first = proof.inventory(tmp_path)
    assert len(first) == 1
    path.write_bytes(b"changed checkpoint")
    assert proof.inventory(tmp_path) != first
    for index in range(32):
        (namespace / f"{index}.pxc").touch()
    with pytest.raises(ValueError, match="bound"):
        proof.inventory(tmp_path)


def test_oversized_log_is_rejected_before_materializing_text(tmp_path):
    path = tmp_path / "large.log"
    with path.open("wb") as output:
        output.truncate(proof.LOG_LIMIT + 1)
    with pytest.raises(ValueError, match="bound"):
        proof.read_measurements(path)


def test_appending_a_suffix_must_preserve_the_consumed_prefix_ids():
    report = valid()
    report["query_prefix_ids_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="exact prefix"):
        proof.verify(report)


def test_existing_runtime_telemetry_wire_format_is_checked_against_independent_work(
    tmp_path,
):
    from sparklab.runtime.scheduler.telemetry import RuntimeTelemetry

    report = valid()
    for mode, run in report["runs"].items():
        metric = RuntimeTelemetry(trace_capacity=128)
        work = {
            "cold": [32, 32, 2],
            "publish": [32, 32, 1],
            "restore": [1],
            "rollback": [32, 32, 2],
        }[mode]
        for index, count in enumerate(work):
            metric.dispatch(
                phase="prefill",
                requests=1,
                tokens=count,
                cached_tokens=65 if mode == "restore" and index == 0 else 0,
                queued=0,
                reason="baseline",
            )
        if mode != "publish":
            for _ in range(3):
                metric.dispatch(
                    phase="decode",
                    requests=1,
                    tokens=1,
                    cached_tokens=0,
                    queued=0,
                    reason="baseline",
                )
        metric.counters["committed_output_tokens"] = 1 if mode == "publish" else 4
        path = tmp_path / f"{mode}.log"
        path.write_text("Runtime measurements: " + json.dumps(metric.snapshot()) + "\n")
        run["metrics"] = proof.read_measurements(path)["metrics"]
    assert proof.verify(report)["computed_prefill_tokens"]["restore"] == 1


def test_completion_counter_cannot_replace_missing_decode_dispatches():
    report = valid()
    report["runs"]["restore"]["metrics"]["trace"].pop()
    with pytest.raises(ValueError, match="decode dispatches"):
        proof.verify(report)


@pytest.mark.parametrize("order", [[1, 0, 2, 3], [0, 1, 2, 3, 0]])
def test_totals_cannot_hide_out_of_order_serial_model_dispatch(order):
    report = valid()
    metric = report["runs"]["restore"]["metrics"]
    original = metric["trace"]
    metric["trace"] = [original[index] for index in order]
    metric["counters"]["prefill_computed_tokens"] = sum(
        event["tokens"] for event in metric["trace"] if event["phase"] == "prefill"
    )
    with pytest.raises(ValueError, match="precedes|follows"):
        proof.verify(report)
