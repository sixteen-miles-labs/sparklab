from __future__ import annotations

import json
import re

from benchmarks.finalize_kimi_k3_result import (
    EXPECTED_EXPERT_ENTRIES,
    EXPECTED_REVISION,
    RUNGS,
    build_result,
    validate_bundle,
)
from benchmarks.bench_decode_moe import extract_answer


def _write(path, value) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _bundle(tmp_path):
    _write(
        tmp_path / "acquisition-final.json",
        {
            "status": "completed",
            "revision": EXPECTED_REVISION,
            "committed_shards": 96,
            "committed_bytes": 100,
            "expected_bytes": 100,
            "partial_bytes": 0,
            "oom_kill_delta": 0,
            "swap_out_pages_delta": 0,
        },
    )
    _write(
        tmp_path / "source-audit-final.json",
        {
            "status": "passed",
            "allow_partial": False,
            "errors": [],
            "index": {
                "completed_shards": 96,
                "missing_shards": [],
                "expert_entries": EXPECTED_EXPERT_ENTRIES,
                "mapped_model_state": 2946,
                "expected_model_state": 2946,
            },
        },
    )
    _write(
        tmp_path / "conversion.json",
        {
            "status": "completed",
            "returncode": 0,
            "termination_reason": None,
            "vmstat": {"delta": {"oom_kill": 0, "pswpout": 0}},
            "swap": {"growth_bytes": 0},
            "memory": {
                "available_min_bytes": 1,
                "process_group_peak_rss_bytes": 1,
            },
            "disk": {"available_min_bytes": 1},
        },
    )
    _write(tmp_path / "conversion-gate.json", {"status": "passed", "errors": []})

    texts = {
        1: "A",
        16: "A coherent response",
        64: "A coherent response that continues through the reasoning",
        256: (
            "A coherent response that continues through the reasoning and final result "
            "\\boxed{42}"
        ),
    }
    for tokens in RUNGS:
        row = {
            "checkpoint": {
                "format": "ftw",
                "fingerprint": "0123456789abcdef",
                "bytes": 123,
            },
            "recipe": {
                "slug": "kimi-k3",
                "recipe_version": "0.2.0",
                "revision": EXPECTED_REVISION,
                "model": "nvidia/Kimi-K3-NVFP4",
            },
            "completion_tokens": tokens,
            "events": tokens,
            "output_sha1": f"hash{tokens}",
            "output_text": texts[tokens],
            "expected_answer": "42",
            "extracted_answer": "42" if tokens == 256 else None,
            "answer_correct": True if tokens == 256 else None,
            "oom_markers": 0,
            "lifecycle_telemetry": {
                "memory_guard_triggered": False,
                "mem_available_gib_min": 20.0,
                "oom_kill_delta": 0,
                "swap_out_pages_delta": 0,
                "swap_growth_bytes": 0,
            },
            "gpu_telemetry": {
                "oom_kill_delta": 0,
                "swap_out_pages_delta": 0,
                "swap_growth_bytes": 0,
                "power_w_avg": 40.0,
                "power_w_peak": 50.0,
                "temperature_c_peak": 55.0,
            },
            "platform": {
                "runtime": {"gpu": "NVIDIA GB10", "cuda": "13.0"},
                "snapshot": {"machine": "aarch64"},
            },
            "provenance": {
                "git_revision": "a" * 40,
                "git_tracked_dirty": True,
                "gpu_driver": "NVIDIA GB10, 580.126.09",
                "packages": {"torch": "2.11", "triton": "3.6"},
            },
            "prompt_tokens": 54,
            "decode_steps": max(0, tokens - 1),
            "decode_tok_s": 1.25,
            "ms_per_token": 800.0,
            "first_request_ttft_ms": 2000.0,
            "ttft_ms": 1000.0,
            "event_ms_p50": 790.0,
            "event_ms_p99": 900.0,
            "vram_gib": 80.0,
            "moe": {
                "miss_rate": 0.25,
                "disk": {"read_ops": 10, "physical_bytes": 1 << 30},
            },
        }
        _write(tmp_path / f"probe-{tokens}.jsonl", row)
        _write(tmp_path / f"gate-{tokens}.json", {"status": "passed", "errors": []})
    return tmp_path


def test_builds_schema_shaped_result_from_complete_promoted_bundle(tmp_path):
    results = _bundle(tmp_path)
    rows, errors = validate_bundle(results)
    assert errors == []

    result = build_result(results, rows)
    assert re.fullmatch(r"GB10-[A-Z]+-[0-9]{3}", result["result_id"])
    assert result["result_id"] == "GB10-KIMI-001"
    assert result["model"]["fingerprint"] == "0123456789abcdef"
    assert result["validation"]["decode_promotion_gates_passed"] == [1, 16, 64, 256]
    assert result["validation"]["oom_count"] == 0
    assert result["validation"]["output_correctness_evaluated"] is True
    assert result["validation"]["output_correctness"] is True
    assert result["metrics"]["decode_tokens_per_second"] == 1.25


def test_rejects_nonzero_swap_out_even_when_gate_file_claims_passed(tmp_path):
    results = _bundle(tmp_path)
    row = json.loads((results / "probe-64.jsonl").read_text(encoding="utf-8"))
    row["lifecycle_telemetry"]["swap_out_pages_delta"] = 1
    _write(results / "probe-64.jsonl", row)

    _, errors = validate_bundle(results)
    assert any("swap_out_pages_delta=1" in error for error in errors)


def test_extract_answer_distinguishes_finished_from_unfinished_reasoning():
    assert extract_answer("reasoning only") is None
    assert extract_answer("therefore the final answer is 42") == "42"
    assert extract_answer("first \\boxed{7}, finally \\boxed{42}") == "42"
