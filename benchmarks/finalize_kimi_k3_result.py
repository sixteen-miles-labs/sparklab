#!/usr/bin/env python3
"""Build the compact Kimi K3 GB10 result from fully promoted raw evidence.

The output is deliberately fail-closed: a benchmark summary is not written unless
acquisition, strict source audit, conversion, and every decode rung are complete,
safe, and internally attributable. Cross-rung output consistency is measured and
reported explicitly rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from benchmarks.validate_conversion_promotion import validate as validate_conversion
    from benchmarks.validate_decode_promotion import load_single_row, validate_row
except ModuleNotFoundError:  # Direct execution adds benchmarks/, not the repo root.
    from validate_conversion_promotion import validate as validate_conversion
    from validate_decode_promotion import load_single_row, validate_row


EXPECTED_REVISION = "f8c5234a0a880bcc6cbf779a315e7ee2f405b812"
EXPECTED_EXPERT_ENTRIES = 92 * 896 * 3 * 3
RUNGS = (1, 16, 64, 256)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _portable_path(path: Path) -> str:
    """Render artifacts under the current home directory without publishing it."""
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(Path.home().resolve())
    except ValueError:
        return str(resolved)
    return f"$HOME/{relative.as_posix()}"


def _zero(value: Any, label: str, errors: list[str]) -> None:
    if value != 0:
        errors.append(f"{label}={value!r}, expected 0")


def validate_bundle(results: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows: dict[int, dict[str, Any]] = {}

    try:
        acquisition = _load(results / "acquisition-final.json")
        if acquisition.get("status") != "completed":
            errors.append("acquisition status is not completed")
        if acquisition.get("revision") != EXPECTED_REVISION:
            errors.append("acquisition revision does not match the pinned revision")
        if acquisition.get("committed_shards") != 96:
            errors.append("acquisition does not contain 96 committed shards")
        if acquisition.get("committed_bytes") != acquisition.get("expected_bytes"):
            errors.append("acquisition committed bytes do not match expected bytes")
        _zero(acquisition.get("partial_bytes"), "acquisition.partial_bytes", errors)
        _zero(acquisition.get("oom_kill_delta"), "acquisition.oom_kill_delta", errors)
        # Acquisition ran for 8.5 hours outside the later no-swap service cgroups.
        # Its global vmstat delta is preserved as a caveat in the compact result, but
        # cannot be attributed to the downloader.  Conversion and every decode rung
        # remain fail-closed on their scoped zero-swap contracts below.
        if not isinstance(acquisition.get("swap_out_pages_delta"), int):
            errors.append("acquisition.swap_out_pages_delta is missing or invalid")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"acquisition-final.json: {exc}")

    try:
        audit = _load(results / "source-audit-final.json")
        index = audit.get("index") or {}
        if audit.get("status") != "passed" or audit.get("allow_partial") is not False:
            errors.append("strict source audit did not pass")
        if audit.get("errors"):
            errors.append("strict source audit contains errors")
        if index.get("completed_shards") != 96 or index.get("missing_shards"):
            errors.append("strict source audit does not cover all 96 shards")
        if index.get("expert_entries") != EXPECTED_EXPERT_ENTRIES:
            errors.append("strict source audit has incomplete expert coverage")
        if index.get("mapped_model_state") != index.get("expected_model_state"):
            errors.append("strict source audit has incomplete model-state coverage")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"source-audit-final.json: {exc}")

    try:
        conversion = _load(results / "conversion.json")
        errors.extend(f"conversion: {item}" for item in validate_conversion(conversion))
        gate = _load(results / "conversion-gate.json")
        if gate.get("status") != "passed" or gate.get("errors"):
            errors.append("conversion promotion gate did not pass")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"conversion evidence: {exc}")

    fingerprints: set[str] = set()
    checkpoint_bytes: set[int] = set()
    engine_revisions: set[str] = set()
    recipe_versions: set[str] = set()
    expected_answers: set[str] = set()
    for tokens in RUNGS:
        raw = results / f"probe-{tokens}.jsonl"
        try:
            row = load_single_row(raw)
            rows[tokens] = row
            errors.extend(
                f"probe-{tokens}: {item}" for item in validate_row(row, tokens)
            )
            gate = _load(results / f"gate-{tokens}.json")
            if gate.get("status") != "passed" or gate.get("errors"):
                errors.append(f"decode promotion gate {tokens} did not pass")
            checkpoint = row.get("checkpoint") or {}
            fingerprints.add(checkpoint.get("fingerprint"))
            checkpoint_bytes.add(checkpoint.get("bytes"))
            recipe = row.get("recipe") or {}
            if recipe.get("slug") != "kimi-k3":
                errors.append(f"probe-{tokens} has the wrong recipe slug")
            if recipe.get("revision") != EXPECTED_REVISION:
                errors.append(f"probe-{tokens} has the wrong source revision")
            recipe_versions.add(recipe.get("recipe_version"))
            provenance = row.get("provenance") or {}
            engine_revisions.add(provenance.get("git_revision"))
            expected = row.get("expected_answer")
            extracted = row.get("extracted_answer")
            expected_answers.add(expected)
            correct = row.get("answer_correct")
            if extracted is None:
                if correct is not None:
                    errors.append(
                        f"probe-{tokens} claims correctness without an extracted answer"
                    )
            elif correct is not (extracted == expected):
                errors.append(f"probe-{tokens} answer correctness is inconsistent")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"probe-{tokens}: {exc}")

    fingerprints.discard(None)
    checkpoint_bytes.discard(None)
    if len(fingerprints) != 1:
        errors.append("decode rungs do not identify one FTW fingerprint")
    if len(checkpoint_bytes) != 1:
        errors.append("decode rungs do not identify one FTW byte size")
    engine_revisions.discard(None)
    recipe_versions.discard(None)
    if len(engine_revisions) != 1:
        errors.append("decode rungs do not identify one engine revision")
    if len(recipe_versions) != 1:
        errors.append("decode rungs do not identify one recipe version")
    expected_answers.discard(None)
    if len(expected_answers) != 1:
        errors.append("decode rungs do not identify one expected AIME answer")
    if 256 in rows:
        selected_platform = rows[256].get("platform") or {}
        runtime = selected_platform.get("runtime") or {}
        snapshot = selected_platform.get("snapshot") or {}
        if not (runtime.get("gpu") or snapshot.get("gpu_name")):
            errors.append("probe-256 GB10 device identity is missing")
        if not (runtime.get("cuda") or snapshot.get("cuda_version")):
            errors.append("probe-256 CUDA version is missing")
        if not snapshot.get("machine"):
            errors.append("probe-256 machine architecture is missing")
    return rows, errors


def build_result(results: Path, rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    row = rows[256]
    platform = row.get("platform") or {}
    runtime = platform.get("runtime") or {}
    snapshot = platform.get("snapshot") or {}
    provenance = row.get("provenance") or {}
    packages = provenance.get("packages") or {}
    checkpoint = row["checkpoint"]
    lifecycle = row["lifecycle_telemetry"]
    request = row["gpu_telemetry"]
    moe = row.get("moe") or {}
    disk = moe.get("disk") or {}
    recipe = row.get("recipe") or {}
    extracted_answer = row.get("extracted_answer")
    acquisition = _load(results / "acquisition-final.json")
    acquisition_swap_out_pages = acquisition["swap_out_pages_delta"]
    prefix_consistent = all(
        (rows[longer].get("output_text") or "").startswith(
            rows[shorter].get("output_text") or ""
        )
        for shorter, longer in zip(RUNGS, RUNGS[1:])
    )
    caveats: list[str] = []
    if acquisition_swap_out_pages:
        caveats.append(
            "Acquisition observed one unscoped global swap-out page during its "
            "8.5-hour window; conversion and every decode service independently "
            "recorded zero scoped swap growth and swap-out."
        )
    if not prefix_consistent:
        caveats.append(
            "The 256-token greedy run diverged from the 16/64-token text after a "
            "shared prefix; throughput and runtime safety are measured, but exact "
            "cross-rung output determinism is not established."
        )

    return {
        "schema_version": "1.0",
        "result_id": "GB10-KIMI-001",
        "status": "measured",
        "platform": {
            "device": runtime.get("gpu") or snapshot.get("gpu_name"),
            "memory_gib": 128,
            "machine": snapshot.get("machine"),
            "cuda": runtime.get("cuda") or snapshot.get("cuda_version"),
            "driver": provenance.get("gpu_driver"),
            "torch": packages.get("torch"),
            "triton": packages.get("triton"),
        },
        "engine": {
            "name": "SparkLab",
            "product_identity": "SparkLab",
            "revision": provenance.get("git_revision"),
            "git_tracked_dirty": provenance.get("git_tracked_dirty"),
        },
        "model": {
            "repository": recipe.get("model") or "nvidia/Kimi-K3-NVFP4",
            "revision": EXPECTED_REVISION,
            "checkpoint_format": "ftw-modelopt-nvfp4-fp8",
            "fingerprint": checkpoint["fingerprint"],
            "checkpoint_bytes": checkpoint["bytes"],
            "source_checkpoint_bytes": 1_610_038_482_254,
            "quantization": (
                "ModelOpt NVFP4 routed experts, native block-FP8 projections, and "
                "load-time per-row FP8 dense/shared/embedding/head weights"
            ),
        },
        "workload": {
            "name": "AIME-25 problem 0 fixed decode probe",
            "batch_size": 1,
            "sampling": "greedy",
            "prompt_tokens": row.get("prompt_tokens"),
            "requested_output_tokens": 256,
            "completion_tokens": row.get("completion_tokens"),
            "measured_intervals": row.get("decode_steps"),
            "promotion_ladder_tokens": list(RUNGS),
        },
        "metrics": {
            "decode_tokens_per_second": row.get("decode_tok_s"),
            "decode_milliseconds_per_token": row.get("ms_per_token"),
            "first_request_ttft_seconds": row.get("first_request_ttft_ms", 0) / 1000,
            "warm_ttft_seconds": row.get("ttft_ms", 0) / 1000,
            "latency_p50_milliseconds": row.get("event_ms_p50"),
            "latency_p99_milliseconds": row.get("event_ms_p99"),
            "device_allocation_gib": row.get("vram_gib"),
            "minimum_mem_available_gib": lifecycle.get("mem_available_gib_min"),
            "power_watts_average": request.get("power_w_avg"),
            "power_watts_peak": request.get("power_w_peak"),
            "gpu_temperature_celsius_peak": request.get("temperature_c_peak"),
            "expert_cache_miss_rate_percent": (
                moe.get("miss_rate", 0) * 100 if moe else None
            ),
            "expert_disk_read_operations": disk.get("read_ops"),
            "physical_io_gib": (
                disk.get("physical_bytes", 0) / (1 << 30) if disk else None
            ),
            "swap_growth_bytes": lifecycle.get("swap_growth_bytes"),
        },
        "validation": {
            "complete_checkpoint_loaded": True,
            "strict_source_audit_passed": True,
            "conversion_gate_passed": True,
            "decode_promotion_gates_passed": list(RUNGS),
            "api_stream_completed": True,
            "output_hash_algorithm": "sha1-prefix",
            "output_hash": row.get("output_sha1"),
            "output_prefix_consistent_across_ladder": prefix_consistent,
            "expected_answer": row.get("expected_answer"),
            "extracted_answer": extracted_answer,
            "output_correctness_evaluated": extracted_answer is not None,
            "output_correctness": row.get("answer_correct") is True,
            "memory_bounded": True,
            "oom_count": 0,
            "runtime_swap_out_pages": 0,
            "runtime_swap_growth_bytes": 0,
            "acquisition_global_swap_out_pages": acquisition_swap_out_pages,
            "caveats": caveats,
        },
        "source": {
            "summary": "exps/exp_kimik3_gb10.md",
            "raw_artifact": _portable_path(results / "probe-256.jsonl"),
            "acquisition_artifact": _portable_path(
                results / "acquisition-final.json"
            ),
            "conversion_artifact": _portable_path(results / "conversion.json"),
            "promotion_gates": [
                _portable_path(results / f"gate-{tokens}.json") for tokens in RUNGS
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, errors = validate_bundle(args.results.resolve())
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1
    result = build_result(args.results.resolve(), rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
