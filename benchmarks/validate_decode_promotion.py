#!/usr/bin/env python3
"""Validate that a decode JSONL row is safe to promote to a longer rung.

The benchmark promotion contract is deliberately stricter than process exit status:
the API response, FTW provenance, output, lifecycle memory telemetry, OOM counters,
and swap counters must all be present and clean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _zero(telemetry: dict[str, Any], key: str, errors: list[str], scope: str) -> None:
    if key not in telemetry:
        errors.append(f"{scope}.{key} is missing")
    elif telemetry[key] != 0:
        errors.append(f"{scope}.{key}={telemetry[key]!r}, expected 0")


def _validate_safety_telemetry(
    telemetry: dict[str, Any], errors: list[str], scope: str
) -> None:
    _zero(telemetry, "oom_kill_delta", errors, scope)
    cgroup = telemetry.get("cgroup") or {}
    end = cgroup.get("end") or {}
    delta = cgroup.get("delta") or {}
    if end.get("swap_max") == "0":
        for key in ("oom_kill", "swap_current_bytes"):
            _zero(delta, key, errors, f"{scope}.cgroup.delta")
        if end.get("swap_current_bytes") != 0:
            errors.append(
                f"{scope}.cgroup.end.swap_current_bytes="
                f"{end.get('swap_current_bytes')!r}, expected 0"
            )
    else:
        for key in ("swap_out_pages_delta", "swap_growth_bytes"):
            _zero(telemetry, key, errors, scope)


def validate_row(row: dict[str, Any], expected_tokens: int) -> list[str]:
    errors: list[str] = []
    if row.get("status") == "failed":
        errors.append(f"benchmark row failed: {row.get('error_type')}: {row.get('error')}")

    checkpoint = row.get("checkpoint") or {}
    if checkpoint.get("format") != "ftw":
        errors.append(f"checkpoint.format={checkpoint.get('format')!r}, expected 'ftw'")
    if not checkpoint.get("fingerprint"):
        errors.append("checkpoint.fingerprint is missing")
    if not isinstance(checkpoint.get("bytes"), int) or checkpoint.get("bytes", 0) <= 0:
        errors.append("checkpoint.bytes is missing or invalid")

    completion = row.get("completion_tokens")
    events = row.get("events")
    if completion != expected_tokens:
        errors.append(f"completion_tokens={completion!r}, expected {expected_tokens}")
    if not isinstance(events, int) or events <= 0:
        errors.append(f"events={events!r}, expected a positive integer")
    if not row.get("output_sha1"):
        errors.append("output_sha1 is missing")

    output = row.get("output_text")
    if not isinstance(output, str) or not output:
        errors.append("output_text is missing or empty")
    else:
        if "\x00" in output or "\ufffd" in output:
            errors.append("output_text contains an invalid NUL or replacement character")
        visible = "".join(ch for ch in output if not ch.isspace())
        if expected_tokens >= 16 and (len(visible) < 8 or len(set(visible)) < 3):
            errors.append("output_text is not plausible for a multi-token response")

    if row.get("oom_markers") != 0:
        errors.append(f"oom_markers={row.get('oom_markers')!r}, expected 0")

    lifecycle = row.get("lifecycle_telemetry") or {}
    if lifecycle.get("memory_guard_triggered") is not False:
        errors.append(
            "lifecycle_telemetry.memory_guard_triggered is missing or not false"
        )
    if not isinstance(lifecycle.get("mem_available_gib_min"), (int, float)):
        errors.append("lifecycle_telemetry.mem_available_gib_min is missing")
    _validate_safety_telemetry(lifecycle, errors, "lifecycle_telemetry")

    request = row.get("gpu_telemetry") or {}
    _validate_safety_telemetry(request, errors, "gpu_telemetry")
    return errors


def load_single_row(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"{path} contains {len(rows)} rows; expected exactly one")
    if not isinstance(rows[0], dict):
        raise ValueError(f"{path} row is not a JSON object")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--expected-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, help="write the machine-readable gate record")
    args = parser.parse_args()
    if args.expected_tokens <= 0:
        parser.error("--expected-tokens must be positive")

    errors: list[str]
    try:
        row = load_single_row(args.jsonl)
        errors = validate_row(row, args.expected_tokens)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors = [str(exc)]
    record = {
        "schema_version": 1,
        "artifact": str(args.jsonl.resolve()),
        "expected_tokens": args.expected_tokens,
        "status": "passed" if not errors else "failed",
        "errors": errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
