#!/usr/bin/env python3
"""Fail closed unless a supervised checkpoint conversion is safe to serve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate(record: dict[str, Any]) -> list[str]:
    errors = []
    if record.get("status") != "completed":
        errors.append(f"status={record.get('status')!r}, expected 'completed'")
    if record.get("returncode") != 0:
        errors.append(f"returncode={record.get('returncode')!r}, expected 0")
    if record.get("termination_reason") is not None:
        errors.append(f"termination_reason={record.get('termination_reason')!r}")

    delta = (record.get("vmstat") or {}).get("delta") or {}
    for key in ("oom_kill", "pswpout"):
        if key not in delta:
            errors.append(f"vmstat.delta.{key} is missing")
        elif delta[key] != 0:
            errors.append(f"vmstat.delta.{key}={delta[key]!r}, expected 0")
    swap = record.get("swap") or {}
    if "growth_bytes" not in swap:
        errors.append("swap.growth_bytes is missing")
    elif swap["growth_bytes"] != 0:
        errors.append(f"swap.growth_bytes={swap['growth_bytes']!r}, expected 0")

    memory = record.get("memory") or {}
    disk = record.get("disk") or {}
    if not isinstance(memory.get("available_min_bytes"), int):
        errors.append("memory.available_min_bytes is missing")
    if not isinstance(memory.get("process_group_peak_rss_bytes"), int):
        errors.append("memory.process_group_peak_rss_bytes is missing")
    if not isinstance(disk.get("available_min_bytes"), int):
        errors.append("disk.available_min_bytes is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    errors = []
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        errors = validate(record)
    except (OSError, json.JSONDecodeError) as exc:
        errors = [str(exc)]
    result = {
        "schema_version": 1,
        "artifact": str(args.record.resolve()),
        "status": "passed" if not errors else "failed",
        "errors": errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
