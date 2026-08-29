#!/usr/bin/env python3
"""Finalize Kimi K3 pull telemetry after the acquisition process exits."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path


def _vmstat() -> dict[str, int]:
    wanted = {"pswpin", "pswpout", "oom_kill"}
    values = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in wanted:
            values[key] = int(value)
    return values


def _mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    started = dt.datetime.fromisoformat(progress["started_at"])
    model = args.model.resolve()
    shards = sorted(model.glob("model-*.safetensors"))
    committed = sum(path.stat().st_size for path in shards)
    partial = sum(
        path.stat().st_size for path in (model / ".cache").rglob("*.incomplete")
    ) if (model / ".cache").exists() else 0

    manifest = None
    manifest_error = None
    source = {}
    validation = {}
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("revision") != args.revision:
            manifest_error = (
                f"manifest revision {manifest.get('revision')!r} != {args.revision!r}"
            )
        source = (manifest.get("artifacts") or {}).get("source") or {}
        validation = source.get("validation") or {}
        source_format = source.get("format") or validation.get("format")
        if source_format != "safetensors":
            manifest_error = "manifest source is not validated safetensors"
    except (OSError, json.JSONDecodeError) as exc:
        manifest_error = str(exc)

    ended = dt.datetime.now(dt.timezone.utc)
    if manifest is not None and manifest.get("completed_at"):
        try:
            ended = dt.datetime.fromisoformat(manifest["completed_at"])
        except (TypeError, ValueError):
            manifest_error = manifest_error or "manifest completed_at is invalid"
    duration = (ended - started).total_seconds()

    vm = _vmstat()
    stat = os.statvfs(model)
    declared_total = int(progress["total_bytes"])
    validated_total = validation.get("physical_bytes")
    expected_total = (
        int(validated_total)
        if isinstance(validated_total, int) and not isinstance(validated_total, bool)
        else declared_total
    )
    complete = (
        manifest_error is None
        and len(shards) == 96
        and committed == expected_total
        and partial == 0
    )
    record = {
        "schema_version": 1,
        "status": "completed" if complete else "failed",
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_s": duration,
        "model": str(model),
        "revision": args.revision,
        "committed_shards": len(shards),
        "committed_bytes": committed,
        "expected_bytes": expected_total,
        "declared_bytes": declared_total,
        "partial_bytes": partial,
        "effective_committed_bytes_per_s": committed / duration if duration > 0 else None,
        "manifest": str(args.manifest.resolve()),
        "manifest_error": manifest_error,
        "memory_available_end_bytes": _mem_available(),
        "disk_available_end_bytes": stat.f_bavail * stat.f_frsize,
        "vmstat_end": vm,
        "oom_kill_delta": vm.get("oom_kill", 0) - int(progress["oom_kill_baseline"]),
        "swap_out_pages_delta": vm.get("pswpout", 0) - int(progress["pswpout_baseline"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
