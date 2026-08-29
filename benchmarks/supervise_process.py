#!/usr/bin/env python3
"""Run a benchmark/build command with fail-closed host-memory telemetry.

The supervisor writes a JSON record even when the child fails or is terminated for
low memory/disk.  It is intended for very large checkpoint preparation and model
startup experiments where an ordinary timing wrapper would lose the OOM evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


GiB = 1 << 30


def _mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def _vmstat() -> dict[str, int]:
    wanted = {"pswpin", "pswpout", "oom_kill"}
    with open("/proc/vmstat") as f:
        values = {key: int(value) for key, value in map(str.split, f) if key in wanted}
    return {key: values.get(key, 0) for key in sorted(wanted)}


def _swap_used_bytes() -> int:
    values = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key = line.split(":", 1)[0]
            if key in {"SwapTotal", "SwapFree"}:
                values[key] = int(line.split()[1]) * 1024
    return max(0, values.get("SwapTotal", 0) - values.get("SwapFree", 0))


def _key_value_file(path: Path) -> dict[str, int]:
    try:
        return {
            key: int(value)
            for key, value in map(str.split, path.read_text().splitlines())
        }
    except (FileNotFoundError, PermissionError, ValueError):
        return {}


def _cgroup_snapshot() -> dict[str, object]:
    try:
        relative = next(
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
    except (FileNotFoundError, PermissionError, StopIteration):
        return {}
    root = Path("/sys/fs/cgroup") / relative.lstrip("/")

    def read_int(name: str) -> int | None:
        try:
            return int((root / name).read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            return None

    try:
        swap_max = (root / "memory.swap.max").read_text().strip()
    except (FileNotFoundError, PermissionError):
        swap_max = None
    return {
        "path": relative,
        "memory_current_bytes": read_int("memory.current"),
        "memory_peak_bytes": read_int("memory.peak"),
        "swap_current_bytes": read_int("memory.swap.current"),
        "swap_peak_bytes": read_int("memory.swap.peak"),
        "swap_max": swap_max,
        "events_local": _key_value_file(root / "memory.events.local"),
    }


def _group_rss(pgid: int) -> int:
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            if int(fields[4]) != pgid:
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safety_reason(
    start_vm: dict[str, int],
    current_vm: dict[str, int],
    swap_start: int,
    swap_current: int,
    *,
    max_oom_kill_delta: int | None,
    max_pswpout_delta: int | None,
    max_swap_growth_bytes: int | None,
) -> str | None:
    checks = (
        ("oom_kill", max_oom_kill_delta),
        ("pswpout", max_pswpout_delta),
    )
    for key, maximum in checks:
        if maximum is None:
            continue
        delta = current_vm[key] - start_vm[key]
        if delta > maximum:
            return f"{key} delta {delta} exceeded {maximum}"
    if max_swap_growth_bytes is not None:
        growth = max(0, swap_current - swap_start)
        if growth > max_swap_growth_bytes:
            return (
                f"swap growth {growth} bytes exceeded "
                f"{max_swap_growth_bytes} bytes"
            )
    return None


def _cgroup_safety_reason(
    start: dict[str, object],
    current: dict[str, object],
    *,
    require_swap_disabled: bool,
    max_oom_kill_delta: int | None,
    max_swap_growth_bytes: int | None,
) -> str | None:
    if require_swap_disabled and current.get("swap_max") != "0":
        return f"cgroup memory.swap.max={current.get('swap_max')!r}, expected '0'"
    start_events = start.get("events_local") or {}
    current_events = current.get("events_local") or {}
    if max_oom_kill_delta is not None:
        if not isinstance(start_events, dict) or not isinstance(current_events, dict):
            return "cgroup local OOM telemetry is unavailable"
        delta = current_events.get("oom_kill", 0) - start_events.get("oom_kill", 0)
        if delta > max_oom_kill_delta:
            return f"cgroup oom_kill delta {delta} exceeded {max_oom_kill_delta}"
    if max_swap_growth_bytes is not None:
        swap_start = start.get("swap_current_bytes")
        swap_current = current.get("swap_current_bytes")
        if not isinstance(swap_start, int) or not isinstance(swap_current, int):
            return "cgroup swap telemetry is unavailable"
        growth = max(0, swap_current - swap_start)
        if growth > max_swap_growth_bytes:
            return (
                f"cgroup swap growth {growth} bytes exceeded "
                f"{max_swap_growth_bytes} bytes"
            )
    return None


def _terminate_group(proc: subprocess.Popen, grace: float) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="JSON telemetry record")
    p.add_argument("--log", required=True, help="combined child stdout/stderr")
    p.add_argument("--disk-path", default=".", help="filesystem whose free space is guarded")
    p.add_argument("--min-memory-gib", type=float, default=12.0)
    p.add_argument("--min-disk-gib", type=float, default=16.0)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--grace", type=float, default=30.0)
    p.add_argument("--max-oom-kill-delta", type=int)
    p.add_argument("--max-pswpout-delta", type=int)
    p.add_argument("--max-swap-growth-bytes", type=int)
    p.add_argument("--require-cgroup-swap-disabled", action="store_true")
    p.add_argument("--max-cgroup-oom-kill-delta", type=int)
    p.add_argument("--max-cgroup-swap-growth-bytes", type=int)
    p.add_argument("command", nargs=argparse.REMAINDER)
    ns = p.parse_args()
    if ns.command[:1] == ["--"]:
        ns.command = ns.command[1:]
    if not ns.command:
        p.error("a command is required after --")
    return ns


def main() -> int:
    args = _parse_args()
    output = Path(args.output).resolve()
    log_path = Path(args.log).resolve()
    disk_path = Path(args.disk_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_vm = _vmstat()
    swap_start = _swap_used_bytes()
    cgroup_start = _cgroup_snapshot()
    start_mem = _mem_available()
    start_disk = os.statvfs(disk_path).f_bavail * os.statvfs(disk_path).f_frsize
    started_at = _iso_now()
    started = time.monotonic()
    proc: subprocess.Popen | None = None
    reason = None
    min_mem = start_mem
    min_disk = start_disk
    peak_rss = 0
    samples = 0

    with log_path.open("w", buffering=1) as log:
        proc = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def tee() -> None:
            assert proc is not None and proc.stdout is not None
            for chunk in iter(proc.stdout.readline, ""):
                log.write(chunk)
                sys.stdout.write(chunk)
                sys.stdout.flush()

        reader = threading.Thread(target=tee, name="supervisor-tee", daemon=True)
        reader.start()
        try:
            while proc.poll() is None:
                mem = _mem_available()
                stat = os.statvfs(disk_path)
                disk = stat.f_bavail * stat.f_frsize
                min_mem = min(min_mem, mem)
                min_disk = min(min_disk, disk)
                peak_rss = max(peak_rss, _group_rss(proc.pid))
                samples += 1
                if mem < args.min_memory_gib * GiB:
                    reason = f"MemAvailable below {args.min_memory_gib:g} GiB"
                    _terminate_group(proc, args.grace)
                    break
                if disk < args.min_disk_gib * GiB:
                    reason = f"free disk below {args.min_disk_gib:g} GiB"
                    _terminate_group(proc, args.grace)
                    break
                reason = _safety_reason(
                    start_vm,
                    _vmstat(),
                    swap_start,
                    _swap_used_bytes(),
                    max_oom_kill_delta=args.max_oom_kill_delta,
                    max_pswpout_delta=args.max_pswpout_delta,
                    max_swap_growth_bytes=args.max_swap_growth_bytes,
                )
                if reason is not None:
                    _terminate_group(proc, args.grace)
                    break
                reason = _cgroup_safety_reason(
                    cgroup_start,
                    _cgroup_snapshot(),
                    require_swap_disabled=args.require_cgroup_swap_disabled,
                    max_oom_kill_delta=args.max_cgroup_oom_kill_delta,
                    max_swap_growth_bytes=args.max_cgroup_swap_growth_bytes,
                )
                if reason is not None:
                    _terminate_group(proc, args.grace)
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            reason = "supervisor interrupted"
            _terminate_group(proc, args.grace)
        finally:
            reader.join(timeout=5.0)

    end_vm = _vmstat()
    swap_end = _swap_used_bytes()
    cgroup_end = _cgroup_snapshot()
    if reason is None:
        reason = _cgroup_safety_reason(
            cgroup_start,
            cgroup_end,
            require_swap_disabled=args.require_cgroup_swap_disabled,
            max_oom_kill_delta=args.max_cgroup_oom_kill_delta,
            max_swap_growth_bytes=args.max_cgroup_swap_growth_bytes,
        )
    end_mem = _mem_available()
    stat = os.statvfs(disk_path)
    end_disk = stat.f_bavail * stat.f_frsize
    returncode = proc.returncode if proc is not None else None
    record = {
        "schema_version": 1,
        "command": args.command,
        "started_at": started_at,
        "ended_at": _iso_now(),
        "duration_s": round(time.monotonic() - started, 3),
        "status": "guard_terminated" if reason else ("completed" if returncode == 0 else "failed"),
        "termination_reason": reason,
        "returncode": returncode,
        "samples": samples,
        "memory": {
            "available_start_bytes": start_mem,
            "available_min_bytes": min_mem,
            "available_end_bytes": end_mem,
            "process_group_peak_rss_bytes": peak_rss,
        },
        "disk": {
            "path": str(disk_path),
            "available_start_bytes": start_disk,
            "available_min_bytes": min_disk,
            "available_end_bytes": end_disk,
        },
        "vmstat": {
            "start": start_vm,
            "end": end_vm,
            "delta": {key: end_vm[key] - start_vm[key] for key in start_vm},
        },
        "swap": {
            "used_start_bytes": swap_start,
            "used_end_bytes": swap_end,
            "growth_bytes": max(0, swap_end - swap_start),
        },
        "cgroup": {
            "start": cgroup_start,
            "end": cgroup_end,
            "delta": {
                "oom_kill": (
                    (cgroup_end.get("events_local") or {}).get("oom_kill", 0)
                    - (cgroup_start.get("events_local") or {}).get("oom_kill", 0)
                ),
                "swap_current_bytes": (
                    cgroup_end.get("swap_current_bytes", 0)
                    - cgroup_start.get("swap_current_bytes", 0)
                ),
            },
        },
        "log": str(log_path),
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if returncode == 0 and reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
