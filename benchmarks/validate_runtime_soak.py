#!/usr/bin/env python3
"""Repeat a declared HTTP workload on an explicitly owned server, retaining evidence.

This collects endurance evidence; elapsed time alone never certifies the runtime.
Use a matching baseline report for strict completed-output preservation. Resource
snapshots and every cycle are streamed to disk rather than retained for hours.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

try:
    from benchmarks.bench_runtime_mixed import run_case, summarize, validate_cases
except ModuleNotFoundError:
    from bench_runtime_mixed import run_case, summarize, validate_cases


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def get_json(base, path, payload=None, timeout=30):
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"content-type": "application/json"},
    )
    busy_error = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # Rebuild rejects in-flight generation/prefix I/O with HTTP 503, and
        # another control operation with 409. Only its explicit busy response
        # can enter the caller's bounded retry loop; other errors stay fatal.
        if path != "/v1/cache/rebuild" or payload is None or exc.code not in (409, 503):
            raise
        busy_error = exc
        response = exc
    with response:
        raw = response.read((2 << 20) + 1)
    if len(raw) > 2 << 20:
        raise ValueError("resource response exceeded 2 MiB")
    result = json.loads(raw)
    if busy_error is not None and (
        not isinstance(result, dict) or result.get("status") != "busy"
    ):
        raise busy_error
    return result


def verify_baseline(workload, identity, baseline):
    cases = sorted(validate_cases(workload["cases"]), key=lambda c: c.get("at", 0))
    if not 2 <= len(cases) <= 64:
        raise ValueError("soak cycles require 2..64 cases")
    if not any("cancel_after_events" in c for c in cases) or not any(
        "cancel_after_events" not in c for c in cases
    ):
        raise ValueError("soak workload must include completed and cancelled requests")
    if any(c.get("at", 0) > 60 or c["request"]["max_tokens"] > 2048 for c in cases):
        raise ValueError("soak cycles require arrivals <=60s and outputs <=2048 tokens")
    if any(
        c["request"].get("temperature") != 0 or not c.get("require_output_budget", True)
        for c in cases
    ):
        raise ValueError(
            "strict soak preservation requires explicit greedy, fixed-output work"
        )
    if baseline["identity"] != identity or baseline["workload_sha256"] != digest(
        workload
    ):
        raise ValueError("baseline checkpoint/backend/workload identity differs")
    observed = baseline["observations"]
    expected = {row["case"]: row for row in observed}
    if len(expected) != len(observed) or set(expected) != {c["id"] for c in cases}:
        raise ValueError("baseline must contain each declared case exactly once")
    verify_outputs(cases, observed, expected)
    return cases, expected


def verify_outputs(cases, observations, expected):
    if len(observations) != len(cases):
        raise ValueError("missing workload observations")
    actual = {row["case"]: row for row in observations}
    if len(actual) != len(cases) or set(actual) != {c["id"] for c in cases}:
        raise ValueError("duplicate or changed case identity")
    for case in cases:
        row, reference = actual[case["id"]], expected[case["id"]]
        if row["error"] is not None:
            raise ValueError(f"request failed: {case['id']}: {row['error']}")
        if "cancel_after_events" in case:
            if (
                not row["cancelled"]
                or len(row["deliveries"]) != case["cancel_after_events"]
            ):
                raise ValueError("requested HTTP cancellation did not occur")
        else:
            if not row["done"] or row["cancelled"] or row["finish_reason"] is None:
                raise ValueError("incomplete declared output work")
            if row["completion_tokens"] != case["request"]["max_tokens"]:
                raise ValueError("completed output budget differs")
            if not row["deliveries"]:
                raise ValueError("reported output has no visible delivery")
            for key in (
                "output",
                "prompt_tokens",
                "completion_tokens",
                "finish_reason",
            ):
                if row[key] != reference[key]:
                    raise ValueError(
                        f"strict baseline output differs: {case['id']}: {key}"
                    )


def host_swap_out():
    for line in Path("/proc/vmstat").read_text().splitlines():
        if line.startswith("pswpout "):
            return int(line.split()[1])
    raise RuntimeError("host swap-out counter unavailable")


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def run_soak(
    *,
    base,
    model,
    workload,
    identity,
    baseline,
    output,
    duration=3600.0,
    concurrency=4,
    timeout=180.0,
    sample_interval=5.0,
    rebuild_every=0,
    max_evidence_mib=256,
    clock=time.monotonic,
):
    if any(
        not math.isfinite(v) or v <= 0 for v in (duration, timeout, sample_interval)
    ):
        raise ValueError(
            "duration, timeout and sample interval must be positive and finite"
        )
    if duration > 28800 or timeout > 600 or not 0.01 <= sample_interval <= 60:
        raise ValueError("soak exceeds duration/timeout/sampling envelope")
    if (
        not 1 <= concurrency <= 16
        or type(rebuild_every) is not int
        or rebuild_every < 0
    ):
        raise ValueError("invalid concurrency or rebuild cadence")
    if type(max_evidence_mib) is not int or not 1 <= max_evidence_mib <= 4096:
        raise ValueError("evidence budget must be 1..4096 MiB")
    cases, expected = verify_baseline(workload, identity, baseline)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = clock()
    report = {
        "schema_version": 1,
        "identity": identity,
        "workload_sha256": digest(workload),
        "baseline_sha256": digest(baseline),
        "requested_seconds": duration,
        "concurrency": concurrency,
        "request_timeout_seconds": timeout,
        "sample_interval_seconds": sample_interval,
        "rebuild_every_cycles": rebuild_every,
        "max_evidence_bytes": max_evidence_mib << 20,
        "evidence_bytes": 0,
        "cycles": 0,
        "warmup_cycles": 0,
        "attempted_requests": 0,
        "observed_requests": 0,
        "failed_requests": 0,
        "completed_requests": 0,
        "cancelled_requests": 0,
        "rebuilds": 0,
        "status": "running",
        "qualification": False,
        "note": "Collection only. Full state, memory drift, swap attribution, latency and release gates require review.",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "client_sha256": hashlib.sha256(
            Path(__file__).with_name("bench_runtime_mixed.py").read_bytes()
        ).hexdigest(),
    }
    write_report(output / "report.json", report)
    instance = None
    geometry = None
    samples = 0
    timed_started = None

    def write_event(stream, value):
        raw = json.dumps(value) + "\n"
        size = len(raw.encode())
        if report["evidence_bytes"] + size > report["max_evidence_bytes"]:
            raise ValueError("evidence budget exhausted; prior evidence retained")
        stream.write(raw)
        stream.flush()
        report["evidence_bytes"] += size

    def sample(stream):
        nonlocal instance, samples
        stats = get_json(base, "/v1/stats", timeout=timeout)
        if not stats.get("instance_id"):
            raise ValueError("server instance identity missing")
        if instance is None:
            instance = stats["instance_id"]
            report["instance_id"] = instance
        elif stats["instance_id"] != instance:
            raise ValueError("server restarted during soak")
        write_event(
            stream,
            {
                "at": clock() - started,
                "stats": stats,
                "host_pswpout_pages": host_swap_out(),
            },
        )
        samples += 1
        return stats

    try:
        with (
            (output / "resources.jsonl").open("x") as resources,
            (output / "cycles.jsonl").open("x") as cycles,
        ):
            geometry = get_json(base, "/v1/cache/status", timeout=timeout)["geometry"]
            report["cache_geometry"] = geometry
            initial = sample(resources)
            if initial["requests"]["active"]:
                raise ValueError("owned server is not initially idle")
            swap_before = host_swap_out()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                while (
                    timed_started is None
                    or not report["cycles"]
                    or clock() - timed_started < duration
                ):
                    warmup = timed_started is None
                    epoch = clock()
                    report["attempted_requests"] += len(cases)
                    futures = [
                        pool.submit(run_case, base, model, c, epoch, timeout)
                        for c in cases
                    ]
                    pending = set(futures)
                    while pending:
                        _, pending = wait(pending, timeout=sample_interval)
                        sample(resources)
                    results = [f.result() for f in futures]
                    rows = [asdict(r) for r in results]
                    summary = summarize(results)
                    write_event(
                        cycles,
                        {
                            "cycle": report["cycles"],
                            "warmup": warmup,
                            "at": epoch - started,
                            "summary": summary,
                            "observations": rows,
                        },
                    )
                    report["observed_requests"] += len(rows)
                    report["failed_requests"] += summary["failed"]
                    report["completed_requests"] += summary["completed"]
                    report["cancelled_requests"] += summary["cancelled"]
                    verify_outputs(cases, rows, expected)
                    # Frontend abort acknowledgement may trail the closed socket.
                    drain_deadline = clock() + timeout
                    while sample(resources)["requests"]["active"]:
                        if clock() >= drain_deadline:
                            raise TimeoutError(
                                "requests did not drain after HTTP completion/cancellation"
                            )
                        time.sleep(min(sample_interval, 0.1))
                    if warmup:
                        report["warmup_cycles"] = 1
                        timed_started = clock()
                    else:
                        report["cycles"] += 1
                    if (
                        not warmup
                        and rebuild_every
                        and report["cycles"] % rebuild_every == 0
                    ):
                        rebuild_deadline = clock() + timeout
                        while True:
                            rebuilt = get_json(
                                base,
                                "/v1/cache/rebuild",
                                {"num_pages": geometry["num_pages"]},
                                timeout,
                            )
                            if (
                                rebuilt.get("status") != "busy"
                                or clock() >= rebuild_deadline
                            ):
                                break
                            sample(resources)
                            time.sleep(min(sample_interval, 0.1))
                        if rebuilt.get("status") != "ok":
                            raise ValueError(f"idle cache rebuild failed: {rebuilt}")
                        if (
                            get_json(base, "/v1/cache/status", timeout=timeout)[
                                "geometry"
                            ]
                            != geometry
                        ):
                            raise ValueError("cache geometry changed during soak")
                        report["rebuilds"] += 1
                    report["elapsed_seconds"] = clock() - started
                    write_report(output / "report.json", report)
            report["host_swap_out_delta_pages"] = host_swap_out() - swap_before
            report["status"] = "collected"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["unobserved_requests"] = (
            report["attempted_requests"] - report["observed_requests"]
        )
        report["resource_samples"] = samples
        report["elapsed_seconds"] = clock() - started
        report["soak_elapsed_seconds"] = (
            clock() - timed_started if timed_started is not None else 0.0
        )
        report["duration_satisfied"] = report["soak_elapsed_seconds"] >= duration
        write_report(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=3600.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--max-evidence-mib", type=int, default=256)
    parser.add_argument(
        "--rebuild-every",
        type=int,
        default=0,
        help="Opt in to idle cache clears on this explicitly owned server",
    )
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text())
    identity["served_model"] = args.model
    result = run_soak(
        base=args.base_url,
        model=args.model,
        workload=json.loads(args.workload.read_text()),
        identity=identity,
        baseline=json.loads(args.baseline.read_text()),
        output=args.output,
        duration=args.duration_seconds,
        concurrency=args.concurrency,
        timeout=args.timeout,
        sample_interval=args.sample_interval,
        rebuild_every=args.rebuild_every,
        max_evidence_mib=args.max_evidence_mib,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
