#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible arrival-scheduled streaming workloads; no inferred per-token times."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
import time
import urllib.request
import urllib.error


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass
class Observation:
    case: str
    scheduled: float
    started: float
    ended: float = 0.0
    deliveries: list[float] = field(default_factory=list)
    delivery_chars: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    finish_reason: str | None = None
    output: str = ""
    error: str | None = None
    cancelled: bool = False
    done: bool = False

    @property
    def success(self):
        return self.error is None and self.done and not self.cancelled


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 10000:
        raise ValueError("workload must contain 1..10000 cases")
    seen = set()
    for case in cases:
        if not isinstance(case.get("id"), str) or case["id"] in seen:
            raise ValueError("each case needs a unique string id")
        seen.add(case["id"])
        at = case.get("at", 0)
        if (
            not isinstance(at, (int, float))
            or not math.isfinite(at)
            or not 0 <= at <= 86400
        ):
            raise ValueError("arrival time must be finite and in [0, 86400]")
        if case.get("endpoint", "/v1/completions") not in (
            "/v1/completions",
            "/v1/chat/completions",
        ):
            raise ValueError("unsupported endpoint")
        payload = case.get("request", {})
        if not isinstance(payload, dict) or payload.get("n", 1) != 1:
            raise ValueError("only one completion per request is supported")
        budget = payload.get("max_tokens")
        if type(budget) is not int or budget <= 0:
            raise ValueError("each request needs a positive max_tokens")
        if "cancel_after_events" in case and (
            type(case["cancel_after_events"]) is not int
            or case["cancel_after_events"] <= 0
        ):
            raise ValueError("cancel_after_events must be positive")
    return cases


def observe_event(result, event, now):
    """One delivery event may contain many tokens. Preserve its single timestamp."""
    if event.get("error"):
        raise ValueError(f"server error: {event['error']}")
    usage = event.get("usage")
    if usage is not None:
        for key in ("prompt_tokens", "completion_tokens"):
            if type(usage.get(key)) is not int or usage[key] < 0:
                raise ValueError(f"invalid or missing usage.{key}")
        result.prompt_tokens = usage["prompt_tokens"]
        result.completion_tokens = usage["completion_tokens"]
        result.cached_tokens = (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        )
        if (
            type(result.cached_tokens) is not int
            or not 0 <= result.cached_tokens <= result.prompt_tokens
        ):
            raise ValueError("invalid cached token usage")
    pieces = []
    for choice in event.get("choices", []):
        if choice.get("finish_reason") is not None:
            result.finish_reason = choice["finish_reason"]
        if choice.get("text"):
            pieces.append(choice["text"])
        delta = choice.get("delta") or {}
        for key in ("content", "reasoning_content"):
            if isinstance(delta.get(key), str) and delta[key]:
                pieces.append(delta[key])
        if delta.get("tool_calls"):
            pieces.append(json.dumps(delta["tool_calls"], sort_keys=True))
    if pieces:
        text = "".join(pieces)
        result.deliveries.append(now)
        result.delivery_chars.append(len(text))
        result.output += text
    return usage is not None


def run_case(base_url, model, case, epoch, timeout, clock=time.monotonic):
    due = epoch + case.get("at", 0)
    delay = due - clock()
    if delay > 0:
        time.sleep(delay)
    result = Observation(case["id"], due - epoch, clock() - epoch)
    payload = dict(case["request"])
    payload.update(model=model, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(
        base_url.rstrip("/") + case.get("endpoint", "/v1/completions"),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    saw_usage = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    result.done = True
                    break
                saw_usage |= observe_event(result, json.loads(data), clock() - epoch)
                if len(result.deliveries) >= case.get("cancel_after_events", math.inf):
                    result.cancelled = True
                    break
                if len(result.deliveries) > 100000:
                    raise ValueError("delivery event limit exceeded")
        if not result.cancelled:
            if not result.done or not saw_usage or result.finish_reason is None:
                raise ValueError("incomplete stream, finish reason or usage missing")
            expected = case.get("expected_prompt_tokens")
            if expected is not None and result.prompt_tokens != expected:
                raise ValueError(
                    f"prompt token count {result.prompt_tokens} != {expected}"
                )
            if (
                case.get("require_output_budget", True)
                and result.completion_tokens != payload["max_tokens"]
            ):
                raise ValueError("request did not complete the declared output work")
            if not result.deliveries and result.completion_tokens:
                raise ValueError("tokens reported with no visible delivery")
    except urllib.error.HTTPError as exc:
        result.error = f"HTTP {exc.code}: {exc.read(8192).decode(errors='replace')}"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    result.ended = clock() - epoch
    return result


def summarize(results):
    successful = [r for r in results if r.success]
    gaps = [b - a for r in successful for a, b in zip(r.deliveries, r.deliveries[1:])]
    ttft = [r.deliveries[0] - r.started for r in successful if r.deliveries]
    elapsed = max((r.ended for r in results), default=0) - min(
        (r.scheduled for r in results), default=0
    )
    tokens = sum(r.completion_tokens for r in successful)
    return {
        "attempted": len(results),
        "completed": len(successful),
        "failed": sum(r.error is not None for r in results),
        "cancelled": sum(r.cancelled for r in results),
        "success_rate": len(successful) / len(results) if results else 0,
        "completed_output_tokens": tokens,
        "elapsed_seconds": elapsed,
        "successful_work_tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
        "delivery_gap_count": len(gaps),
        "ttft_count": len(ttft),
        "delivery_gap_p50_seconds": percentile(gaps, 0.5),
        "delivery_gap_p95_seconds": percentile(gaps, 0.95),
        "delivery_gap_p99_seconds": percentile(gaps, 0.99)
        if len(gaps) >= 1000
        else None,
        "ttft_p50_seconds": percentile(ttft, 0.5),
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "ttft_p99_seconds": percentile(ttft, 0.99) if len(ttft) >= 1000 else None,
        "client_queue_delay_p95_seconds": percentile(
            [r.started - r.scheduled for r in results], 0.95
        ),
        "visible_events": sum(len(r.deliveries) for r in results),
        "note": "Delivery timestamps describe SSE events, not individual speculative tokens.",
    }


def compare(baseline, candidate):
    if baseline["workload_sha256"] != candidate["workload_sha256"]:
        raise ValueError("workload identity differs")
    if baseline["identity"] != candidate["identity"]:
        raise ValueError("checkpoint/backend/sampling/cache identity differs")
    if (
        baseline["summary"]["completed"] != baseline["summary"]["attempted"]
        or candidate["summary"]["completed"] != candidate["summary"]["attempted"]
    ):
        raise ValueError("benefit comparison requires identical fully completed work")
    keys = (
        "successful_work_tokens_per_second",
        "delivery_gap_p95_seconds",
        "ttft_p95_seconds",
    )
    return {
        k: {
            "baseline": baseline["summary"][k],
            "candidate": candidate["summary"][k],
            "change_percent": 100
            * (candidate["summary"][k] / baseline["summary"][k] - 1)
            if baseline["summary"][k] and candidate["summary"][k] is not None
            else None,
        }
        for k in keys
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--identity",
        type=Path,
        required=True,
        help="Exact checkpoint/backend/cache/precision metadata JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if (
        not 1 <= args.concurrency <= 128
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("concurrency must be 1..128 and timeout positive and finite")
    workload = json.loads(args.workload.read_text())
    cases = sorted(validate_cases(workload["cases"]), key=lambda c: c.get("at", 0))
    identity = json.loads(args.identity.read_text())
    identity["served_model"] = args.model
    epoch = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_case, args.base_url, args.model, c, epoch, args.timeout)
            for c in cases
        ]
        results = [f.result() for f in futures]
    provenance = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines(),
        "platform": platform.platform(),
        "base_url": args.base_url,
        "client_concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
    }
    report = {
        "schema_version": "1.0",
        "label": args.label,
        "identity": identity,
        "provenance": provenance,
        "workload_sha256": hashlib.sha256(
            json.dumps(workload, sort_keys=True).encode()
        ).hexdigest(),
        "workload": workload,
        "summary": summarize(results),
        "observations": [asdict(r) for r in results],
    }
    if args.baseline:
        report["comparison"] = compare(json.loads(args.baseline.read_text()), report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
