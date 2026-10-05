#!/usr/bin/env python3
"""Measure block-diffusion SSE latency and end-to-end throughput, batch one."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

PROMPT = (
    "Explain how a hash table works, including collisions, resizing, and its time "
    "complexity. Use a concrete example and give a detailed, self-contained explanation."
)


def stream_once(base_url, model, max_tokens, thinking, *, opener=None, clock=None):
    opener = opener or urllib.request.urlopen
    clock = clock or time.perf_counter
    body = {
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = clock()
    emissions = []
    content, reasoning = [], []
    usage = None
    finish_reason = None
    done = False
    with opener(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"server stream error: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if text or thought:
                    emissions.append(clock() - started)
                    content.append(text)
                    reasoning.append(thought)
    elapsed = clock() - started
    if not done or not emissions:
        raise RuntimeError("incomplete stream or no content/reasoning emitted")
    if not usage or type(usage.get("completion_tokens")) is not int:
        raise RuntimeError("server did not return exact completion-token usage")
    tokens = usage["completion_tokens"]
    if tokens <= 0:
        raise RuntimeError("completion-token usage must be positive")
    text, thought = "".join(content), "".join(reasoning)
    return {
        "ttft_seconds": emissions[0], "request_seconds": elapsed,
        "completion_tokens": tokens, "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens_per_second": tokens / elapsed,
        "content_deltas": len(emissions), "emission_times_seconds": emissions,
        "finish_reason": finish_reason, "content": text, "reasoning": thought,
        "output_sha256": hashlib.sha256((thought + text).encode()).hexdigest(),
    }


def summarize(trials):
    return {
        "warm_ttft_seconds_median": statistics.median(t["ttft_seconds"] for t in trials),
        "request_seconds_median": statistics.median(t["request_seconds"] for t in trials),
        "completion_tokens_per_second_median": statistics.median(
            t["completion_tokens_per_second"] for t in trials),
        "completion_tokens_per_second_aggregate": (
            sum(t["completion_tokens"] for t in trials) / sum(t["request_seconds"] for t in trials)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="nvidia/diffusiongemma-26B-A4B-it-NVFP4")
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--thinking", choices=["on", "off", "both"], default="both")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, help="JSON containing backend/hardware/weight identity")
    args = parser.parse_args()
    if args.trials < 1 or any(t < 1 for t in args.max_tokens):
        parser.error("trials and token limits must be positive")
    result = {
        "schema_version": "1.0", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "provenance": json.loads(args.provenance.read_text()) if args.provenance else {},
        "method": {
            "concurrency": 1, "prompt": PROMPT,
            "sampling": "checkpoint entropy-bound sampler; ordinary temperature and seed are unsupported by vLLM 0.28 diffusion",
            "warmup_requests_per_case": 1, "measured_requests_per_case": args.trials,
            "ttft": "request start to first nonempty content or reasoning SSE delta",
            "throughput": "exact completion_tokens / full request elapsed seconds, including TTFT",
            "decode_tokens_per_second": None,
            "note": "SSE delta count is not a token count. Block emission invalidates autoregressive decode timing.",
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    modes = [False, True] if args.thinking == "both" else [args.thinking == "on"]
    save()
    try:
        for thinking in modes:
            for limit in args.max_tokens:
                case = {"thinking": thinking, "max_tokens": limit, "trials": []}
                result["cases"].append(case)
                case["warmup"] = stream_once(args.base_url, args.model, limit, thinking)
                save()
                for _ in range(args.trials):
                    case["trials"].append(stream_once(args.base_url, args.model, limit, thinking))
                    save()
                case["metrics"] = summarize(case["trials"])
                save()
                print(json.dumps({"thinking": thinking, "max_tokens": limit, **case["metrics"]}), flush=True)
        result["status"] = "measured"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
