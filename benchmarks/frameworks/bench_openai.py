#!/usr/bin/env python3
"""Measure warm TTFT and single-stream decode throughput over an OpenAI API.

The server is warmed once, then measured repeatedly. Decode throughput excludes
the first token: ``(completion_tokens - 1) / (last_token_time - first_token_time)``.
The final streamed usage chunk supplies the exact completion-token count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


PROMPT = (
    "Solve this problem carefully and show your reasoning. Find the smallest positive "
    "integer n for which 7n is a perfect square and 5n is a perfect cube."
)


def stream_once(base_url: str, model: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    last = None
    completion_tokens = None
    pieces: list[str] = []
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Reasoning deltas are not named consistently across compatible APIs:
            # SGLang uses reasoning_content, while Ollama releases have used
            # reasoning or thinking.
            text = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("thinking")
                or delta.get("content")
                or ""
            )
            if text:
                now = time.perf_counter()
                first = first or now
                last = now
                pieces.append(text)
    ended = time.perf_counter()
    if first is None or last is None:
        raise RuntimeError("stream completed without a content or reasoning token")
    if completion_tokens is None:
        raise RuntimeError("server did not return streamed completion-token usage")
    decode_seconds = last - first
    return {
        "ttft_seconds": first - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (
            (completion_tokens - 1) / decode_seconds
            if completion_tokens > 1 and decode_seconds > 0
            else None
        ),
        "completion_tokens": completion_tokens,
        "request_seconds": ended - started,
        "output_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--version", default="unknown")
    parser.add_argument("--weight-source", default="unknown")
    parser.add_argument("--weight-format", default="unknown")
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    warmup = stream_once(args.base_url, args.model, args.warmup_tokens)
    trials = [
        stream_once(args.base_url, args.model, args.max_tokens)
        for _ in range(args.trials)
    ]
    result = {
        "schema_version": "1.0",
        "framework": args.framework,
        "framework_version": args.version,
        "status": "measured",
        "weights": {"source": args.weight_source, "format": args.weight_format},
        "method": {
            "concurrency": 1,
            "temperature": 0,
            "warmup_requests": 1,
            "measured_requests": args.trials,
            "max_output_tokens": args.max_tokens,
            "prompt": PROMPT,
            "ttft": "request start to first non-empty content/reasoning SSE delta",
            "decode": "(completion_tokens - 1) / (last delta - first delta)",
        },
        "warmup": warmup,
        "trials": trials,
        "metrics": {
            "warm_ttft_seconds_median": statistics.median(
                trial["ttft_seconds"] for trial in trials
            ),
            "decode_tokens_per_second_median": statistics.median(
                trial["decode_tokens_per_second"] for trial in trials
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
