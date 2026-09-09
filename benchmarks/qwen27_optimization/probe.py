#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay the six-prompt optimization screen from a frozen AIPerf raw export."""

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def request(base_url, model, payload):
    body = dict(payload)
    body.update(
        model=model, max_completion_tokens=128, temperature=0, ignore_eos=True,
        stream=True, stream_options={"include_usage": True},
    )
    body.pop("max_tokens", None)
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = usage = None
    parts = []
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            value = line[5:].strip()
            if value == b"[DONE]":
                break
            event = json.loads(value)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = (delta.get("reasoning_content") or delta.get("reasoning") or "")
                text += delta.get("content") or ""
                if text:
                    last = time.perf_counter()
                    first = first or last
                    parts.append(text)
    elapsed = time.perf_counter() - start
    if not usage or usage.get("completion_tokens") != 128 or first is None or last <= first:
        raise RuntimeError(f"Incomplete 128-token response: {usage}")
    text = "".join(parts)
    return {
        "ttft_s": first - start, "elapsed_s": elapsed,
        "decode_tps": 127 / (last - first), "usage": usage, "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1938")
    parser.add_argument("--model", default="qwen27-opt")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    rows = [json.loads(line) for line in args.input_export.read_text().splitlines() if line]
    rows.sort(key=lambda row: row["metadata"]["request_start_ns"])
    if len(rows) < 7:
        parser.error("input export must contain at least seven requests")
    report = {
        "input_export_sha256": hashlib.sha256(args.input_export.read_bytes()).hexdigest(),
        "warmup": request(args.base_url, args.model, rows[6]["payload"]),
        "requests": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows[:6]:
        result = request(args.base_url, args.model, row["payload"])
        report["requests"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print({key: round(result[key], 3) for key in ("decode_tps", "ttft_s", "elapsed_s")}, flush=True)
    report["mean"] = {
        key: statistics.mean(row[key] for row in report["requests"])
        for key in ("decode_tps", "ttft_s", "elapsed_s")
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
