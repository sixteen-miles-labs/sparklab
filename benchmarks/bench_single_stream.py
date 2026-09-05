#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Matched single-stream math, coding, and prose probes against a running server."""
import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

PROMPTS = {
    "math": "Find the sum of all integer bases b > 9 for which 17_b is a divisor of 97_b. Explain your reasoning and put the final answer in \\boxed{}.",
    "code": "Implement a Python LRU cache using a doubly linked list and a dictionary. Include get and put methods, a capacity check, and tests covering updates, eviction, and capacity one. Explain the invariants.",
    "prose": "Explain how ocean navigation developed from coastal landmarks to celestial navigation and modern satellite systems. Write continuous prose with concrete examples of the limitations of each approach.",
}

def generate(base_url, prompt, tokens, thinking):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    stamps, pieces, usage = [], [], None
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if text:
                    stamps.append(time.perf_counter())
                    pieces.append(text)
    if usage is None or len(stamps) < 2:
        raise RuntimeError("Incomplete streaming response")
    if usage.get("completion_tokens") != tokens:
        raise RuntimeError(f"Expected {tokens} completion tokens, got {usage}")
    text = "".join(pieces)
    return {"decode_tokens_per_second": (usage["completion_tokens"] - 1) / (stamps[-1] - stamps[0]),
            "ttft_seconds": stamps[0] - start, "elapsed_seconds": time.perf_counter() - start,
            "usage": usage, "output_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:1927")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--workloads", default="math,code,prose")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.tokens < 2 or args.trials < 1:
        p.error("--tokens must be at least two and --trials must be positive")
    workloads = args.workloads.split(",")
    if "aime" in workloads:
        from bench_decode_moe import load_problem

        PROMPTS["aime"], _ = load_problem(None, 0)
    if any(name not in PROMPTS for name in workloads):
        p.error("--workloads must contain math, code, prose, or aime")
    report = {"label": args.label, "tokens": args.tokens, "thinking": args.thinking, "workloads": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in workloads:
        prompt = PROMPTS[name]
        generate(args.base_url, prompt, args.tokens, args.thinking)
        trials = [generate(args.base_url, prompt, args.tokens, args.thinking) for _ in range(args.trials)]
        result = {"prompt": prompt, "trials": trials,
                  "decode_tokens_per_second_median": statistics.median(t["decode_tokens_per_second"] for t in trials),
                  "ttft_seconds_median": statistics.median(t["ttft_seconds"] for t in trials)}
        report["workloads"][name] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, round(result["decode_tokens_per_second_median"], 2), "tok/s", flush=True)

if __name__ == "__main__":
    main()
