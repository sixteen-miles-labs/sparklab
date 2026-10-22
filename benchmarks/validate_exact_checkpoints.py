#!/usr/bin/env python3
"""Real-server branch/full-prompt/rollback smoke on an explicitly owned test server.

--reset-cache performs idle cache rebuilds to obtain cold references. This is a
small correctness gate, not a quality, performance, or endurance certification.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lengths", default="1,15,16,17,31,63,64,65")
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument(
        "--churn-requests",
        type=int,
        default=0,
        help="Use controlled prefix pressure instead of rebuilding",
    )
    args = parser.parse_args()
    if not args.reset_cache and not 1 <= args.churn_requests <= 128:
        parser.error(
            "choose --reset-cache or --churn-requests 1..128 for cold references"
        )

    def call(path, payload=None):
        req = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"HTTP {exc.code}: {exc.read(8192).decode(errors='replace')}"
            ) from exc

    geo = call("/v1/cache/status")["geometry"]

    def reset():
        reply = call("/v1/cache/rebuild", {"num_pages": geo["num_pages"]})
        if reply.get("status") != "ok":
            raise RuntimeError(f"cache reset failed: {reply}")

    def generate(prompt, budget):
        start = time.monotonic()
        reply = call(
            "/v1/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": budget,
                "ignore_eos": True,
            },
        )
        usage = reply.get("usage", {})
        if usage.get("completion_tokens") != budget:
            raise RuntimeError("incomplete output work")
        return {
            "text": reply["choices"][0]["text"],
            "usage": usage,
            "seconds": time.monotonic() - start,
        }

    churn_serial = 0

    def cold_start():
        nonlocal churn_serial
        if args.reset_cache:
            reset()
        else:
            for _ in range(args.churn_requests):
                churn_serial += 1
                generate(
                    f"Cache control {churn_serial}: a different synthetic prefix."
                    + " route" * 37,
                    1,
                )

    report = {
        "schema_version": "1.0",
        "geometry": geo,
        "cases": [],
        "cold_control": "rebuild"
        if args.reset_cache
        else {"churn_requests": args.churn_requests},
        "note": "Synthetic small endpoint smoke; no logits or broad quality claim.",
    }
    try:
        for n in [int(v) for v in args.lengths.split(",")]:
            prefix = " navigation" * n
            cold_start()
            generate(prefix, 1)
            restored = []
            for suffix in [" map", " compass"]:
                restored.append(generate(prefix + suffix, 4))
            for suffix, hit in zip([" map", " compass"], restored):
                cold_start()
                cold = generate(prefix + suffix, 4)
                cold_hit = (cold["usage"].get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                )
                if cold_hit:
                    raise AssertionError(
                        f"cold reference unexpectedly reused {cold_hit} tokens"
                    )
                cached = (hit["usage"].get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                )
                result = {
                    "consumed_endpoint": n,
                    "suffix": suffix,
                    "cached_tokens": cached,
                    "restored": hit,
                    "cold": cold,
                    "output_equal": hit["text"] == cold["text"],
                    "restored_sha256": hashlib.sha256(hit["text"].encode()).hexdigest(),
                    "cold_sha256": hashlib.sha256(cold["text"].encode()).hexdigest(),
                }
                report["cases"].append(result)
                if cached < n or not result["output_equal"]:
                    raise AssertionError(f"exact continuation failed at {n}: {result}")
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(report["cases"]),
                "error": report.get("error"),
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
