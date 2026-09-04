#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure matched OpenAI streaming throughput at several concurrency levels."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


PROMPT = (
    "Write continuous narrative prose about the history of maritime navigation. "
    "Do not use headings or lists."
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


@dataclass
class RequestResult:
    prompt_tokens: int
    completion_tokens: int
    ttft_seconds: float
    elapsed_seconds: float
    output_sha256: str


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class OpenAIClient:
    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=self.timeout)

    def get_json(self, path: str) -> dict:
        with self._request("GET", path) as response:
            return json.load(response)

    def generate(self, prompt: str, output_tokens: int) -> RequestResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        first_token: float | None = None
        usage: dict = {}
        output: list[str] = []
        with self._request("POST", "/v1/chat/completions", payload) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                pieces = [
                    delta[key]
                    for key in ("reasoning_content", "content")
                    if isinstance(delta.get(key), str) and delta[key]
                ]
                if pieces:
                    first_token = first_token or time.perf_counter()
                    output.extend(pieces)
        ended = time.perf_counter()
        return RequestResult(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            ttft_seconds=(first_token or ended) - started,
            elapsed_seconds=ended - started,
            output_sha256=hashlib.sha256("".join(output).encode()).hexdigest(),
        )


def _run_trial(
    client: OpenAIClient,
    concurrency: int,
    trial: int,
    output_tokens: int,
) -> dict:
    gate = threading.Event()

    def request(index: int) -> RequestResult:
        gate.wait()
        prompt = f"Benchmark request {index}, trial {trial}, concurrency {concurrency}.\n{PROMPT}"
        return client.generate(prompt, output_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request, index) for index in range(concurrency)]
        started = time.perf_counter()
        gate.set()
        results = [future.result() for future in futures]
        ended = time.perf_counter()

    completion_tokens = sum(result.completion_tokens for result in results)
    wall_seconds = ended - started
    return {
        "aggregate_output_tokens_per_second": completion_tokens / wall_seconds,
        "wall_seconds": wall_seconds,
        "prompt_tokens": sum(result.prompt_tokens for result in results),
        "completion_tokens": completion_tokens,
        "ttft_p50_seconds": _percentile(
            [result.ttft_seconds for result in results], 0.50
        ),
        "ttft_p95_seconds": _percentile(
            [result.ttft_seconds for result in results], 0.95
        ),
        "e2e_p95_seconds": _percentile(
            [result.elapsed_seconds for result in results], 0.95
        ),
        "requests": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--trials", type=_positive_int, default=3)
    parser.add_argument("--output-tokens", type=_positive_int, default=128)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    concurrency_levels = [
        _positive_int(value.strip()) for value in args.concurrency.split(",")
    ]
    client = OpenAIClient(args.base_url, args.model, args.timeout)
    client.get_json("/health")
    client.generate("Warm up the measured generation path.", 32)

    report: dict = {
        "schema_version": "1.0",
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parameters": {
            "base_url": args.base_url,
            "model": args.model,
            "concurrency": args.concurrency,
            "trials": args.trials,
            "output_tokens": args.output_tokens,
        },
        "levels": {},
    }
    for concurrency in concurrency_levels:
        trials = [
            _run_trial(client, concurrency, trial, args.output_tokens)
            for trial in range(args.trials)
        ]
        throughput = [trial["aggregate_output_tokens_per_second"] for trial in trials]
        ttft = [trial["ttft_p95_seconds"] for trial in trials]
        e2e = [trial["e2e_p95_seconds"] for trial in trials]
        report["levels"][str(concurrency)] = {
            "aggregate_output_tokens_per_second_median": statistics.median(throughput),
            "ttft_p95_seconds_median": statistics.median(ttft),
            "e2e_p95_seconds_median": statistics.median(e2e),
            "trials": trials,
        }
        print(
            f"C{concurrency}: {statistics.median(throughput):.2f} aggregate tok/s, "
            f"TTFT p95 {statistics.median(ttft):.3f}s, "
            f"E2E p95 {statistics.median(e2e):.3f}s",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
