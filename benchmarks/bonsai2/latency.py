"""Matched fixed-corpus latency probe, using the portfolio's streaming metric.

Pass the saved SPEED-Bench JSONL files used by the Qwen comparison. --limit
selects an explicitly reported prefix of that corpus; this is not certification.
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_single_stream import generate


def run(args):
    records = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    warmups = [
        json.loads(line)
        for line in args.warmup.read_text().splitlines()
        if line.strip()
    ]

    def one(record):
        messages = record["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("This probe requires single-user-message corpus entries")
        result = generate(args.url, messages[0]["content"], 256, True, "bonsai2")
        result["question_id"] = record["question_id"]
        result.pop("text")
        return result

    report = {
        "label": args.label,
        "concurrency": 1,
        "output_tokens": 256,
        "temperature": 0,
        "thinking": True,
        "ignore_eos": True,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "warmup_sha256": hashlib.sha256(args.warmup.read_bytes()).hexdigest(),
        "corpus_requests": len(records),
        "measured_requests": min(args.limit, len(records)),
        "warmups": [],
        "trials": [],
    }
    for record in warmups[: args.warmups]:
        report["warmups"].append(one(record))
        print("warmup", flush=True)
    for record in records[: args.limit]:
        row = one(record)
        report["trials"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            row["question_id"],
            row["usage"],
            row["ttft_seconds"],
            row["decode_tokens_per_second"],
            flush=True,
        )
    report["metrics"] = {
        "decode_tokens_per_second_median": statistics.median(
            r["decode_tokens_per_second"] for r in report["trials"]
        ),
        "ttft_seconds_median": statistics.median(
            r["ttft_seconds"] for r in report["trials"]
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--warmup", type=Path, required=True)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.limit < 1 or a.warmups < 1:
        p.error("Counts must be positive")
    run(a)
