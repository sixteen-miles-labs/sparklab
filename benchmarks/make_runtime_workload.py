#!/usr/bin/env python3
"""Create exact-token text inputs using the local serving tokenizer (no downloads)."""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def exact_text(tokenizer, length, stem=" navigation"):
    # Stable public synthetic text; validate the actual text after encoding rather
    # than assuming repeated words or decode(token_ids) preserve token boundaries.
    text = stem * length
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == length:
        return text
    for end in range(
        max(1, len(text) - 4 * abs(len(ids) - length) - 32), len(text) + 1
    ):
        candidate = text[:end]
        if len(tokenizer.encode(candidate, add_special_tokens=False)) == length:
            return candidate
    raise ValueError(
        "could not construct exact input length with this tokenizer; choose a different stem"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--long-tokens", type=int, default=8192)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--arrival-gap", type=float, default=1)
    args = p.parse_args()
    if (
        args.long_tokens <= 0
        or args.output_tokens <= 0
        or not 0 <= args.arrival_gap <= 86400
    ):
        p.error("token budgets must be positive and arrival gap in [0,86400]")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cases = []
    for name, at, length, budget in [
        ("active", 0, 128, args.output_tokens),
        ("long", args.arrival_gap, args.long_tokens, args.output_tokens),
        ("branch", 2 * args.arrival_gap, 512, args.output_tokens),
    ]:
        cases.append(
            {
                "id": name,
                "at": at,
                "expected_prompt_tokens": length,
                "request": {
                    "prompt": exact_text(tokenizer, length),
                    "max_tokens": budget,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "cache_condition": "cold first request, repeated synthetic prefix thereafter",
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
