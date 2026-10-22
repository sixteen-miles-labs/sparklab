"""Bounded greedy-logit evidence for instrumented correctness runs, not timing.

Only the top entries and a digest leave each synchronous observation. These
diagnostics neither alter target selection nor qualify distributional quality.
"""

import hashlib

import torch


class GreedyLogitTrace:
    def __init__(self, limit=64, top_k=8):
        if not 1 <= limit <= 4096 or not 2 <= top_k <= 32:
            raise ValueError("invalid logit diagnostic bound")
        self.limit = limit
        self.top_k = top_k
        self.tokens = []

    def observe(self, logits, chosen, *, phase):
        ids = chosen.detach().reshape(-1).cpu().tolist()
        if logits.ndim != 2 or logits.shape[0] < len(ids):
            raise ValueError("logit rows do not cover emitted tokens")
        if len(self.tokens) + len(ids) > self.limit:
            raise ValueError("logit diagnostic token bound exceeded")
        rows = logits[: len(ids)].detach().float().cpu().contiguous()
        records = []
        for row, token in zip(rows, ids):
            if not torch.isfinite(row).all():
                raise ValueError("nonfinite target logits")
            if type(token) is not int or not 0 <= token < row.numel():
                raise ValueError("chosen token outside vocabulary")
            values, indices = row.topk(min(self.top_k, row.numel()))
            maximum = float(values[0])
            argmax = int(row.argmax())
            if token != argmax:
                raise ValueError("greedy target selection disagrees with logits")
            records.append(
                {
                    "token": token,
                    "phase": phase,
                    "argmax": argmax,
                    "maximum_ties": int((row == values[0]).sum()),
                    "top_margin": maximum - float(values[1])
                    if len(values) > 1
                    else None,
                    "top_logits": [
                        [int(index), float(value)]
                        for index, value in zip(indices, values)
                    ],
                    "logits_sha256": hashlib.sha256(
                        row.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
            )
        self.tokens.extend(records)

    def validate_output(self, ids):
        if ids != [row["token"] for row in self.tokens]:
            raise ValueError("diagnostic emissions do not match completed output")


def first_divergence(reference, candidate):
    """Compare token-aligned evidence only up to the first trajectory change."""
    for position, (left, right) in enumerate(zip(reference.tokens, candidate.tokens)):
        if left["token"] != right["token"]:
            return {
                "position": position,
                "equal_preceding_output_tokens": position,
                "reference": left,
                "candidate": right,
                "scope": "First divergence after the same output prefix; state equivalence is a separate gate.",
            }
    if len(reference.tokens) != len(candidate.tokens):
        return {
            "position": min(len(reference.tokens), len(candidate.tokens)),
            "length_mismatch": True,
        }
    return None
