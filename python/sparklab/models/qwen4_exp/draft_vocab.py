"""Optional draft-only vocabulary restriction; target logits stay full-sized.

Independent implementation of reduced-vocabulary speculative drafting, evaluated
after reviewing MiaAI-Lab's single-Spark recipe. No third-party patch code is used.
"""
from __future__ import annotations

import torch


def draft_token_ids(vocab_size: int, budget: int, required_ids: list[int]) -> list[int]:
    """Keep a low-token-ID prefix plus every required control/added token.

    This is a deterministic, corpus-free candidate set, not a frequency-trained
    multilingual vocabulary. Missing tokens can reduce draft acceptance.
    """
    if not 0 < budget <= vocab_size:
        raise ValueError("draft vocabulary budget must be within the target vocabulary")
    required = set(required_ids)
    if any(token < 0 or token >= vocab_size for token in required):
        raise ValueError("required draft token is outside the target vocabulary")
    if len(required) > budget:
        raise ValueError("draft vocabulary budget cannot hold required tokens")
    selected = set(range(budget - len(required))) | required
    token = budget - len(required)
    while len(selected) < budget:
        selected.add(token)
        token += 1
    return sorted(selected)


class DraftVocabulary:
    """Private resident head copy for batch-one greedy draft sampling only."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, ids: list[int]):
        self.ids = torch.tensor(ids, device=weight.device, dtype=torch.long)
        self.weight = weight.index_select(0, self.ids).contiguous()
        self.bias = bias.index_select(0, self.ids).contiguous() if bias is not None else None

    def select(self, hidden: torch.Tensor) -> torch.Tensor:
        from sparklab.layers.linear import _linear_forward

        local = _linear_forward(hidden, self.weight, self.bias).argmax(-1)
        return self.ids[local]
