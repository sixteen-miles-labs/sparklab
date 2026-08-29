from __future__ import annotations

import torch

from .base import BaseMoeBackend


class OffloadMoeBackend(BaseMoeBackend):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise RuntimeError("Offload MoE is handled by OffloadMoELayer")
