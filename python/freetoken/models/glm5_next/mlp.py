"""Checkpoint-faithful clamped SwiGLU layers for GLM-5.3-Flash."""

from __future__ import annotations

import torch
from freetoken.kernel.triton.fp8_block_linear import Fp8BlockLinear
from freetoken.layers import BaseOP, LinearReplicated


class Glm5Fp8BlockLinear(Fp8BlockLinear):
    """GLM's block-FP8 linear keeps the checkpoint's fp32 inverse scales."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        super().__init__(in_features, out_features, has_bias)
        self.weight_scale_inv = torch.empty(
            out_features // 128, in_features // 128, dtype=torch.float32
        )


def _linear(quantized: bool, in_features: int, out_features: int) -> BaseOP:
    if quantized:
        return Glm5Fp8BlockLinear(in_features, out_features, has_bias=False)
    return LinearReplicated(in_features, out_features, has_bias=False)


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    if gate.is_cuda:
        from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu

        return fused_swiglu(gate, up, limit, gate.dtype)
    # CPU reference/testing path; compute the nonlinear epilogue in fp32 like HF.
    gate_f = gate.float().clamp(max=limit)
    up_f = up.float().clamp(min=-limit, max=limit)
    return (torch.nn.functional.silu(gate_f) * up_f).to(gate.dtype)


class Glm5NextMLP(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int, limit: float, *, quantized: bool):
        self.gate_proj = _linear(quantized, hidden_size, intermediate_size)
        self.up_proj = _linear(quantized, hidden_size, intermediate_size)
        self.down_proj = _linear(quantized, intermediate_size, hidden_size)
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        return self.down_proj.forward(clamped_swiglu(gate, up, self.limit))


__all__ = ["Glm5Fp8BlockLinear", "Glm5NextMLP", "clamped_swiglu"]
