from __future__ import annotations

import torch

from sparklab.kernels.triton.fp8_block_linear import Fp8BlockLinear
from sparklab.layers import BaseOP, LinearColParallelMerged, LinearReplicated, LinearRowParallel


class KimiFp8BlockLinear(Fp8BlockLinear):
    """K3 ModelOpt block-FP8 linear with the checkpoint's fp32 scales."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        super().__init__(in_features, out_features, has_bias)
        self.weight_scale_inv = torch.empty(
            out_features // 128, in_features // 128, dtype=torch.float32
        )


def kimi_linear(
    quantization: str,
    in_features: int,
    out_features: int,
    *,
    force_bf16: bool = False,
) -> BaseOP:
    """Keep aligned NVIDIA FP8 projections native; use BF16 for small tails.

    ModelOpt pads the scale grid for projections such as KDA's 96-row beta
    projection.  The current block-FP8 kernel requires exact 128-row blocks, so
    those small projections are dequantized once by the weight loader.
    """
    if (
        quantization == "fp8_block"
        and not force_bf16
        and in_features % 128 == 0
        and out_features % 128 == 0
    ):
        return KimiFp8BlockLinear(in_features, out_features, has_bias=False)
    return LinearReplicated(in_features, out_features, has_bias=False)


def situ_and_mul(x: torch.Tensor, beta: float = 4.0, linear_beta: float | None = 25.0):
    input_dtype = x.dtype
    gate, up = x.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(input_dtype)


def apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Attention Residual mixture, evaluated in fp32 like the reference."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_fp32 = values.float()
    normalized = values_fp32 * torch.rsqrt(
        values_fp32.square().mean(dim=-1, keepdim=True) + eps
    )
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    probs = (normalized * score_weight).sum(dim=-1).softmax(dim=-1).unsqueeze(1)
    return torch.matmul(probs, values_fp32).squeeze(1).to(values.dtype)


class KimiSituMLP(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int, beta: float, linear_beta: float):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(
            situ_and_mul(self.gate_up_proj.forward(x), self.beta, self.linear_beta)
        )


__all__ = [
    "KimiFp8BlockLinear",
    "KimiSituMLP",
    "apply_attention_residual",
    "kimi_linear",
    "situ_and_mul",
]
