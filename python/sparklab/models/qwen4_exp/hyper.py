from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from sparklab.layers import BaseOP, LinearReplicated


class GroupedPlusOneRMSNorm(BaseOP):
    """Qwen4 grouped (1+w) RMSNorm over independent H-wide streams."""

    def __init__(self, width: int, group_size: int, eps: float):
        if width % group_size:
            raise ValueError(f"grouped RMS width {width} is not divisible by {group_size}")
        self.weight = torch.empty(width)
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from sparklab.kernels.triton.qwen4 import grouped_plus_one_rmsnorm

            return grouped_plus_one_rmsnorm(
                x, self.weight, self.group_size, self.eps
            )
        shape = x.shape
        grouped = x.float().reshape(*shape[:-1], -1, self.group_size)
        inv = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        # Qwen4 multiplies the normalized fp32 value by (1 + weight) in fp32,
        # then performs exactly one cast back to the activation dtype. Casting
        # before the affine would introduce an extra BF16 rounding at every HC
        # and PLE norm.
        out = (grouped * inv).reshape(shape)
        return (out * (1.0 + self.weight.float())).to(x.dtype)


class Qwen4GatedResidual(BaseOP):
    """Reference-exact Hyper-Connection input mixer and residual injector."""

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        hc_lowrank: int,
        eps: float,
        *,
        use_combine: bool = True,
    ):
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.width = hidden_size * hc_count
        self.hc_norm = GroupedPlusOneRMSNorm(self.width, hidden_size, eps)
        self.input_mix_weight_down = LinearReplicated(
            self.width, hc_lowrank, has_bias=False
        )
        self.input_mix_weight_up = LinearReplicated(
            hc_lowrank, self.width, has_bias=False
        )
        self.block_inject_weight = (
            LinearReplicated(self.width, hc_count, has_bias=False) if use_combine else None
        )
        self._fused = os.getenv(
            "SPARKLAB_DISABLE_QWEN4_HYPER_FUSION", "0"
        ).lower() not in {"1", "true", "yes"}

    def forward(self, hyper_input: torch.Tensor):
        if hyper_input.shape[-1] != self.width:
            raise ValueError(
                f"Qwen4 Hyper-Connection expected width {self.width}, "
                f"got {hyper_input.shape[-1]}"
            )
        normed = self.hc_norm.forward(hyper_input)
        down = self.input_mix_weight_down.forward(normed)
        if self._fused and down.is_cuda and down.dtype == torch.bfloat16:
            from sparklab.kernels.triton.qwen4 import hyper_mix, scaled_silu

            mix = scaled_silu(down.contiguous(), self.hc_count)
            mixed = hyper_mix(
                self.input_mix_weight_up.forward(mix).contiguous(),
                normed.contiguous(),
                groups=self.hc_count,
                group_size=self.hidden_size,
            )
        else:
            mix = F.silu(down / self.hc_count)
            mix = torch.sigmoid(self.input_mix_weight_up.forward(mix))
            mix = mix.view(*mix.shape[:-1], self.hc_count, self.hidden_size)
            mixed = (
                mix
                * normed.view(
                    *normed.shape[:-1], self.hc_count, self.hidden_size
                )
            ).mean(-2)
        if self.block_inject_weight is None:
            return mixed
        injection_logits = self.block_inject_weight.forward(normed)
        if (
            self._fused
            and injection_logits.is_cuda
            and injection_logits.dtype == torch.bfloat16
        ):
            from sparklab.kernels.triton.qwen4 import hyper_injection_weights

            injection = hyper_injection_weights(
                injection_logits.contiguous(), self.hc_count
            )
        else:
            injection = 2 * torch.sigmoid(injection_logits / self.hc_count)
        return mixed, hyper_input, injection

    def inject(
        self,
        branch: torch.Tensor,
        hyper_input: torch.Tensor,
        weights: torch.Tensor,
    ):
        if self._fused and branch.is_cuda and branch.dtype == torch.bfloat16:
            from sparklab.kernels.triton.qwen4 import hyper_residual_inject

            return hyper_residual_inject(
                branch.contiguous(),
                hyper_input.contiguous(),
                weights.contiguous(),
                groups=self.hc_count,
            )
        update = branch.unsqueeze(-2) * weights.unsqueeze(-1)
        return hyper_input + update.flatten(-2)


__all__ = ["GroupedPlusOneRMSNorm", "Qwen4GatedResidual"]
