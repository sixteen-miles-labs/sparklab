from __future__ import annotations

import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from sparklab.kernels.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        from flashinfer import silu_and_mul
    else:
        from sparklab.kernels.triton.activation import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from sparklab.kernels.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        from flashinfer import gelu_and_mul
    else:
        from sparklab.kernels.triton.activation import gelu_and_mul

    return gelu_and_mul(x, out=out)


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """tanh-approximate GELU gate (`gelu_pytorch_tanh`) followed by elementwise mul."""
    from sparklab.kernels.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        from flashinfer import gelu_tanh_and_mul
    else:
        from sparklab.kernels.triton.activation import gelu_tanh_and_mul

    return gelu_tanh_and_mul(x, out=out)


def swigluoai_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
):
    """SwiGLU-OAI (gpt-oss / MiniMax-M3 ``swigluoai``) over UNINTERLEAVED halves
    (gate ``x[..., :d]``, up ``x[..., d:]``): ``clamp(gate, max=limit) *
    sigmoid(alpha * gate) * (clamp(up, +-limit) + 1)``. Always the in-repo Triton
    kernel (flashinfer ships no clamped-swiglu *_and_mul)."""
    from sparklab.kernels.triton.activation import swigluoai_and_mul

    return swigluoai_and_mul(x, out=out, alpha=alpha, limit=limit)


def situ_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    beta: float = 4.0,
    linear_beta: float = 25.0,
):
    """Kimi K3 SiTU-GLU over uninterleaved gate/up halves."""
    gate, up = x.float().chunk(2, dim=-1)
    value = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    value = value * (linear_beta * torch.tanh(up / linear_beta))
    value = value.to(x.dtype)
    if out is None:
        return value
    out.copy_(value)
    return out


__all__ = [
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "swigluoai_and_mul",
    "situ_and_mul",
]
