"""Fused RMSNorm for DeepSeek-V4 decode.

The eager ``RMSNorm`` (``x.float(); x*rsqrt(mean(x^2)+eps); (w*x).to(dtype)``) is a
chain of ~5 elementwise/reduction launches, run ~5x per layer (q/kv/attn/ffn/
compressor norms). At bs=1 these are latency-bound; collapsing each into one kernel
(row in registers, single reduction) removes most of the decode "tail" of tiny ops.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr, M, D, eps,
    stride_xm, stride_om,
    BLOCK_D: tl.constexpr, HAS_W: tl.constexpr, compute_type: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    y = x * tl.rsqrt(var + eps)
    if HAS_W:
        y = y * tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_om + offs, y.to(compute_type), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """``y = x * rsqrt(mean(x^2, -1) + eps) * weight`` (weight optional), fused."""
    D = x.shape[-1]
    x2d = x.reshape(-1, D)
    M = x2d.shape[0]
    out_dtype = x.dtype if x.dtype in _TL else torch.bfloat16
    out = torch.empty_like(x2d, dtype=out_dtype)
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    _rmsnorm_kernel[(M,)](
        x2d, weight, out, M, D, eps,
        x2d.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D, HAS_W=weight is not None,
        compute_type=_TL[out_dtype], num_warps=num_warps,
    )
    return out.reshape(x.shape)


__all__ = ["rms_norm"]
