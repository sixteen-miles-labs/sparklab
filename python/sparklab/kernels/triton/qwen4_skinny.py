"""Small-M BF16 linear kernel for Qwen3.8 on GB10 (SM121)."""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bf16_skinny_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def bf16_skinny_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute a contiguous bias-free BF16 linear for one to three rows."""
    lead = x.shape[:-1]
    k = x.shape[-1]
    m = x.numel() // k
    n = weight.shape[0]
    if not (1 <= m <= 3):
        raise ValueError(f"Qwen4 skinny linear requires 1 <= M <= 3, got {m}")
    x2 = x.reshape(m, k).contiguous()
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_n = 2 if m == 1 else 1
    _bf16_skinny_kernel[(triton.cdiv(n, block_n),)](
        x2,
        weight,
        out,
        M=m,
        N=n,
        K=k,
        stride_xm=x2.stride(0),
        stride_wn=weight.stride(0),
        BLOCK_M=triton.next_power_of_2(m),
        BLOCK_N=block_n,
        BLOCK_K=min(triton.next_power_of_2(k), 4096),
        num_warps=4,
    )
    return out.reshape(*lead, n)


# Native TP=1 Qwen3.8 shapes that beat F.linear in a controlled GB10 sweep.
# Small projections (notably 320x10240 and 640x2560) deliberately stay on
# cuBLAS because launch/under-occupancy costs erase their apparent specialization.
_SM121_PLANS = {
    (13312, 2560),  # QSA fused q/gate/k/v
    (16480, 2560),  # GDN fused q/k/v/z/b/a
    (2560, 6144),   # QSA/GDN output projection
    (248320, 2560),  # full-vocabulary LM head
}


@functools.cache
def _is_sm121(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (12, 1)


def qwen4_skinny_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Use the measured SM121 kernel when eligible, otherwise ``F.linear``."""
    k = x.shape[-1]
    m = x.numel() // k
    eligible = (
        os.getenv("SPARKLAB_DISABLE_QWEN4_SKINNY_GEMM", "0").lower()
        not in {"1", "true", "yes"}
        and bias is None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and 1 <= m <= 3
        and tuple(weight.shape) in _SM121_PLANS
        and _is_sm121(x.device.index or 0)
    )
    return bf16_skinny_linear(x, weight) if eligible else F.linear(x, weight, bias)


__all__ = ["bf16_skinny_linear", "qwen4_skinny_linear"]
