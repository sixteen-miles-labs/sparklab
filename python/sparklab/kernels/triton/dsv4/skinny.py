"""Small-row BF16 kernels for DeepSeek-V4 decode and DSpark."""

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
    """Compute a contiguous bias-free BF16 linear for one to six rows."""
    *lead, k = x.shape
    m = x.numel() // k
    n = weight.shape[0]
    if not 1 <= m <= 6:
        raise ValueError(f"DSV4 skinny linear requires 1 <= M <= 6, got {m}")
    x2 = x.reshape(m, k).contiguous()
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    num_warps = 8 if m == 1 else 2 if m == 3 else 4
    _bf16_skinny_kernel[(n,)](
        x2,
        weight,
        out,
        M=m,
        N=n,
        K=k,
        stride_xm=x2.stride(0),
        stride_wn=weight.stride(0),
        BLOCK_M=triton.next_power_of_2(m),
        BLOCK_N=1,
        BLOCK_K=min(triton.next_power_of_2(k), 4096),
        num_warps=num_warps,
    )
    return out.reshape(*lead, n)


@functools.cache
def _is_sm121(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (12, 1)


def dsv4_head_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Use the measured SM121 vocabulary kernel when eligible."""
    k = x.shape[-1]
    m = x.numel() // k
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and tuple(weight.shape) == (129280, 4096)
        and 1 <= m <= 6
        and _is_sm121(x.device.index or 0)
        and os.getenv("SPARKLAB_DISABLE_DSV4_SMALL_M", "0").strip().lower()
        not in {"1", "true", "yes", "on"}
    ):
        return F.linear(x, weight)
    return bf16_skinny_linear(x, weight)


@triton.jit
def _markov_partial_argmax_kernel(
    base_ptr,
    markov_ptr,
    weight_ptr,
    values_ptr,
    indices_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, K)
    markov = tl.load(markov_ptr + offs_k).to(tl.float32)
    weight = tl.load(
        weight_ptr + offs_n[:, None] * K + offs_k[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    # Match torch's BF16 linear followed by its BF16 add before comparing.
    projected = tl.sum(weight * markov[None, :], axis=1).to(tl.bfloat16)
    base = tl.load(base_ptr + offs_n, mask=n_mask, other=-float("inf"))
    score = (projected + base).to(tl.bfloat16).to(tl.float32)
    local = tl.argmax(score, axis=0)
    tl.store(values_ptr + pid, tl.max(score, axis=0))
    tl.store(indices_ptr + pid, pid * BLOCK_N + local)


def dsv4_markov_argmax(
    base_logits: torch.Tensor,
    markov: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Fuse the DSpark Markov projection, base-logit add, and greedy argmax."""
    n, k = weight.shape
    eligible = (
        os.getenv("SPARKLAB_DISABLE_DSV4_MARKOV_FUSION", "0").strip().lower()
        not in {"1", "true", "yes", "on"}
        and base_logits.is_cuda
        and markov.is_cuda
        and weight.is_cuda
        and base_logits.dtype == torch.bfloat16
        and markov.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and tuple(base_logits.shape) == (1, n)
        and tuple(markov.shape) == (1, k)
        and k == 256
        and _is_sm121(weight.device.index or 0)
    )
    if not eligible:
        return torch.argmax(base_logits + F.linear(markov, weight), dim=-1)
    block_n = 8
    blocks = triton.cdiv(n, block_n)
    values = torch.empty(blocks, dtype=torch.float32, device=weight.device)
    indices = torch.empty(blocks, dtype=torch.int64, device=weight.device)
    _markov_partial_argmax_kernel[(blocks,)](
        base_logits.reshape(-1),
        markov.reshape(-1),
        weight,
        values,
        indices,
        N=n,
        K=k,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return indices[torch.argmax(values)].reshape(1)


__all__ = ["bf16_skinny_linear", "dsv4_head_linear", "dsv4_markov_argmax"]
