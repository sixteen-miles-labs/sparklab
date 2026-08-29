"""bf16-weight GEMV with on-chip upcast + fp32 accumulate/output (DeepSeek-V4).

Several DSV4 decode ops need fp32 *math* on bf16 weights (the compressor wkv/wgate
gated pool, the MoE router) for numerical stability. Doing it as
``F.linear(x.float(), w.float())`` materializes an fp32 copy of the weight in HBM
every step (read bf16 + write fp32 + re-read fp32) and runs a heavier fp32 GEMM.

This kernel instead streams the bf16 weight from HBM once, upcasts to fp32 in
registers (on-chip), and accumulates in fp32 -> fp32 output. Same precision as the
reference (only the fp32 accumulation *order* differs, ~1e-6), at the HBM cost of a
plain bf16 read. Decode is M==1 (a GEMV); M>1 (prefill) falls back to F.linear.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bf16_gemv_fp32_kernel(
    x_ptr, w_ptr, out_ptr, N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    w_row = w_ptr + offs_n[:, None] * stride_wn
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        w = tl.load(w_row + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        xk = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        acc += tl.sum(w * xk[None, :], axis=1)
    tl.store(out_ptr + offs_n, acc, mask=n_mask)


def bf16_linear_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``out = x @ weight.T`` in fp32, reading ``weight`` (bf16) straight from HBM.

    ``x``: ``[..., K]`` (leading dims collapse to M). ``weight``: ``[N, K]`` bf16.
    Returns ``[..., N]`` fp32. Bit-exact to ``F.linear(x.float(), weight.float())``
    up to fp32 accumulation order; for M>1 it *is* that call (prefill)."""
    *lead, K = x.shape
    N = weight.shape[0]
    M = 1
    for d in lead:
        M *= d
    if M != 1:
        return F.linear(x.float(), weight.float())
    x1 = x.reshape(K).contiguous()
    out = torch.empty(N, dtype=torch.float32, device=x.device)
    # BN=2 -> N/2 CTAs spreads the (tiny) GEMV across SMs (occupancy-bound, not BW);
    # BLOCK_K covering all of K avoids a K-loop. Tuned on H100 (~2 TB/s at N=1024).
    BLOCK_N = 2
    BLOCK_K = min(triton.next_power_of_2(K), 4096)
    _bf16_gemv_fp32_kernel[(triton.cdiv(N, BLOCK_N),)](
        x1, weight, out, N, K,
        weight.stride(0), weight.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4,
    )
    return out.reshape(*lead, N)


__all__ = ["bf16_linear_fp32"]
