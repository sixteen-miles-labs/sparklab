"""Fused manifold-constrained Hyper-Connections (mHC) stream mixing for DeepSeek-V4.

The residual stream is ``hc_mult`` parallel copies. Per block, ``hc_pre`` collapses
them to one stream (``y[d] = sum_h pre[h]*x[h,d]``) and ``hc_post`` re-expands the
sublayer output back to ``hc_mult`` streams (``y[h,d] = post[h]*a[d] +
sum_h' comb[h,h']*res[h',d]``). In eager torch these are broadcast (``[hc,hc,dim]``)
+ reduce chains that dominate the decode "tail"; here each is a single launch with
the tiny ``hc x hc`` mix in registers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _hc_pre_combine_kernel(
    x_ptr, pre_ptr, y_ptr, M, D,
    stride_xm, stride_xh, stride_pm, stride_ym,
    HC: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for h in tl.static_range(HC):
        xh = tl.load(x_ptr + m * stride_xm + h * stride_xh + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(pre_ptr + m * stride_pm + h).to(tl.float32)
        acc += p * xh
    tl.store(y_ptr + m * stride_ym + offs, acc.to(OUT), mask=mask)


@triton.jit
def _hc_post_combine_kernel(
    a_ptr, res_ptr, post_ptr, comb_ptr, y_ptr, M, D,
    stride_am, stride_rm, stride_rh, stride_pm, stride_cm, stride_ym, stride_yh,
    HC: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    a = tl.load(a_ptr + m * stride_am + offs, mask=mask, other=0.0).to(tl.float32)
    # y[q,d] = post[q]*a[d] + sum_p comb[p,q]*res[p,d]  (reduction over comb's first axis)
    for q in tl.static_range(HC):
        acc = tl.load(post_ptr + m * stride_pm + q).to(tl.float32) * a
        for p in tl.static_range(HC):
            c = tl.load(comb_ptr + m * stride_cm + p * HC + q).to(tl.float32)
            r = tl.load(res_ptr + m * stride_rm + p * stride_rh + offs, mask=mask, other=0.0).to(tl.float32)
            acc += c * r
        tl.store(y_ptr + m * stride_ym + q * stride_yh + offs, acc.to(OUT), mask=mask)


def hc_pre_combine(x: torch.Tensor, pre: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """``y[m,d] = sum_h pre[m,h] * x[m,h,d]``. ``x``: [M,HC,D] (fp32), ``pre``: [M,HC]."""
    M, HC, D = x.shape
    x = x.contiguous()
    pre = pre.contiguous()
    y = torch.empty((M, D), dtype=out_dtype, device=x.device)
    BLOCK = 1024
    grid = (M, triton.cdiv(D, BLOCK))
    _hc_pre_combine_kernel[grid](
        x, pre, y, M, D, x.stride(0), x.stride(1), pre.stride(0), y.stride(0),
        HC=HC, BLOCK=BLOCK, OUT=_TL[out_dtype], num_warps=4,
    )
    return y


def hc_post_combine(a: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """``y[m,h,d] = post[m,h]*a[m,d] + sum_h' comb[m,h,h']*residual[m,h',d]``.

    ``a``: [M,D], ``residual``: [M,HC,D], ``post``: [M,HC], ``comb``: [M,HC,HC]."""
    M, HC, D = residual.shape
    out_dtype = a.dtype
    a = a.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    comb = comb.contiguous()
    y = torch.empty((M, HC, D), dtype=out_dtype, device=a.device)
    BLOCK = 1024
    grid = (M, triton.cdiv(D, BLOCK))
    _hc_post_combine_kernel[grid](
        a, residual, post, comb, y, M, D,
        a.stride(0), residual.stride(0), residual.stride(1), post.stride(0), comb.stride(0),
        y.stride(0), y.stride(1),
        HC=HC, BLOCK=BLOCK, OUT=_TL[out_dtype], num_warps=4,
    )
    return y


__all__ = ["hc_pre_combine", "hc_post_combine"]
