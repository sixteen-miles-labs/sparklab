"""Fused mHC split + Sinkhorn normalization (DeepSeek-V4 Hyper-Connections).

Per token the reference computes, from a ``[(2+hc)*hc]`` mix vector, three pieces:
``pre[hc]`` (sigmoid gate), ``post[hc]`` (2*sigmoid), and ``comb[hc,hc]`` (a softmax
then ``sinkhorn_iters`` of alternating row/col normalization -> doubly stochastic).
In torch that is ~40 tiny reductions per call x86 calls/token = thousands of kernel
launches. This collapses each call into a single launch (one program per token,
the ``hc x hc`` matrix lives in registers). Matches ``ops.hc_split_sinkhorn``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _hc_sinkhorn_kernel(
    mixes_ptr, scale_ptr, base_ptr,
    pre_ptr, post_ptr, comb_ptr,
    n,
    stride_mn, stride_pn, stride_pon, stride_cn,
    HC: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n:
        return
    h = tl.arange(0, HC)
    m = mixes_ptr + row * stride_mn
    sc0 = tl.load(scale_ptr + 0)
    sc1 = tl.load(scale_ptr + 1)
    sc2 = tl.load(scale_ptr + 2)

    pre = tl.sigmoid(tl.load(m + h) * sc0 + tl.load(base_ptr + h)) + EPS
    tl.store(pre_ptr + row * stride_pn + h, pre)

    post = 2.0 * tl.sigmoid(tl.load(m + HC + h) * sc1 + tl.load(base_ptr + HC + h))
    tl.store(post_ptr + row * stride_pon + h, post)

    idx = h[:, None] * HC + h[None, :]  # [HC, HC] row-major within the comb block
    c = tl.load(m + 2 * HC + idx) * sc2 + tl.load(base_ptr + 2 * HC + idx)
    # softmax over the last axis (dim=-1)
    c = c - tl.max(c, axis=1)[:, None]
    c = tl.exp(c)
    c = c / tl.sum(c, axis=1)[:, None]
    c = c + EPS
    # initial column normalization (sum over rows = axis 0)
    c = c / (tl.sum(c, axis=0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        c = c / (tl.sum(c, axis=1)[:, None] + EPS)
        c = c / (tl.sum(c, axis=0)[None, :] + EPS)
    tl.store(comb_ptr + row * stride_cn + idx, c)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """Triton drop-in for :func:`sparklab.models.deepseek_v4.ops.hc_split_sinkhorn`.

    ``mixes`` ``[n, (2+hc)*hc]`` -> ``pre[n,hc]``, ``post[n,hc]``, ``comb[n,hc,hc]`` (fp32).
    """
    assert hc_mult & (hc_mult - 1) == 0, "hc_mult must be a power of two"
    mixes = mixes.contiguous().float()
    n = mixes.shape[0]
    dev = mixes.device
    pre = torch.empty(n, hc_mult, device=dev, dtype=torch.float32)
    post = torch.empty(n, hc_mult, device=dev, dtype=torch.float32)
    comb = torch.empty(n, hc_mult, hc_mult, device=dev, dtype=torch.float32)
    _hc_sinkhorn_kernel[(n,)](
        mixes, hc_scale.float().contiguous(), hc_base.float().contiguous(),
        pre, post, comb,
        n, mixes.stride(0), pre.stride(0), post.stride(0), comb.stride(0),
        HC=hc_mult, ITERS=sinkhorn_iters, EPS=eps, num_warps=1,
    )
    return pre, post, comb


@triton.jit
def _hc_sinkhorn_pre_combine_kernel(
    mixes_ptr, x_ptr, scale_ptr, base_ptr,
    post_ptr, comb_ptr, collapsed_ptr,
    n, D,
    stride_mn, stride_xn, stride_xh, stride_pon, stride_cn, stride_yn,
    HC: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    """Compute Sinkhorn outputs and stream collapse in one heterogeneous grid."""
    row = tl.program_id(0)
    part = tl.program_id(1)
    if row >= n:
        return
    h = tl.arange(0, HC)
    m = mixes_ptr + row * stride_mn

    if part == 0:
        sc1 = tl.load(scale_ptr + 1)
        sc2 = tl.load(scale_ptr + 2)
        post = 2.0 * tl.sigmoid(
            tl.load(m + HC + h) * sc1 + tl.load(base_ptr + HC + h)
        )
        tl.store(post_ptr + row * stride_pon + h, post)

        idx = h[:, None] * HC + h[None, :]
        c = (
            tl.load(m + 2 * HC + idx) * sc2
            + tl.load(base_ptr + 2 * HC + idx)
        )
        c = c - tl.max(c, axis=1)[:, None]
        c = tl.exp(c)
        c = c / tl.sum(c, axis=1)[:, None]
        c = c + EPS
        c = c / (tl.sum(c, axis=0)[None, :] + EPS)
        for _ in range(ITERS - 1):
            c = c / (tl.sum(c, axis=1)[:, None] + EPS)
            c = c / (tl.sum(c, axis=0)[None, :] + EPS)
        tl.store(comb_ptr + row * stride_cn + idx, c)
    else:
        offs = (part - 1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        sc0 = tl.load(scale_ptr + 0)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for p in tl.static_range(HC):
            pre_p = tl.sigmoid(
                tl.load(m + p) * sc0 + tl.load(base_ptr + p)
            ) + EPS
            xp = tl.load(
                x_ptr + row * stride_xn + p * stride_xh + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += pre_p * xp
        tl.store(
            collapsed_ptr + row * stride_yn + offs,
            acc.to(OUT),
            mask=mask,
        )


def hc_sinkhorn_pre_combine(
    mixes,
    streams,
    hc_scale,
    hc_base,
    hc_mult,
    sinkhorn_iters,
    eps,
):
    """Fused ``hc_split_sinkhorn`` plus mHC input-stream collapse.

    ``pre`` is an intermediate used only for the collapse, so each stream block
    evaluates its four gates directly while the first program in the same launch
    computes ``post`` and ``comb``.  This removes one launch and the temporary
    ``pre`` tensor without changing either reduction order.
    """
    assert hc_mult & (hc_mult - 1) == 0, "hc_mult must be a power of two"
    mixes = mixes.contiguous().float()
    streams = streams.contiguous()
    n, stream_mult, D = streams.shape
    assert mixes.shape[0] == n and stream_mult == hc_mult
    out_dtype = streams.dtype
    if out_dtype not in _TL:
        raise ValueError(f"unsupported mHC stream dtype: {out_dtype}")
    dev = mixes.device
    post = torch.empty(n, hc_mult, device=dev, dtype=torch.float32)
    comb = torch.empty(n, hc_mult, hc_mult, device=dev, dtype=torch.float32)
    collapsed = torch.empty(n, D, device=dev, dtype=out_dtype)
    BLOCK = 1024
    _hc_sinkhorn_pre_combine_kernel[
        (n, 1 + triton.cdiv(D, BLOCK))
    ](
        mixes,
        streams,
        hc_scale.float().contiguous(),
        hc_base.float().contiguous(),
        post,
        comb,
        collapsed,
        n,
        D,
        mixes.stride(0),
        streams.stride(0),
        streams.stride(1),
        post.stride(0),
        comb.stride(0),
        collapsed.stride(0),
        HC=hc_mult,
        ITERS=sinkhorn_iters,
        EPS=eps,
        BLOCK=BLOCK,
        OUT=_TL[out_dtype],
        num_warps=4,
    )
    return post, comb, collapsed


__all__ = ["hc_sinkhorn_pre_combine", "hc_split_sinkhorn"]
