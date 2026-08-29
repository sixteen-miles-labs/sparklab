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


__all__ = ["hc_split_sinkhorn"]
