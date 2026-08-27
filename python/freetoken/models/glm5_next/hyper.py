"""Manifold-constrained Hyper-Connections (mHC) for GLM-5.3-Flash."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Upstream's strict-fp32, unweighted RMS normalization."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


class Glm5NextHyperConnection(BaseOP):
    def __init__(self, hidden_size: int, mult: int, eps: float, sinkhorn_iters: int, norm_eps: float):
        self.mult = mult
        self.eps = eps
        self.sinkhorn_iters = sinkhorn_iters
        self.norm_eps = norm_eps
        mix = (2 + mult) * mult
        self.fn = torch.empty(mix, mult * hidden_size)
        # The release keeps the low-rank mapping in BF16 but its additive bases
        # and three learned output scales in FP32.
        self.base = torch.empty(mix, dtype=torch.float32)
        self.scale = torch.empty(3, dtype=torch.float32)

    def forward(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``post``, doubly-stochastic ``comb``, and collapsed input.

        FreeToken packs requests on the token axis, so the upstream ``[B,S,M,H]``
        shape is represented as ``[T,M,H]`` here; the mapping is otherwise exact.
        """
        mult = self.mult
        flat = streams.flatten(start_dim=1).float()
        flat = unweighted_rms_norm(flat, self.norm_eps)
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(
            [mult, mult, mult * mult], dim=-1
        )
        pre_b, post_b, comb_b = self.base.float().split([mult, mult, mult * mult])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.eps
        post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(-1, mult, mult) * comb_scale + comb_b.view(mult, mult)
        comb = torch.softmax(comb_logits, dim=-1) + self.eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(streams.dtype)
        return post, comb, collapsed

    @staticmethod
    def expand(
        output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        dtype = residual.dtype
        placed = post.to(dtype).unsqueeze(-1) * output.unsqueeze(1)
        mixed = torch.matmul(comb.to(dtype).transpose(-1, -2), residual)
        return placed + mixed


__all__ = ["Glm5NextHyperConnection", "unweighted_rms_norm"]
