"""Small pure-torch KDA oracle used for reduced-shape correctness tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference KDA recurrence; inputs are ``[B,T,H,D]`` except beta ``[B,T,H]``."""
    input_dtype = q.dtype
    q = q.float()
    k = k.float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (q.shape[-1] ** -0.5)
    v = v.float()
    batch, steps, heads, dim = q.shape
    a = a.float().view(batch, steps, heads, dim)
    dt_bias = dt_bias.float().view(heads, dim)
    if A_log.numel() != dim:
        raise ValueError(f"Kimi KDA A_log must have head_dim={dim} values")
    decay = -A_log.float().exp().view(1, 1, 1, dim) * F.softplus(
        a + dt_bias.view(1, 1, heads, dim)
    )
    if lower_bound is not None:
        decay = decay.clamp_min(lower_bound)
    beta = beta.float().sigmoid()
    state = (
        torch.zeros(batch, heads, dim, dim, device=q.device)
        if initial_state is None
        else initial_state.float().clone()
    )
    outputs = []
    for t in range(steps):
        state = state * decay[:, t].exp().unsqueeze(-1)
        memory = torch.einsum("bhkv,bhk->bhv", state, k[:, t])
        delta = (v[:, t] - memory) * beta[:, t].unsqueeze(-1)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]))
    return torch.stack(outputs, dim=1).to(input_dtype), state


__all__ = ["recurrent_kda"]
