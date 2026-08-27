"""GLM-5.3 Kimi Delta Attention on FreeToken's recurrent FLA kernel."""

from __future__ import annotations

import torch
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearReplicated


class _DepthwiseConv1d(BaseOP):
    def __init__(self, dim: int, kernel: int):
        self.weight = torch.empty(dim, 1, kernel)


class _SigmoidGatedRMSNorm(BaseOP):
    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from freetoken.kernel.fla import rms_norm_gated

            return rms_norm_gated(
                x=x,
                weight=self.weight,
                bias=None,
                z=gate,
                eps=self.eps,
                is_rms_norm=True,
                norm_before_gate=True,
                activation="sigmoid",
            )
        xf = x.float()
        normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float() * torch.sigmoid(gate.float())).to(x.dtype)


class Glm5NextDeltaAttention(BaseOP):
    def __init__(self, config, layer_id: int):
        args = config.glm5_next_args
        assert args is not None
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = args.kda_num_heads
        self.head_dim = args.kda_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.conv_kernel_size = args.kda_conv_kernel
        self.gate_lower_bound = args.kda_gate_lower_bound

        # The public checkpoint deliberately leaves KDA projections in BF16.
        self.q_proj = LinearReplicated(self.hidden_size, self.projection_size, has_bias=False)
        self.k_proj = LinearReplicated(self.hidden_size, self.projection_size, has_bias=False)
        self.v_proj = LinearReplicated(self.hidden_size, self.projection_size, has_bias=False)
        self.q_conv1d = _DepthwiseConv1d(self.projection_size, self.conv_kernel_size)
        self.k_conv1d = _DepthwiseConv1d(self.projection_size, self.conv_kernel_size)
        self.v_conv1d = _DepthwiseConv1d(self.projection_size, self.conv_kernel_size)
        self.A_log = torch.empty(self.num_heads, dtype=torch.float32)
        self.f_a_proj = LinearReplicated(self.hidden_size, self.head_dim, has_bias=False)
        self.f_b_proj = LinearReplicated(self.head_dim, self.projection_size, has_bias=False)
        self.dt_bias = torch.empty(self.projection_size, dtype=torch.float32)
        self.b_proj = LinearReplicated(self.hidden_size, self.num_heads, has_bias=False)
        self.g_a_proj = LinearReplicated(self.hidden_size, self.head_dim, has_bias=False)
        self.g_b_proj = LinearReplicated(self.head_dim, self.projection_size, has_bias=False)
        self.o_norm = _SigmoidGatedRMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = LinearReplicated(self.projection_size, self.hidden_size, has_bias=False)

    def _conv_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.q_conv1d.weight.squeeze(1),
                self.k_conv1d.weight.squeeze(1),
                self.v_conv1d.weight.squeeze(1),
            ),
            dim=0,
        )

    def _conv(self, x: torch.Tensor, fla, pool, decode: bool) -> torch.Tensor:
        local = pool.local_index(self.layer_id)
        if decode:
            return causal_conv1d_decode(
                x, pool.conv_states[local], self._conv_weight(), fla.cache_indices
            )
        transposed = x.transpose(0, 1).contiguous()
        out = causal_conv1d_varlen(
            transposed,
            self._conv_weight(),
            pool.conv_states[local],
            fla.cu_seqlens,
            fla.cache_indices,
            fla.has_initial_state,
        )
        return out.transpose(0, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from freetoken.attention.linear import build_fla_metadata
        from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        fla = batch.fla_metadata
        if fla is None:
            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla
        if fla.track_dst is not None:
            raise RuntimeError("GLM-5.3 KDA state snapshots are not implemented")

        raw = torch.cat(
            [
                self.q_proj.forward(hidden_states),
                self.k_proj.forward(hidden_states),
                self.v_proj.forward(hidden_states),
            ],
            dim=-1,
        )
        mixed = self._conv(raw, fla, pool, batch.is_decode)
        q, k, v = mixed.split([self.projection_size] * 3, dim=-1)
        total = hidden_states.shape[0]
        q = q.view(1, total, self.num_heads, self.head_dim)
        k = k.view(1, total, self.num_heads, self.head_dim)
        v = v.view(1, total, self.num_heads, self.head_dim)
        a = self.f_b_proj.forward(self.f_a_proj.forward(hidden_states))
        beta_logits = self.b_proj.forward(hidden_states).float()

        local = pool.local_index(self.layer_id)
        if fla.fresh_state_indices is not None:
            pool.recurrent_states[local].index_fill_(0, fla.fresh_state_indices, 0.0)
        core = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a,
            dt_bias=self.dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=beta_logits,
            initial_state_source=pool.recurrent_states[local],
            initial_state_indices=fla.cache_indices,
            scale=self.head_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=fla.cu_seqlens,
            is_kda=True,
            kda_a_log_per_head=True,
            lower_bound=self.gate_lower_bound,
        )
        gate = self.g_b_proj.forward(self.g_a_proj.forward(hidden_states))
        gate = gate.view(-1, self.head_dim)
        core = self.o_norm.forward(core.reshape(-1, self.head_dim), gate)
        return self.o_proj.forward(core.reshape(total, self.projection_size))


__all__ = ["Glm5NextDeltaAttention"]
