"""GLM-5.3 Kimi Delta Attention on SparkLab's recurrent FLA kernel."""

from __future__ import annotations

import torch
from sparklab.core import get_global_ctx
from sparklab.kernels.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from sparklab.layers import BaseOP, LinearReplicated


def _main_projection(quant: str, in_features: int, out_features: int) -> BaseOP:
    """Build one bandwidth-dominant KDA projection in the artifact's stored format."""
    if quant == "fp8_pertensor":
        from sparklab.kernels.triton.fp8_pertensor_linear import Fp8PerTensorLinear

        return Fp8PerTensorLinear(in_features, out_features, has_bias=False)
    return LinearReplicated(in_features, out_features, has_bias=False)


class _DepthwiseConv1d(BaseOP):
    def __init__(self, dim: int, kernel: int):
        self.weight = torch.empty(dim, 1, kernel)


class _SigmoidGatedRMSNorm(BaseOP):
    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from sparklab.kernels.fla import rms_norm_gated

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

        # Publisher artifacts leave KDA in BF16. An optimized FTW can store only the four
        # bandwidth-dominant projections as per-row FP8, freeing unified memory for the
        # routed-expert cache. Recurrent gates stay BF16 to preserve their sensitive state.
        self.q_proj = _main_projection(
            args.kda_quant, self.hidden_size, self.projection_size
        )
        self.k_proj = _main_projection(
            args.kda_quant, self.hidden_size, self.projection_size
        )
        self.v_proj = _main_projection(
            args.kda_quant, self.hidden_size, self.projection_size
        )
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
        self.o_proj = _main_projection(
            args.kda_quant, self.projection_size, self.hidden_size
        )
        # Populated after checkpoint loading.  Keeping this private prevents the
        # derived tensor from becoming part of the checkpoint state dict.
        self._packed_conv_weight: torch.Tensor | None = None

    def prepare_for_runtime(self) -> None:
        # Q/K/V use one fused causal-convolution call, so its weight is invariant
        # for the entire generation.  Packing it once removes one allocation and
        # three device copies from every KDA layer on every generated token.
        if self._packed_conv_weight is not None:
            return
        self._packed_conv_weight = torch.cat(
            (
                self.q_conv1d.weight.squeeze(1),
                self.k_conv1d.weight.squeeze(1),
                self.v_conv1d.weight.squeeze(1),
            ),
            dim=0,
        ).contiguous()

    def _conv_weight(self) -> torch.Tensor:
        if self._packed_conv_weight is not None:
            return self._packed_conv_weight
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
        # The recurrent kernel addresses features within each head contiguously.
        # Materialize the channel-major convolution output once here so the three
        # Q/K/V views can share one dense allocation.
        return out.transpose(0, 1).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from sparklab.attention.linear import build_fla_metadata
        from sparklab.kernels.fla import fused_sigmoid_gating_delta_rule_update

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
        # Verification is a decode lifecycle event but contains a contiguous
        # candidate sequence. Run it through the varlen path so convolution and
        # recurrent state advance in token order instead of treating every row
        # as an independent one-token request.
        mixed = self._conv(raw, fla, pool, not batch.uses_prefill_kernels)
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
