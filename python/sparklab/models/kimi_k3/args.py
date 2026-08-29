from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KimiK3Args:
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel: int
    kda_full_rank_gate: bool
    kda_gate_lower_bound: float | None
    attn_res_block_size: int
    routed_expert_hidden_size: int
    latent_moe_use_norm: bool
    situ_beta: float
    situ_linear_beta: float | None
    mla_output_gate: bool


__all__ = ["KimiK3Args"]
