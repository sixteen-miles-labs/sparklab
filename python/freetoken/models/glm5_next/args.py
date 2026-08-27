"""Pinned GLM-5.3-Flash text-tower geometry used by the serving path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Glm5NextArgs:
    hidden_size: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    norm_eps: float
    max_position: int
    layer_types: Tuple[str, ...]
    kda_layer_ids: Tuple[int, ...]
    dsa_layer_ids: Tuple[int, ...]
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel: int
    kda_gate_lower_bound: float | None
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_always_select_tail: bool
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def packed_index_dim(self) -> int:
        # Cached index state is normalized K followed by the KPool gate logits.
        return 2 * self.index_head_dim


__all__ = ["Glm5NextArgs"]
