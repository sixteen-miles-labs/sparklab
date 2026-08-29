from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import sparklab.layers.moe as moe_layers
from sparklab.layers import BaseOP, LinearReplicated, MoELayer, OffloadMoELayer
from sparklab.moe import is_offload_moe_backend
from sparklab.moe.fused_mxfp4 import (
    MXFP4_DECODE_MAX_TOKENS,
    _transpose_mxfp4_for_decode,
    run_mxfp4_prefill_experts_t,
    run_mxfp4_splitk_decode_experts,
)
from sparklab.utils import nvtx_annotate

from .weight import local_mxfp4_intermediate_range

if TYPE_CHECKING:
    from sparklab.models.config import ModelConfig


class GptOssMxfp4TritonMoELayer(MoELayer):
    def __init__(self, config: ModelConfig):
        tp_info = moe_layers.get_tp_info()
        _, _, local_intermediate = local_mxfp4_intermediate_range(
            config.moe_intermediate_size,
            rank=tp_info.rank,
            world_size=tp_info.size,
        )
        if config.hidden_size % 32 != 0:
            raise ValueError("GPT-OSS MXFP4 hidden size must be divisible by 32")

        super().__init__(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            activation="gpt_oss_swiglu",
            allocate_experts=False,
            # Descriptive only: this subclass owns its allocation and forward, but the
            # field must not claim bf16 for mxfp4 tensors.
            weight_format="mxfp4_triton",
        )
        self.local_intermediate_size = local_intermediate
        self.hidden_act_alpha = config.hidden_act_alpha
        self.swiglu_limit = config.swiglu_limit

        hidden_blocks = config.hidden_size // 32
        intermediate_blocks = local_intermediate // 32
        self.gate_up_proj_blocks = torch.empty(
            config.num_experts,
            2 * local_intermediate,
            hidden_blocks,
            16,
            dtype=torch.uint8,
        )
        self.gate_up_proj_scales = torch.empty(
            config.num_experts,
            2 * local_intermediate,
            hidden_blocks,
            dtype=torch.uint8,
        )
        self.gate_up_proj_bias = torch.empty(
            config.num_experts,
            2 * local_intermediate,
            dtype=torch.bfloat16,
        )
        self.down_proj_blocks = torch.empty(
            config.num_experts,
            config.hidden_size,
            intermediate_blocks,
            16,
            dtype=torch.uint8,
        )
        self.down_proj_scales = torch.empty(
            config.num_experts,
            config.hidden_size,
            intermediate_blocks,
            dtype=torch.uint8,
        )
        self.down_proj_bias = torch.empty(
            config.num_experts,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        # Transposed split-K decode weights, built lazily in _ensure_decode_weights.
        self._gu_blocks_t = None
        self._gu_scales_t = None
        self._dn_blocks_t = None
        self._dn_scales_t = None

    def _topk(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from sparklab.kernels import gpt_oss_fused_routing

        return gpt_oss_fused_routing(router_logits, self.top_k)

    def _ensure_decode_weights(self) -> None:
        """Build the transposed split-K/prefill weights once and free the HF blocks/
        scales (single transposed layout ~61GB so 120B fits). Both prefill and decode
        read the transposed weights. Must run before CUDA graph capture."""
        if self._gu_blocks_t is not None:
            return
        self._gu_blocks_t, self._gu_scales_t = _transpose_mxfp4_for_decode(
            self.gate_up_proj_blocks, self.gate_up_proj_scales
        )
        self._dn_blocks_t, self._dn_scales_t = _transpose_mxfp4_for_decode(
            self.down_proj_blocks, self.down_proj_scales
        )
        self.gate_up_proj_blocks = None
        self.gate_up_proj_scales = None
        self.down_proj_blocks = None
        self.down_proj_scales = None
        torch.cuda.empty_cache()

    def prepare_for_runtime(self) -> None:
        self._ensure_decode_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise RuntimeError("GPT-OSS MXFP4 MoE requires the Triton CUDA kernel")
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not router_logits.is_contiguous():
            router_logits = router_logits.contiguous()

        topk_weights, topk_ids = self._topk(router_logits)
        self._ensure_decode_weights()
        if hidden_states.shape[0] <= MXFP4_DECODE_MAX_TOKENS:
            output = run_mxfp4_splitk_decode_experts(
                hidden_states, topk_weights, topk_ids,
                self._gu_blocks_t, self._gu_scales_t, self.gate_up_proj_bias,
                self._dn_blocks_t, self._dn_scales_t, self.down_proj_bias,
                top_k=self.top_k,
                hidden_act_alpha=self.hidden_act_alpha,
                swiglu_limit=self.swiglu_limit,
            )
        else:
            output = run_mxfp4_prefill_experts_t(
                hidden_states, topk_weights, topk_ids,
                self._gu_blocks_t, self._gu_scales_t, self.gate_up_proj_bias,
                self._dn_blocks_t, self._dn_scales_t, self.down_proj_bias,
                top_k=self.top_k,
                hidden_act_alpha=self.hidden_act_alpha,
                swiglu_limit=self.swiglu_limit,
            )
        return self._maybe_all_reduce(output)


class GptOssMxfp4OffloadMoELayer(OffloadMoELayer):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp_info = moe_layers.get_tp_info()
        _, _, local_intermediate = local_mxfp4_intermediate_range(
            config.moe_intermediate_size,
            rank=tp_info.rank,
            world_size=tp_info.size,
        )
        super().__init__(
            layer_id=layer_id,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            activation="gpt_oss_swiglu",
        )
        self.local_intermediate_size = local_intermediate
        self.hidden_act_alpha = config.hidden_act_alpha
        self.swiglu_limit = config.swiglu_limit

    def _topk(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from sparklab.kernels import gpt_oss_fused_routing

        return gpt_oss_fused_routing(router_logits, self.top_k)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert router_logits is not None
        if not router_logits.is_contiguous():
            router_logits = router_logits.contiguous()
        topk_weights, topk_ids = self._topk(router_logits)
        # routed_forward dispatches prefill/decode movement and all-reduces once;
        # topk_ids is cloned because decode rewrites it in place into slot ids.
        return self.routed_forward(hidden_states, topk_weights, topk_ids.clone())


class GptOssMLP(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        self.router = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=config.has_router_bias,
        )
        if config.moe_weight_format != "mxfp4":
            raise ValueError(
                f"gpt-oss supports only mxfp4 expert weights, got "
                f"moe_weight_format={config.moe_weight_format!r}"
            )
        if is_offload_moe_backend(config.moe_backend):
            assert layer_id is not None
            self.experts = GptOssMxfp4OffloadMoELayer(config, layer_id)
        else:
            self.experts = GptOssMxfp4TritonMoELayer(config)
        self._layer_id = layer_id

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.router.forward(hidden_states)
        final_hidden_states = self.experts.forward(hidden_states, router_logits)
        return final_hidden_states.view(num_tokens, hidden_dim)

    def prepare_for_runtime(self) -> None:
        if hasattr(self.experts, "prepare_for_runtime"):
            self.experts.prepare_for_runtime()


__all__ = [
    "GptOssMLP",
    "GptOssMxfp4OffloadMoELayer",
    "GptOssMxfp4TritonMoELayer",
]
