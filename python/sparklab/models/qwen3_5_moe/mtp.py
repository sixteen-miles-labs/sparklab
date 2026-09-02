"""Checkpoint-native multi-token predictor for Qwen3.5/3.6.

The NVIDIA Qwen3.6 NVFP4 checkpoint deliberately leaves the complete MTP head
in BF16.  It is a projection over the shifted token embedding and target hidden
state followed by one full-attention decoder layer.  The draft's small routed
expert bank stays resident and is independent of the target offload cache.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from sparklab.layers import (
    BaseOP,
    GemmaRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    OPList,
    silu_and_mul,
)
from sparklab.moe.fused import fused_topk

from .attention import Qwen3_5Attention


class _Bf16SharedExpert(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(hidden)))


class _ResidentBf16MTPMoE(BaseOP):
    """One resident BF16 routed layer plus Qwen's gated shared expert."""

    def __init__(self, config):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=True,
            weight_format="bf16",
        )
        self.gate = LinearReplicated(
            config.hidden_size, config.num_experts, has_bias=False
        )
        self.shared_expert = _Bf16SharedExpert(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(
            config.hidden_size, 1, has_bias=False
        )
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate.forward(hidden)
        # The resident grouped GEMM may reuse its input storage.
        shared_input = hidden.clone()
        shared = self.shared_expert.forward(shared_input)
        shared.mul_(torch.sigmoid(self.shared_expert_gate.forward(shared_input)))
        weights, ids = fused_topk(
            hidden_states=hidden,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=True,
        )
        return self.experts.routed_forward(hidden, weights, ids) + shared


class _MTPDecoderLayer(BaseOP):
    def __init__(self, config):
        # All published MTP tensors are BF16, even when the target is NVFP4.
        draft_config = replace(
            config,
            attn_quant="none",
            dense_quant="none",
            expert_quant="none",
            shared_expert_quant="none",
        )
        self.self_attn = Qwen3_5Attention(draft_config, config.num_layers)
        self.mlp = _ResidentBf16MTPMoE(draft_config)
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm.forward(hidden)
        hidden = self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(
            hidden, residual
        )
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5MultiTokenPredictor(BaseOP):
    def __init__(self, config):
        h = config.hidden_size
        self.fc = LinearReplicated(2 * h, h, has_bias=False)
        self.layers = OPList([_MTPDecoderLayer(config)])
        self.norm = GemmaRMSNorm(h, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = GemmaRMSNorm(h, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(h, eps=config.rms_norm_eps)

    def forward(
        self, token_embeddings: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        embed = self.pre_fc_norm_embedding.forward(token_embeddings)
        hidden = self.pre_fc_norm_hidden.forward(target_hidden)
        hidden = self.fc.forward(torch.cat((embed, hidden), dim=-1))
        hidden, residual = self.layers.op_list[0].forward(hidden)
        hidden, _ = self.norm.forward_add_residual(hidden, residual)
        return hidden


__all__ = ["Qwen3_5MultiTokenPredictor"]
