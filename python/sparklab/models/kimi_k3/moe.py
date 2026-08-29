from __future__ import annotations

import torch
import torch.nn.functional as F
from sparklab.layers import BaseOP, LinearReplicated, RMSNorm, make_moe_layer

from .ops import KimiSituMLP


class KimiSparseMoeBlock(BaseOP):
    def __init__(self, config, layer_id: int):
        args = config.kimi_k3_args
        assert args is not None
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.scaling = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.e_score_correction_bias = torch.empty(config.num_experts)
        self.routed_expert_down_proj = LinearReplicated(
            config.hidden_size, args.routed_expert_hidden_size, has_bias=False
        )
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            activation="situ",
            weight_format={
                "none": "bf16",
                "mxfp4": "mxfp4_triton",
                "nvfp4": "nvfp4",
            }[config.expert_quant],
            hidden_size=args.routed_expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            extra_attrs={
                "hidden_act_alpha": args.situ_beta,
                "swiglu_limit": args.situ_linear_beta,
            },
        )
        self.routed_expert_norm = (
            RMSNorm(args.routed_expert_hidden_size, eps=config.rms_norm_eps)
            if args.latent_moe_use_norm
            else None
        )
        self.routed_expert_up_proj = LinearReplicated(
            args.routed_expert_hidden_size, config.hidden_size, has_bias=False
        )
        self.shared_experts = KimiSituMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            args.situ_beta,
            args.situ_linear_beta,
            quantization=config.dense_quant,
        )

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        choice = scores + self.e_score_correction_bias.float()
        if self.n_group > 1 and self.n_group > self.topk_group:
            grouped = choice.view(x.shape[0], self.n_group, self.num_experts // self.n_group)
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            groups = group_scores.topk(self.topk_group, dim=-1, sorted=False).indices
            mask = torch.zeros_like(group_scores, dtype=torch.bool)
            mask.scatter_(1, groups, True)
            choice = choice.masked_fill(
                ~mask.unsqueeze(-1).expand_as(grouped).reshape_as(choice), float("-inf")
            )
        ids = choice.topk(self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        if self.top_k > 1 and self.renormalize:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.scaling
        return weights.float().contiguous(), ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.view(-1, shape[-1])
        weights, ids = self._route(x)
        latent = self.routed_expert_down_proj.forward(x)
        routed = self.experts.routed_forward(latent, weights, ids)
        if self.routed_expert_norm is not None:
            routed = self.routed_expert_norm.forward(routed)
        routed = self.routed_expert_up_proj.forward(routed)
        return (routed + self.shared_experts.forward(x)).view(shape)


__all__ = ["KimiSparseMoeBlock"]
