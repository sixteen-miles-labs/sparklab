"""GLM-5.3 sigmoid/noaux router, FP8 routed experts, and shared expert."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

from .mlp import Glm5NextMLP


class Glm5NextRouter(BaseOP):
    def __init__(self, hidden_size: int, num_experts: int):
        # These names intentionally mirror ``mlp.gate.{weight,e_score_correction_bias}``.
        self.weight = torch.empty(num_experts, hidden_size)
        self.e_score_correction_bias = torch.empty(num_experts, dtype=torch.float32)


class Glm5NextSparseMoe(BaseOP):
    def __init__(self, config, layer_id: int):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.gate = Glm5NextRouter(config.hidden_size, config.num_experts)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            weight_format="fp8_block",
            extra_attrs={"swiglu_limit": config.swiglu_limit},
        )
        self.shared_experts = Glm5NextMLP(
            config.hidden_size,
            config.moe_intermediate_size * max(1, config.n_shared_experts),
            config.swiglu_limit,
            quantization=config.dense_quant,
        )

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(x.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        choice = scores + self.gate.e_score_correction_bias.float()
        if self.n_group > 1:
            m, groups = choice.shape[0], self.n_group
            grouped = choice.view(m, groups, self.num_experts // groups)
            group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = group_scores.topk(self.topk_group, dim=-1, sorted=False)[1]
            mask = torch.zeros_like(group_scores)
            mask.scatter_(1, group_idx, 1.0)
            choice = choice.masked_fill(
                ~mask.unsqueeze(-1).expand_as(grouped).reshape_as(choice).bool(),
                float("-inf"),
            )
        ids = choice.topk(self.top_k, dim=-1, sorted=False)[1]
        weights = scores.gather(-1, ids)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return weights.float().contiguous(), ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.view(-1, shape[-1])
        weights, ids = self._route(x)
        cache = getattr(self.experts, "offload_cache", None)
        overlap = bool(
            cache is not None
            and cache.shared_expert_overlap
            and cache.disk_source is not None
            and x.is_cuda
        )
        if overlap:
            current = torch.cuda.current_stream(x.device)
            if cache.shared_expert_stream is None:
                cache.shared_expert_stream = torch.cuda.Stream(device=x.device)
            shared_stream = cache.shared_expert_stream
            shared_stream.wait_stream(current)
            with torch.cuda.stream(shared_stream):
                shared = self.shared_experts.forward(x)
            # Routing is already known, so disk staging can proceed on the host while
            # the resident shared MLP executes on its own CUDA stream. Join only when
            # the elementwise sum consumes both results.
            routed = self.experts.routed_forward(x, weights, ids)
            current.wait_stream(shared_stream)
            cache.shared_expert_overlap_calls += 1
        else:
            # Shared expert runs from the unmodified input; the offload path may reuse
            # the routed input buffer internally.
            shared = self.shared_experts.forward(x)
            routed = self.experts.routed_forward(x, weights, ids)
        return (routed + shared).view(shape)


__all__ = ["Glm5NextRouter", "Glm5NextSparseMoe"]
