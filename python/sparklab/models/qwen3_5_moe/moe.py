from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sparklab.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)

from sparklab.kernels.triton.fp8_block_linear import Fp8BlockColMerged, Fp8BlockLinear

if TYPE_CHECKING:
    from sparklab.models.config import ModelConfig


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(self, config: ModelConfig, hidden_size: int, intermediate_size: int):
        shared_quant = getattr(config, "shared_expert_quant", None)
        if shared_quant is None:
            shared_quant = getattr(config, "expert_quant", "none")
        if shared_quant == "fp8_block":
            self.gate_up_proj = Fp8BlockColMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = Fp8BlockLinear(intermediate_size, hidden_size, has_bias=False)
        elif shared_quant == "fp8_pertensor":
            from sparklab.kernels.triton.fp8_pertensor_linear import (
                Fp8PerTensorColMerged,
                Fp8PerTensorLinear,
            )

            self.gate_up_proj = Fp8PerTensorColMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = Fp8PerTensorLinear(
                intermediate_size, hidden_size, has_bias=False
            )
        elif shared_quant == "nvfp4" or getattr(config, "dense_quant", "none") == "nvfp4":
            # NVFP4 checkpoint: keep the shared expert's NVFP4 weights native (W4A16).
            from sparklab.kernels.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear

            self.gate_up_proj = Nvfp4DenseColMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = Nvfp4DenseLinear(intermediate_size, hidden_size, has_bias=False)
        else:
            self.gate_up_proj = LinearColParallelMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure (and quant
    dispatch) as the shared expert -- NVFP4 (W4A16) when ``dense_quant=="nvfp4"``, else bf16 --
    so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, config.hidden_size, config.intermediate_size)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        weight_format = (
            "fp8_block" if getattr(config, "expert_quant", "none") == "fp8_block" else "bf16"
        )
        self.experts = make_moe_layer(
            config, layer_id=layer_id, renormalize=True, weight_format=weight_format
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        router_logits = self.gate.forward(hidden_states)
        cache = getattr(self.experts, "offload_cache", None)
        overlap = bool(
            cache is not None
            and cache.shared_expert_overlap
            and cache.disk_source is not None
            and hidden_states.is_cuda
        )
        if overlap:
            current = torch.cuda.current_stream(hidden_states.device)
            if cache.shared_expert_stream is None:
                cache.shared_expert_stream = torch.cuda.Stream(device=hidden_states.device)
            shared_stream = cache.shared_expert_stream
            # The routed fused kernel may overwrite ``hidden_states`` in place. Copy
            # it on the current stream before forking work so the shared branch has an
            # immutable input; stream ordering completes this small copy before either
            # consumer can race with it.
            shared_input = hidden_states.clone()
            shared_stream.wait_stream(current)
            with torch.cuda.stream(shared_stream):
                shared = self.shared_expert.forward(shared_input)
                shared = shared * torch.sigmoid(
                    self.shared_expert_gate.forward(shared_input)
                )
            # Routing is already available. Disk staging or immutable-cache expert
            # compute can proceed on the main stream while the always-on shared expert
            # consumes otherwise-idle SM capacity on the auxiliary stream.
            routed = self.experts.forward(
                hidden_states=hidden_states, router_logits=router_logits
            )
            current.wait_stream(shared_stream)
            # The result was allocated on the auxiliary stream and is consumed by
            # the current stream's add below. Tell the caching allocator about that
            # cross-stream lifetime so its storage cannot be recycled after only the
            # producer stream completes.
            shared.record_stream(current)
            cache.shared_expert_overlap_calls += 1
        else:
            shared = self.shared_expert.forward(hidden_states)
            shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
            routed = self.experts.forward(
                hidden_states=hidden_states, router_logits=router_logits
            )
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
