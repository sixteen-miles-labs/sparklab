from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from sparklab.layers import BaseOP
from sparklab.utils import nvtx_annotate

from .nvfp4_linear import LinearNVFP4

if TYPE_CHECKING:
    import torch


class GlmGatedMLP(BaseOP):
    """SwiGLU MLP whose projections keep the checkpoint's native NVFP4 weights.

    Used for both the leading dense layers (``intermediate_size``) and each MoE layer's
    always-on shared expert (``moe_intermediate_size``). GLM-4 ships these in NVFP4; we keep
    them NVFP4 and dequantize in the forward (:class:`LinearNVFP4`) -- identical math to the
    routed experts (faithful to the checkpoint) and the smallest footprint, which matters
    because GLM activates 89*8 experts per token. Activations stay bf16.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_proj = LinearNVFP4(hidden_size, intermediate_size)
        self.up_proj = LinearNVFP4(hidden_size, intermediate_size)
        self.down_proj = LinearNVFP4(intermediate_size, hidden_size)

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        del x
        return self.down_proj.forward(F.silu(gate) * up)


__all__ = ["GlmGatedMLP"]
