"""SwiGLU MLP for GLM-5.2's leading dense layers and per-layer shared experts.

The checkpoint ships these bf16 (NVIDIA's NVFP4 recipe leaves them unquantized; only
the routed experts are FP4). Serving default is W8A16 fp8 with per-row scales,
quantized at load (``ModelConfig.dense_quant == "fp8_pertensor"``, from
SPARKLAB_GLM_MLP_FP8): decode reads every shared expert each token, so this halves
~5.6 GiB/token of weight traffic and frees the same VRAM for expert-cache slots.
``SPARKLAB_GLM_MLP_FP8=0`` restores the checkpoint-faithful bf16 weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from sparklab.layers import BaseOP
from sparklab.utils import nvtx_annotate

from .attention import _make_proj

if TYPE_CHECKING:
    import torch


class GlmDsaGatedMLP(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int, quant: str = "none"):
        self.gate_proj = _make_proj(quant, hidden_size, intermediate_size)
        self.up_proj = _make_proj(quant, hidden_size, intermediate_size)
        self.down_proj = _make_proj(quant, intermediate_size, hidden_size)

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        del x
        return self.down_proj.forward(F.silu(gate) * up)


__all__ = ["GlmDsaGatedMLP"]
