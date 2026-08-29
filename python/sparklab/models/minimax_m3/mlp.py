"""SwiGLU-OAI MLP for MiniMax-M3's leading dense layers and per-layer shared experts.

The checkpoint ships these MXFP8 (fp8-e4m3 weight + uint8 e8m0 block-32
``weight_scale_inv``); serving default keeps them native W8A16
(``ModelConfig.dense_quant == "mxfp8"``, from SPARKLAB_M3_MLP_MXFP8) -- decode reads
every shared expert each token, so this halves the resident-MLP weight traffic and
frees the same VRAM for expert-cache slots. ``SPARKLAB_M3_MLP_MXFP8=0`` dequantizes
to bf16 at load (bring-up / ablation).

gate/up are stored merged (``gate_up_proj``, [gate; up] halves -- the loader
concatenates the two checkpoint projections output-wise; MXFP8 scales are per-output-
row so the fusion is exact), which feeds the same uninterleaved ``swigluoai_and_mul``
kernel the NVFP4 expert path uses:
``clamp(gate, max=limit) * sigmoid(alpha * gate) * (clamp(up, +-limit) + 1)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sparklab.layers import BaseOP, LinearReplicated, swigluoai_and_mul
from sparklab.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


def make_proj(quant: str, in_features: int, out_features: int) -> BaseOP:
    """A resident projection in the model's resolved quant mode: ``"mxfp8"`` (W8A16,
    the checkpoint's native block-32 e8m0 format -- see weight.py) or bf16."""
    if quant == "mxfp8":
        from sparklab.kernels.triton.mxfp8_linear import Mxfp8Linear

        return Mxfp8Linear(in_features, out_features, has_bias=False)
    return LinearReplicated(in_features, out_features, has_bias=False)


class MiniMaxM3MLP(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        quant: str,
        alpha: float,
        limit: float,
    ):
        self.gate_up_proj = make_proj(quant, hidden_size, 2 * intermediate_size)
        self.down_proj = make_proj(quant, intermediate_size, hidden_size)
        self._alpha = alpha
        self._limit = limit

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = swigluoai_and_mul(gate_up, alpha=self._alpha, limit=self._limit)
        del gate_up
        return self.down_proj.forward(y)


__all__ = ["MiniMaxM3MLP", "make_proj"]
