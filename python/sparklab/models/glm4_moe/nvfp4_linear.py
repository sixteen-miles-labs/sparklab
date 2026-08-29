from __future__ import annotations

import os

import torch
from sparklab.layers import BaseOP
from sparklab.layers.base import _concat_prefix
from sparklab.kernels.triton.nvfp4_linear import (
    nvfp4_dense_linear_t,
    nvfp4_transpose_resident,
)

# Resident NVFP4 linears default to vLLM's fused FP4 Marlin GEMM (W4A16) on Marlin-capable GPUs
# (cc in [8.0, 10.0), which lack native FP4 tensor cores): ~2-7x faster per-linear at bs=1 and it
# frees the bf16 dequant scratch, with numerically-faithful output (validated under the model's
# sampling regime). When Marlin is unavailable (e.g. sm_120, or vLLM not installed) the
# forward runs SparkLab's own shared
# W4A16 kernel (sparklab.kernels.triton.nvfp4_linear) -- reading the FP4 weight directly (fp32
# accumulation, no per-forward bf16 dequant + scratch), which is faster *and* strictly more
# accurate than the old dequant fallback. Set GLM_NVFP4_NOMARLIN=1 to force the shared kernel.
# (Marlin's ~bf16-level difference can perturb *greedy* decoding, but GLM's recommended decoding
# is sampling, where it is correctness-equivalent.)
_DISABLE_MARLIN = os.environ.get("GLM_NVFP4_NOMARLIN", "0") == "1"


class LinearNVFP4(BaseOP):
    """Replicated (TP=1) linear that keeps the checkpoint's native NVFP4 weight as-is.

    Used for GLM-4's always-resident dense MLPs and shared experts (W4A16: bf16 activation,
    FP4 weight). By default the weight is repacked to Marlin layout at load and the forward is a
    single fused dequant+GEMM (vLLM Marlin). With GLM_NVFP4_NOMARLIN=1 (or on non-Marlin GPUs /
    when vLLM is absent) it runs the shared SparkLab W4A16 kernel directly on the packed FP4
    weight (decode int32 GEMV / prefill GEMM, fp32 accumulation) -- no bf16 dequant scratch.

    Keeping NVFP4 (4 bits) is faithful to the checkpoint (identical math to the routed experts)
    and the smallest option -- re-quantizing to fp8 would add ~2.6% error *and* use twice the
    memory, and the budget is tight once attention is lossless DF11.
    """

    def __init__(self, in_features: int, out_features: int):
        assert in_features % 16 == 0
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features // 2, dtype=torch.uint8)
        self.weight_scale = torch.empty(out_features, in_features // 16, dtype=torch.float8_e4m3fn)
        self.weight_global = torch.empty(out_features, dtype=torch.float16)
        self._marlin = False

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        weight = state_dict.pop(_concat_prefix(prefix, "weight"))
        weight_scale = state_dict.pop(_concat_prefix(prefix, "weight_scale"))
        weight_global = state_dict.pop(_concat_prefix(prefix, "weight_global"))

        if not _DISABLE_MARLIN:
            try:
                from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
                    apply_fp4_marlin_linear,
                    is_fp4_marlin_supported,
                    prepare_fp4_layer_for_marlin,
                )

                marlin_supported = is_fp4_marlin_supported()
            except Exception:
                marlin_supported = False

            if marlin_supported:
                from types import SimpleNamespace

                # weight_global is the per-tensor weight_scale_2 broadcast to [out]; Marlin
                # wants the scalar.
                layer = SimpleNamespace(
                    weight=weight,
                    weight_scale=weight_scale,
                    weight_scale_2=weight_global.reshape(-1)[0].to(torch.bfloat16),
                    output_size_per_partition=self.out_features,
                    input_size_per_partition=self.in_features,
                    params_dtype=torch.bfloat16,
                )
                prepare_fp4_layer_for_marlin(layer)  # repacks weight + processes scales in-place
                self._mweight = layer.weight
                self._mscale = layer.weight_scale
                self._mglobal = layer.weight_scale_2
                self._mworkspace = layer.workspace
                self._apply = apply_fp4_marlin_linear
                self._marlin = True
                dev = weight.device
                self.weight = torch.empty(0, dtype=torch.uint8, device=dev)
                self.weight_scale = torch.empty(0, dtype=torch.float8_e4m3fn, device=dev)
                self.weight_global = torch.empty(0, dtype=torch.float16, device=dev)

        if not self._marlin:
            # K-major resident repack: coalesced weight loads in the shared W4A16 kernels.
            self.weight, self.weight_scale = nvfp4_transpose_resident(weight, weight_scale)
            self.weight_global = weight_global

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._marlin:
            return self._apply(
                input=x,
                weight=self._mweight,
                weight_scale=self._mscale,
                weight_scale_2=self._mglobal,
                workspace=self._mworkspace,
                size_n=self.out_features,
                size_k=self.in_features,
            )
        # Shared SparkLab W4A16 kernel: read the packed FP4 weight directly (no bf16 scratch).
        return nvfp4_dense_linear_t(x, self.weight, self.weight_scale, self.weight_global)


__all__ = ["LinearNVFP4"]
