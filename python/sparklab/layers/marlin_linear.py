"""Optional vLLM Marlin adapter for a checkpoint-native NVFP4 linear."""
from types import SimpleNamespace

import torch


class MarlinNVFP4Linear:
    def __init__(self, weight, scales, global_scale):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )

        if not torch.all(global_scale == global_scale[0]):
            raise ValueError("Marlin linear requires one global scale per projection")
        self.size_n, packed_k = weight.shape
        self.size_k = packed_k * 2
        self.layer = SimpleNamespace(
            weight=weight,
            weight_scale=scales,
            weight_global_scale=global_scale[:1].float(),
            output_size_per_partition=self.size_n,
            input_size_per_partition=self.size_k,
            params_dtype=torch.bfloat16,
        )
        prepare_fp4_layer_for_marlin(self.layer)
        self.apply = apply_fp4_marlin_linear

    def forward(self, hidden):
        layer = self.layer
        return self.apply(
            hidden, layer.weight, layer.weight_scale, layer.weight_global_scale,
            layer.workspace, self.size_n, self.size_k,
        )
