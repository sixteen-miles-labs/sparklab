"""Opt-in W4A4 prefill for dense Qwen NVFP4 MLPs on GB10.

This keeps a row-major weight layout alongside the native W4A16 layout. Activations
use dynamic NVFP4 scaling, which changes numerics relative to W4A16 and to static
checkpoint activation scales. Imports that initialize CUDA stay inside load/forward.
"""
from __future__ import annotations

import torch


class Nvfp4Prefill:
    def __init__(self, weight, block_scale, global_scale, output_sizes):
        if weight.device.type != "cuda" or torch.cuda.get_device_capability(weight.device) != (12, 1):
            raise ValueError("FlashInfer NVFP4 prefill currently requires NVIDIA GB10 (SM121)")
        if weight.shape[1] * 2 % 128 or any(size % 128 for size in output_sizes):
            raise ValueError("FlashInfer NVFP4 prefill requires projection dimensions divisible by 128")
        if sum(output_sizes) != weight.shape[0]:
            raise ValueError("NVFP4 prefill projection sizes do not cover the weight")
        from flashinfer.quantization import block_scale_interleave

        self.parts = []
        offset = 0
        for size in output_sizes:
            w = weight[offset:offset + size].contiguous()
            s = block_scale[offset:offset + size].contiguous()
            g = global_scale[offset:offset + size]
            if not bool(torch.all(g == g[0])):
                raise ValueError("NVFP4 prefill requires one global weight scale per projection")
            self.parts.append((w, block_scale_interleave(s.view(torch.uint8)), g[:1].float()))
            offset += size

    def forward(self, x):
        import flashinfer
        from flashinfer.quantization import SfLayout

        if x.dtype != torch.bfloat16:
            raise ValueError("FlashInfer NVFP4 prefill requires BF16 activations")
        activation = x.reshape(-1, x.shape[-1]).contiguous()
        # E4M3 scale range (448) times E2M1 value range (6).
        inverse_scale = 2688.0 / activation.float().abs().amax().clamp_min(1.e-6).reshape(1)
        quantized, scales = flashinfer.nvfp4_quantize(
            activation, inverse_scale, sfLayout=SfLayout.layout_128x4,
        )
        with flashinfer.autotune(tune_mode=activation.shape[0] in (2048, 8192)):
            outputs = [
                flashinfer.mm_fp4(
                    quantized, w.T, scales, s, g / inverse_scale, backend="cutlass",
                )
                for w, s, g in self.parts
            ]
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        return output.reshape(*x.shape[:-1], output.shape[-1])
