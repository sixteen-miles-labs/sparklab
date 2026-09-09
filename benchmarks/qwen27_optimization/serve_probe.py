#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Experimental Qwen3.8-27B prefill probe; not a production recipe.

Q27_W4A4_PREFILL=1 enables dynamic FP4 activation quantization for dense
NVFP4 projections with at least 128 input rows. Smaller batches retain the
native W4A16 implementation. This changes numerics and retains a second packed
weight layout. It does not use the checkpoint's calibrated activation scales.

Q27_SCRATCH_MIB optionally changes the native BF16 prefill scratch size.
All command-line arguments are forwarded to the normal SparkLab server.
"""

import os
import sys

import torch

from sparklab.kernels.triton import nvfp4_linear as kernels


def install_probe():
    if os.getenv("Q27_SCRATCH_MIB"):
        kernels._SCRATCH_CHUNK_BYTES = int(os.environ["Q27_SCRATCH_MIB"]) * 1024**2
    if os.getenv("Q27_W4A4_PREFILL") != "1":
        return

    original_load = kernels.Nvfp4DenseLinear.load_state_dict
    original_forward = kernels.Nvfp4DenseLinear.forward

    def load(self, state_dict, *, prefix="", _internal=False):
        # Import lazily: FlashInfer can initialize CUDA, which must happen after
        # the engine establishes its process and device state.
        from flashinfer.quantization import block_scale_interleave

        def key(suffix):
            return prefix + "." + suffix if prefix else suffix

        weight = state_dict[key("weight")]
        scale = state_dict[key("weight_scale")]
        global_scale = state_dict[key("weight_global")]
        sizes = getattr(self, "output_sizes", [self.out_features])
        parts, offset = [], 0
        for size in sizes:
            w = weight[offset:offset + size].contiguous()
            s = scale[offset:offset + size].contiguous()
            g = global_scale[offset:offset + size]
            assert torch.all(g == g[0]), (prefix, "nonuniform global scale")
            parts.append((w, block_scale_interleave(s.view(torch.uint8)), g[:1].float()))
            offset += size
        self._q27_fi_parts = parts
        original_load(self, state_dict, prefix=prefix, _internal=_internal)

    def forward(self, x):
        import flashinfer
        from flashinfer.quantization import SfLayout

        if x.numel() // x.shape[-1] < 128:
            return original_forward(self, x)
        lead = x.shape[:-1]
        activation = x.reshape(-1, x.shape[-1]).contiguous()
        # 448 (E4M3 scale range) * 6 (E2M1 value range).
        inverse_scale = 2688.0 / activation.float().abs().amax().clamp_min(1.e-6).reshape(1)
        quantized, block_scales = flashinfer.nvfp4_quantize(
            activation, inverse_scale, sfLayout=SfLayout.layout_128x4,
        )
        with flashinfer.autotune(tune_mode=activation.shape[0] in (2048, 8192)):
            parts = [
                flashinfer.mm_fp4(
                    quantized, w.T, block_scales, s, g / inverse_scale,
                    backend="cutlass",
                )
                for w, s, g in self._q27_fi_parts
            ]
        output = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*lead, self.out_features)

    kernels.Nvfp4DenseLinear.load_state_dict = load
    kernels.Nvfp4DenseLinear.forward = forward


# The engine spawns workers that import this entrypoint as __mp_main__. Install
# in those workers as well as the parent, before either starts loading weights.
install_probe()


if __name__ == "__main__":
    from sparklab.serving import launch_server

    launch_server(argv=sys.argv[1:])
