from __future__ import annotations

import torch
import torch.nn.functional as F
from sparklab.kernels.triton.df11 import DF11_LMAX, df11_compress
from sparklab.kernels.triton.df11_decode import df11_decompress
from sparklab.layers import BaseOP
from sparklab.layers.base import _concat_prefix

# One decode scratch per device, shared by every LinearDF11 (decodes are sequential, so the
# weight is materialized, consumed by the GEMM, then overwritten by the next projection). It
# grows to the largest projection during warmup and is then stable -> CUDA-graph safe.
_SCRATCH: dict[torch.device, torch.Tensor] = {}


def _scratch(out_features: int, in_features: int, device: torch.device) -> torch.Tensor:
    n = out_features * in_features
    buf = _SCRATCH.get(device)
    if buf is None or buf.numel() < n:
        buf = torch.empty(n, dtype=torch.int16, device=device)
        _SCRATCH[device] = buf
    return buf[:n].view(out_features, in_features)


def compress_df11_weight(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compress a BF16 ``[out, in]`` weight into the 5 DF11 buffers (keyed by attr name)."""
    c = df11_compress(weight)
    return {k: c[k] for k in LinearDF11._DF11_KEYS}


class LinearDF11(BaseOP):
    """Replicated (TP=1) linear whose weight is stored losslessly as DF11 (compressed BF16).

    The forward decodes the weight **bit-for-bit** into a shared scratch buffer and runs a
    normal bf16 GEMM. This reproduces GLM-4's intended full-precision attention -- NVIDIA's
    NVFP4 recipe deliberately keeps every ``self_attn`` (qkvo) out of quantization -- while
    fitting a 32 GB VRAM target (~10.7 bits/weight vs bf16's 16, and lossless unlike fp8).

    The DF11 buffers are entropy-sized (data-dependent), so :meth:`load_state_dict` adopts
    whatever shapes arrive instead of asserting against placeholders.
    """

    _DF11_KEYS = ("low8", "bitstream", "chunk_start", "lut")

    def __init__(self, in_features: int, out_features: int, has_bias: bool):
        self.in_features = in_features
        self.out_features = out_features
        self.n = out_features * in_features
        # Placeholders; real (entropy-sized) buffers are installed by load_state_dict.
        self.low8 = torch.empty(0, dtype=torch.uint8)
        self.bitstream = torch.empty(0, dtype=torch.int32)
        self.chunk_start = torch.empty(0, dtype=torch.int32)
        self.lut = torch.empty(0, dtype=torch.int32)
        self.bias = torch.empty(out_features, dtype=torch.bfloat16) if has_bias else None

    def _bundle(self) -> dict:
        g = self.chunk_start.numel()  # number of chunks (interleave stride)
        rows = (self.n + g - 1) // g
        return {
            "low8": self.low8,
            "bitstream": self.bitstream,
            "chunk_start": self.chunk_start,
            "lut": self.lut,
            "meta": (self.out_features, self.in_features, self.n, g, rows, DF11_LMAX),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out16 = _scratch(self.out_features, self.in_features, x.device)
        w = df11_decompress(self._bundle(), out=out16)  # bit-exact bf16 [out, in]
        return F.linear(x, w, self.bias)

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        for name in self._DF11_KEYS:
            setattr(self, name, state_dict.pop(_concat_prefix(prefix, name)))
        if self.bias is not None:
            self.bias = state_dict.pop(_concat_prefix(prefix, "bias"))
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


__all__ = ["LinearDF11", "compress_df11_weight"]
