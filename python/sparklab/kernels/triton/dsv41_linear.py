"""Small-row packed linears for the DeepSeek V4.1 reference decoder."""

from __future__ import annotations

import functools

import torch

from sparklab.moe.fused_ds_fp4 import _grouped_decode


@functools.lru_cache(maxsize=None)
def _single_slot(device_index: int, rows: int) -> torch.Tensor:
    return torch.zeros((rows, 1), dtype=torch.int32, device=torch.device("cuda", device_index))


def mxfp4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale_codes: torch.Tensor,
) -> torch.Tensor:
    """Compute ``x @ W.T`` directly from block-32 E2M1/E8M0 checkpoint data.

    ``x`` has already undergone the checkpoint's FP8 activation round-trip.
    The shared DSV4 decode kernel performs FP32 accumulation while expanding
    each packed nibble and scale in registers.
    """
    *lead, width = x.shape
    rows = x.numel() // width
    if not (
        x.is_cuda
        and packed_weight.is_cuda
        and scale_codes.is_cuda
        and packed_weight.ndim == 2
        and packed_weight.shape[1] * 2 == width
        and scale_codes.shape == (packed_weight.shape[0], width // 32)
    ):
        raise ValueError("V4.1 packed FP4 linear requires compatible CUDA tensors")
    x2 = x.reshape(rows, width).contiguous()
    slots = _single_slot(x.device.index or 0, rows)
    result = _grouped_decode(
        x2,
        packed_weight.unsqueeze(0),
        scale_codes.view(torch.uint8).unsqueeze(0),
        slots,
        None,
        a_row_is_route=False,
        mul_routed_weight=False,
    )
    return result[:, 0].reshape(*lead, packed_weight.shape[0])


__all__ = ["mxfp4_linear"]
