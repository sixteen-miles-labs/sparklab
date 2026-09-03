"""Packed E2M1 cache rows for the DSV4 Lightning Indexer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fp8_linear import _log2_ceil, _round_fp4


@triton.jit
def decode_e2m1(code):
    mag = code & 7
    value = tl.where(
        mag == 0, 0.0,
        tl.where(mag == 1, 0.5,
        tl.where(mag == 2, 1.0,
        tl.where(mag == 3, 1.5,
        tl.where(mag == 4, 2.0,
        tl.where(mag == 5, 3.0,
        tl.where(mag == 6, 4.0, 6.0)))))))
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _encode_e2m1(value):
    value = _round_fp4(value)
    a = tl.abs(value)
    mag = tl.where(
        a == 0.0, 0,
        tl.where(a == 0.5, 1,
        tl.where(a == 1.0, 2,
        tl.where(a == 1.5, 3,
        tl.where(a == 2.0, 4,
        tl.where(a == 3.0, 5,
        tl.where(a == 4.0, 6, 7)))))))
    return mag | tl.where(value < 0, 8, 0)


@triton.jit
def _pack_fp4_rows_kernel(
    x_ptr, rows_ptr, packed_ptr, scales_ptr,
    M: tl.constexpr, D: tl.constexpr,
    stride_xm, stride_xd, stride_pr, stride_pb, stride_sr, stride_sb,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + m * stride_xm + offs * stride_xd).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 6.0 * (2.0 ** -126))
    exponent = _log2_ceil(amax / 6.0)
    scale = tl.exp2(exponent.to(tl.float32))
    pairs = tl.arange(0, BLOCK // 2)
    even = tl.load(
        x_ptr + m * stride_xm + (b * BLOCK + pairs * 2) * stride_xd
    ).to(tl.float32)
    odd = tl.load(
        x_ptr + m * stride_xm + (b * BLOCK + pairs * 2 + 1) * stride_xd
    ).to(tl.float32)
    lo = _encode_e2m1(tl.clamp(even / scale, -6.0, 6.0)).to(tl.uint8)
    hi = _encode_e2m1(tl.clamp(odd / scale, -6.0, 6.0)).to(tl.uint8)
    byte = lo | (hi << 4)
    row = tl.load(rows_ptr + m)
    tl.store(packed_ptr + row * stride_pr + (b * (BLOCK // 2) + pairs) * stride_pb, byte)
    tl.store(scales_ptr + row * stride_sr + b * stride_sb, scale)


def pack_fp4_rows(
    x: torch.Tensor,
    rows: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    """Quantize ``x[M,D]`` into selected packed-pool rows, block size 32."""
    x = x.reshape(-1, x.shape[-1]).contiguous()
    rows = rows.reshape(-1).to(torch.int64)
    M, D = x.shape
    assert rows.numel() == M and D % 32 == 0
    assert packed.dtype == torch.uint8 and packed.shape[1] == D // 2
    assert scales.dtype == torch.float32 and scales.shape[1] == D // 32
    if x.device.type == "cpu":
        blocks = x.float().view(M, D // 32, 32)
        amax = blocks.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
        q = blocks / scale.unsqueeze(-1)
        lut = torch.tensor(
            [0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=x.device
        )
        distance = (q.abs().unsqueeze(-1) - lut).abs()
        code = distance.argmin(-1).to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)
        code = code.view(M, D)
        bytes_ = code[:, 0::2] | (code[:, 1::2] << 4)
        packed.index_copy_(0, rows, bytes_)
        scales.index_copy_(0, rows, scale)
        return
    _pack_fp4_rows_kernel[(M, D // 32)](
        x, rows, packed, scales, M=M, D=D,
        stride_xm=x.stride(0), stride_xd=x.stride(1),
        stride_pr=packed.stride(0), stride_pb=packed.stride(1),
        stride_sr=scales.stride(0), stride_sb=scales.stride(1), BLOCK=32,
    )


@triton.jit
def _unpack_fp4_rows_kernel(
    packed_ptr, scales_ptr, rows_ptr, out_ptr,
    M: tl.constexpr, D: tl.constexpr,
    stride_pr, stride_pb, stride_sr, stride_sb, stride_om, stride_od,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(rows_ptr + m)
    byte = tl.load(packed_ptr + row * stride_pr + (offs // 2) * stride_pb)
    code = tl.where((offs & 1) == 0, byte & 15, byte >> 4)
    scale = tl.load(scales_ptr + row * stride_sr + b * stride_sb)
    value = decode_e2m1(code) * scale
    tl.store(out_ptr + m * stride_om + offs * stride_od, value)


def unpack_fp4_rows(
    packed: torch.Tensor, scales: torch.Tensor, rows: torch.Tensor, D: int
) -> torch.Tensor:
    """Gather/dequantize selected cache rows into BF16."""
    rows = rows.reshape(-1).to(torch.int64)
    out = torch.empty((rows.numel(), D), dtype=torch.bfloat16, device=packed.device)
    if packed.device.type == "cpu":
        byte = packed.index_select(0, rows)
        code = torch.empty((rows.numel(), D), dtype=torch.uint8)
        code[:, 0::2], code[:, 1::2] = byte & 15, byte >> 4
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
        value = lut[(code & 7).long()] * torch.where(code & 8 != 0, -1.0, 1.0)
        return (value.view(-1, D // 32, 32) * scales.index_select(0, rows).unsqueeze(-1)).view(-1, D).to(torch.bfloat16)
    _unpack_fp4_rows_kernel[(rows.numel(), D // 32)](
        packed, scales, rows, out, M=rows.numel(), D=D,
        stride_pr=packed.stride(0), stride_pb=packed.stride(1),
        stride_sr=scales.stride(0), stride_sb=scales.stride(1),
        stride_om=out.stride(0), stride_od=out.stride(1), BLOCK=32,
    )
    return out


__all__ = ["decode_e2m1", "pack_fp4_rows", "unpack_fp4_rows"]
