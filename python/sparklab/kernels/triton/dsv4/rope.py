"""Fused interleaved RoPE for DeepSeek-V4 decode (borrowed from sglang's Triton rope).

SparkLab's decode rope was a chain of torch ops (``view_as_complex`` -> complex mul ->
``view_as_real`` -> ``copy_``), ~3-5 small ``at::native`` kernels per call, called for q/kv/o
in every layer. This collapses each call into ONE ``@triton.jit`` kernel.

The math is identical to ``ops.apply_rotary_emb_decode``: pairing the interleaved last
``rope_dim`` of ``x`` as ``(real, imag)`` and multiplying by the per-row complex ``freqs``
(``inverse`` uses the conjugate). All compute in fp32, stored back to ``x``'s dtype -- so it is
bit-identical to the torch path (modulo a possible 1-ULP FMA, gated by the decode parity check).

Kernel adapted from sglang ``srt/layers/deepseek_v4_rope.py:apply_rotary_emb_triton_kernel``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_decode_kernel(
    x_ptr, freqs_ptr,
    rope_dim,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_freq_pos, stride_freq_dim,
    IS_INVERSE: tl.constexpr,
    IS_3D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_dim = tl.program_id(2)

    base = pid_batch * stride_x_batch + (pid_head * stride_x_head if IS_3D else 0)
    offs_pair = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_pair < (rope_dim // 2)

    offs_real = base + offs_pair * 2 * stride_x_dim
    offs_imag = base + (offs_pair * 2 + 1) * stride_x_dim
    x_real = tl.load(x_ptr + offs_real, mask=mask, other=0.0).to(tl.float32)
    x_imag = tl.load(x_ptr + offs_imag, mask=mask, other=0.0).to(tl.float32)

    f_off_real = pid_batch * stride_freq_pos + offs_pair * 2 * stride_freq_dim
    f_off_imag = pid_batch * stride_freq_pos + (offs_pair * 2 + 1) * stride_freq_dim
    f_real = tl.load(freqs_ptr + f_off_real, mask=mask, other=0.0)
    f_imag = tl.load(freqs_ptr + f_off_imag, mask=mask, other=0.0)

    if IS_INVERSE:  # multiply by conj(freqs)
        out_real = x_real * f_real + x_imag * f_imag
        out_imag = x_imag * f_real - x_real * f_imag
    else:
        out_real = x_real * f_real - x_imag * f_imag
        out_imag = x_real * f_imag + x_imag * f_real

    tl.store(x_ptr + offs_real, out_real, mask=mask)
    tl.store(x_ptr + offs_imag, out_imag, mask=mask)


def rope_decode_inplace(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place interleaved RoPE on the last ``rope_dim`` of ``x``, per-row complex ``freqs_cis``.

    ``x``: ``[B, 1, (H,) rope_dim]`` (one decode token per row; rope applies to the last dim, which
    may be a non-contiguous slice -- strides are honored). ``freqs_cis``: ``[B, rope_dim//2]``
    complex (already gathered per row). Mutates and returns ``x``.
    """
    xv = x.squeeze(1)  # drop the seq==1 dim -> [B, (H,) rope_dim]
    is_3d = xv.ndim == 3
    if is_3d:
        B, H, rope_dim = xv.shape
    else:
        B, rope_dim = xv.shape
        H = 1
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()  # [B, rope_dim] (cos,sin)
    grid = (B, H, triton.cdiv(rope_dim // 2, 128))
    _rope_decode_kernel[grid](
        xv, freqs_real,
        rope_dim,
        xv.stride(0), xv.stride(1) if is_3d else 0, xv.stride(-1),
        freqs_real.stride(0), freqs_real.stride(1),
        IS_INVERSE=inverse, IS_3D=is_3d, BLOCK_SIZE=128,
    )
    return x


__all__ = ["rope_decode_inplace"]
