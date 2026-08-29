"""Triton GPU decoder for the DF11 (lossless BF16) format produced by ``df11_compress``.

Huffman codes are variable-length, so the exponent stream of a chunk must be decoded
sequentially. We therefore map **one chunk (``DF11_CHUNK`` weights) to one program** and run
many chunks in parallel -- a big tensor yields hundreds of thousands of chunks, so the grid
saturates the GPU even though each program is serial. The decoded BF16 is materialized into a
caller-provided buffer (fixed shape => CUDA-graph safe) and discarded after the matmul.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .df11 import DF11_CHUNK, DF11_LMAX


@triton.jit
def _df11_decode_kernel(
    low8_ptr,        # [N]       uint8  : (sign<<7)|mantissa, original order
    bitstream_ptr,   # [NWORDS]  int32  : MSB-first packed exponent codes
    chunk_start_ptr, # [G]       int32  : starting bit of each chunk
    lut_ptr,         # [2^LMAX]  int32  : peek-bits -> (symbol << 8) | length
    out_ptr,         # [N]       int16  : reconstructed bf16 bit pattern
    N,
    G,               # number of chunks (= interleave stride)
    NWORDS,
    ROWS: tl.constexpr,
    LMAX: tl.constexpr,
    BLOCK: tl.constexpr,   # chunks decoded in parallel per program (one per lane)
):
    # Each lane owns one chunk j and decodes its symbols serially (Huffman is variable-length).
    # Interleaving means lane j handles position i*G+j at step i, so consecutive lanes touch
    # consecutive output addresses => coalesced low8 reads and bf16 writes.
    #
    # Each lane keeps a 64-bit MSB-aligned bit buffer (`acc`, with `nbits` valid high bits) and
    # only loads a new 32-bit stream word when the buffer would drop below a full peek window.
    # That turns the old 2-loads-per-symbol into ~1-load-per-2.6-symbols of the (scattered)
    # bitstream, which is what was bounding the kernel; low8/out stay coalesced.
    cid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cvalid = cid < G
    start_bit = tl.load(chunk_start_ptr + cid, mask=cvalid, other=0)
    w0 = start_bit >> 5
    off0 = start_bit & 31
    word0 = tl.load(bitstream_ptr + w0, mask=cvalid & (w0 < NWORDS), other=0).to(tl.uint32)
    acc = (word0 << off0).to(tl.int64) << 32  # chunk's first bit -> acc bit 63
    nbits = 32 - off0
    wordidx = w0 + 1
    peek_sh = 64 - LMAX
    peek_mask = (1 << LMAX) - 1
    for i in tl.range(0, ROWS):
        idx = i * G + cid
        valid = cvalid & (idx < N)
        # Refill: keep >= 32 valid bits (>= LMAX) before each peek.
        need = nbits < 32
        word = tl.load(
            bitstream_ptr + wordidx, mask=need & cvalid & (wordidx < NWORDS), other=0
        ).to(tl.uint32)
        sh_ins = tl.maximum(32 - nbits, 0)
        acc = acc | tl.where(need, word.to(tl.int64) << sh_ins, 0)
        nbits = tl.where(need, nbits + 32, nbits)
        wordidx = tl.where(need, wordidx + 1, wordidx)
        # Peek LMAX bits (masking the low LMAX bits makes the arithmetic >> sign-fill irrelevant).
        v = ((acc >> peek_sh) & peek_mask).to(tl.int32)
        combined = tl.load(lut_ptr + v)
        length = combined & 0xFF
        sym = (combined >> 8) & 0xFF
        acc = acc << length
        nbits = nbits - length
        low = tl.load(low8_ptr + idx, mask=valid, other=0).to(tl.int32)
        sign = (low >> 7) & 1
        mant = low & 0x7F
        u16 = (sign << 15) | (sym << 7) | mant
        tl.store(out_ptr + idx, u16.to(tl.int16), mask=valid)


def df11_decompress(c: dict, out: torch.Tensor | None = None, block: int = 128) -> torch.Tensor:
    """Decode a DF11 bundle back to its exact BF16 tensor.

    If ``out`` (an int16 buffer of shape ``[OUT, IN]``) is given it is filled in place; this
    lets callers preallocate and reuse it under CUDA graphs. Returns a bf16 view.
    """
    out_features, in_features, n, num_chunks, rows, lmax = c["meta"]
    if out is None:
        out = torch.empty((out_features, in_features), dtype=torch.int16, device=c["low8"].device)
    assert out.dtype == torch.int16 and out.is_contiguous()
    grid = ((num_chunks + block - 1) // block,)
    _df11_decode_kernel[grid](
        c["low8"], c["bitstream"], c["chunk_start"], c["lut"],
        out, n, num_chunks, c["bitstream"].numel(),
        ROWS=rows, LMAX=lmax, BLOCK=block, num_warps=block // 32,
    )
    return out.view(torch.bfloat16)


@triton.jit
def _df11_gather_kernel(
    low8_ptr,        # [R*C]     uint8  : (sign<<7)|mantissa, row-major
    bitstream_ptr,   # [NWORDS]  int32  : MSB-first packed exponent codes
    chunk_start_ptr, # [R]       int64  : starting bit of each row's codes
    lut_ptr,         # [2^LMAX]  int32  : peek-bits -> (symbol << 8) | length
    ids_ptr,         # [T]              : row (token) id to decode for each output slot
    out_ptr,         # [T*C]     int16  : reconstructed bf16 bit pattern
    T,
    C,               # embedding dim (row length)
    NWORDS,
    LMAX: tl.constexpr,
    BLOCK: tl.constexpr,   # rows decoded in parallel per program (one per lane)
):
    # One lane decodes one gathered row r = ids[t] serially, reading its contiguous code stream
    # via the same 64-bit bit-buffer as the matmul decoder. Embedding lookups touch very few rows
    # (one per token), so per-row serial decode is cheap and coalescing is unimportant here.
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tvalid = t < T
    r = tl.load(ids_ptr + t, mask=tvalid, other=0).to(tl.int64)
    start_bit = tl.load(chunk_start_ptr + r, mask=tvalid, other=0)  # int64
    w0 = start_bit >> 5
    off0 = (start_bit & 31).to(tl.int32)
    word0 = tl.load(bitstream_ptr + w0, mask=tvalid & (w0 < NWORDS), other=0).to(tl.uint32)
    acc = (word0 << off0).to(tl.int64) << 32
    nbits = 32 - off0
    wordidx = w0 + 1
    peek_sh = 64 - LMAX
    peek_mask = (1 << LMAX) - 1
    row_base = r * C
    out_base = t.to(tl.int64) * C
    for k in tl.range(0, C):
        need = nbits < 32
        word = tl.load(
            bitstream_ptr + wordidx, mask=need & tvalid & (wordidx < NWORDS), other=0
        ).to(tl.uint32)
        sh_ins = tl.maximum(32 - nbits, 0)
        acc = acc | tl.where(need, word.to(tl.int64) << sh_ins, 0)
        nbits = tl.where(need, nbits + 32, nbits)
        wordidx = tl.where(need, wordidx + 1, wordidx)
        v = ((acc >> peek_sh) & peek_mask).to(tl.int32)
        combined = tl.load(lut_ptr + v)
        length = combined & 0xFF
        sym = (combined >> 8) & 0xFF
        acc = acc << length
        nbits = nbits - length
        low = tl.load(low8_ptr + row_base + k, mask=tvalid, other=0).to(tl.int32)
        sign = (low >> 7) & 1
        mant = low & 0x7F
        u16 = (sign << 15) | (sym << 7) | mant
        tl.store(out_ptr + out_base + k, u16.to(tl.int16), mask=tvalid)


def df11_gather_decode(
    c: dict, ids: torch.Tensor, out: torch.Tensor | None = None, block: int = 128
) -> torch.Tensor:
    """Decode the embedding rows ``ids`` from a row-contiguous DF11 bundle into ``[T, C]`` bf16.

    Pass a preallocated int16 ``out`` to stay CUDA-graph safe (decode batch size is fixed).
    """
    rows, cols, n, lmax = c["meta"]
    ids = ids.reshape(-1).contiguous()
    t = ids.numel()
    if out is None:
        out = torch.empty((t, cols), dtype=torch.int16, device=c["low8"].device)
    assert out.dtype == torch.int16 and out.is_contiguous()
    grid = ((t + block - 1) // block,)
    _df11_gather_kernel[grid](
        c["low8"], c["bitstream"], c["chunk_start"], c["lut"], ids, out,
        t, cols, c["bitstream"].numel(), LMAX=lmax, BLOCK=block, num_warps=block // 32,
    )
    return out.view(torch.bfloat16)


__all__ = ["df11_decompress", "df11_gather_decode"]
