"""DFloat11-style lossless BF16 compression (self-implemented, no external deps).

BF16's 8-bit exponent carries only ~2.6 bits of entropy, so we Huffman-code the exponent
per-tensor and leave the sign+mantissa (8 bits) raw. A weight averages ~11 bits instead of
16 (~30% smaller) and decompresses **bit-for-bit** back to the original BF16.

Layout produced by :func:`df11_compress` (all tensors, ready to ship to the GPU):
  - ``low8``        [N]            uint8   : (sign<<7) | mantissa, raw, original order.
  - ``bitstream``   [W]            int32   : exponent Huffman codes, packed MSB-first
                                            (32-bit big-endian words; int32 holds the bits).
  - ``chunk_start`` [G]            int32   : bit offset of each chunk's code stream.
  - ``lut``         [2^DF11_LMAX]  int32   : next-``DF11_LMAX``-bits -> ``(symbol << 8) | length``.

**Interleaved chunking (for coalesced GPU decode).** With ``G`` chunks, chunk ``j`` owns the
weights at flat positions ``{j, j+G, j+2G, ...}``. So at decode step ``i`` lane ``j`` handles
position ``i*G+j``; consecutive lanes => consecutive addresses => coalesced ``low8`` reads and
``bf16`` writes (the bulk of the traffic). Only the per-chunk bitstream peek stays scattered.

Decoding (reference here, Triton in ``df11_decode.py``) reconstructs
``u16 = (sign<<15) | (exp<<7) | mantissa`` and views it as bf16.
"""

from __future__ import annotations

import heapq

import torch

# Max Huffman code length -> direct LUT of 2^DF11_LMAX entries (length-limited to fit SRAM and
# bound the decoder's peek width). DF11_CHUNK weights are decoded sequentially per GPU program.
DF11_LMAX = 12
DF11_CHUNK = 256


def _huffman_lengths(freq: dict[int, int]) -> dict[int, int]:
    """Code length per symbol from a Huffman tree (length >= 1, even for a lone symbol)."""
    if len(freq) == 1:
        return {next(iter(freq)): 1}
    order = 0
    heap: list = []
    for sym, f in freq.items():
        heapq.heappush(heap, (f, order, ("leaf", sym)))
        order += 1
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, order, ("node", n1, n2)))
        order += 1
    lengths: dict[int, int] = {}

    def walk(node, depth):
        if node[0] == "leaf":
            lengths[node[1]] = max(depth, 1)
        else:
            walk(node[1], depth + 1)
            walk(node[2], depth + 1)

    walk(heap[0][2], 0)
    return lengths


def _length_limited_lengths(freq: dict[int, int], lmax: int) -> dict[int, int]:
    """Huffman lengths capped at ``lmax`` by flooring rare-symbol frequencies (stays a valid
    prefix code -> still lossless, just marginally off optimal for the rarest exponents)."""
    f = dict(freq)
    floor = 1
    while True:
        lengths = _huffman_lengths(f)
        if max(lengths.values()) <= lmax:
            return lengths
        floor *= 2
        f = {s: max(c, floor) for s, c in f.items()}


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Canonical Huffman codes: sort by (length, symbol), increment-and-shift."""
    items = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    prev_len = items[0][1]
    for i, (sym, length) in enumerate(items):
        if i > 0:
            code = (code + 1) << (length - prev_len)
            prev_len = length
        codes[sym] = (code, length)
    return codes


def _build_lut(codes: dict[int, tuple[int, int]], lmax: int) -> torch.Tensor:
    """One ``2^lmax`` LUT mapping the next ``lmax`` peeked bits -> ``(symbol << 8) | length``.

    Merging symbol+length into a single int32 entry halves the decoder's LUT traffic to one
    load per symbol.
    """
    size = 1 << lmax
    lut = torch.zeros(size, dtype=torch.int32)
    for sym, (code, length) in codes.items():
        base = code << (lmax - length)
        span = 1 << (lmax - length)
        lut[base : base + span] = (sym << 8) | length
    return lut


def _huffman_tables(
    hist: torch.Tensor, lmax: int, dev: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """From a 256-bin exponent histogram build the decode ``lut`` and the per-symbol
    ``code``/``length`` arrays (on ``dev``) used to pack the bitstream."""
    freq = {int(s): int(hist[s]) for s in torch.nonzero(hist).flatten().tolist()}
    codes = _canonical_codes(_length_limited_lengths(freq, lmax))
    lut = _build_lut(codes, lmax).to(dev)
    code_arr = torch.zeros(256, dtype=torch.int64, device=dev)
    len_arr = torch.zeros(256, dtype=torch.int64, device=dev)
    for sym, (code, length) in codes.items():
        code_arr[sym] = code
        len_arr[sym] = length
    return lut, code_arr, len_arr


def df11_compress(weight: torch.Tensor, chunk: int = DF11_CHUNK, lmax: int = DF11_LMAX) -> dict:
    """Compress a 2-D BF16 ``weight`` into the DF11 tensor bundle (see module docstring)."""
    assert weight.dtype == torch.bfloat16 and weight.dim() == 2
    out_features, in_features = weight.shape
    dev = weight.device
    bits = weight.reshape(-1).view(torch.int16).to(torch.int64) & 0xFFFF
    n = bits.numel()
    exp = ((bits >> 7) & 0xFF).to(torch.int64)
    low8 = (((bits >> 15) & 1) << 7 | (bits & 0x7F)).to(torch.uint8)

    # Per-tensor canonical Huffman over the exponent histogram.
    hist = torch.bincount(exp, minlength=256)
    lut, code_arr, len_arr = _huffman_tables(hist, lmax, dev)

    w_code = code_arr[exp]  # [N]
    w_len = len_arr[exp]  # [N]

    # Interleaved layout: G chunks, chunk j owns positions {j, j+G, ...}. Bit offset of a
    # position p=i*G+j is chunk_start[j] + (sum of code lengths of rows 0..i-1 in column j).
    num_chunks = (n + chunk - 1) // chunk  # G
    rows = (n + num_chunks - 1) // num_chunks  # max symbols per chunk (~= chunk)
    len_pad = torch.zeros(rows * num_chunks, dtype=torch.int64, device=dev)
    len_pad[:n] = w_len
    len_2d = len_pad.view(rows, num_chunks)  # [i, j] -> position i*G+j (0-length where padded)
    col_bits = len_2d.sum(0)  # [G] bits per chunk
    chunk_start = torch.zeros(num_chunks, dtype=torch.int64, device=dev)
    chunk_start[1:] = torch.cumsum(col_bits, 0)[:-1]  # chunks concatenated in j order
    col_prefix = torch.zeros_like(len_2d)
    col_prefix[1:] = torch.cumsum(len_2d, 0)[:-1]  # exclusive prefix down each column
    off = (chunk_start[None, :] + col_prefix).reshape(-1)[:n]  # [N] bit offset per position
    total_bits = int(chunk_start[-1].item() + col_bits[-1].item()) if n else 0

    # Scatter each MSB-first code into 32-bit words. A code touches word w (high part) and
    # possibly w+1 (low part); the two parts occupy disjoint bits, so index_add (sum) == OR.
    num_words = (total_bits + 31) // 32 + 2  # +2 pad so the decoder can always read w and w+1
    o = off & 31
    word = off >> 5
    bits_in_w = torch.minimum(w_len, 32 - o)
    rem = w_len - bits_in_w
    high = (w_code >> rem) << (32 - o - bits_in_w)
    low = (w_code & ((1 << rem) - 1)) << (32 - rem)
    words = torch.zeros(num_words, dtype=torch.int64, device=dev)
    words.index_add_(0, word, high)
    words.index_add_(0, word + 1, low)
    bitstream = (words & 0xFFFFFFFF).to(torch.int32)

    return {
        "low8": low8.to(dev),
        "bitstream": bitstream.to(dev),
        "chunk_start": chunk_start.to(torch.int32).to(dev),
        "lut": lut.to(dev),
        "meta": (out_features, in_features, n, num_chunks, rows, lmax),
    }


def df11_decompress_ref(c: dict) -> torch.Tensor:
    """Pure-Python reference decoder (slow; for correctness tests on small tensors)."""
    out_features, in_features, n, num_chunks, rows, lmax = c["meta"]
    stream = [int(x) & 0xFFFFFFFF for x in c["bitstream"].tolist()]
    chunk_start = c["chunk_start"].tolist()
    lut = c["lut"].tolist()
    low8 = c["low8"].tolist()
    nwords = len(stream)
    g = num_chunks
    exp_out = [0] * n

    def peek(pos: int) -> int:
        w = pos >> 5
        o = pos & 31
        hi = stream[w] if w < nwords else 0
        lo = stream[w + 1] if w + 1 < nwords else 0
        combined = (hi << 32) | lo
        return (combined >> (64 - o - lmax)) & ((1 << lmax) - 1)

    for j in range(g):
        pos = chunk_start[j]
        for i in range(rows):
            p = i * g + j
            if p >= n:
                break
            v = peek(pos)
            combined = lut[v]
            exp_out[p] = combined >> 8
            pos += combined & 0xFF

    exp_t = torch.tensor(exp_out, dtype=torch.int64)
    low_t = torch.tensor(low8, dtype=torch.int64)
    sign = (low_t >> 7) & 1
    mant = low_t & 0x7F
    u16 = ((sign << 15) | (exp_t << 7) | mant).to(torch.int16)
    return u16.view(torch.bfloat16).reshape(out_features, in_features)


def df11_compress_rows(
    weight: torch.Tensor, lmax: int = DF11_LMAX, row_block: int = 8192
) -> dict:
    """Row-contiguous DF11: every ROW is an independently-decodable chunk.

    Unlike :func:`df11_compress` (interleaved, tuned for decoding the *whole* matrix at once),
    this lays each row's codes out contiguously with a per-row ``chunk_start``, so a gather
    kernel can decode only the looked-up rows -- the embedding-table use case, where decode
    touches a single row per token. ``chunk_start`` is **int64** (a full vocab table's bit
    offsets exceed 2^31). Compression is row-blocked to bound peak memory on the big table.
    """
    assert weight.dtype == torch.bfloat16 and weight.dim() == 2
    rows, cols = weight.shape
    dev = weight.device
    n = rows * cols
    low8 = torch.empty(n, dtype=torch.uint8, device=dev)

    # Pass 1: exponent histogram + raw sign|mantissa (row-blocked).
    hist = torch.zeros(256, dtype=torch.int64, device=dev)
    for r0 in range(0, rows, row_block):
        r1 = min(r0 + row_block, rows)
        b = weight[r0:r1].reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
        hist += torch.bincount((b >> 7) & 0xFF, minlength=256)
        low8[r0 * cols : r1 * cols] = (((b >> 15) & 1) << 7 | (b & 0x7F)).to(torch.uint8)
    lut, code_arr, len_arr = _huffman_tables(hist, lmax, dev)

    # Pass 2: per-row bit lengths -> exclusive-cumsum chunk_start (int64).
    row_bits = torch.empty(rows, dtype=torch.int64, device=dev)
    for r0 in range(0, rows, row_block):
        r1 = min(r0 + row_block, rows)
        exp = (weight[r0:r1].reshape(-1).view(torch.int16).to(torch.int32) >> 7) & 0xFF
        row_bits[r0:r1] = len_arr[exp].view(r1 - r0, cols).sum(1)
    chunk_start = torch.zeros(rows, dtype=torch.int64, device=dev)
    chunk_start[1:] = torch.cumsum(row_bits, 0)[:-1]
    total_bits = int(chunk_start[-1].item() + row_bits[-1].item()) if rows else 0
    num_words = (total_bits + 31) // 32 + 2
    words = torch.zeros(num_words, dtype=torch.int64, device=dev)

    # Pass 3: pack codes MSB-first; off[p] = chunk_start[row] + within-row prefix (row-blocked).
    for r0 in range(0, rows, row_block):
        r1 = min(r0 + row_block, rows)
        b = weight[r0:r1].reshape(-1).view(torch.int16).to(torch.int64) & 0xFFFF
        exp = (b >> 7) & 0xFF
        w_code = code_arr[exp]
        w_len = len_arr[exp]
        len_2d = w_len.view(r1 - r0, cols)
        row_prefix = torch.zeros_like(len_2d)
        row_prefix[:, 1:] = torch.cumsum(len_2d, 1)[:, :-1]
        off = (chunk_start[r0:r1, None] + row_prefix).reshape(-1)
        o = off & 31
        word = off >> 5
        bits_in_w = torch.minimum(w_len, 32 - o)
        rem = w_len - bits_in_w
        high = (w_code >> rem) << (32 - o - bits_in_w)
        low = (w_code & ((1 << rem) - 1)) << (32 - rem)
        words.index_add_(0, word, high)
        words.index_add_(0, word + 1, low)
    bitstream = (words & 0xFFFFFFFF).to(torch.int32)

    return {
        "low8": low8,
        "bitstream": bitstream,
        "chunk_start": chunk_start,  # int64 bit offsets
        "lut": lut,
        "meta": (rows, cols, n, lmax),
    }


def df11_decompress_rows_ref(c: dict) -> torch.Tensor:
    """Pure-Python reference decoder for the row-contiguous layout (small tensors only)."""
    rows, cols, n, lmax = c["meta"]
    stream = [int(x) & 0xFFFFFFFF for x in c["bitstream"].tolist()]
    chunk_start = c["chunk_start"].tolist()
    lut = c["lut"].tolist()
    low8 = c["low8"].tolist()
    nwords = len(stream)
    exp_out = [0] * n

    def peek(pos: int) -> int:
        w = pos >> 5
        o = pos & 31
        hi = stream[w] if w < nwords else 0
        lo = stream[w + 1] if w + 1 < nwords else 0
        return (((hi << 32) | lo) >> (64 - o - lmax)) & ((1 << lmax) - 1)

    for r in range(rows):
        pos = chunk_start[r]
        for k in range(cols):
            combined = lut[peek(pos)]
            exp_out[r * cols + k] = combined >> 8
            pos += combined & 0xFF

    exp_t = torch.tensor(exp_out, dtype=torch.int64)
    low_t = torch.tensor(low8, dtype=torch.int64)
    u16 = (((low_t >> 7) & 1) << 15 | (exp_t << 7) | (low_t & 0x7F)).to(torch.int16)
    return u16.view(torch.bfloat16).reshape(rows, cols)


__all__ = [
    "df11_compress",
    "df11_compress_rows",
    "df11_decompress_ref",
    "df11_decompress_rows_ref",
    "DF11_LMAX",
    "DF11_CHUNK",
]
