"""Packed Prism ternary W2A16 kernels and normalized block Hadamard transform.

Layouts follow PrismML-Eng/llama.cpp (MIT), commit
1a07bfa5f4144274c8f1c9963821dd9d9a51854b, ggml/src/ggml-quants.c.
Resident weights remain packed. Large prefills use bounded per-projection BF16
scratch for cuBLAS; decode and small prefills unpack directly in registers.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _values(W, row, k, K: tl.constexpr, TYPE: tl.constexpr):
    SIZE: tl.constexpr = 34 if TYPE == 142 else 28
    base = row * (K // 128 * SIZE) + k // 128 * SIZE
    SCALE_OFFSET: tl.constexpr = 0 if TYPE == 142 else 26
    lo = tl.load(W + base + SCALE_OFFSET).to(tl.uint16)
    hi = tl.load(W + base + SCALE_OFFSET + 1).to(tl.uint16)
    d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    j = k % 128
    if TYPE == 142:
        b = tl.load(W + base + 2 + j // 4).to(tl.int32)
        v = ((b >> (2 * (j % 4))) & 3) - 1
    else:
        offset = tl.where(
            j < 80, j % 16, tl.where(j < 120, 16 + (j - 80) % 8, 24 + (j - 120) % 2)
        )
        n = tl.where(j < 80, j // 16, tl.where(j < 120, (j - 80) // 8, (j - 120) // 2))
        pow3 = tl.where(
            n == 0,
            1,
            tl.where(n == 1, 3, tl.where(n == 2, 9, tl.where(n == 3, 27, 81))),
        )
        b = tl.load(W + base + offset).to(tl.int32)
        v = (((b * pow3) & 255) * 3 >> 8) - 1
    return d * v


@tr.jit
def _gemv(
    X,
    W,
    P,
    K: tl.constexpr,
    N: tl.constexpr,
    TYPE: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    # Shape constraints are validated at dispatch; dimensions divide the tiles.
    w = _values(W, rows[:, None], k[None, :], K, TYPE)
    x = tl.load(X + tl.program_id(2) * K + k)
    out = tl.sum(w * x[None, :].to(tl.float32), 1)
    tl.store(P + tl.program_id(2) * N * SPLITS + rows * SPLITS + tl.program_id(1), out)


@tr.jit
def _gemv_blocks(
    X,
    W,
    P,
    K: tl.constexpr,
    N: tl.constexpr,
    TYPE: tl.constexpr,
    GROUPS: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    # Decode each packed byte once and reuse it across its four/five weights.
    # Keep the group scale outside the inner dot product.
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    groups = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    slot = tl.arange(0, 32)
    SIZE: tl.constexpr = 34 if TYPE == 142 else 28
    SO: tl.constexpr = 0 if TYPE == 142 else 26
    QO: tl.constexpr = 2 if TYPE == 142 else 0
    base = rows[:, None] * (K // 128 * SIZE) + groups[None, :] * SIZE
    lo = tl.load(W + base + SO).to(tl.uint16)
    hi = tl.load(W + base + SO + 1).to(tl.uint16)
    scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    valid = slot < (32 if TYPE == 142 else 26)
    q = tl.load(
        W + base[:, :, None] + QO + slot[None, None, :], valid[None, None, :], 0
    ).to(tl.int32)
    acc = tl.full((BN, GROUPS, 32), 0, tl.float32)
    for digit in tl.static_range(4 if TYPE == 142 else 5):
        if TYPE == 142:
            j = slot * 4 + digit
            v = ((q >> (2 * digit)) & 3) - 1
            mask = valid
        else:
            j = tl.where(
                slot < 16,
                slot + 16 * digit,
                tl.where(
                    slot < 24, 80 + slot - 16 + 8 * digit, 120 + slot - 24 + 2 * digit
                ),
            )
            v = (((q * (3**digit)) & 255) * 3 >> 8) - 1
            mask = valid & ((digit < 4) | (slot < 24))
        x = tl.load(
            X + tl.program_id(2) * K + groups[:, None] * 128 + j[None, :],
            mask[None, :],
            0,
        ).to(tl.float32)
        acc += v.to(tl.float32) * x[None, :, :]
    out = tl.sum(tl.sum(acc, 2) * scale, 1)
    tl.store(P + tl.program_id(2) * N * SPLITS + rows * SPLITS + tl.program_id(1), out)


@tr.jit
def _gemm(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TYPE: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 128,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(tr.cdiv(K, BK)):
        k = base * BK + kk
        x = tl.load(X + m[:, None] * K + k[None, :], m[:, None] < M, 0)
        w = _values(W, n[None, :], k[:, None], K, TYPE).to(x.dtype)
        acc += tl.dot(x, w)
    tl.store(Y + m[:, None] * N + n[None, :], acc, m[:, None] < M)


@tr.jit
def _rows(W, IDS, Y, K: tl.constexpr, TYPE: tl.constexpr, BK: tl.constexpr = 128):
    r = tl.program_id(0)
    idx = tl.load(IDS + r)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    w = _values(W, idx, k, K, TYPE)
    tl.store(Y + r * K + k, w)


@tr.jit
def _dequant(W, Y, K: tl.constexpr, N: tl.constexpr, TYPE: tl.constexpr):
    n = tl.program_id(0) * 16 + tl.arange(0, 16)
    k = tl.program_id(1) * 128 + tl.arange(0, 128)
    values = _values(W, n[:, None], k[None, :], K, TYPE)
    tl.store(Y + n[:, None] * K + k[None, :], values)


@tr.jit
def _rotate(X, S, Y, K: tl.constexpr, B: tl.constexpr, INVERSE: tl.constexpr):
    i = tl.arange(0, B)
    offset = tl.program_id(0) * B + i
    signs = tl.load(S + offset % K).to(tl.float32)
    v = tl.load(X + offset).to(tl.float32)
    if not INVERSE:
        v *= signs
    for bit in tl.static_range(0, B.bit_length() - 1):
        other = tl.gather(v, i ^ (1 << bit), 0)
        v = tl.where((i & (1 << bit)) == 0, v + other, other - v)
    v *= 1.0 / tl.sqrt(float(B))
    if INVERSE:
        v *= signs
    tl.store(Y + offset, v)


def rotate(x, signs, block=1024, inverse=False):
    if x.shape[-1] % block or signs.numel() != x.shape[-1]:
        raise ValueError("Invalid Bonsai Hadamard geometry")
    x = x.contiguous()
    y = torch.empty_like(x)
    _rotate[(x.numel() // block,)](x, signs, y, x.shape[-1], block, inverse)
    return y


def linear(x, weight, kind):
    if kind not in (142, 143):
        raise ValueError(f"Unsupported ternary type {kind}")
    x = x.contiguous()
    m, k = x.shape
    n = weight.shape[0]
    if k % 128 or n % 32:
        raise ValueError("Bonsai matrices require K divisible by 128 and N by 32")
    if m >= 64:
        # One projection, never the entire model. The temporary dies after GEMM.
        dense = torch.empty((n, k), device=x.device, dtype=x.dtype)
        _dequant[(n // 16, k // 128)](weight, dense, k, n, kind)
        return torch.nn.functional.linear(x, dense)
    if m <= 4:
        bk = 1024 if k % 1024 == 0 else 128
        splits = k // bk
        partial = torch.empty((m, n, splits), device=x.device, dtype=torch.float32)
        if kind == 143:
            _gemv_blocks[(n // 16, splits, m)](
                x, weight, partial, k, n, kind, bk // 128, 16, splits
            )
        else:
            _gemv[(n // 4, splits, m)](x, weight, partial, k, n, kind, bk, 4, splits)
        return partial.sum(-1).to(x.dtype)
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _gemm[(tr.cdiv(m, 16), n // 32)](x, weight, y, m, n, k, kind)
    return y


def embedding(ids, weight, kind, width, dtype=torch.bfloat16):
    ids = ids.contiguous()
    y = torch.empty((*ids.shape, width), device=ids.device, dtype=dtype)
    _rows[(ids.numel(), width // 128)](weight, ids, y, width, kind)
    return y
