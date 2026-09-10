"""Torch reference numerics for native V4.1 research execution.

Quantized weights remain packed in the disk/device cache. A projection is
dequantized only while computing it; activations follow the upstream rounding.
"""
import torch
import torch.nn.functional as F

from sparklab.models.deepseek_v4.ops import (
    apply_rotary_emb, precompute_freqs_cis, hc_split_sinkhorn,
)


def fp8_roundtrip(x, block=32):
    z = x.float().unflatten(-1, (-1, block))
    scale = torch.exp2(torch.ceil(torch.log2(z.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    return ((z / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).flatten(-2).to(x.dtype)


def fp4_roundtrip(x, block=32, e4m3_scale=False):
    z = x.float().unflatten(-1, (-1, block))
    amax = z.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        scale = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6 * 2**-126) / 6)))
    value = (z / scale).clamp(-6, 6)
    # E2M1 round-to-nearest-even: the even code wins exact midpoint ties.
    levels = value.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    distance = (value.abs().unsqueeze(-1) - levels).abs()
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    code = order[distance[..., order].argmin(-1)]
    quantized = torch.copysign(levels[code], value)
    return (quantized * scale).flatten(-2).to(x.dtype)


def dequant(weight, scale, fp4=False):
    scales = torch.exp2(scale.view(torch.uint8).float() - 127)
    if fp4:
        packed = weight.view(torch.uint8)
        code = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        levels = weight.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                                   0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
        return levels[code.long()] * scales.repeat_interleave(32, dim=-1)
    return weight.float() * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)[:weight.shape[0], :weight.shape[1]]


def linear(store, name, x):
    weight = store.get(name + ".weight")
    if name + ".scale" in store.metadata:
        scale = store.get(name + ".scale")
        fp4 = weight.dtype in (torch.uint8, torch.int8)
        weight = dequant(weight, scale, fp4)
        return F.linear(fp8_roundtrip(x).float(), weight).to(x.dtype)
    return F.linear(x, weight.to(x.dtype))


def norm(store, name, x, eps):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (y * store.get(name + ".weight").float()).to(x.dtype)


def sparse_attention(q, kv, sink, scale):
    scores = torch.einsum("hd,td->ht", q.float(), kv.float()) * scale
    scores = torch.cat((scores, sink.float().unsqueeze(-1)), dim=-1)
    weights = scores.softmax(-1)[..., :-1]
    return (weights @ kv.float()).to(q.dtype)


def candidate_mask(scores, topk_blocks, block_size):
    width = scores.numel()
    blocks = F.pad(scores, (0, -width % block_size), value=-torch.inf).view(-1, block_size).amax(-1)
    blocks[-1] = torch.inf  # always retain the newest partial block
    top = blocks.topk(min(topk_blocks, blocks.numel()))
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(0, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size)[:width]


__all__ = ["apply_rotary_emb", "precompute_freqs_cis", "hc_split_sinkhorn",
           "fp8_roundtrip", "fp4_roundtrip", "dequant", "linear", "norm",
           "sparse_attention", "candidate_mask"]
