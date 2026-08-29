# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/gemma4_fused_ops.py
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_dual_rmsnorm_residual_kernel(
    X1_ptr,
    W1_ptr,
    X2_ptr,
    W2_ptr,
    W3_ptr,
    Residual_ptr,
    Scalar_ptr,
    Out_ptr,
    stride_x1,
    stride_x2,
    stride_r,
    stride_o,
    N,
    eps1,
    eps2,
    eps3,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x1 = tl.load(X1_ptr + row * stride_x1 + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X2_ptr + row * stride_x2 + cols, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        Residual_ptr + row * stride_r + cols, mask=mask, other=0.0
    ).to(tl.float32)

    rrms1 = tl.rsqrt(tl.sum(x1 * x1, axis=0) / N + eps1)
    norm1 = x1 * rrms1 * w1

    rrms2 = tl.rsqrt(tl.sum(x2 * x2, axis=0) / N + eps2)
    norm2 = x2 * rrms2 * w2

    combined = norm1 + norm2
    rrms3 = tl.rsqrt(tl.sum(combined * combined, axis=0) / N + eps3)
    norm3 = combined * rrms3 * w3

    scalar = tl.load(Scalar_ptr).to(tl.float32)
    out = (residual + norm3) * scalar
    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


def gemma_dual_rmsnorm_residual_scalar(
    x1: torch.Tensor,
    weight1: torch.Tensor,
    x2: torch.Tensor,
    weight2: torch.Tensor,
    weight3: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps1: float,
    eps2: float,
    eps3: float,
) -> torch.Tensor:
    assert x1.is_cuda and x2.is_cuda and residual.is_cuda
    assert x1.dim() == 2 and x2.dim() == 2 and residual.dim() == 2
    assert x1.stride(-1) == 1 and x2.stride(-1) == 1 and residual.stride(-1) == 1
    tokens, hidden_size = x1.shape
    assert x2.shape == x1.shape and residual.shape == x1.shape
    assert weight1.shape[-1] == hidden_size
    assert weight2.shape[-1] == hidden_size
    assert weight3.shape[-1] == hidden_size

    out = torch.empty_like(x1)
    _gemma_dual_rmsnorm_residual_kernel[(tokens,)](
        x1,
        weight1,
        x2,
        weight2,
        weight3,
        residual,
        scalar,
        out,
        x1.stride(0),
        x2.stride(0),
        residual.stride(0),
        out.stride(0),
        hidden_size,
        eps1,
        eps2,
        eps3,
        BLOCK_SIZE=triton.next_power_of_2(hidden_size),
    )
    return out


# REUSE-NOTE(gpt-oss): the sort + softmax-over-topK core below is a generic fused
# softmax+top-k router. GPT-OSS adds a trimmed local copy (without per_expert_scale)
# under kernel/triton for its MXFP4 MoE. These should be consolidated into one shared
# fused-routing kernel (per_expert_scale optional via a HAS_SCALE constexpr) later.
@triton.jit
def _gemma4_routing_kernel(
    logits_ptr,
    per_expert_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    stride_logits_t,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E

    logits = tl.load(
        logits_ptr + token_id * stride_logits_t + offs_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    min_i32 = -2147483648
    logit_bits = logits.to(tl.int32, bitcast=True)
    sign = logit_bits >> 31
    key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ min_i32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    packed = ((key.to(tl.int64) & 0x00000000FFFFFFFF) << 32) | offs_e.to(tl.int64)

    sorted_packed = tl.sort(packed, descending=False)
    all_keys = ((sorted_packed >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_packed & 0x00000000FFFFFFFF).to(tl.int32)

    sign_k = all_keys >> 31
    all_bits = tl.where(sign_k < 0, all_keys ^ -1, all_keys ^ min_i32)
    all_logits = all_bits.to(tl.float32, bitcast=True)

    top_mask = offs_e < K
    max_l = tl.max(tl.where(top_mask, all_logits, -float("inf")), axis=0)
    raw_exp = tl.where(top_mask, tl.exp(all_logits - max_l), 0.0)
    denom = tl.sum(raw_exp, axis=0)
    weights = raw_exp / tl.where(denom > 0.0, denom, 1.0)

    scales = tl.load(
        per_expert_scale_ptr + all_ids.to(tl.int64),
        mask=top_mask,
        other=1.0,
    ).to(tl.float32)
    weights = weights * scales

    out_off = token_id * K + offs_e
    tl.store(topk_weights_ptr + out_off, weights, mask=top_mask)
    tl.store(topk_ids_ptr + out_off, all_ids, mask=top_mask)


def gemma4_fused_routing(
    logits: torch.Tensor,
    per_expert_scale: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert logits.is_cuda and logits.dim() == 2
    assert per_expert_scale.is_cuda and per_expert_scale.dim() == 1
    tokens, num_experts = logits.shape
    assert per_expert_scale.shape[0] == num_experts
    assert topk <= num_experts
    assert num_experts <= 1024

    logits = logits.contiguous()
    per_expert_scale = per_expert_scale.contiguous()
    topk_weights = torch.empty(
        (tokens, topk), dtype=torch.float32, device=logits.device
    )
    topk_ids = torch.empty((tokens, topk), dtype=torch.int32, device=logits.device)
    if tokens == 0:
        return topk_weights, topk_ids

    _gemma4_routing_kernel[(tokens,)](
        logits,
        per_expert_scale,
        topk_weights,
        topk_ids,
        logits.stride(0),
        E=num_experts,
        K=topk,
        BLOCK_E=triton.next_power_of_2(num_experts),
        num_warps=1,
    )
    return topk_weights, topk_ids
