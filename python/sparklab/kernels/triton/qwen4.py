"""Small fused kernels for Qwen3.8/Qwen4-Exp hot paths."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TL_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


@triton.jit
def _round_fp32_to_bf16_fp32(value):
    """Force an observable BF16 round while keeping the value in registers."""
    bits = value.to(tl.int32, bitcast=True)
    bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + bias) & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _grouped_plus_one_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    group_size: tl.constexpr,
    groups: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
    out_dtype: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    offsets = tl.arange(0, block)
    mask = offsets < group_size
    x = tl.load(
        x_ptr + row * group_size + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / group_size + eps)
    weight = tl.load(
        weight_ptr + (row % groups) * group_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out_ptr + row * group_size + offsets,
        (x * inv * (1.0 + weight)).to(out_dtype),
        mask=mask,
    )


@triton.jit
def _scaled_silu_kernel(
    x_ptr,
    out_ptr,
    size,
    divisor: tl.constexpr,
    block: tl.constexpr,
    out_dtype: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < size
    value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # The divisor is a power of two for Qwen4, so the BF16 intermediate has
    # exactly the same value as the eager division before SiLU.
    value = (value / divisor).to(out_dtype).to(tl.float32)
    tl.store(
        out_ptr + offsets,
        (value * tl.sigmoid(value)).to(out_dtype),
        mask=mask,
    )


@triton.jit
def _hyper_mix_kernel(
    logits_ptr,
    normed_ptr,
    out_ptr,
    rows,
    width: tl.constexpr,
    group_size: tl.constexpr,
    groups: tl.constexpr,
    block: tl.constexpr,
    out_dtype: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    mask = (row < rows) & (offsets < group_size)
    mixed = tl.zeros((block,), dtype=tl.float32)
    for group in range(groups):
        index = row * width + group * group_size + offsets
        logits = tl.load(logits_ptr + index, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(normed_ptr + index, mask=mask, other=0.0).to(tl.float32)
        gate = tl.sigmoid(logits).to(out_dtype).to(tl.float32)
        # Eager BF16 multiplication rounds before the mean reduction.
        mixed += (gate * normed).to(out_dtype).to(tl.float32)
    tl.store(
        out_ptr + row * group_size + offsets,
        (mixed / groups).to(out_dtype),
        mask=mask,
    )


@triton.jit
def _hyper_injection_weights_kernel(
    logits_ptr,
    out_ptr,
    size,
    divisor: tl.constexpr,
    block: tl.constexpr,
    out_dtype: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < size
    value = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    value = (value / divisor).to(out_dtype).to(tl.float32)
    gate = tl.sigmoid(value).to(out_dtype).to(tl.float32)
    tl.store(out_ptr + offsets, (2.0 * gate).to(out_dtype), mask=mask)


@triton.jit
def _hyper_residual_inject_kernel(
    branch_ptr,
    hyper_ptr,
    weights_ptr,
    out_ptr,
    size,
    group_size: tl.constexpr,
    groups: tl.constexpr,
    block: tl.constexpr,
    out_dtype: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < size
    group = (offsets % (group_size * groups)) // group_size
    row = offsets // (group_size * groups)
    hidden = offsets % group_size
    branch = tl.load(
        branch_ptr + row * group_size + hidden, mask=mask, other=0.0
    ).to(tl.float32)
    weight = tl.load(
        weights_ptr + row * groups + group, mask=mask, other=0.0
    ).to(tl.float32)
    residual = tl.load(hyper_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # Eager materializes the BF16 multiply before launching the residual add.
    # A cast-only expression is optimized back into an fp32 FMA by Triton, so
    # force the intervening round with the IEEE-754 round-to-nearest-even bit
    # operation used for fp32 -> BF16 conversion.
    update = _round_fp32_to_bf16_fp32(branch * weight)
    tl.store(out_ptr + offsets, (residual + update).to(out_dtype), mask=mask)


@triton.jit
def _ple_conv_decode_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    indices_ptr,
    out_ptr,
    width,
    stride_state_slot,
    stride_state_channel,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    state_len: tl.constexpr,
    state_block: tl.constexpr,
    block: tl.constexpr,
    out_dtype: tl.constexpr,
):
    batch = tl.program_id(0)
    channels = tl.program_id(1) * block + tl.arange(0, block)
    channel_mask = channels < width
    slot = tl.load(indices_ptr + batch)
    current = tl.load(
        x_ptr + batch * width + channels, mask=channel_mask, other=0.0
    ).to(tl.float32)
    state_base = (
        state_ptr + slot * stride_state_slot + channels * stride_state_channel
    )

    taps = tl.arange(0, kernel_size)
    positions = taps * dilation
    from_state = positions < state_len
    history = tl.load(
        state_base[:, None] + positions[None, :],
        mask=channel_mask[:, None] & from_state[None, :],
        other=0.0,
    ).to(tl.float32)
    history = tl.where(from_state[None, :], history, current[:, None])
    weight = tl.load(
        weight_ptr + channels[:, None] * kernel_size + taps[None, :],
        mask=channel_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    conv = tl.sum(history * weight, axis=1).to(out_dtype).to(tl.float32)
    tl.store(
        out_ptr + batch * width + channels,
        (conv * tl.sigmoid(conv)).to(out_dtype),
        mask=channel_mask,
    )

    positions = tl.arange(0, state_block)
    state_mask = channel_mask[:, None] & (positions[None, :] < state_len - 1)
    shifted = tl.load(
        state_base[:, None] + positions[None, :] + 1,
        mask=state_mask,
        other=0.0,
    )
    tl.debug_barrier()
    tl.store(
        state_base[:, None] + positions[None, :], shifted, mask=state_mask
    )
    tl.store(
        state_base + state_len - 1,
        current.to(out_dtype),
        mask=channel_mask,
    )


@triton.jit
def _qsa_index_scores_kernel(
    query_ptr,
    key_ptr,
    score_ptr,
    stride_qh,
    stride_kb,
    rows,
    heads: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
    scale: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_d)
    mask = offsets < dim
    key = tl.load(
        key_ptr + row * stride_kb + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    score = 0.0
    for head in range(heads):
        query = tl.load(
            query_ptr + head * stride_qh + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        dot = tl.sum(query * key, axis=0)
        score += tl.maximum(dot, 0.0)
    tl.store(score_ptr + row, score * scale)


@triton.jit
def _qsa_expand_selected_rows_kernel(
    block_ids_ptr,
    physical_rows_ptr,
    output_ptr,
    tail_start,
    tail_count,
    selected_blocks: tl.constexpr,
    ratio: tl.constexpr,
):
    block_offsets = tl.arange(0, selected_blocks)
    blocks = tl.load(block_ids_ptr + block_offsets).to(tl.int32)
    blocks = tl.sort(blocks, descending=False)
    token_offsets = tl.arange(0, ratio)
    logical = blocks[:, None] * ratio + token_offsets[None, :]
    physical = tl.load(physical_rows_ptr + logical)
    output_offsets = block_offsets[:, None] * ratio + token_offsets[None, :]
    tl.store(output_ptr + output_offsets, physical)

    tail_offsets = tl.arange(0, ratio)
    tail_mask = tail_offsets < tail_count
    tail = tl.load(
        physical_rows_ptr + tail_start + tail_offsets,
        mask=tail_mask,
        other=0,
    )
    tl.store(
        output_ptr + selected_blocks * ratio + tail_offsets,
        tail,
        mask=tail_mask,
    )


def grouped_plus_one_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    """Fused grouped ``(1 + weight)`` RMSNorm with Qwen4 fp32 semantics."""
    if not x.is_cuda:
        raise ValueError("Qwen4 fused grouped RMSNorm requires CUDA")
    if x.dtype not in _TL_DTYPE:
        raise TypeError(f"unsupported Qwen4 RMSNorm dtype: {x.dtype}")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Qwen4 fused grouped RMSNorm requires contiguous tensors")
    if x.shape[-1] != weight.numel() or weight.numel() % group_size:
        raise ValueError(
            f"invalid grouped RMSNorm geometry: x={x.shape}, "
            f"weight={weight.shape}, group_size={group_size}"
        )
    groups = weight.numel() // group_size
    rows = x.numel() // group_size
    output = torch.empty_like(x)
    block = triton.next_power_of_2(group_size)
    num_warps = 4 if block <= 1024 else (8 if block <= 4096 else 16)
    _grouped_plus_one_rmsnorm_kernel[(rows,)](
        x,
        weight,
        output,
        rows,
        group_size=group_size,
        groups=groups,
        eps=eps,
        block=block,
        out_dtype=_TL_DTYPE[x.dtype],
        num_warps=num_warps,
    )
    return output


def scaled_silu(x: torch.Tensor, divisor: int) -> torch.Tensor:
    """Fuse the Qwen4 power-of-two scale and SiLU activation."""
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("Qwen4 fused scaled SiLU requires a contiguous CUDA tensor")
    if x.dtype not in _TL_DTYPE:
        raise TypeError(f"unsupported Qwen4 scaled SiLU dtype: {x.dtype}")
    output = torch.empty_like(x)
    block = 256
    _scaled_silu_kernel[(triton.cdiv(x.numel(), block),)](
        x,
        output,
        x.numel(),
        divisor=divisor,
        block=block,
        out_dtype=_TL_DTYPE[x.dtype],
        num_warps=4,
    )
    return output


def hyper_mix(
    logits: torch.Tensor,
    normed: torch.Tensor,
    *,
    groups: int,
    group_size: int,
) -> torch.Tensor:
    """Fuse sigmoid gating, four-stream multiplication, and mean reduction."""
    if not logits.is_cuda or not normed.is_cuda:
        raise ValueError("Qwen4 fused hyper mix requires CUDA tensors")
    if not logits.is_contiguous() or not normed.is_contiguous():
        raise ValueError("Qwen4 fused hyper mix requires contiguous tensors")
    if logits.shape != normed.shape or logits.shape[-1] != groups * group_size:
        raise ValueError(
            f"invalid Qwen4 hyper mix geometry: {logits.shape}, "
            f"groups={groups}, group_size={group_size}"
        )
    if logits.dtype != normed.dtype or logits.dtype not in _TL_DTYPE:
        raise TypeError("Qwen4 hyper mix requires matching supported dtypes")
    rows = logits.numel() // logits.shape[-1]
    output = torch.empty(
        (*logits.shape[:-1], group_size),
        dtype=logits.dtype,
        device=logits.device,
    )
    block = 256
    _hyper_mix_kernel[(rows, triton.cdiv(group_size, block))](
        logits,
        normed,
        output,
        rows,
        width=logits.shape[-1],
        group_size=group_size,
        groups=groups,
        block=block,
        out_dtype=_TL_DTYPE[logits.dtype],
        num_warps=4,
    )
    return output


def hyper_injection_weights(logits: torch.Tensor, divisor: int) -> torch.Tensor:
    """Fuse Qwen4 injection scaling, sigmoid, and factor-of-two affine."""
    if not logits.is_cuda or not logits.is_contiguous():
        raise ValueError("Qwen4 fused injection weights require contiguous CUDA logits")
    if logits.dtype not in _TL_DTYPE:
        raise TypeError(f"unsupported Qwen4 injection dtype: {logits.dtype}")
    output = torch.empty_like(logits)
    block = 128
    _hyper_injection_weights_kernel[(triton.cdiv(logits.numel(), block),)](
        logits,
        output,
        logits.numel(),
        divisor=divisor,
        block=block,
        out_dtype=_TL_DTYPE[logits.dtype],
        num_warps=4,
    )
    return output


def hyper_residual_inject(
    branch: torch.Tensor,
    hyper_input: torch.Tensor,
    weights: torch.Tensor,
    *,
    groups: int,
) -> torch.Tensor:
    """Fuse broadcast branch weighting and the Qwen4 residual addition."""
    if not branch.is_cuda or not hyper_input.is_cuda or not weights.is_cuda:
        raise ValueError("Qwen4 fused residual injection requires CUDA tensors")
    if not branch.is_contiguous() or not hyper_input.is_contiguous() or not weights.is_contiguous():
        raise ValueError("Qwen4 fused residual injection requires contiguous tensors")
    group_size = branch.shape[-1]
    if hyper_input.shape[:-1] != branch.shape[:-1] or hyper_input.shape[-1] != groups * group_size:
        raise ValueError("invalid Qwen4 residual injection geometry")
    if weights.shape != (*branch.shape[:-1], groups):
        raise ValueError("invalid Qwen4 residual injection weights")
    if branch.dtype != hyper_input.dtype or branch.dtype != weights.dtype:
        raise TypeError("Qwen4 residual injection requires matching dtypes")
    if branch.dtype != torch.bfloat16:
        raise TypeError("Qwen4 fused residual injection requires BF16 tensors")
    output = torch.empty_like(hyper_input)
    block = 256
    _hyper_residual_inject_kernel[(triton.cdiv(output.numel(), block),)](
        branch,
        hyper_input,
        weights,
        output,
        output.numel(),
        group_size=group_size,
        groups=groups,
        block=block,
        out_dtype=_TL_DTYPE[output.dtype],
        num_warps=4,
    )
    return output


def ple_conv_decode(
    x: torch.Tensor,
    states: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Fused one-token dilated depthwise conv, SiLU, and state update."""
    batch, width = x.shape
    kernel_size = weight.shape[-1]
    state_len = states.shape[-1]
    if state_len != (kernel_size - 1) * dilation:
        raise ValueError(
            f"invalid PLE conv state: {state_len} != "
            f"({kernel_size} - 1) * {dilation}"
        )
    output = torch.empty_like(x)
    block = 256
    _ple_conv_decode_kernel[(batch, triton.cdiv(width, block))](
        x,
        states,
        weight,
        indices,
        output,
        width,
        states.stride(0),
        states.stride(1),
        kernel_size=kernel_size,
        dilation=dilation,
        state_len=state_len,
        state_block=triton.next_power_of_2(state_len),
        block=block,
        out_dtype=_TL_DTYPE[x.dtype],
        num_warps=4,
    )
    return output


def qsa_index_scores(
    query: torch.Tensor, pooled_keys: torch.Tensor
) -> torch.Tensor:
    """Fused Qwen QSA ``relu(Q @ K.T).sum(heads) / sqrt(dim)`` score."""
    if not query.is_cuda or not pooled_keys.is_cuda:
        raise ValueError("QSA fused index scoring requires CUDA tensors")
    if query.ndim != 2 or pooled_keys.ndim != 2:
        raise ValueError(
            f"QSA index scoring expects [heads, dim] and [rows, dim], got "
            f"{query.shape} and {pooled_keys.shape}"
        )
    if query.shape[1] != pooled_keys.shape[1]:
        raise ValueError("QSA query/key index dimensions must match")
    if not query.is_contiguous() or not pooled_keys.is_contiguous():
        raise ValueError("QSA fused index scoring requires contiguous tensors")
    rows, dim = pooled_keys.shape
    output = torch.empty(rows, dtype=torch.float32, device=query.device)
    if rows == 0:
        return output
    _qsa_index_scores_kernel[(rows,)](
        query,
        pooled_keys,
        output,
        query.stride(0),
        pooled_keys.stride(0),
        rows,
        heads=query.shape[0],
        dim=dim,
        block_d=triton.next_power_of_2(dim),
        scale=dim**-0.5,
        num_warps=4,
    )
    return output


def qsa_expand_selected_rows(
    block_ids: torch.Tensor,
    physical_rows: torch.Tensor,
    *,
    ratio: int,
    visible: int,
) -> torch.Tensor:
    """Sort selected QSA blocks and expand them directly to physical token rows."""
    if not block_ids.is_cuda or not physical_rows.is_cuda:
        raise ValueError("QSA fused row expansion requires CUDA tensors")
    if block_ids.ndim != 1 or physical_rows.ndim != 1:
        raise ValueError("QSA block ids and physical rows must be one-dimensional")
    if not block_ids.is_contiguous() or not physical_rows.is_contiguous():
        raise ValueError("QSA fused row expansion requires contiguous tensors")
    selected_blocks = block_ids.numel()
    if selected_blocks <= 0 or selected_blocks & (selected_blocks - 1):
        raise ValueError("QSA fused row expansion requires power-of-two selected blocks")
    complete_tokens = visible // ratio * ratio
    tail_count = visible - complete_tokens
    output = torch.empty(
        selected_blocks * ratio + tail_count,
        dtype=physical_rows.dtype,
        device=physical_rows.device,
    )
    _qsa_expand_selected_rows_kernel[(1,)](
        block_ids,
        physical_rows,
        output,
        complete_tokens,
        tail_count,
        selected_blocks=selected_blocks,
        ratio=ratio,
        num_warps=8,
    )
    return output


__all__ = [
    "grouped_plus_one_rmsnorm",
    "hyper_injection_weights",
    "hyper_mix",
    "hyper_residual_inject",
    "ple_conv_decode",
    "qsa_expand_selected_rows",
    "qsa_index_scores",
    "scaled_silu",
]
