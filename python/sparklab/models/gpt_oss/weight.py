from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from sparklab.runtime.distributed import get_tp_info
from sparklab.models.loader import (
    MergeRule,
    iter_merged_tensors,
    iter_root_safetensor_files_from_index,
    shard_tensor,
)
from sparklab.utils import cached_load_hf_config

from .config import parse_config

_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
}


def local_mxfp4_intermediate_range(
    intermediate_size: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, int, int]:
    if intermediate_size % 32 != 0:
        raise ValueError("GPT-OSS MXFP4 intermediate size must be divisible by 32")
    blocks = intermediate_size // 32
    blocks_per_rank = (blocks + world_size - 1) // world_size
    local_intermediate = blocks_per_rank * 32
    start = rank * local_intermediate
    end = min((rank + 1) * local_intermediate, intermediate_size)
    return start, end, local_intermediate


def _shard_dim(value: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    slices = [slice(None)] * value.ndim
    slices[dim] = slice(start, end)
    return value[tuple(slices)].clone()


def _shard_dim_pad(
    value: torch.Tensor,
    dim: int,
    start: int,
    end: int,
    target_size: int,
) -> torch.Tensor:
    shard = _shard_dim(value, dim, start, end)
    pad_size = target_size - shard.shape[dim]
    if pad_size <= 0:
        return shard
    pad_shape = list(shard.shape)
    pad_shape[dim] = pad_size
    return torch.cat([shard, shard.new_zeros(pad_shape)], dim=dim)


def _attn_range(
    *,
    kind: str,
    rank: int,
    world_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    if kind == "q":
        if num_q_heads % world_size != 0:
            raise ValueError("GPT-OSS query heads must be divisible by TP size")
        heads_per_rank = num_q_heads // world_size
        head_idx = rank * heads_per_rank
    else:
        if world_size > num_kv_heads:
            if world_size % num_kv_heads != 0:
                raise ValueError("GPT-OSS TP size must divide or replicate KV heads")
            heads_per_rank = 1
            head_idx = rank * num_kv_heads // world_size
        else:
            if num_kv_heads % world_size != 0:
                raise ValueError("GPT-OSS KV heads must be divisible by TP size")
            heads_per_rank = num_kv_heads // world_size
            head_idx = rank * heads_per_rank
    return head_idx * head_dim, (head_idx + heads_per_rank) * head_dim


def shard_gpt_oss_tensor(
    name: str,
    value: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> torch.Tensor:
    if name.endswith(".self_attn.sinks"):
        start, end = _attn_range(
            kind="q",
            rank=rank,
            world_size=world_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=1,
        )
        return value[start:end].clone()

    for suffix, kind in (
        (".self_attn.q_proj.weight", "q"),
        (".self_attn.q_proj.bias", "q"),
        (".self_attn.k_proj.weight", "kv"),
        (".self_attn.k_proj.bias", "kv"),
        (".self_attn.v_proj.weight", "kv"),
        (".self_attn.v_proj.bias", "kv"),
    ):
        if name.endswith(suffix):
            start, end = _attn_range(
                kind=kind,
                rank=rank,
                world_size=world_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            return _shard_dim(value, 0, start, end)

    if name.endswith(".self_attn.o_proj.weight"):
        start, end = _attn_range(
            kind="q",
            rank=rank,
            world_size=world_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        return _shard_dim(value, 1, start, end)
    if name.endswith(".self_attn.o_proj.bias"):
        return value.clone() if rank == 0 else torch.zeros_like(value)

    if ".mlp.experts." in name:
        tp_start, tp_end, local_intermediate = local_mxfp4_intermediate_range(
            intermediate_size,
            rank=rank,
            world_size=world_size,
        )
        if name.endswith(".mlp.experts.gate_up_proj_blocks"):
            return _shard_dim_pad(
                value,
                1,
                2 * tp_start,
                2 * tp_end,
                2 * local_intermediate,
            )
        if name.endswith(".mlp.experts.gate_up_proj_scales"):
            return _shard_dim_pad(
                value,
                1,
                2 * tp_start,
                2 * tp_end,
                2 * local_intermediate,
            )
        if name.endswith(".mlp.experts.gate_up_proj_bias"):
            return _shard_dim_pad(
                value,
                1,
                2 * tp_start,
                2 * tp_end,
                2 * local_intermediate,
            )
        if name.endswith(".mlp.experts.down_proj_blocks"):
            return _shard_dim_pad(
                value,
                2,
                tp_start // 32,
                tp_end // 32,
                local_intermediate // 32,
            )
        if name.endswith(".mlp.experts.down_proj_scales"):
            return _shard_dim_pad(
                value,
                value.ndim - 1,
                tp_start // 32,
                tp_end // 32,
                local_intermediate // 32,
            )
        if name.endswith(".mlp.experts.down_proj_bias"):
            return value.clone() if rank == 0 else torch.zeros_like(value)

    if "embed_tokens" in name or "lm_head" in name:
        return shard_tensor(
            name,
            value,
            rank=rank,
            world_size=world_size,
            num_kv_heads=num_kv_heads,
        )
    return value.clone()


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def sharded_tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in iter_root_safetensor_files_from_index(model_path):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    is_expert = ".mlp.experts." in raw_name
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue
                    raw = f.get_tensor(raw_name)
                    yield raw_name, shard_gpt_oss_tensor(
                        raw_name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_q_heads=config.num_qo_heads,
                        num_kv_heads=config.num_kv_heads,
                        head_dim=config.head_dim,
                        intermediate_size=config.moe_intermediate_size,
                    )

    yield from iter_merged_tensors(
        sharded_tensors(),
        _MERGE_RULES,
        model_name="gpt_oss",
    )


def _expert_layer_and_name(key: str) -> tuple[int, str] | None:
    parts = key.split(".")
    if len(parts) < 6 or parts[0] != "model" or parts[1] != "layers":
        return None
    if parts[4] != "experts":
        return None
    try:
        layer_id = int(parts[2])
    except ValueError:
        return None
    return layer_id, parts[5]


def _empty_mxfp4_triton_banks(
    config,
    *,
    dtype: torch.dtype,
    tp_info=None,
):
    """returns ``(banks, host_banks)``, unpinned. ``banks[name]`` is one ``[E, ...]``
    tensor per layer (independent allocations). host banks are lazy anon mmaps (already
    zero, so no ``zero_`` pass); caller fills then pins (pin-after-fill skips cudaHostAlloc's
    slow commit) -- per-layer via ``PinPipeline`` for the real streaming loaders, or
    ``pin_banks`` for the dummy path. unwritten padding stays zero."""
    if tp_info is None:
        tp_info = get_tp_info()
    _, _, local_intermediate = local_mxfp4_intermediate_range(
        config.moe_intermediate_size,
        rank=tp_info.rank,
        world_size=tp_info.size,
    )
    num_layers, E = config.num_layers, config.num_experts
    hidden_blocks = config.hidden_size // 32
    intermediate_blocks = local_intermediate // 32
    specs = {
        "gate_up_blocks": ((E, config.hidden_size // 2, 2 * local_intermediate), torch.uint8),
        "gate_up_scales": ((E, hidden_blocks, 2 * local_intermediate), torch.uint8),
        "gate_up_bias": ((E, 2 * local_intermediate), dtype),
        "down_blocks": ((E, local_intermediate // 2, config.hidden_size), torch.uint8),
        "down_scales": ((E, intermediate_blocks, config.hidden_size), torch.uint8),
        "down_bias": ((E, config.hidden_size), dtype),
    }
    if torch.cuda.is_available():
        from sparklab.moe.host_banks import alloc_layer_banks

        hb = alloc_layer_banks(specs, num_layers)  # lazy anon mmap -> zero-initialized, unpinned
        return {name: [b.tensor for b in hb[name]] for name in specs}, hb
    return {
        name: [torch.zeros(shape, dtype=dt) for _ in range(num_layers)]
        for name, (shape, dt) in specs.items()
    }, {}


def _make_dummy_mxfp4_triton_banks(
    config,
    *,
    dtype: torch.dtype,
    tp_info=None,
) -> dict[str, list[torch.Tensor]]:
    from sparklab.moe.host_banks import pin_banks

    banks, _hb = _empty_mxfp4_triton_banks(config, dtype=dtype, tp_info=tp_info)
    for t in banks["gate_up_blocks"]:
        t.random_(0, 256)
    for t in banks["gate_up_scales"]:
        t.fill_(127)
    for t in banks["gate_up_bias"]:
        t.normal_()
    for t in banks["down_blocks"]:
        t.random_(0, 256)
    for t in banks["down_scales"]:
        t.fill_(127)
    for t in banks["down_bias"]:
        t.normal_()
    pin_banks(_hb)  # pin-after-fill
    return banks


def _read_safetensor_slice(
    f,
    name: str,
    slices: tuple[slice, ...],
) -> torch.Tensor:
    if all(s == slice(None) for s in slices):
        return f.get_tensor(name)
    return f.get_slice(name)[slices]


def _copy_transposed_mxfp4_blocks(dst: torch.Tensor, raw: torch.Tensor) -> None:
    # HF [E, N, K//32, 16] -> split-K [E, K//2, N].
    if raw.numel() == 0:
        return
    source = raw.reshape(raw.shape[0], raw.shape[1], raw.shape[2] * raw.shape[3]).permute(
        0,
        2,
        1,
    )
    dst[:, : source.shape[1], : source.shape[2]].copy_(source)


def _copy_transposed_mxfp4_scales(dst: torch.Tensor, raw: torch.Tensor) -> None:
    # HF [E, N, K//32] -> split-K [E, K//32, N].
    if raw.numel() == 0:
        return
    source = raw.permute(0, 2, 1)
    dst[:, : source.shape[1], : source.shape[2]].copy_(source)


def _copy_prefix(dst: torch.Tensor, raw: torch.Tensor) -> None:
    dst[:, : raw.shape[1]].copy_(raw)


def load_mxfp4_triton_banks_streaming(
    model_path: str,
    model_config=None,
    *,
    dtype: torch.dtype,
    tp_info=None,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """``layer_sink=None`` (serving): pin each layer as its writes complete, via an
    internally-owned :class:`PinPipeline` (or no pinning at all in the CPU-only fallback,
    where ``_hb`` is empty). ``layer_sink`` given (converter): the completion tracker
    fires into it instead -- nothing here is pinned, and the sink may release banks it
    has written out, so the returned tensors are only valid until then (the caller owns
    that tradeoff)."""
    config = model_config if model_config is not None else parse_config(cached_load_hf_config(model_path))
    if config.moe_weight_format != "mxfp4":
        raise ValueError("GPT-OSS offload requires MXFP4 expert weights")

    if tp_info is None:
        tp_info = get_tp_info()
    tp_start, tp_end, local_intermediate = local_mxfp4_intermediate_range(
        config.moe_intermediate_size,
        rank=tp_info.rank,
        world_size=tp_info.size,
    )
    tp_end = max(tp_start, tp_end)
    banks, _hb = _empty_mxfp4_triton_banks(config, dtype=dtype, tp_info=tp_info)
    seen: set[tuple[int, str]] = set()
    all_expert_sources = (
        "gate_up_proj_blocks",
        "gate_up_proj_scales",
        "gate_up_proj_bias",
        "down_proj_blocks",
        "down_proj_scales",
        "down_proj_bias",
    )
    copy_plan = {
        "gate_up_proj_blocks": (
            "gate_up_blocks",
            (slice(None), slice(2 * tp_start, 2 * tp_end), slice(None), slice(None)),
            _copy_transposed_mxfp4_blocks,
        ),
        "gate_up_proj_scales": (
            "gate_up_scales",
            (slice(None), slice(2 * tp_start, 2 * tp_end), slice(None)),
            _copy_transposed_mxfp4_scales,
        ),
        "gate_up_proj_bias": (
            "gate_up_bias",
            (slice(None), slice(2 * tp_start, 2 * tp_end)),
            _copy_prefix,
        ),
        "down_proj_blocks": (
            "down_blocks",
            (
                slice(None),
                slice(None),
                slice(tp_start // 32, tp_end // 32),
                slice(None),
            ),
            _copy_transposed_mxfp4_blocks,
        ),
        "down_proj_scales": (
            "down_scales",
            (slice(None), slice(None), slice(tp_start // 32, tp_end // 32)),
            _copy_transposed_mxfp4_scales,
        ),
    }

    from sparklab.moe.host_banks import LayerCompletionTracker, PinPipeline

    def _load(sink) -> None:
        # _hb is empty in the CPU-only fallback (no banks to pin/stream): no-op tracker.
        tracker = LayerCompletionTracker(len(all_expert_sources), _hb, sink) if _hb else None
        for file in iter_root_safetensor_files_from_index(model_path):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    info = _expert_layer_and_name(name)
                    if info is None:
                        continue
                    layer_id, source_name = info
                    if layer_id < 0 or layer_id >= config.num_layers:
                        raise ValueError(f"Unexpected GPT-OSS expert layer in checkpoint: {name}")
                    seen.add((layer_id, source_name))

                    if source_name == "down_proj_bias":
                        if tp_info.rank == 0:
                            raw = f.get_tensor(name)
                            banks["down_bias"][layer_id].copy_(raw)
                        if tracker is not None:
                            tracker.note(layer_id)
                        continue

                    plan = copy_plan.get(source_name)
                    if plan is None:
                        raise ValueError(f"Unexpected GPT-OSS expert source: {name}")
                    bank_name, source_slice, copy_fn = plan
                    raw = _read_safetensor_slice(f, name, source_slice)
                    copy_fn(banks[bank_name][layer_id], raw)
                    if tracker is not None:
                        tracker.note(layer_id)

    if layer_sink is not None:
        _load(layer_sink)
    elif _hb:
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    expected = {
        (layer_id, source_name)
        for layer_id in range(config.num_layers)
        for source_name in all_expert_sources
    }
    missing = expected - seen
    if missing:
        raise ValueError(f"Missing GPT-OSS expert tensors: {sorted(missing)[:8]}")
    assert all(
        len(per_layer) == config.num_layers
        and all(t.is_contiguous() and t.size(0) == config.num_experts for t in per_layer)
        for per_layer in banks.values()
    )
    return banks


def load_mxfp4_triton_banks_streaming_parallel(
    model_path: str,
    model_config=None,
    *,
    dtype: torch.dtype,
    tp_info=None,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """parallel counterpart of load_mxfp4_triton_banks_streaming: same banks + copy plan,
    but bulk expert tensors come from the common chunked multi-threaded O_DIRECT reader.
    source_slice applied to the whole tensor == the serial result, so correct for any TP.
    ``layer_sink``: see :func:`load_mxfp4_triton_banks_streaming`."""
    from sparklab.models.weight import iter_expert_tensors_parallel

    config = model_config if model_config is not None else parse_config(cached_load_hf_config(model_path))
    if config.moe_weight_format != "mxfp4":
        raise ValueError("GPT-OSS offload requires MXFP4 expert weights")
    if tp_info is None:
        tp_info = get_tp_info()
    tp_start, tp_end, _local = local_mxfp4_intermediate_range(
        config.moe_intermediate_size, rank=tp_info.rank, world_size=tp_info.size
    )
    tp_end = max(tp_start, tp_end)
    banks, _hb = _empty_mxfp4_triton_banks(config, dtype=dtype, tp_info=tp_info)
    all_expert_sources = (
        "gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias",
        "down_proj_blocks", "down_proj_scales", "down_proj_bias",
    )
    copy_plan = {
        "gate_up_proj_blocks": ("gate_up_blocks",
            (slice(None), slice(2 * tp_start, 2 * tp_end), slice(None), slice(None)),
            _copy_transposed_mxfp4_blocks),
        "gate_up_proj_scales": ("gate_up_scales",
            (slice(None), slice(2 * tp_start, 2 * tp_end), slice(None)),
            _copy_transposed_mxfp4_scales),
        "gate_up_proj_bias": ("gate_up_bias",
            (slice(None), slice(2 * tp_start, 2 * tp_end)), _copy_prefix),
        "down_proj_blocks": ("down_blocks",
            (slice(None), slice(None), slice(tp_start // 32, tp_end // 32), slice(None)),
            _copy_transposed_mxfp4_blocks),
        "down_proj_scales": ("down_scales",
            (slice(None), slice(None), slice(tp_start // 32, tp_end // 32)),
            _copy_transposed_mxfp4_scales),
    }

    def _is_expert(name: str) -> bool:
        info = _expert_layer_and_name(name)
        return info is not None and 0 <= info[0] < config.num_layers

    from sparklab.moe.host_banks import LayerCompletionTracker, PinPipeline

    seen: set[tuple[int, str]] = set()

    def _load(sink) -> None:
        # _hb is empty in the CPU-only fallback (no banks to pin/stream): no-op tracker.
        tracker = LayerCompletionTracker(len(all_expert_sources), _hb, sink) if _hb else None
        for name, whole in iter_expert_tensors_parallel(model_path, _is_expert, workers=workers, chunk=chunk):
            layer_id, source_name = _expert_layer_and_name(name)
            seen.add((layer_id, source_name))
            if source_name == "down_proj_bias":
                if tp_info.rank == 0:
                    banks["down_bias"][layer_id].copy_(whole)
                if tracker is not None:
                    tracker.note(layer_id)
                continue
            bank_name, source_slice, copy_fn = copy_plan[source_name]
            copy_fn(banks[bank_name][layer_id], whole[source_slice])
            if tracker is not None:
                tracker.note(layer_id)

    if layer_sink is not None:
        _load(layer_sink)
    elif _hb:
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    expected = {(layer_id, src) for layer_id in range(config.num_layers) for src in all_expert_sources}
    missing = expected - seen
    if missing:
        raise ValueError(f"Missing GPT-OSS expert tensors: {sorted(missing)[:8]}")
    assert all(
        len(per_layer) == config.num_layers
        and all(t.is_contiguous() and t.size(0) == config.num_experts for t in per_layer)
        for per_layer in banks.values()
    )
    return banks


def setup_offload_expert_banks(
    model_path: str,
    model_config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
    tp_info=None,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
):
    from sparklab.moe.expert_banks import ExpertBanks

    if model_config.moe_weight_format != "mxfp4":
        raise ValueError("GPT-OSS offload currently supports MXFP4 expert weights")
    if dummy:
        sources = _make_dummy_mxfp4_triton_banks(model_config, dtype=dtype, tp_info=tp_info)
        streamed = False
    elif parallel:  # parallel: common chunked multi-threaded O_DIRECT reader
        sources = load_mxfp4_triton_banks_streaming_parallel(
            model_path, model_config, dtype=dtype, tp_info=tp_info, workers=workers, chunk=chunk,
            layer_sink=layer_sink,
        )
        streamed = layer_sink is not None
    else:
        sources = load_mxfp4_triton_banks_streaming(
            model_path, model_config, dtype=dtype, tp_info=tp_info, layer_sink=layer_sink
        )
        streamed = layer_sink is not None
    return ExpertBanks("mxfp4_triton", sources, streamed=streamed)


__all__ = [
    "iter_weights",
    "local_mxfp4_intermediate_range",
    "load_mxfp4_triton_banks_streaming",
    "setup_offload_expert_banks",
    "shard_gpt_oss_tensor",
]
