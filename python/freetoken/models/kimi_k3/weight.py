"""Kimi K3 text weights and compressed-tensors MXFP4 expert-bank conversion."""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule,
    iter_merged_tensors,
    iter_root_safetensor_files_from_index,
)

from .config import parse_config


_MERGE_RULES = {
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}
_EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
    r"(w[123])\.(weight_packed|weight_scale)$"
)


def map_weight_name(raw_name: str) -> str | None:
    """Map the multimodal wrapper onto FreeToken's text-only module tree."""
    if not raw_name.startswith("language_model."):
        return None
    name = raw_name.removeprefix("language_model.")
    if ".block_sparse_moe.experts." in name:
        return None
    name = name.replace(
        ".block_sparse_moe.gate.e_score_correction_bias",
        ".block_sparse_moe.e_score_correction_bias",
    )
    return name


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if include_moe_experts:
        raise ValueError(
            "Kimi K3 MXFP4 experts are loaded as native offload banks; they are not "
            "materialized as module tensors"
        )
    assert include_non_moe

    def mapped():
        for file in iter_root_safetensor_files_from_index(model_path):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as handle:
                for raw_name in handle.keys():
                    name = map_weight_name(raw_name)
                    if name is not None:
                        value = handle.get_tensor(raw_name)
                        # Recurrence-gating parameters are consumed as fp32 by the
                        # kernel; keep their checkpoint precision during loading.
                        if name.endswith((".A_log", ".dt_bias")):
                            value = value.float()
                        yield name, value

    yield from iter_merged_tensors(mapped(), _MERGE_RULES, model_name="kimi_k3")


def _copy_expert(dst: torch.Tensor, raw: torch.Tensor, out_slice: slice) -> None:
    # compressed-tensors [N, K//2] packed or [N, K//32] e8m0 scale ->
    # FreeToken split-K [K//2|K//32, N], with N kept contiguous.
    dst[:, out_slice].copy_(raw.transpose(0, 1).contiguous())


def load_mxfp4_banks_streaming(model_path: str, config, *, layer_sink):
    """Stream K3's per-expert MXFP4 tensors into per-layer transposed banks.

    The released checkpoint is far larger than GB10 memory, so direct serving from
    Hugging Face shards is intentionally rejected: this loader is the conversion
    path and requires a layer sink that persists and releases each completed layer.
    """
    if layer_sink is None:
        raise RuntimeError(
            "Kimi K3 must be converted to an FTW disk checkpoint before GB10 serving; "
            "direct loading would require materializing roughly 1.5 TB of host banks"
        )
    tp = get_tp_info()
    if tp.size != 1:
        raise NotImplementedError("Kimi K3 MXFP4 conversion currently supports TP=1")
    from freetoken.moe.host_banks import alloc_layer_banks

    layers = config.num_moe_layers
    experts = config.num_experts
    latent = config.kimi_k3_args.routed_expert_hidden_size
    intermediate = config.moe_intermediate_size
    specs = {
        "gate_up_blocks": ((experts, latent // 2, 2 * intermediate), torch.uint8),
        "gate_up_scales": ((experts, latent // 32, 2 * intermediate), torch.uint8),
        "gate_up_bias": ((experts, 2 * intermediate), torch.bfloat16),
        "down_blocks": ((experts, intermediate // 2, latent), torch.uint8),
        "down_scales": ((experts, intermediate // 32, latent), torch.uint8),
        "down_bias": ((experts, latent), torch.bfloat16),
    }
    host = alloc_layer_banks(specs, layers)
    banks = {name: [bank.tensor for bank in per_layer] for name, per_layer in host.items()}
    expected_per_layer = experts * 6
    seen = [0] * layers
    completed: set[int] = set()

    for file in iter_root_safetensor_files_from_index(model_path):
        with safetensors.safe_open(file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                match = _EXPERT_RE.match(name)
                if match is None:
                    continue
                global_layer, expert, proj, kind = match.groups()
                local_layer = int(global_layer) - config.first_k_dense_replace
                if not (0 <= local_layer < layers):
                    continue
                expert_id = int(expert)
                raw = handle.get_tensor(name)
                if proj in ("w1", "w3"):
                    bank = "gate_up_blocks" if kind == "weight_packed" else "gate_up_scales"
                    start = 0 if proj == "w1" else intermediate
                    _copy_expert(
                        banks[bank][local_layer][expert_id],
                        raw,
                        slice(start, start + intermediate),
                    )
                else:
                    bank = "down_blocks" if kind == "weight_packed" else "down_scales"
                    _copy_expert(banks[bank][local_layer][expert_id], raw, slice(None))
                seen[local_layer] += 1
                if seen[local_layer] == expected_per_layer:
                    # Bias banks are zero-filled anonymous mappings; K3 experts have
                    # no bias but the shared MXFP4 kernels use one six-bank schema.
                    layer_sink(
                        local_layer,
                        {key: value[local_layer] for key, value in host.items()},
                    )
                    completed.add(local_layer)

    missing = [layer for layer, count in enumerate(seen) if count != expected_per_layer]
    if missing:
        raise ValueError(
            f"incomplete Kimi K3 MXFP4 expert layers: {missing[:8]} "
            f"(first count={seen[missing[0]]}/{expected_per_layer})"
        )
    if completed != set(range(layers)):
        raise AssertionError("Kimi K3 conversion sink did not receive every expert layer")
    return banks


def setup_offload_expert_banks(
    model_path: str,
    model_config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    decode_target: str = "gpu",
    layer_sink=None,
):
    del device, dtype, workers, chunk
    if decode_target != "gpu":
        raise NotImplementedError("Kimi SiTU MXFP4 currently supports GPU expert execution only")
    if parallel:
        raise NotImplementedError("parallel Kimi K3 expert conversion is not implemented yet")
    if dummy:
        raise NotImplementedError("dummy Kimi K3 MXFP4 banks are not implemented")
    from freetoken.moe.expert_banks import ExpertBanks

    sources = load_mxfp4_banks_streaming(model_path, model_config, layer_sink=layer_sink)
    return ExpertBanks("mxfp4_triton", sources, streamed=True)


__all__ = [
    "iter_weights",
    "load_mxfp4_banks_streaming",
    "map_weight_name",
    "setup_offload_expert_banks",
]
