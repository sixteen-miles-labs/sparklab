"""Kimi K3 text weights and compressed-tensors MXFP4 expert-bank conversion."""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from sparklab.runtime.distributed import get_tp_info
from sparklab.kernels.triton.fp8_block_linear import dequant_block_fp8
from sparklab.models.loader import (
    MergeRule,
    drop_page_cache,
    iter_merged_tensors,
    iter_root_safetensor_files_from_index,
)
from sparklab.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)
from sparklab.utils import cached_load_hf_config

from .config import parse_config


_MERGE_RULES = {
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}
_EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
    r"(w[123])\.(weight_packed|weight_scale)$"
)
_MODEL_OPT_EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>w1|w2|w3)\."
    r"(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_MODEL_OPT_EXPERT_RE,
    proj_to_role={"w1": "gate", "w3": "up", "w2": "down"},
    layer_to_bank=lambda layer, config: layer - config.first_k_dense_replace,
    desc="Kimi K3 ModelOpt NVFP4 experts",
)


def map_weight_name(raw_name: str) -> str | None:
    """Map the multimodal wrapper onto SparkLab's text-only module tree."""
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


def _block_scale(scale: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Normalize ModelOpt's singleton-rich 128x128 scale grid."""
    out_blocks = (weight.shape[0] + 127) // 128
    in_blocks = (weight.shape[1] + 127) // 128
    return scale.reshape(out_blocks, in_blocks).contiguous()


def _keep_fp8(name: str, weight: torch.Tensor) -> bool:
    """Whether the K3 module consumes this projection through Fp8BlockLinear."""
    if weight.ndim != 2 or any(dim % 128 for dim in weight.shape):
        return False
    # MLA's kv_b matrix is absorbed into two resident BMM operands at runtime.
    # Dequantize it once; torch.bmm does not consume the block scale metadata.
    return not name.endswith(".self_attn.kv_b_proj.weight")


def _iter_modelopt_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if include_moe_experts:
        raise ValueError(
            "Kimi K3 ModelOpt NVFP4 experts are offload banks and cannot be "
            "materialized as module tensors"
        )
    if not include_non_moe:
        return

    def mapped():
        for file in iter_root_safetensor_files_from_index(model_path):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as handle:
                keyset = set(handle.keys())
                for raw_name in handle.keys():
                    if ".block_sparse_moe.experts." in raw_name:
                        continue
                    if raw_name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
                        continue
                    name = map_weight_name(raw_name)
                    if name is None:
                        continue
                    value = handle.get_tensor(raw_name)
                    if raw_name.endswith(".weight"):
                        raw_base = raw_name.removesuffix(".weight")
                        scale_name = raw_base + ".weight_scale"
                        if scale_name in keyset:
                            scale = _block_scale(handle.get_tensor(scale_name), value)
                            if _keep_fp8(name, value):
                                yield name, value
                                yield name.removesuffix(".weight") + ".weight_scale_inv", scale
                                continue
                            value = dequant_block_fp8(value, scale)
                    if name.endswith((".A_log", ".dt_bias")):
                        value = value.float()
                    yield name, value
            drop_page_cache(file)

    yield from iter_merged_tensors(mapped(), _MERGE_RULES, model_name="kimi_k3_modelopt")


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    if config.expert_quant == "nvfp4":
        yield from _iter_modelopt_weights(
            model_path,
            device,
            include_moe_experts=include_moe_experts,
            include_non_moe=include_non_moe,
        )
        return
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


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str,
    config,
    *,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
):
    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


def _copy_expert(dst: torch.Tensor, raw: torch.Tensor, out_slice: slice) -> None:
    # compressed-tensors [N, K//2] packed or [N, K//32] e8m0 scale ->
    # SparkLab split-K [K//2|K//32, N], with N kept contiguous.
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
    from sparklab.moe.host_banks import alloc_layer_banks

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
    if model_config.expert_quant == "nvfp4":
        from sparklab.moe.expert_banks import _nvfp4_banks

        return _nvfp4_banks(
            model_path,
            model_config,
            device,
            dtype,
            dummy,
            parallel=parallel,
            workers=workers,
            chunk=chunk,
            decode_target=decode_target,
            layer_sink=layer_sink,
        )
    del device, dtype, workers, chunk
    if decode_target != "gpu":
        raise NotImplementedError("Kimi SiTU MXFP4 currently supports GPU expert execution only")
    if parallel:
        raise NotImplementedError("parallel Kimi K3 expert conversion is not implemented yet")
    if dummy:
        raise NotImplementedError("dummy Kimi K3 MXFP4 banks are not implemented")
    from sparklab.moe.expert_banks import ExpertBanks

    sources = load_mxfp4_banks_streaming(model_path, model_config, layer_sink=layer_sink)
    return ExpertBanks("mxfp4_triton", sources, streamed=True)


__all__ = [
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_mxfp4_banks_streaming",
    "map_weight_name",
    "setup_offload_expert_banks",
]
