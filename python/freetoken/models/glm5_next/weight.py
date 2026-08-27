"""GLM-5.3-Flash block-FP8 checkpoint reader and expert-bank hook."""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import iter_weight_files
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config


_EXPERT = re.compile(r"\.mlp\.experts\.\d+\.")
_LAYER = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")
_HC = re.compile(
    r"^(?P<prefix>model\.layers\.\d+)\.hc_(?P<site>attn|ffn)_(?P<part>fn|base|scale)$"
)


def map_weight_name(raw_name: str) -> str | None:
    """Map the multimodal HF wrapper onto FreeToken's text-only model."""
    if raw_name.startswith(("mtp.", "model.mtp.", "model.visual.", "visual.")):
        return None
    name = raw_name
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model.") :]
    if name.startswith(("model.mtp.", "model.visual.", "model.vision_tower.")):
        return None
    match = _HC.match(name)
    if match:
        site = "attn_hc" if match["site"] == "attn" else "ffn_hc"
        return f"{match['prefix']}.{site}.{match['part']}"
    return name


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size > 1:
        raise NotImplementedError("GLM-5.3-Flash weight loading supports TP=1 only")
    config = parse_config(cached_load_hf_config(model_path))

    if include_non_moe:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading GLM-5.3 weights",
            disable=not get_tp_info().is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as handle:
                for raw_name in handle.keys():
                    name = map_weight_name(raw_name)
                    if name is None or _EXPERT.search(name):
                        continue
                    layer_match = _LAYER.match(name)
                    if layer_match and int(layer_match["layer"]) >= config.num_layers:
                        continue  # trailing MTP layer, or a dev-layer-cap tail
                    # Static activation/cache scales are calibration metadata, not
                    # model state. Block weight scales end in weight_scale_inv and
                    # pass through to Fp8BlockLinear unchanged.
                    if name.endswith((".input_scale", ".q_scale", ".k_scale", ".v_scale")):
                        continue
                    yield name, handle.get_tensor(raw_name)

    if include_moe_experts:
        # The common Qwen block-FP8 builder has exactly the same raw per-expert
        # layout and bank schema; reuse it so resident and offload paths cannot
        # disagree about gate|up stacking or scale placement.
        from freetoken.models.qwen3_5_moe.weight import _build_fp8_expert_banks

        banks = _build_fp8_expert_banks(model_path, config, dummy=False, pin=False)
        for local_layer in range(config.num_moe_layers):
            layer = config.first_k_dense_replace + local_layer
            prefix = f"model.layers.{layer}.mlp.experts"
            yield f"{prefix}.gate_up_proj", banks["gate_up"][local_layer]
            yield f"{prefix}.gate_up_scale_inv", banks["gate_up_scale"][local_layer]
            yield f"{prefix}.down_proj", banks["down"][local_layer]
            yield f"{prefix}.down_scale_inv", banks["down_scale"][local_layer]


def setup_offload_expert_banks(*args, **kwargs):
    """Delegate to the shared per-expert 128x128 block-FP8 bank builder."""
    from freetoken.models.qwen3_5_moe.weight import setup_offload_expert_banks as setup

    return setup(*args, **kwargs)


__all__ = ["iter_weights", "map_weight_name", "setup_offload_expert_banks"]
