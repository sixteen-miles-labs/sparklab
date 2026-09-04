"""GLM-5.3-Flash block-FP8 checkpoint reader and expert-bank hook."""

from __future__ import annotations

import os
import re
import shutil
from typing import Iterator

import safetensors
import torch
from sparklab.runtime.distributed import get_tp_info
from sparklab.models.loader import drop_page_cache, iter_weight_files
from sparklab.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_ct_nvfp4_expert_source_banks,
)
from sparklab.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config


_EXPERT = re.compile(r"\.mlp\.experts\.\d+\.")
_LAYER = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")
_HC = re.compile(
    r"^(?P<prefix>model\.layers\.\d+)\.hc_(?P<site>attn|ffn)_(?P<part>fn|base|scale)$"
)
_CT_NVFP4_EXPERT = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight_packed|weight_scale|weight_global_scale)$"
)
_CT_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_CT_NVFP4_EXPERT,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer - config.first_k_dense_replace,
    desc="GLM-5.3 compressed-tensors NVFP4 experts",
)

_KDA_MAIN_WEIGHT = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)\.weight$"
)
_FP8_MAX = 448.0


def _quant_fp8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 matrix to the KDA artifact's W8A16 per-output-row format."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError(
            "GLM-5.3 KDA FP8 conversion expects a BF16 matrix, got "
            f"shape={tuple(weight.shape)}, dtype={weight.dtype}"
        )
    source = weight.float()
    scale = (source.abs().amax(dim=1) / _FP8_MAX).clamp(min=1e-12)
    quantized = (source / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def _is_kda_main_weight(name: str, config) -> bool:
    match = _KDA_MAIN_WEIGHT.match(name)
    return bool(match and config.is_linear_layer(int(match["layer"])))


def map_weight_name(raw_name: str) -> str | None:
    """Map the multimodal HF wrapper onto SparkLab's text-only model."""
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


def find_mtp_sidecar(model_path: str) -> str | None:
    """Resolve the opt-in layer-45 payload for source or prepared checkpoints."""
    override = os.environ.get("SPARKLAB_GLM5_MTP_PATH")
    candidates = [override] if override else []
    candidates.extend(
        [
            os.path.join(model_path, "model_mtp.safetensors"),
            os.path.join(model_path, "mtp", "model_mtp.safetensors"),
        ]
    )
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def copy_external_artifacts(
    model_path: str,
    out_dir: str,
    model_config,
    *,
    ngram_dtype: str = "preserve",
) -> list[dict]:
    """Copy the optional publisher MTP payload beside a prepared FTW artifact."""
    del model_config, ngram_dtype  # shared converter-hook arguments
    source = find_mtp_sidecar(model_path)
    if source is None:
        return []

    filename = "model_mtp.safetensors"
    destination = os.path.join(out_dir, filename)
    temporary = destination + ".tmp"
    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(source, "rb", buffering=0) as src, open(
            temporary, "wb", buffering=0
        ) as dst:
            shutil.copyfileobj(src, dst, length=16 << 20)
            os.fsync(dst.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    return [
        {
            "kind": "glm5_mtp",
            "file": filename,
            "format": "safetensors-fp8-block",
            "nbytes": os.path.getsize(destination),
        }
    ]


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
                    weight = handle.get_tensor(raw_name)
                    if (
                        config.glm5_next_args.kda_quant == "fp8_pertensor"
                        and _is_kda_main_weight(name, config)
                    ):
                        quantized, scale = _quant_fp8_per_row(weight)
                        yield name, quantized
                        yield name.removesuffix(".weight") + ".weight_scale", scale
                    else:
                        yield name, weight

    if include_moe_experts:
        # The common Qwen block-FP8 builder has exactly the same raw per-expert
        # layout and bank schema; reuse it so resident and offload paths cannot
        # disagree about gate|up stacking or scale placement.
        from sparklab.models.qwen3_5_moe.weight import _build_fp8_expert_banks

        banks = _build_fp8_expert_banks(model_path, config, dummy=False, pin=False)
        for local_layer in range(config.num_moe_layers):
            layer = config.first_k_dense_replace + local_layer
            prefix = f"model.layers.{layer}.mlp.experts"
            yield f"{prefix}.gate_up_proj", banks["gate_up"][local_layer]
            yield f"{prefix}.gate_up_scale_inv", banks["gate_up_scale"][local_layer]
            yield f"{prefix}.down_proj", banks["down"][local_layer]
            yield f"{prefix}.down_scale_inv", banks["down_scale"][local_layer]


def setup_offload_expert_banks(*args, **kwargs):
    """Build block-FP8 or compressed-tensors NVFP4 expert banks."""
    if len(args) >= 2 and getattr(args[1], "expert_quant", None) == "nvfp4":
        if kwargs.get("parallel"):
            raise NotImplementedError(
                "parallel GLM-5.3 compressed-tensors NVFP4 loading is not implemented"
            )
        from sparklab.moe.expert_banks import _nvfp4_banks

        return _nvfp4_banks(
            args[0],
            args[1],
            kwargs.get("device", torch.device("cuda:0")),
            kwargs.get("dtype", torch.bfloat16),
            kwargs.get("dummy", False),
            parallel=False,
            workers=kwargs.get("workers", 8),
            chunk=kwargs.get("chunk", 8 << 20),
            decode_target=kwargs.get("decode_target", "gpu"),
            layer_sink=kwargs.get("layer_sink"),
        )
    from sparklab.models.qwen3_5_moe.weight import setup_offload_expert_banks as setup

    return setup(*args, **kwargs)


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    return load_ct_nvfp4_expert_source_banks(
        model_path,
        config,
        _CT_NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


__all__ = [
    "copy_external_artifacts",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "map_weight_name",
    "find_mtp_sidecar",
    "setup_offload_expert_banks",
    "_quant_fp8_per_row",
    "_is_kda_main_weight",
]
