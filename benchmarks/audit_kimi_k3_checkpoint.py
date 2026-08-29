#!/usr/bin/env python3
"""Metadata-only audit for NVIDIA's ModelOpt Kimi K3 checkpoint.

The audit never materializes tensor payloads. It proves source-index coverage against
SparkLab's text model and checks every locally completed safetensors header. Partial
mode is useful while a resumable snapshot is downloading; the final gate requires all
indexed shards to be present.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import safetensors
import torch
from transformers import AutoConfig

from sparklab.runtime.distributed import set_tp_info, try_get_tp_info
from sparklab.models.kimi_k3.config import parse_config
from sparklab.models.kimi_k3.model import KimiK3ForCausalLM
from sparklab.models.kimi_k3.weight import _MODEL_OPT_EXPERT_RE, map_weight_name


PROJECTIONS = {"w1": 0, "w2": 1, "w3": 2}
KINDS = {"weight": 0, "weight_scale": 1, "weight_scale_2": 2}
EXPERT_SHAPES = {
    ("w1", "weight"): ((3072, 1792), "U8"),
    ("w3", "weight"): ((3072, 1792), "U8"),
    ("w2", "weight"): ((3584, 1536), "U8"),
    ("w1", "weight_scale"): ((3072, 224), "F8_E4M3"),
    ("w3", "weight_scale"): ((3072, 224), "F8_E4M3"),
    ("w2", "weight_scale"): ((3584, 192), "F8_E4M3"),
    ("w1", "weight_scale_2"): ((), "F32"),
    ("w3", "weight_scale_2"): ((), "F32"),
    ("w2", "weight_scale_2"): ((), "F32"),
    ("w1", "input_scale"): ((), "F32"),
    ("w2", "input_scale"): ((), "F32"),
    ("w3", "input_scale"): ((), "F32"),
}
EXPERT_HEADER_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>w1|w2|w3)\."
    r"(?P<kind>weight|weight_scale|weight_scale_2|input_scale)$"
)


def _expected_state(model_path: Path) -> set[str]:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = parse_config(AutoConfig.from_pretrained(model_path, trust_remote_code=True))
    object.__setattr__(config, "moe_backend", "offload")
    with torch.device("meta"):
        model = KimiK3ForCausalLM(config)
    return set(model.state_dict())


def _mapped_state(raw_names: set[str], expected: set[str]) -> set[str]:
    mapped = set()
    for raw_name in raw_names:
        if ".block_sparse_moe.experts." in raw_name or raw_name.endswith(
            (".weight_scale", ".weight_scale_2", ".input_scale")
        ):
            continue
        name = map_weight_name(raw_name)
        if name is None:
            continue
        name = name.replace(".gate_proj.", ".gate_up_proj.")
        name = name.replace(".up_proj.", ".gate_up_proj.")
        mapped.add(name)
    for raw_name in raw_names:
        if not raw_name.endswith(".weight_scale") or ".block_sparse_moe.experts." in raw_name:
            continue
        name = map_weight_name(raw_name)
        if name is None:
            continue
        name = name.removesuffix(".weight_scale") + ".weight_scale_inv"
        if name in expected:
            mapped.add(name)
    return mapped


def audit(model_path: Path, *, allow_partial: bool) -> dict:
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    raw_names = set(weight_map)
    indexed_shards = sorted(set(weight_map.values()))
    completed = sorted(path.name for path in model_path.glob("model-*.safetensors"))
    completed_set = set(completed)
    missing_shards = sorted(set(indexed_shards) - completed_set)
    unexpected_shards = sorted(completed_set - set(indexed_shards))
    errors: list[dict] = []

    if unexpected_shards:
        errors.append({"unexpected_shards": unexpected_shards})
    if missing_shards and not allow_partial:
        errors.append({"missing_shards": missing_shards})

    expected_state = _expected_state(model_path)
    mapped_state = _mapped_state(raw_names, expected_state)
    missing_state = sorted(expected_state - mapped_state)
    unexpected_state = sorted(mapped_state - expected_state)
    if missing_state:
        errors.append({"missing_state": missing_state})
    if unexpected_state:
        errors.append({"unexpected_state": unexpected_state})

    seen_experts: set[int] = set()
    invalid_expert_names = []
    duplicate_experts = 0
    for name in raw_names:
        if ".block_sparse_moe.experts." not in name or name.endswith(".input_scale"):
            continue
        match = _MODEL_OPT_EXPERT_RE.match(name)
        if match is None:
            invalid_expert_names.append(name)
            continue
        layer = int(match["layer"])
        expert = int(match["expert"])
        proj = match["proj"]
        kind = match["kind"]
        if not (1 <= layer <= 92 and 0 <= expert < 896):
            invalid_expert_names.append(name)
            continue
        code = ((((layer - 1) * 896 + expert) * 3 + PROJECTIONS[proj]) * 3 + KINDS[kind])
        if code in seen_experts:
            duplicate_experts += 1
        seen_experts.add(code)
    expected_expert_entries = 92 * 896 * 3 * 3
    if len(seen_experts) != expected_expert_entries:
        errors.append({
            "expert_entry_count": len(seen_experts),
            "expected_expert_entry_count": expected_expert_entries,
        })
    if invalid_expert_names:
        errors.append({"invalid_expert_names": sorted(invalid_expert_names)[:20]})
    if duplicate_experts:
        errors.append({"duplicate_expert_entries": duplicate_experts})

    header_summary = []
    for filename in completed:
        path = model_path / filename
        counts = {
            "expert_entries": 0,
            "expert_input_scales": 0,
            "resident_entries": 0,
            "resident_scale_pairs": 0,
        }
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for name in keys:
                match = EXPERT_HEADER_RE.match(name)
                if match is not None:
                    counts["expert_entries"] += 1
                    if match["kind"] == "input_scale":
                        counts["expert_input_scales"] += 1
                    shape = tuple(handle.get_slice(name).get_shape())
                    dtype = handle.get_slice(name).get_dtype()
                    expected_shape, expected_dtype = EXPERT_SHAPES[(match["proj"], match["kind"])]
                    if shape != expected_shape or dtype != expected_dtype:
                        errors.append({
                            "tensor": name,
                            "shape": shape,
                            "dtype": dtype,
                            "expected_shape": expected_shape,
                            "expected_dtype": expected_dtype,
                        })
                    continue
                if name.endswith(".weight_scale"):
                    weight_name = name.removesuffix("_scale")
                    if weight_name in keys:
                        counts["resident_scale_pairs"] += 1
                        weight_shape = tuple(handle.get_slice(weight_name).get_shape())
                        weight_dtype = handle.get_slice(weight_name).get_dtype()
                        scale_shape = tuple(handle.get_slice(name).get_shape())
                        scale_dtype = handle.get_slice(name).get_dtype()
                        expected_scales = math.ceil(weight_shape[0] / 128) * math.ceil(
                            weight_shape[1] / 128
                        )
                        if (
                            math.prod(scale_shape) != expected_scales
                            or weight_dtype != "F8_E4M3"
                            or scale_dtype != "F32"
                        ):
                            errors.append({
                                "tensor": name,
                                "scale_shape": scale_shape,
                                "scale_dtype": scale_dtype,
                                "weight_shape": weight_shape,
                                "weight_dtype": weight_dtype,
                                "expected_scale_elements": expected_scales,
                                "expected_scale_dtype": "F32",
                                "expected_weight_dtype": "F8_E4M3",
                            })
                        mapped_weight = map_weight_name(weight_name)
                        if mapped_weight is not None:
                            mapped_scale = (
                                mapped_weight.removesuffix(".weight")
                                + ".weight_scale_inv"
                            )
                            native_path = (
                                len(weight_shape) == 2
                                and all(dim % 128 == 0 for dim in weight_shape)
                                and not mapped_weight.endswith(
                                    ".self_attn.kv_b_proj.weight"
                                )
                            )
                            module_has_scale = mapped_scale in expected_state
                            if native_path != module_has_scale:
                                errors.append({
                                    "tensor": weight_name,
                                    "mapped_weight": mapped_weight,
                                    "native_fp8_path": native_path,
                                    "module_has_scale": module_has_scale,
                                    "module_scale": mapped_scale,
                                })
                        continue
                counts["resident_entries"] += 1
        header_summary.append({"file": filename, "bytes": path.stat().st_size, **counts})

    cross_shard_scales = []
    for name, filename in weight_map.items():
        if not name.endswith(".weight_scale") or ".block_sparse_moe.experts." in name:
            continue
        weight_name = name.removesuffix("_scale")
        if weight_name in weight_map and weight_map[weight_name] != filename:
            cross_shard_scales.append((weight_name, weight_map[weight_name], filename))
    if cross_shard_scales:
        errors.append({"cross_shard_resident_scales": cross_shard_scales[:20]})

    return {
        "schema_version": 1,
        "model_path": str(model_path.resolve()),
        "allow_partial": allow_partial,
        "status": "passed" if not errors else "failed",
        "index": {
            "tensor_count": len(raw_names),
            "logical_tensor_bytes": int(index.get("metadata", {}).get("total_size", 0)),
            "indexed_shards": len(indexed_shards),
            "completed_shards": len(completed),
            "completed_shard_bytes": sum((model_path / name).stat().st_size for name in completed),
            "missing_shards": missing_shards,
            "unexpected_shards": unexpected_shards,
            "expected_model_state": len(expected_state),
            "mapped_model_state": len(mapped_state),
            "expert_entries": len(seen_experts),
            "cross_shard_resident_scales": len(cross_shard_scales),
        },
        "headers": header_summary,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.model.resolve(), allow_partial=args.allow_partial)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
