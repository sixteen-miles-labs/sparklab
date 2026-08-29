#!/usr/bin/env python3
"""Round-trip one genuine Kimi K3 expert layer through the streaming FTW path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import torch

from sparklab.checkpoint.convert import _ConvertSink
from sparklab.checkpoint.ftw import DEFAULT_SHARD_LIMIT, FTWReader, FTWWriter
from sparklab.models.kimi_k3.weight import _MODEL_OPT_EXPERT_RE, _NVFP4_SOURCE_SPEC
from sparklab.models.loader import drop_page_cache
from sparklab.models.nvfp4_banks import load_nvfp4_expert_source_banks


BANK_NAMES = (
    "gate_up_packed", "gate_up_scale", "gate_up_global",
    "down_packed", "down_scale", "down_global",
)


def _sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in BANK_NAMES:
        tensor = tensors[name].contiguous()
        if tensor.dtype == torch.float8_e4m3fn:
            tensor = tensor.view(torch.uint8)
        digest.update(name.encode())
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.model.resolve()
    if not 1 <= args.layer <= 92:
        parser.error("--layer must be in [1, 92]")

    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    selected = {
        name: shard
        for name, shard in index["weight_map"].items()
        if (match := _MODEL_OPT_EXPERT_RE.match(name)) is not None
        and int(match["layer"]) == args.layer
    }
    expected_entries = 896 * 3 * 3
    if len(selected) != expected_entries:
        raise RuntimeError(f"layer {args.layer}: {len(selected)} entries, expected {expected_entries}")
    shards = sorted(set(selected.values()))
    if len(shards) != 1 or not (root / shards[0]).is_file():
        raise RuntimeError(f"layer {args.layer} is not locally complete: {shards}")

    config = SimpleNamespace(
        num_layers=2,
        first_k_dense_replace=1,
        num_experts=896,
        hidden_size=7168,
        expert_hidden_size=3584,
        moe_intermediate_size=3072,
    )
    sample_ids = torch.linspace(0, 895, 16).round().to(torch.long).tolist()
    source_samples: dict[str, torch.Tensor] = {}

    with tempfile.TemporaryDirectory(prefix="kimi-k3-source-") as source_name, tempfile.TemporaryDirectory(
        prefix="kimi-k3-ftw-"
    ) as ftw_name:
        source = Path(source_name)
        os.symlink(root / shards[0], source / shards[0])
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": selected}), encoding="utf-8"
        )

        writer = FTWWriter(ftw_name, shard_limit=DEFAULT_SHARD_LIMIT)
        convert_sink = _ConvertSink(writer, desc="Writing one Kimi K3 FTW layer")

        def sink(layer_id: int, banks: dict) -> None:
            if layer_id != 0:
                raise RuntimeError(f"unexpected local layer {layer_id}")
            for name in BANK_NAMES:
                source_samples[name] = banks[name].tensor[sample_ids].clone()
            convert_sink(layer_id, banks)  # writes all six banks, then releases them

        started = time.perf_counter()
        load_nvfp4_expert_source_banks(
            str(source), config, _NVFP4_SOURCE_SPEC,
            drop_page_cache=drop_page_cache, primary=True, layer_sink=sink,
        )
        convert_sink.close()
        ftw_index = writer.finalize({
            "source_model_path": str(root),
            "fingerprint": "one-layer-roundtrip-probe",
            "quant_format": "nvfp4",
            "expert_bank_num_layers": 1,
            "counts": {"weight": 0, "experts_bank": 6},
        })
        write_seconds = time.perf_counter() - started
        source_sha = _sha256(source_samples)

        started = time.perf_counter()
        reader = FTWReader(ftw_name)
        try:
            descriptors = reader.expert_row_descriptors(num_layers=1)
            roundtrip_samples = {
                name: torch.stack([
                    reader.read_expert_row(descriptors[(0, expert, name)])
                    for expert in sample_ids
                ])
                for name in BANK_NAMES
            }
        finally:
            reader.close()
        read_seconds = time.perf_counter() - started
        roundtrip_sha = _sha256(roundtrip_samples)
        descriptor_count = len(descriptors)
        shard_evidence = [
            {"file": shard["file"], "bytes": shard["nbytes"]}
            for shard in ftw_index["shards"]
        ]
        tensor_evidence = [
            {
                "name": tensor["name"],
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "bytes": tensor["nbytes"],
            }
            for tensor in ftw_index["tensors"]
        ]

    expected_descriptors = 896 * len(BANK_NAMES)
    passed = source_sha == roundtrip_sha and descriptor_count == expected_descriptors
    record = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "model": str(root),
        "global_layer": args.layer,
        "source_shard": shards[0],
        "sample_expert_ids": sample_ids,
        "source_sample_sha256": source_sha,
        "roundtrip_sample_sha256": roundtrip_sha,
        "descriptor_count": descriptor_count,
        "expected_descriptor_count": expected_descriptors,
        "ftw_total_bytes": ftw_index["total_bytes"],
        "ftw_shards": shard_evidence,
        "ftw_tensors": tensor_evidence,
        "write_seconds": write_seconds,
        "sample_read_seconds": read_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
