#!/usr/bin/env python3
"""Guard-friendly real-payload probe for one completed NVIDIA Kimi K3 expert layer.

The full checkpoint is acquired one expert layer per shard. This probe builds a
temporary one-layer index, drives the production serial ModelOpt NVFP4 loader,
samples 16 experts, releases the 15.52-GiB layer banks, and compares both GB10
decode and prefill SiTU kernels with literal E2M1 dequantization.
"""

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

from sparklab.models.kimi_k3.weight import _MODEL_OPT_EXPERT_RE, _NVFP4_SOURCE_SPEC
from sparklab.models.loader import drop_page_cache
from sparklab.models.nvfp4_banks import load_nvfp4_expert_source_banks


E2M1 = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
])
BANK_NAMES = (
    "gate_up_packed", "gate_up_scale", "gate_up_global",
    "down_packed", "down_scale", "down_global",
)


def _sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in BANK_NAMES:
        tensor = tensors[name].contiguous()
        # NumPy has no float8 scalar type; hash its exact E4M3 storage bytes.
        if tensor.dtype == torch.float8_e4m3fn:
            tensor = tensor.view(torch.uint8)
        value = tensor.numpy()
        digest.update(name.encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _dequant(packed: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
    rows, half_k = packed.shape
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(rows, 2 * half_k).long()
    values = E2M1.to(packed.device)[codes]
    return values * scale.float().repeat_interleave(16, dim=1) * glob.float().unsqueeze(1)


@torch.no_grad()
def _reference(
    hidden: torch.Tensor,
    banks: dict[str, torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(weights.shape[1]):
        gate_up = _dequant(
            banks["gate_up_packed"][expert],
            banks["gate_up_scale"][expert],
            banks["gate_up_global"][expert],
        )
        down = _dequant(
            banks["down_packed"][expert],
            banks["down_scale"][expert],
            banks["down_global"][expert],
        )
        gate, up = (hidden.float() @ gate_up.T).chunk(2, dim=-1)
        active = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate))
        active *= 25.0 * torch.tanh(up / 25.0)
        out += (active @ down.T) * weights[:, expert : expert + 1]
    return out


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    a, b = actual.float(), expected.float()
    rel = torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)
    cosine = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    return {
        "finite": bool(torch.isfinite(a).all()),
        "max_abs_error": float((a - b).abs().max()),
        "relative_l2_error": float(rel),
        "cosine": float(cosine),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=1, help="global decoder layer (1..92)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.model.resolve()
    if not 1 <= args.layer <= 92:
        parser.error("--layer must be in [1, 92]")

    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    selected = {}
    for name, shard in index["weight_map"].items():
        match = _MODEL_OPT_EXPERT_RE.match(name)
        if match is not None and int(match["layer"]) == args.layer:
            selected[name] = shard
    expected_entries = 896 * 3 * 3
    if len(selected) != expected_entries:
        raise RuntimeError(f"layer {args.layer}: found {len(selected)} entries, expected {expected_entries}")
    shards = sorted(set(selected.values()))
    if len(shards) != 1 or not (root / shards[0]).is_file():
        raise RuntimeError(f"layer {args.layer} is not locally complete: {shards}")

    sampled: dict[str, torch.Tensor] = {}
    bank_evidence = {}
    sample_ids = torch.linspace(0, 895, 16).round().to(torch.long)
    config = SimpleNamespace(
        num_layers=2,
        first_k_dense_replace=1,
        num_experts=896,
        hidden_size=7168,
        expert_hidden_size=3584,
        moe_intermediate_size=3072,
    )

    def sink(layer_id: int, banks: dict) -> None:
        if layer_id != 0:
            raise RuntimeError(f"unexpected local layer {layer_id}")
        for name in BANK_NAMES:
            bank = banks[name]
            bank_evidence[name] = {
                "shape": list(bank.tensor.shape),
                "dtype": str(bank.tensor.dtype),
                "bytes": bank.nbytes,
            }
            sampled[name] = bank.tensor[sample_ids].clone()
        for bank in banks.values():
            bank.release()

    with tempfile.TemporaryDirectory(prefix="kimi-k3-layer-") as tmp_name:
        tmp = Path(tmp_name)
        os.symlink(root / shards[0], tmp / shards[0])
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": selected}), encoding="utf-8"
        )
        started = time.perf_counter()
        load_nvfp4_expert_source_banks(
            str(tmp), config, _NVFP4_SOURCE_SPEC,
            drop_page_cache=drop_page_cache, primary=True, layer_sink=sink,
        )
        load_seconds = time.perf_counter() - started

    sample_hash = _sha256(sampled)
    device = torch.device("cuda")
    gpu = {name: tensor.to(device) for name, tensor in sampled.items()}
    torch.manual_seed(20260828)
    hidden = torch.randn(8, 3584, dtype=torch.bfloat16, device=device) / 4
    weights = torch.rand(8, 16, dtype=torch.float32, device=device)
    ids = torch.arange(16, dtype=torch.int32, device=device).expand(8, -1).contiguous()

    expected = _reference(hidden, gpu, weights)
    from sparklab.moe.fused_nvfp4 import (
        fused_experts_decode_nvfp4_marlin,
        fused_experts_nvfp4,
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    prefill = fused_experts_nvfp4(
        hidden, *(gpu[name] for name in BANK_NAMES), weights, ids, 16,
        "situ", False, 4.0, 25.0,
    )
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started

    torch.cuda.synchronize()
    started = time.perf_counter()
    decode = fused_experts_decode_nvfp4_marlin(
        hidden[:1], *(gpu[name] for name in BANK_NAMES), weights[:1], ids[:1],
        "situ", False, 4.0, 25.0,
    )
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started

    prefill_metrics = _metrics(prefill, expected)
    decode_metrics = _metrics(decode, expected[:1])
    passed = all(
        metrics["finite"]
        and metrics["relative_l2_error"] < 0.08
        and metrics["cosine"] > 0.995
        for metrics in (prefill_metrics, decode_metrics)
    )
    record = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "model": str(root),
        "global_layer": args.layer,
        "source_shard": shards[0],
        "source_shard_bytes": (root / shards[0]).stat().st_size,
        "source_entries": len(selected),
        "expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "sample_expert_ids": sample_ids.tolist(),
        "sample_sha256": sample_hash,
        "bank_bytes": sum(item["bytes"] for item in bank_evidence.values()),
        "banks": bank_evidence,
        "load_seconds": load_seconds,
        "prefill": {**prefill_metrics, "seconds": prefill_seconds, "tokens": 8},
        "decode": {**decode_metrics, "seconds": decode_seconds, "tokens": 1},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
