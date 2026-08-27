from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from sparklab.acquire import AcquisitionError, validate_safetensors_snapshot


def _indexed_snapshot(path):
    tensors = {
        "model.a": torch.arange(12, dtype=torch.bfloat16).view(3, 4),
        "model.b": torch.arange(5, dtype=torch.float32),
    }
    weight_map = {}
    total = 0
    for index, (name, tensor) in enumerate(tensors.items()):
        filename = f"model-{index}.safetensors"
        save_file({name: tensor}, path / filename)
        weight_map[name] = filename
        total += tensor.numel() * tensor.element_size()
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return weight_map, total


def test_validate_safetensors_snapshot_checks_all_headers(tmp_path):
    weight_map, total = _indexed_snapshot(tmp_path)
    result = validate_safetensors_snapshot(tmp_path)
    assert result["shards"] == 2
    assert result["tensors"] == len(weight_map)
    assert result["logical_bytes"] == total
    assert result["physical_bytes"] > total


def test_validate_safetensors_snapshot_rejects_missing_shard(tmp_path):
    weight_map, _ = _indexed_snapshot(tmp_path)
    (tmp_path / next(iter(weight_map.values()))).unlink()
    with pytest.raises(AcquisitionError, match="missing shard"):
        validate_safetensors_snapshot(tmp_path)


def test_validate_safetensors_snapshot_allows_unindexed_checkpoint(tmp_path):
    assert validate_safetensors_snapshot(tmp_path) is None
