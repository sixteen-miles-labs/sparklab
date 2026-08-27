from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.checkpoint.ftw import FTWWriter
from sparklab.acquire import (
    AcquisitionError,
    validate_ftw_checkpoint,
    validate_safetensors_snapshot,
)


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


def _ftw_checkpoint(path):
    writer = FTWWriter(str(path), shard_limit=4096)
    writer.add_tensor("model.a", torch.arange(12, dtype=torch.bfloat16).view(3, 4))
    writer.add_tensor("model.b", torch.arange(1024, dtype=torch.float32))
    external = path / "external.bin"
    external.write_bytes(b"payload")
    return writer.finalize(
        {
            "fingerprint": "0123456789abcdef",
            "counts": {"weight": 2},
            "external_artifacts": [{"file": external.name, "nbytes": 7}],
        }
    )


def test_validate_ftw_checkpoint_checks_shards_tensors_and_external_artifacts(tmp_path):
    index = _ftw_checkpoint(tmp_path)
    result = validate_ftw_checkpoint(tmp_path)
    assert result["fingerprint"] == index["fingerprint"]
    assert result["shards"] == 2
    assert result["tensors"] == 2
    assert result["external_bytes"] == 7
    assert result["kind_counts"] == {"weight": 2}


def test_validate_ftw_checkpoint_rejects_truncated_shard(tmp_path):
    index = _ftw_checkpoint(tmp_path)
    shard = tmp_path / index["shards"][0]["file"]
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(AcquisitionError, match="shard size mismatch"):
        validate_ftw_checkpoint(tmp_path)


def test_validate_ftw_checkpoint_rejects_stale_shard(tmp_path):
    _ftw_checkpoint(tmp_path)
    (tmp_path / "stale.ftw").write_bytes(b"stale")
    with pytest.raises(AcquisitionError, match="shard set mismatch"):
        validate_ftw_checkpoint(tmp_path)


def test_validate_ftw_checkpoint_rejects_bad_tensor_range(tmp_path):
    _ftw_checkpoint(tmp_path)
    index_path = tmp_path / "freetoken_weight.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["tensors"][1]["global_off"] += 4096
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(AcquisitionError, match="tensor range"):
        validate_ftw_checkpoint(tmp_path)
