from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sparklab.checkpoint.ftw import FTWWriter
from sparklab.acquire import (
    AcquisitionError,
    acquire_recipe,
    validate_ftw_checkpoint,
    validate_safetensors_snapshot,
)
from sparklab.catalog import RuntimeArtifact, get_recipe


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


def test_validate_safetensors_snapshot_reports_stale_published_total(tmp_path):
    _, total = _indexed_snapshot(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["metadata"]["total_size"] = total - 4
    index_path.write_text(json.dumps(index), encoding="utf-8")

    result = validate_safetensors_snapshot(tmp_path)
    assert result["logical_bytes"] == total
    assert result["published_logical_bytes"] == total - 4
    assert result["published_logical_bytes_delta"] == 4


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


def test_validate_ftw_checkpoint_ignores_explicit_zero_kind_count(tmp_path):
    _ftw_checkpoint(tmp_path)
    index_path = tmp_path / "freetoken_weight.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["counts"]["experts_bank"] = 0
    index_path.write_text(json.dumps(index), encoding="utf-8")

    result = validate_ftw_checkpoint(tmp_path)

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


def test_acquire_recipe_downloads_and_manifests_pinned_prebuilt_ftw(tmp_path):
    recipe = replace(
        get_recipe("qwen3.8-flash-next"),
        source_bytes=1,
        prepared_bytes=1,
        minimum_free_bytes=2,
        runtime_artifact=RuntimeArtifact(
            repo_id="sparklab/qwen-ftw",
            revision="a" * 40,
            bytes=1,
            fingerprint="0123456789abcdef",
        ),
    )
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        _ftw_checkpoint(destination)
        return str(destination)

    result = acquire_recipe(
        recipe,
        root=str(tmp_path),
        prepare=True,
        downloader=downloader,
    )

    assert calls == [
        {
            "repo_id": "sparklab/qwen-ftw",
            "revision": "a" * 40,
            "local_dir": result["artifact_plan"]["prepared_path"],
        }
    ]
    assert result["acquisition"] == "prebuilt-runtime"
    assert result["manifest"]["artifacts"]["source"] is None
    runtime = result["manifest"]["artifacts"]["runtime"]
    assert runtime["repository"] == "sparklab/qwen-ftw"
    assert runtime["revision"] == "a" * 40
    assert runtime["fingerprint"] == "0123456789abcdef"


def test_acquire_recipe_rejects_wrong_prebuilt_fingerprint(tmp_path):
    recipe = replace(
        get_recipe("qwen3.8-flash-next"),
        source_bytes=1,
        prepared_bytes=1,
        minimum_free_bytes=2,
        runtime_artifact=RuntimeArtifact(
            repo_id="sparklab/qwen-ftw",
            revision="b" * 40,
            bytes=1,
            fingerprint="wrong",
        ),
    )

    def downloader(**kwargs):
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        _ftw_checkpoint(destination)
        return str(destination)

    with pytest.raises(AcquisitionError, match="fingerprint mismatch"):
        acquire_recipe(
            recipe,
            root=str(tmp_path),
            prepare=True,
            downloader=downloader,
        )


def test_acquire_recipe_can_force_source_instead_of_prebuilt(tmp_path):
    recipe = replace(
        get_recipe("qwen3.8-flash-next"),
        source_bytes=1,
        prepared_bytes=1,
        minimum_free_bytes=2,
        runtime_artifact=RuntimeArtifact(
            repo_id="sparklab/qwen-ftw",
            revision="c" * 40,
            bytes=1,
            fingerprint="0123456789abcdef",
        ),
    )
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        return str(destination)

    def converter(_source, destination, **_kwargs):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        return _ftw_checkpoint(path)

    result = acquire_recipe(
        recipe,
        root=str(tmp_path),
        prepare=True,
        from_source=True,
        downloader=downloader,
        converter=converter,
    )

    assert calls[0]["repo_id"] == recipe.model
    assert result["acquisition"] == "source-conversion"
    assert result["manifest"]["artifacts"]["source"] is not None
