from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from freetoken.checkpoint.ftw import FTWWriter
from sparklab.catalog import RuntimeArtifact, get_recipe
from sparklab.paths import manifest_path, prepared_path, source_path
from sparklab.platform import GB10Snapshot
from sparklab.runtime import RuntimePlanError, plan_invocation

GIB = 1 << 30


def _ready_snapshot() -> GB10Snapshot:
    return GB10Snapshot(
        os_name="Linux",
        machine="aarch64",
        gpu_name="NVIDIA GB10",
        cuda_available=True,
        cuda_version="13.0",
        compute_capability=(12, 1),
        gpu_total_bytes=128 * GIB,
        cuda_free_bytes=100 * GIB,
        integrated_gpu=True,
        memory_total_bytes=121 * GIB,
        memory_available_bytes=121 * GIB,
        swap_total_bytes=8 * GIB,
        swap_free_bytes=8 * GIB,
        storage_path="/models",
        storage_total_bytes=4 * 1024 * GIB,
        storage_free_bytes=2 * 1024 * GIB,
        filesystem="ext4",
        block_device="/dev/nvme0n1p2",
        nvme=True,
        dependencies={},
    )


def test_runtime_routes_resident_recipe_through_selected_backend(tmp_path):
    recipe = replace(
        get_recipe("qwen3.6-35b-a3b"),
        runtime_memory={"total_bytes": 64 * GIB},
    )
    checkpoint = source_path(recipe, str(tmp_path))
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    invocation = plan_invocation(
        recipe,
        _ready_snapshot(),
        root=str(tmp_path),
        extra_args=("--port", "1919"),
    )

    assert invocation.backend == "native"
    assert invocation.checkpoint == str(checkpoint.resolve())
    assert invocation.arguments[-2:] == ("--port", "1919")
    assert invocation.to_dict()["backend_version"]


def test_runtime_rejects_prebuilt_artifact_with_wrong_fingerprint(tmp_path):
    recipe = replace(
        get_recipe("qwen3.8-flash-next"),
        runtime_memory={"total_bytes": 64 * GIB},
        runtime_artifact=RuntimeArtifact(
            repo_id="freetoken/qwen-ftw",
            revision="e" * 40,
            bytes=4096,
            fingerprint="expected",
        ),
    )
    checkpoint = prepared_path(recipe, str(tmp_path))
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    writer = FTWWriter(str(checkpoint), shard_limit=4096)
    writer.add_tensor("weight", torch.ones(1))
    writer.finalize({"fingerprint": "actual", "counts": {"weight": 1}})
    manifest = {
        "schema_version": "2.0",
        "artifacts": {
            "source": None,
            "runtime": {
                "path": str(checkpoint),
                "repository": "freetoken/qwen-ftw",
                "revision": "e" * 40,
            },
        },
    }
    manifest_path(recipe, str(tmp_path)).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(recipe, str(tmp_path)).write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(RuntimePlanError, match="fingerprint mismatch"):
        plan_invocation(recipe, _ready_snapshot(), root=str(tmp_path))
