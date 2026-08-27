from __future__ import annotations

from dataclasses import replace

from sparklab.catalog import get_recipe
from sparklab.paths import source_path
from sparklab.platform import GB10Snapshot
from sparklab.runtime import plan_invocation

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
