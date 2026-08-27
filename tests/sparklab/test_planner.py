from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from freetoken.platform.gb10 import GB10Snapshot
from sparklab.acquire import acquire_recipe
from sparklab.catalog import get_recipe
from sparklab.planner import plan_artifacts, plan_runtime

GIB = 1 << 30


def _snapshot(*, available: int = 100 * GIB, swap_used: int = 0) -> GB10Snapshot:
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
        memory_available_bytes=available,
        swap_total_bytes=8 * GIB,
        swap_free_bytes=8 * GIB - swap_used,
        storage_path="/models",
        storage_total_bytes=4 * 1024 * GIB,
        storage_free_bytes=2 * 1024 * GIB,
        filesystem="ext4",
        block_device="/dev/nvme0n1p2",
        nvme=True,
        dependencies={},
    )


def test_runtime_plan_reserves_physical_memory_and_fails_without_measurement():
    recipe = replace(get_recipe("qwen3.8-flash-next"), runtime_memory=None)
    unknown = plan_runtime(recipe, _snapshot())
    assert unknown.ready is False
    assert "no measured GB10 runtime-memory budget" in unknown.reasons[0]

    measured = replace(recipe, runtime_memory={"total_bytes": 80 * GIB})
    plan = plan_runtime(measured, _snapshot())
    assert plan.ready is True
    assert plan.usable_bytes == 88 * GIB
    assert plan.headroom_bytes == 8 * GIB


def test_runtime_plan_never_counts_swap_as_capacity():
    recipe = replace(
        get_recipe("qwen3.8-flash-next"),
        runtime_memory={"total_bytes": 80 * GIB},
    )
    plan = plan_runtime(recipe, _snapshot(swap_used=GIB))
    assert plan.ready is False
    assert plan.swap_used_bytes == GIB
    assert any("swap is in use" in reason for reason in plan.reasons)


def test_artifact_plan_accounts_for_source_prepare_and_safety_margin(tmp_path, monkeypatch):
    recipe = replace(
        get_recipe("qwen3.8-flash-next"),
        source_bytes=10 * GIB,
        prepared_bytes=9 * GIB,
        minimum_free_bytes=24 * GIB,
    )
    monkeypatch.setattr(
        "sparklab.planner.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=100 * GIB, used=70 * GIB, free=30 * GIB),
    )
    source_only = plan_artifacts(recipe, root=str(tmp_path), include_prepared=False)
    complete = plan_artifacts(recipe, root=str(tmp_path), include_prepared=True)
    assert source_only.required_bytes == 15 * GIB
    assert complete.required_bytes == 24 * GIB
    assert source_only.ready is complete.ready is True


def test_acquisition_is_pinned_and_manifest_marks_only_completed_download(tmp_path):
    recipe = replace(
        get_recipe("qwen3.8-flash-next"),
        source_bytes=1,
        prepared_bytes=1,
        minimum_free_bytes=2,
    )
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        return str(destination)

    result = acquire_recipe(recipe, root=str(tmp_path), downloader=downloader)
    assert calls[0]["revision"] == recipe.revision
    assert calls[0]["repo_id"] == recipe.model
    assert result["manifest"]["revision"] == recipe.revision
    assert Path(result["manifest"]["source_path"], "config.json").is_file()
