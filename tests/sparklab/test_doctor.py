from __future__ import annotations

from dataclasses import replace

from freetoken.platform.gb10 import GB10Snapshot, assess_gb10

GIB = 1 << 30


def _snapshot() -> GB10Snapshot:
    return GB10Snapshot(
        os_name="Linux",
        machine="aarch64",
        gpu_name="NVIDIA GB10",
        cuda_available=True,
        cuda_version="13.0",
        compute_capability=(12, 1),
        gpu_total_bytes=128 * GIB,
        cuda_free_bytes=80 * GIB,
        integrated_gpu=True,
        memory_total_bytes=121 * GIB,
        memory_available_bytes=80 * GIB,
        swap_total_bytes=8 * GIB,
        swap_free_bytes=8 * GIB,
        storage_path="/models",
        storage_total_bytes=4 * 1024 * GIB,
        storage_free_bytes=2 * 1024 * GIB,
        filesystem="ext4",
        block_device="/dev/nvme0n1p2",
        nvme=True,
        dependencies={"torch": "2.11.0", "triton": "3.6.0"},
    )


def test_gb10_profile_passes_and_preserves_safety_reserve():
    report = assess_gb10(_snapshot())
    assert report["status"] == "supported"
    assert report["supported"] is True and report["ready"] is True
    assert report["memory"]["safe_available_bytes"] == 68 * GIB
    assert report["memory"]["swap_used_bytes"] == 0
    assert report["failures"] == [] and report["warnings"] == []


def test_non_sm121_platform_fails_closed():
    report = assess_gb10(replace(_snapshot(), compute_capability=(12, 0), gpu_name="RTX"))
    assert report["status"] == "unsupported_platform"
    assert report["supported"] is False and report["ready"] is False
    assert "compute_capability" in report["platform_failures"]


def test_swap_usage_blocks_readiness_but_not_gb10_identity():
    report = assess_gb10(replace(_snapshot(), swap_free_bytes=7 * GIB))
    assert report["status"] == "supported_not_ready"
    assert report["supported"] is True and report["ready"] is False
    assert report["failures"] == ["swap_usage"]
    assert report["memory"]["swap_used_bytes"] == GIB
    assert any("return swap usage to zero" in item for item in report["recommendations"])


def test_non_nvme_or_low_capacity_is_an_explicit_warning():
    report = assess_gb10(
        replace(
            _snapshot(),
            nvme=None,
            block_device="/dev/mapper/root",
            storage_free_bytes=100 * GIB,
        )
    )
    assert report["status"] == "supported_with_warnings"
    assert report["ready"] is True
    assert report["warnings"] == ["nvme_storage", "storage_capacity"]
