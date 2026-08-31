"""Read-only NVIDIA GB10 platform inspection.

SparkLab owns this product policy. Collection is separated from assessment so the
policy is deterministic and unit-testable without CUDA hardware.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform as host_platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GIB = 1 << 30
GB10_MIN_PHYSICAL_BYTES = 115 * GIB
GB10_SAFETY_RESERVE_BYTES = 12 * GIB
GB10_RECOMMENDED_STORAGE_FREE_BYTES = 512 * GIB


@dataclass(frozen=True)
class GB10Snapshot:
    os_name: str
    machine: str
    gpu_name: str | None
    cuda_available: bool
    cuda_version: str | None
    compute_capability: tuple[int, int] | None
    gpu_total_bytes: int | None
    cuda_free_bytes: int | None
    integrated_gpu: bool | None
    memory_total_bytes: int
    memory_available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int
    storage_path: str
    storage_total_bytes: int
    storage_free_bytes: int
    filesystem: str | None
    block_device: str | None
    nvme: bool | None
    dependencies: dict[str, str | None]


def _read_meminfo(path: str = "/proc/meminfo") -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                key, sep, raw = line.partition(":")
                if not sep:
                    continue
                fields = raw.strip().split()
                if fields:
                    values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def _mount_for(path: str, mountinfo: str = "/proc/self/mountinfo") -> tuple[str | None, str | None]:
    resolved = os.path.realpath(path)
    best: tuple[int, str, str] | None = None
    try:
        with open(mountinfo, encoding="utf-8") as handle:
            for line in handle:
                left, sep, right = line.rstrip().partition(" - ")
                if not sep:
                    continue
                left_fields = left.split()
                right_fields = right.split()
                if len(left_fields) < 5 or len(right_fields) < 2:
                    continue
                mountpoint = left_fields[4].replace("\\040", " ")
                if resolved != mountpoint and not resolved.startswith(mountpoint.rstrip("/") + "/"):
                    continue
                candidate = (len(mountpoint), right_fields[0], right_fields[1])
                if best is None or candidate[0] > best[0]:
                    best = candidate
    except OSError:
        return None, None
    return (best[1], best[2]) if best is not None else (None, None)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_gb10_snapshot(storage_path: str = ".") -> GB10Snapshot:
    """Collect platform facts without allocating model weights or running kernels."""
    target = str(Path(storage_path).expanduser().resolve())
    usage = shutil.disk_usage(target)
    mem = _read_meminfo()
    filesystem, block_device = _mount_for(target)

    cuda_available = False
    cuda_version = None
    capability = None
    gpu_name = None
    gpu_total = None
    cuda_free = None
    integrated = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda_available:
            capability = tuple(int(x) for x in torch.cuda.get_device_capability())
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_total = int(props.total_memory)
            integrated_value = getattr(props, "is_integrated", None)
            integrated = bool(integrated_value) if integrated_value is not None else None
            try:
                cuda_free = int(torch.cuda.mem_get_info()[0])
            except RuntimeError:
                cuda_free = None
    except (ImportError, RuntimeError):
        pass

    swap_total = mem.get("SwapTotal", 0)
    swap_free = mem.get("SwapFree", 0)
    try:
        from sparklab.version import __version__ as runtime_version
    except ImportError:
        runtime_version = None

    return GB10Snapshot(
        os_name=host_platform.system(),
        machine=host_platform.machine(),
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        compute_capability=capability,
        gpu_total_bytes=gpu_total,
        cuda_free_bytes=cuda_free,
        integrated_gpu=integrated,
        memory_total_bytes=mem.get("MemTotal", 0),
        memory_available_bytes=mem.get("MemAvailable", 0),
        swap_total_bytes=swap_total,
        swap_free_bytes=swap_free,
        storage_path=target,
        storage_total_bytes=int(usage.total),
        storage_free_bytes=int(usage.free),
        filesystem=filesystem,
        block_device=block_device,
        nvme=("nvme" in block_device.lower()) if block_device else None,
        dependencies={
            # Use the imported code version for SparkLab, not distribution metadata. An
            # editable checkout can move to a new release while its installed companion
            # cache remains on the previous one -- the exact drift doctor must detect.
            "sparklab": runtime_version,
            "sparklab-kernel-cache": _package_version("sparklab-kernel-cache"),
            "torch": _package_version("torch"),
            "triton": _package_version("triton"),
            "flashinfer-python": _package_version("flashinfer-python"),
            "sglang-kernel": _package_version("sglang-kernel"),
        },
    )


def _check(name: str, status: str, observed: Any, expected: str, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "observed": observed,
        "expected": expected,
        "detail": detail,
    }


def assess_gb10(snapshot: GB10Snapshot) -> dict[str, Any]:
    """Apply SparkLab's GB10 production policy to a collected snapshot."""
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "operating_system",
            "pass" if snapshot.os_name == "Linux" else "fail",
            snapshot.os_name,
            "Linux / DGX OS",
            "SparkLab's supported runtime is Linux-only.",
        )
    )
    checks.append(
        _check(
            "cpu_architecture",
            "pass" if snapshot.machine.lower() in {"aarch64", "arm64"} else "fail",
            snapshot.machine,
            "aarch64",
            "GB10 pairs the Blackwell GPU with a Grace ARM64 CPU.",
        )
    )
    checks.append(
        _check(
            "cuda",
            "pass" if snapshot.cuda_available else "fail",
            snapshot.cuda_version,
            "CUDA 13.x available",
            "CUDA must be visible to the Python runtime.",
        )
    )
    cuda_major = None
    try:
        cuda_major = int((snapshot.cuda_version or "").split(".", 1)[0])
    except ValueError:
        pass
    checks.append(
        _check(
            "cuda_version",
            "pass" if cuda_major == 13 else "fail",
            snapshot.cuda_version,
            "13.x",
            "SparkLab's GB10 kernel contract targets CUDA 13.",
        )
    )
    checks.append(
        _check(
            "compute_capability",
            "pass" if snapshot.compute_capability == (12, 1) else "fail",
            list(snapshot.compute_capability) if snapshot.compute_capability else None,
            "SM121 (12.1)",
            "Architecture-specific kernels must fail closed on a different GPU.",
        )
    )
    gpu_marker = (snapshot.gpu_name or "").lower()
    checks.append(
        _check(
            "gpu_identity",
            "pass" if "gb10" in gpu_marker else "warn",
            snapshot.gpu_name,
            "NVIDIA GB10",
            "SM121 is authoritative; the device name is retained as supporting evidence.",
        )
    )
    checks.append(
        _check(
            "unified_memory",
            "pass" if snapshot.memory_total_bytes >= GB10_MIN_PHYSICAL_BYTES else "fail",
            snapshot.memory_total_bytes,
            f">= {GB10_MIN_PHYSICAL_BYTES} bytes",
            "The 128 GB product profile exposes roughly 119 GiB to Linux.",
        )
    )
    available_status = (
        "pass" if snapshot.memory_available_bytes >= 24 * GIB else
        "warn" if snapshot.memory_available_bytes >= GB10_SAFETY_RESERVE_BYTES else "fail"
    )
    checks.append(
        _check(
            "available_memory",
            available_status,
            snapshot.memory_available_bytes,
            f">= {GB10_SAFETY_RESERVE_BYTES} bytes safety reserve",
            "Model recipes consume only memory above the operational reserve.",
        )
    )
    swap_used = max(0, snapshot.swap_total_bytes - snapshot.swap_free_bytes)
    checks.append(
        _check(
            "swap_usage",
            "pass" if swap_used == 0 else "fail",
            swap_used,
            "0 bytes used",
            "Swap is never counted as model capacity.",
        )
    )
    checks.append(
        _check(
            "nvme_storage",
            "pass" if snapshot.nvme is True else "warn",
            snapshot.block_device,
            "local NVMe-backed filesystem",
            "Device-mapper filesystems may require manual NVMe verification.",
        )
    )
    checks.append(
        _check(
            "storage_capacity",
            "pass" if snapshot.storage_free_bytes >= GB10_RECOMMENDED_STORAGE_FREE_BYTES else "warn",
            snapshot.storage_free_bytes,
            f">= {GB10_RECOMMENDED_STORAGE_FREE_BYTES} bytes free",
            "Large FTW and Kimi K3 workflows require substantial local capacity.",
        )
    )

    runtime_version = snapshot.dependencies.get("sparklab")
    kernel_cache_version = snapshot.dependencies.get("sparklab-kernel-cache")
    if runtime_version is not None and kernel_cache_version is not None:
        from sparklab.kernels.utils import _kernel_cache_version_ok

        cache_matches = _kernel_cache_version_ok(kernel_cache_version, runtime_version)
        checks.append(
            _check(
                "kernel_cache_version",
                "pass" if cache_matches else "fail",
                {
                    "sparklab": runtime_version,
                    "sparklab-kernel-cache": kernel_cache_version,
                },
                "same release and build stamp",
                (
                    "The prebuilt CUDA kernels must come from the same SparkLab release "
                    "and, when stamped, the same source build."
                ),
            )
        )

    failures = [item["name"] for item in checks if item["status"] == "fail"]
    warnings = [item["name"] for item in checks if item["status"] == "warn"]
    identity_checks = {
        "operating_system",
        "cpu_architecture",
        "cuda",
        "cuda_version",
        "compute_capability",
        "unified_memory",
    }
    platform_failures = [name for name in failures if name in identity_checks]
    safe_available = max(0, snapshot.memory_available_bytes - GB10_SAFETY_RESERVE_BYTES)
    if platform_failures:
        status = "unsupported_platform"
    elif failures:
        status = "supported_not_ready"
    elif warnings:
        status = "supported_with_warnings"
    else:
        status = "supported"
    recommendations = []
    if failures:
        recommendations.append("Resolve failed platform checks before loading a model.")
    if "swap_usage" in failures:
        recommendations.append(
            "Finish active workloads and return swap usage to zero before starting a certified recipe."
        )
    if "nvme_storage" in warnings:
        recommendations.append("Confirm the selected model directory is backed by local NVMe.")
    if "storage_capacity" in warnings:
        recommendations.append("Choose a storage path with at least 512 GiB free for frontier recipes.")
    if "kernel_cache_version" in failures:
        recommendations.append(
            "Reinstall sparklab and sparklab-kernel-cache as a matched wheel pair before "
            "loading a model."
        )
    if not recommendations:
        recommendations.append("Platform identity is ready for recipe-level model validation.")

    return {
        "schema_version": "1.0",
        "product": "SparkLab",
        "profile": "gb10",
        "status": status,
        "supported": not platform_failures,
        "ready": not failures,
        "failures": failures,
        "platform_failures": platform_failures,
        "warnings": warnings,
        "checks": checks,
        "memory": {
            "physical_bytes": snapshot.memory_total_bytes,
            "available_bytes": snapshot.memory_available_bytes,
            "safe_available_bytes": safe_available,
            "safety_reserve_bytes": GB10_SAFETY_RESERVE_BYTES,
            "cuda_free_bytes": snapshot.cuda_free_bytes,
            "swap_total_bytes": snapshot.swap_total_bytes,
            "swap_used_bytes": swap_used,
        },
        "storage": {
            "path": snapshot.storage_path,
            "filesystem": snapshot.filesystem,
            "block_device": snapshot.block_device,
            "total_bytes": snapshot.storage_total_bytes,
            "free_bytes": snapshot.storage_free_bytes,
            "nvme": snapshot.nvme,
        },
        "runtime": {
            "gpu": snapshot.gpu_name,
            "cuda": snapshot.cuda_version,
            "compute_capability": (
                list(snapshot.compute_capability) if snapshot.compute_capability else None
            ),
            "integrated_gpu": snapshot.integrated_gpu,
            "dependencies": snapshot.dependencies,
        },
        "snapshot": asdict(snapshot),
        "recommendations": recommendations,
    }


__all__ = ["GB10Snapshot", "assess_gb10", "collect_gb10_snapshot"]
