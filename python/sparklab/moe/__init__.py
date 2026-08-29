from __future__ import annotations

from typing import Protocol

from sparklab.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")

# Backends that serve experts from CPU (pinned) host banks through an
# ``OffloadMoeCache`` -- the GPU only holds the two-layer prefill double buffer.
# They differ only in how *decode* gets the experts: ``offload`` streams the
# missing experts over PCIe into a GPU slot cache and runs the GEMM on the GPU;
# ``cpu`` ships the activations to the CPU, computes the experts there (high RAM
# bandwidth), and ships the results back; ``hybrid`` keeps a full GPU slot cache
# AND a CPU executor -- it fetches at most K missing experts/layer over PCIe
# (computed on the GPU with the cache hits) and computes the remaining misses on
# the CPU, overlapped, then merges (capped PCIe + CPU overflow). All build their
# cache the same way, so model layer construction and the engine wiring key off
# this set rather than a bare ``== "offload"`` check.
OFFLOAD_MOE_BACKENDS = frozenset({"offload", "cpu", "hybrid"})


def is_offload_moe_backend(backend: str) -> bool:
    return backend in OFFLOAD_MOE_BACKENDS


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()


@SUPPORTED_MOE_BACKENDS.register("offload")
def create_offload_moe_backend():
    from .offload import OffloadMoeBackend

    return OffloadMoeBackend()


@SUPPORTED_MOE_BACKENDS.register("cpu")
def create_cpu_moe_backend():
    from .cpu_offload import CpuOffloadMoeBackend

    return CpuOffloadMoeBackend()


@SUPPORTED_MOE_BACKENDS.register("hybrid")
def create_hybrid_moe_backend():
    from .cpu_offload import HybridMoeBackend

    return HybridMoeBackend()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
    "OFFLOAD_MOE_BACKENDS",
    "is_offload_moe_backend",
]
