from __future__ import annotations

import torch

from .base import BaseMoeBackend


class CpuOffloadMoeBackend(BaseMoeBackend):
    """Marker backend for CPU-compute expert offload.

    Like :class:`OffloadMoeBackend`, the real work lives in ``OffloadMoELayer``:
    prefill streams whole expert layers into the GPU double buffer and runs the
    GEMM on the GPU (identical to ``offload``), while decode ships the
    activations to the CPU, computes the experts there (reading the pinned host
    banks at full RAM bandwidth), and ships the results back. The whole decode
    path is CUDA-graph capturable via ``cudaLaunchHostFunc`` submit/sync nodes.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise RuntimeError("CPU-offload MoE is handled by OffloadMoELayer")


class HybridMoeBackend(BaseMoeBackend):
    """Marker backend for hybrid GPU-cache + CPU-overflow expert decode.

    Prefill is identical to ``offload`` (stream whole layers into the GPU double
    buffer, GEMM on the GPU). Decode keeps a full GPU slot cache but caps PCIe
    fetches per (layer, step) -- at ``hybrid_max_fetch`` experts, or at the
    bandwidth-matched ``hybrid_fetch_fraction`` of that step's misses (default:
    the benched pcie_bw/cpu_bw ratio, so the fetch and the CPU compute overlap
    perfectly): the cache hits plus the freshly-fetched experts run on the GPU,
    the remaining misses run on the CPU (reading the pinned host banks), and the
    two partial results are summed. The real work lives in the model's MoE layer
    (``OffloadMoELayer`` / DSV4 ``MoE``).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise RuntimeError("Hybrid MoE is handled by the model's MoE layer")
