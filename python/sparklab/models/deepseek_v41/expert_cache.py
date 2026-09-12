"""Bounded packed expert bank for DeepSeek V4.1 Flash."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import math
import os

import torch


class ExpertBank:
    """LRU of complete routed experts in checkpoint-native GPU tensors."""

    def __init__(self, args, store):
        raw = os.getenv("SPARKLAB_DSV41_EXPERT_CACHE_GB", "64")
        try:
            budget = int(raw) << 30
        except ValueError as exc:
            raise ValueError(f"invalid SPARKLAB_DSV41_EXPERT_CACHE_GB={raw!r}") from exc
        if budget < 0:
            raise ValueError("SPARKLAB_DSV41_EXPERT_CACHE_GB must be non-negative")
        self.args = args
        self.store = store
        self.device = store.device
        self.entries: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dsv41-expert")
        self.free_slots: list[int] = []
        self.gate_up = self.gate_up_scale = self.down = self.down_scale = None

        prefix = "layers.0.ffn.experts.0"
        specs = [
            store.metadata[prefix + suffix]
            for suffix in (
                ".w1.weight", ".w3.weight", ".w1.scale", ".w3.scale",
                ".w2.weight", ".w2.scale",
            )
        ]
        bytes_per = sum(
            torch.empty((), dtype=spec[3]).element_size()
            * math.prod(spec[2])
            for spec in specs
        )
        self.capacity = budget // bytes_per
        if self.capacity and self.capacity < args.n_routed_experts:
            raise ValueError("V4.1 expert cache must hold one layer's routed experts")

    def _allocate(self):
        if self.gate_up is not None or not self.capacity:
            return
        a = self.args
        self.gate_up = torch.empty(
            self.capacity, 2 * a.moe_inter_dim, a.dim // 2,
            dtype=torch.uint8, device=self.device,
        )
        self.gate_up_scale = torch.empty(
            self.capacity, 2 * a.moe_inter_dim, a.dim // 32,
            dtype=torch.uint8, device=self.device,
        )
        self.down = torch.empty(
            self.capacity, a.dim, a.moe_inter_dim // 2,
            dtype=torch.uint8, device=self.device,
        )
        self.down_scale = torch.empty(
            self.capacity, a.dim, a.moe_inter_dim // 32,
            dtype=torch.uint8, device=self.device,
        )
        self.free_slots = list(range(self.capacity - 1, -1, -1))

    def _read_expert(self, key):
        layer, expert = key
        root = f"layers.{layer}.ffn.experts.{expert}"
        return [self.store._read_host(root + suffix) for suffix in (
            ".w1.weight", ".w3.weight", ".w1.scale", ".w3.scale",
            ".w2.weight", ".w2.scale",
        )]

    def _copy_expert(self, tensors, slot: int):
        width = self.args.moe_inter_dim
        for tensor, start in zip(tensors[:2], (0, width)):
            self.gate_up[slot, start:start + width].copy_(
                tensor.view(torch.uint8)
            )
        for tensor, start in zip(tensors[2:4], (0, width)):
            self.gate_up_scale[slot, start:start + width].copy_(
                tensor.view(torch.uint8)
            )
        self.down[slot].copy_(tensors[4].view(torch.uint8))
        self.down_scale[slot].copy_(tensors[5].view(torch.uint8))

    def slots(self, layer: int, experts: torch.Tensor) -> torch.Tensor | None:
        if not self.capacity:
            return None
        self._allocate()
        keys = [(layer, int(expert)) for expert in experts.reshape(-1).cpu().tolist()]
        unique = list(dict.fromkeys(keys))
        protected = set(unique)
        missing_count = sum(key not in self.entries for key in unique)
        while len(self.free_slots) < missing_count:
            victim = next(key for key in self.entries if key not in protected)
            self.free_slots.append(self.entries.pop(victim))

        missing = []
        for key in unique:
            slot = self.entries.get(key)
            if slot is None:
                slot = self.free_slots.pop()
                self.entries[key] = slot
                missing.append((key, slot))
            self.entries.move_to_end(key)
        result = [self.entries[key] for key in keys]
        # Limit staged host data to roughly 150 MiB while keeping NVMe queue
        # depth high enough for the large, independently located tensors.
        for start in range(0, len(missing), 8):
            chunk = missing[start:start + 8]
            loaded = self.executor.map(self._read_expert, (key for key, _ in chunk))
            for (key, slot), tensors in zip(chunk, loaded):
                self._copy_expert(tensors, slot)
        return torch.tensor(result, dtype=torch.int32, device=self.device).reshape(experts.shape)

    def close(self):
        self.executor.shutdown(wait=True)

    def __del__(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

__all__ = ["ExpertBank"]
