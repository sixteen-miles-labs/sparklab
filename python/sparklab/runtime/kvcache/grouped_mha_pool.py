from __future__ import annotations

from typing import Sequence

import torch

from .base import BaseKVCachePool, spec_kv_bytes_per_token
from .mha_pool import MHAKVCache


class GroupedMHAKVCache(BaseKVCachePool):
    """Paged GQA caches for models with more than one KV geometry.

    Target and draft towers can share the scheduler's physical token locations while
    using different head counts and head dimensions.  Each geometry therefore owns a
    compact :class:`MHAKVCache`; this facade routes global layer ids to the right one.
    """

    def __init__(
        self,
        *,
        groups: Sequence,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._groups: list[MHAKVCache] = []
        self._layer_to_group: dict[int, MHAKVCache] = {}
        self._num_layers = num_layers
        self._device = device
        self._dtype = dtype
        for group in groups:
            if not group.layer_ids:
                continue
            pool = MHAKVCache(
                num_kv_heads=group.num_kv_heads,
                num_layers=num_layers,
                head_dim=group.head_dim,
                num_pages=num_pages,
                page_size=page_size,
                dtype=dtype,
                device=device,
                layer_ids=group.layer_ids,
            )
            self._groups.append(pool)
            for layer_id in group.layer_ids:
                if layer_id in self._layer_to_group:
                    raise ValueError(f"KV layer {layer_id} belongs to multiple groups")
                self._layer_to_group[layer_id] = pool

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        for pool in self._groups:
            pool.rebuild(num_pages + 1)

    def unit_bytes(self) -> tuple[int, int]:
        return sum(pool.unit_bytes()[0] for pool in self._groups), 0

    def _pool(self, layer_id: int) -> MHAKVCache:
        try:
            return self._layer_to_group[layer_id]
        except KeyError as exc:
            raise KeyError(f"layer {layer_id} has no paged KV storage") from exc

    def k_cache(self, index: int) -> torch.Tensor:
        return self._pool(index).k_cache(index)

    def v_cache(self, index: int) -> torch.Tensor:
        return self._pool(index).v_cache(index)

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        self._pool(layer_id).store_kv(k, v, out_loc, layer_id)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
