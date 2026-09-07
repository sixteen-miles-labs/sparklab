"""CPU execution of the actual Triton admission kernel, without touching the GPU.

Run with CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 pytest <this file>.
CUDA parity tests in test_offload.py remain required before promoting a recipe.
"""

import os

import pytest
import torch
import triton

from sparklab.moe.offload_cache import OffloadMoeCache
from sparklab.moe.offload_kernels import _layer_lru_ensure_kernel, layer_lru_ensure


pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="requires TRITON_INTERPRET=1"
)


def _interpret(cache, layer, ids):
    _layer_lru_ensure_kernel[(1,)](
        ids, cache.slot_for_id, cache.id_of_slot, cache.usage, cache.step,
        cache.layer_counts, cache.layer_quotas, ids,
        cache.src_indices, cache.evict_slots, cache.num_indices, None,
        layer, ids.numel(), cache.num_layers, cache.num_experts, cache.cache_size,
        BLOCK_K=triton.next_power_of_2(ids.numel()),
        BLOCK_L=triton.next_power_of_2(cache.num_layers),
        BLOCK_C=triton.next_power_of_2(cache.cache_size),
        USAGE_MAX=torch.iinfo(cache.usage.dtype).max,
        COLLECT_STATS=False,
    )


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_layer_lru_interpreted_kernel_matches_reference(seed, monkeypatch):
    from triton.runtime import interpreter

    # Triton 3.6's interpreter uses int(ndarray) for loop bounds; NumPy 2.4
    # requires extracting the scalar explicitly. Keep this compatibility shim
    # confined to the CPU test, without changing the production kernel.
    original_patch = interpreter._patch_lang_tensor

    def patch_tensor(tensor, scope):
        original_patch(tensor, scope)
        scope.set_attr(tensor, "__index__", lambda value: int(value.handle.data.item()))

    monkeypatch.setattr(interpreter, "_patch_lang_tensor", patch_tensor)
    reference = OffloadMoeCache(3, 8, 12, torch.device("cpu"), cache_policy="layer_lru")
    interpreted = OffloadMoeCache(3, 8, 12, torch.device("cpu"), cache_policy="layer_lru")
    # Include unequal quotas: admission must use the actual reservations.
    for cache in (reference, interpreted):
        cache.reserve_layer_quotas({2: 6})
    generator = torch.Generator().manual_seed(seed)
    queries = [(layer, torch.arange(quota, dtype=torch.int32))
               for layer, quota in enumerate(reference.layer_quotas.tolist())]
    # Full-layer requests force borrowing below protected quotas. Later small
    # queries must reclaim loans without evicting any route in their own query.
    queries += [(layer, torch.arange(8, dtype=torch.int32)) for layer in range(3)]
    for i in range(24):
        queries.append((i % 3, torch.randint(
            0, 8, (1 + i % 9,), dtype=torch.int32, generator=generator,
        )))
    for layer, raw in queries:
        expected, actual = raw.clone(), raw.clone()
        layer_lru_ensure(reference, layer, expected)
        _interpret(interpreted, layer, actual)
        assert torch.equal(actual, expected)
        assert torch.equal(interpreted.id_of_slot[actual.long()], layer * 8 + raw)
        for name in ("slot_for_id", "id_of_slot", "usage", "step", "layer_counts", "num_indices"):
            assert torch.equal(getattr(interpreted, name), getattr(reference, name)), name
        n = int(reference.num_indices.item())
        assert torch.equal(interpreted.src_indices[:n], reference.src_indices[:n])
        assert torch.equal(interpreted.evict_slots[:n], reference.evict_slots[:n])
