from contextlib import contextmanager

import pytest
import torch

from sparklab.runtime.distributed import set_tp_info, try_get_tp_info


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _make_layer_and_cache():
    from sparklab.layers.moe import OffloadMoELayer
    from sparklab.moe.offload_cache import OffloadMoeCache

    _init_tp()
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=4,
        top_k=2,
        hidden_size=8,
        intermediate_size=16,
    )
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    layer.offload_cache = cache
    return layer, cache


def _exercise_layer_lru(cache):
    plans = []
    for layer_id, experts in (
        (0, [0, 1]),
        (0, [2]),       # borrow one still-empty slot above layer 0's quota
        (1, [0, 1]),
        (2, [0, 1]),    # fill the last empty slot, then reclaim layer 0's loan
        (0, [3]),       # all layers at quota: replace within layer 0
    ):
        ids = torch.tensor(experts, dtype=torch.int32, device=cache.device)
        cache.ensure_experts(layer_id, ids)
        n = int(cache.num_indices.item())
        plans.append((ids.cpu().tolist(), cache.src_indices[:n].cpu().tolist()))
    return plans


def test_layer_lru_borrows_then_protects_each_layers_quota_cpu():
    from sparklab.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=3,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
        cache_policy="layer_lru",
    )

    _exercise_layer_lru(cache)

    assert cache.layer_counts.tolist() == [2, 2, 2]
    assert (cache.slot_for_id[1, :2] >= 0).all()
    assert (cache.slot_for_id[2, :2] >= 0).all()
    assert int((cache.slot_for_id[0] >= 0).sum()) == 2
    assert cache.slot_for_id[0, 3] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_layer_lru_cuda_matches_cpu_reference():
    from sparklab.moe.offload_cache import OffloadMoeCache

    cpu = OffloadMoeCache(3, 4, 6, torch.device("cpu"), cache_policy="layer_lru")
    gpu = OffloadMoeCache(3, 4, 6, torch.device("cuda"), cache_policy="layer_lru")

    assert _exercise_layer_lru(gpu) == _exercise_layer_lru(cpu)
    assert torch.equal(gpu.slot_for_id.cpu(), cpu.slot_for_id)
    assert torch.equal(gpu.id_of_slot.cpu(), cpu.id_of_slot)
    assert torch.equal(gpu.usage.cpu(), cpu.usage)
    assert torch.equal(gpu.layer_counts.cpu(), cpu.layer_counts)


def test_layer_lru_rejects_prefill_buffer_slot_borrowing():
    from sparklab.moe.offload_cache import OffloadMoeCache

    with pytest.raises(AssertionError, match="incompatible with prefill overlap"):
        OffloadMoeCache(
            3,
            4,
            8,
            torch.device("cpu"),
            cache_policy="layer_lru",
            prefill_overlap=True,
        )


def test_stage_disk_experts_filters_deduplicates_and_admits():
    from types import SimpleNamespace

    _, cache = _make_layer_and_cache()
    calls = []
    cache.disk_source = SimpleNamespace(
        stage=lambda layer_id, expert_ids, *, admit: calls.append(
            (layer_id, expert_ids, admit)
        )
    )

    cache.stage_disk_experts(0, torch.tensor([[3, -1], [1, 3]], dtype=torch.int32))

    assert calls == [(0, [1, 3], True)]


def test_stage_disk_experts_is_noop_without_source_or_cpu_routes():
    from types import SimpleNamespace

    _, cache = _make_layer_and_cache()
    cache.stage_disk_experts(0, torch.tensor([[1, 2]], dtype=torch.int32))
    cache.disk_source = SimpleNamespace(
        stage=lambda *args, **kwargs: pytest.fail("empty CPU route set must not stage")
    )
    cache.stage_disk_experts(0, torch.full((1, 2), -1, dtype=torch.int32))


def test_stage_disk_hybrid_combines_cpu_and_gpu_routes_once():
    from types import SimpleNamespace

    _, cache = _make_layer_and_cache()
    calls = []
    cache.disk_source = SimpleNamespace(
        stage=lambda layer_id, expert_ids, *, admit: calls.append(
            (layer_id, expert_ids, admit)
        )
    )
    cache.num_indices.fill_(2)
    cache.src_indices[:2] = torch.tensor([2, 3], dtype=torch.int32)

    cache.stage_disk_hybrid(
        0, torch.tensor([[3, -1], [1, 3]], dtype=torch.int32)
    )

    assert calls == [(0, [1, 3, 2], True)]
    assert cache._pending_disk_stage_layer == 0


def test_copy_missing_skips_disk_stage_after_combined_hybrid_stage(monkeypatch):
    from types import SimpleNamespace

    _, cache = _make_layer_and_cache()
    calls = []
    cache.disk_source = SimpleNamespace(
        stage=lambda layer_id, expert_ids, *, admit: calls.append(
            (layer_id, expert_ids, admit)
        )
    )
    cache._pending_src_layer = 0
    cache.num_indices.fill_(1)
    cache.src_indices[0] = 2
    cache.stage_disk_hybrid(0, torch.tensor([1], dtype=torch.int32))
    monkeypatch.setattr(
        "sparklab.kernels.fast_index_copy_jit", lambda *args, **kwargs: None
    )

    cache.copy_missing()

    assert calls == [(0, [1, 2], True)]
    assert cache._pending_disk_stage_layer is None


def test_dummy_expert_sources_use_moe_layer_count(monkeypatch):
    from types import SimpleNamespace

    import sparklab.models.weight as weight

    _init_tp()
    config = SimpleNamespace(
        num_layers=5,
        num_moe_layers=3,
        num_experts=4,
        hidden_size=6,
        moe_intermediate_size=8,
    )

    gate_up, down = weight.dummy_moe_expert_sources(config, dtype=torch.float16)

    assert len(gate_up) == 3 and all(t.shape == (4, 16, 6) for t in gate_up)
    assert len(down) == 3 and all(t.shape == (4, 6, 8) for t in down)

    monkeypatch.setattr(
        weight,
        "alloc_pinned_tensor",
        lambda *shape, dtype: torch.empty(*shape, dtype=dtype),
    )
    banks = weight.dummy_nvfp4_expert_sources(config)

    assert {len(layers) for layers in banks.values()} == {3}
    assert {t.shape[0] for layers in banks.values() for t in layers} == {4}


def test_offload_moe_layer_prefill_forward_uses_single_layer_cache_view(monkeypatch):
    layer, cache = _make_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    calls = {}

    monkeypatch.setattr(
        "sparklab.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )
    monkeypatch.setattr(cache, "materialize_layer", lambda layer_id: calls.setdefault("layer_id", layer_id))
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.setdefault("copied", True))

    def fake_fused(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
    ):
        calls["w1"] = w1
        calls["w2"] = w2
        calls["topk_weights"] = got_topk_weights
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("sparklab.layers.moe.fused_experts_impl", fake_fused)

    out = layer.prefill_forward(hidden_states, router_logits)

    assert out is hidden_states
    assert calls["layer_id"] == 0
    assert calls["copied"] is True
    assert calls["w1"].shape[0] == layer.num_experts
    assert calls["w2"].shape[0] == layer.num_experts
    assert calls["w1"].data_ptr() == cache.bank_caches["gate_up"].data_ptr()
    assert calls["w2"].data_ptr() == cache.bank_caches["down"].data_ptr()
    assert calls["topk_weights"] is topk_weights
    assert calls["topk_ids"].dtype == torch.int32
    # slot == expert id after materialize, so the routing ids pass through unmapped
    assert calls["topk_ids"].tolist() == [[2, 1]]


def test_offload_moe_layer_sparse_prefill_routes_through_persistent_cache(monkeypatch):
    layer, cache = _make_layer_and_cache()
    cache.prefill_sparse_max_tokens = 4
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    calls = {}

    monkeypatch.setattr(
        cache,
        "materialize_layer",
        lambda *_: pytest.fail("sparse prefill must not materialize a full layer"),
    )

    def fake_ensure(layer_id, ids, *, is_prefill=False):
        calls["ensure"] = (layer_id, is_prefill)
        assert ids.tolist() == [1, 2]
        cache.slot_for_id[layer_id, 1] = 4
        cache.slot_for_id[layer_id, 2] = 5
        cache.num_indices.fill_(2)

    monkeypatch.setattr(cache, "ensure_experts", fake_ensure)
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.setdefault("copied", True))

    def fake_fused(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
    ):
        calls["w1_rows"] = w1.shape[0]
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("sparklab.layers.moe.fused_experts_impl", fake_fused)

    out = layer._prefill_routed(hidden_states, topk_weights, topk_ids.clone())

    assert out is hidden_states
    assert calls["w1_rows"] == cache.cache_size
    assert calls["topk_ids"].tolist() == [[5, 4]]
    assert calls["ensure"] == (0, True)
    assert calls["copied"] is True
    assert cache.sparse_prefill_layers == 1
    assert cache.sparse_prefill_routes == 2
    assert cache.sparse_prefill_unique_rows == 2
    assert int(cache.num_indices.item()) == 2


def test_native_nvfp4_sparse_prefill_sorts_logical_ids_and_maps_slots(monkeypatch):
    """Native NVFP4 sparse prefill keeps E as its sort domain and maps slots in-kernel."""
    from types import SimpleNamespace

    from sparklab.layers.moe import OffloadMoELayer

    _init_tp()
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=4,
        top_k=2,
        hidden_size=8,
        intermediate_size=16,
    )
    hidden_states = torch.randn(1, 8)
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[5, 4]], dtype=torch.int32)
    views = tuple(torch.empty(6, 1) for _ in range(6))
    calls = {}

    def fake_fused(
        hidden_states,
        gate_up_packed,
        gate_up_scale,
        gate_up_global,
        down_packed,
        down_scale,
        down_global,
        topk_weights,
        topk_ids,
        num_experts,
        *args,
        slot_map=None,
    ):
        calls["num_experts"] = num_experts
        calls["topk_ids"] = topk_ids.clone()
        calls["slot_map"] = slot_map
        return hidden_states

    monkeypatch.setattr("sparklab.moe.fused_nvfp4.fused_experts_nvfp4", fake_fused)
    out = layer._expert_gemm(
        SimpleNamespace(quant_format="nvfp4"),
        hidden_states,
        topk_weights,
        torch.tensor([[1, 2]], dtype=torch.int32),
        views=views,
        n=None,
        alphas=None,
        is_prefill=True,
        prefill_slot_map=torch.tensor([3, 5, 4, 2]),
    )

    assert out is hidden_states
    assert calls["num_experts"] == 4
    assert calls["topk_ids"].tolist() == [[1, 2]]
    assert calls["slot_map"].tolist() == [3, 5, 4, 2]


def test_fp8_sparse_prefill_sorts_logical_ids_and_maps_slots(monkeypatch):
    """Block-FP8 sparse prefill keeps E as its sort domain and maps slots in-kernel."""
    from types import SimpleNamespace

    from sparklab.layers.moe import OffloadMoELayer

    _init_tp()
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=4,
        top_k=2,
        hidden_size=8,
        intermediate_size=16,
    )
    hidden_states = torch.randn(1, 8)
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[5, 4]], dtype=torch.int32)
    views = tuple(torch.empty(6, 1) for _ in range(4))
    calls = {}

    def fake_fused(
        hidden_states,
        gate_up,
        gate_up_scale,
        down,
        down_scale,
        topk_weights,
        topk_ids,
        num_experts,
        *args,
        slot_map=None,
    ):
        calls["num_experts"] = num_experts
        calls["topk_ids"] = topk_ids.clone()
        calls["slot_map"] = slot_map
        return hidden_states

    monkeypatch.setattr("sparklab.moe.fused_fp8_block.fused_experts_fp8_block", fake_fused)
    out = layer._expert_gemm(
        SimpleNamespace(quant_format="fp8_block"),
        hidden_states,
        topk_weights,
        torch.tensor([[1, 2]], dtype=torch.int32),
        views=views,
        n=None,
        alphas=None,
        is_prefill=True,
        prefill_slot_map=torch.tensor([3, 5, 4, 2]),
    )

    assert out is hidden_states
    assert calls["num_experts"] == 4
    assert calls["topk_ids"].tolist() == [[1, 2]]
    assert calls["slot_map"].tolist() == [3, 5, 4, 2]


def test_offload_moe_layer_prefill_overlap_prefetches_layers_into_two_buffers(monkeypatch):
    from sparklab.layers.moe import OffloadMoELayer
    from sparklab.moe.offload_cache import OffloadMoeCache

    _init_tp()
    num_layers = 3
    num_experts = 4
    layers = [
        OffloadMoELayer(
            layer_id=layer_id,
            num_experts=num_experts,
            top_k=2,
            hidden_size=8,
            intermediate_size=16,
        )
        for layer_id in range(num_layers)
    ]
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.arange(num_layers * num_experts * 32 * 8, dtype=torch.float32).reshape(
        num_layers * num_experts, 32, 8
    ).split(num_experts))
    down_source = list(torch.arange(num_layers * num_experts * 8 * 16, dtype=torch.float32).reshape(
        num_layers * num_experts, 8, 16
    ).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})
    for layer in layers:
        layer.offload_cache = cache

    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, num_experts)
    fused_calls = []

    monkeypatch.setattr(
        "sparklab.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (
            topk_weights,
            topk_ids.clone(),
        ),
    )

    def unexpected_fast_index_copy(*args, **kwargs):
        raise AssertionError("prefill overlap should use direct async copy")

    monkeypatch.setattr("sparklab.kernels.fast_index_copy_jit", unexpected_fast_index_copy)

    def fake_fused(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
    ):
        layer_id = len(fused_calls)
        fused_calls.append(
            {
                "w1_ptr": w1.data_ptr(),
                "w2_ptr": w2.data_ptr(),
                "w1": w1.clone(),
                "w2": w2.clone(),
                "topk_weights": got_topk_weights,
                "topk_ids": got_topk_ids.clone(),
            }
        )
        return hidden_states + layer_id

    monkeypatch.setattr("sparklab.layers.moe.fused_experts_impl", fake_fused)

    out = hidden_states
    for layer in layers:
        out = layer.prefill_forward(out, router_logits)

    assert torch.allclose(out, hidden_states + 3)
    for layer_id in range(num_layers):
        assert fused_calls[layer_id]["topk_weights"] is topk_weights
        assert fused_calls[layer_id]["topk_ids"].tolist() == [[2, 1]]
        assert torch.equal(fused_calls[layer_id]["w1"], gate_up_source[layer_id])
        assert torch.equal(fused_calls[layer_id]["w2"], down_source[layer_id])

    assert fused_calls[0]["w1_ptr"] == fused_calls[2]["w1_ptr"]
    assert fused_calls[0]["w2_ptr"] == fused_calls[2]["w2_ptr"]
    assert fused_calls[0]["w1_ptr"] != fused_calls[1]["w1_ptr"]
    assert fused_calls[0]["w2_ptr"] != fused_calls[1]["w2_ptr"]
    prefill_gate_up_buffer, prefill_down_buffer = cache.prefill_bank_buffers
    assert prefill_gate_up_buffer.data_ptr() == cache.bank_caches["gate_up"].data_ptr()
    assert prefill_down_buffer.data_ptr() == cache.bank_caches["down"].data_ptr()


def test_offload_moe_cache_prefill_overlap_requires_two_layer_slots():
    from sparklab.moe.offload_cache import OffloadMoeCache

    with pytest.raises(AssertionError):
        OffloadMoeCache(
            num_layers=3,
            num_experts=4,
            cache_size=7,
            device=torch.device("cpu"),
            prefill_overlap=True,
        )


def test_offload_moe_cache_marlin_rejects_slot_count_beyond_kernel_limit():
    from sparklab.moe.offload_cache import OffloadMoeCache

    with pytest.raises(ValueError, match="992"):
        OffloadMoeCache(
            num_layers=2,
            num_experts=8,
            cache_size=1024,
            device=torch.device("cpu"),
            quant_format="nvfp4_marlin",
        )


def test_prefill_overlap_prefetch_invalidates_borrowed_unified_cache_slots():
    from sparklab.moe.offload_cache import OffloadMoeCache

    num_layers = 3
    num_experts = 4
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.arange(num_layers * num_experts * 32 * 8, dtype=torch.float32).reshape(
        num_layers * num_experts, 32, 8
    ).split(num_experts))
    down_source = list(torch.arange(num_layers * num_experts * 8 * 16, dtype=torch.float32).reshape(
        num_layers * num_experts, 8, 16
    ).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})

    old_layers = torch.tensor([2, 2, 1, 1], dtype=torch.int32)
    old_experts = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    cache.id_of_slot[:num_experts] = old_layers * num_experts + old_experts
    cache.usage[:num_experts] = torch.arange(1, num_experts + 1, dtype=torch.int64)
    for slot, (layer_id, expert_id) in enumerate(zip(old_layers.tolist(), old_experts.tolist())):
        cache.slot_for_id[layer_id, expert_id] = slot

    cache.prefetch_prefill_layer(0)

    assert cache.id_of_slot[:num_experts].tolist() == [-1] * num_experts
    assert cache.usage[:num_experts].tolist() == [0] * num_experts
    for layer_id, expert_id in zip(old_layers.tolist(), old_experts.tolist()):
        assert int(cache.slot_for_id[layer_id, expert_id].item()) == -1
    assert torch.equal(cache.bank_caches["gate_up"][:num_experts], gate_up_source[0])
    assert torch.equal(cache.bank_caches["down"][:num_experts], down_source[0])


def test_prefill_overlap_waits_for_previous_prefill_release_after_begin(monkeypatch):
    from sparklab.moe.offload_cache import OffloadMoeCache

    num_layers = 2
    num_experts = 4
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.zeros(num_layers * num_experts, 32, 8).split(num_experts))
    down_source = list(torch.zeros(num_layers * num_experts, 8, 16).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})

    class FakeStream:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            self.waited.append(event.name)

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream=None):
            pass

    @contextmanager
    def fake_cuda_stream(stream):
        yield

    copy_stream = FakeStream()
    cache.prefill_copy_stream = copy_stream
    cache.prefill_begin_event = FakeEvent("begin")
    cache.prefill_ready_events = [FakeEvent("ready0"), FakeEvent("ready1")]
    cache.prefill_release_events = [FakeEvent("release0"), FakeEvent("release1")]
    monkeypatch.setattr("torch.cuda.stream", fake_cuda_stream)
    monkeypatch.setattr("torch.cuda.current_stream", lambda device=None: object())

    cache.prefetch_prefill_layer(0)
    cache.release_prefill_layer(0)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)

    # begin_prefill fences the copy stream behind the compute stream (so a prefetch
    # cannot race the preceding decode batch), then the buffer reuse waits on the
    # previous prefill's release event.
    assert copy_stream.waited == ["begin", "release0"]


def test_offload_moe_layer_decode_forward_uses_remapped_slot_ids(monkeypatch):
    layer, cache = _make_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    calls = {}

    monkeypatch.setattr(
        "sparklab.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )

    def fake_ensure(layer_id, expert_ids):
        calls["ensure_layer_id"] = layer_id
        calls["ensure_expert_ids"] = expert_ids.clone()
        expert_ids.copy_(torch.tensor([[5, 0]], dtype=torch.int32))

    monkeypatch.setattr(cache, "ensure_experts", fake_ensure)
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.setdefault("copied", True))

    def fake_fused_decode(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
    ):
        calls["w1"] = w1
        calls["w2"] = w2
        calls["topk_weights"] = got_topk_weights
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("sparklab.layers.moe.fused_experts_decode_impl", fake_fused_decode)

    out = layer.decode_forward(hidden_states, router_logits)

    assert out is hidden_states
    assert calls["ensure_layer_id"] == 0
    assert calls["ensure_expert_ids"].tolist() == [[2, 1]]
    assert calls["copied"] is True
    assert calls["w1"] is cache.bank_caches["gate_up"]
    assert calls["w2"] is cache.bank_caches["down"]
    assert calls["topk_weights"] is topk_weights
    assert calls["topk_ids"].dtype == torch.int32
    assert calls["topk_ids"].tolist() == [[5, 0]]



def test_lru_gpu_cache_assigns_unique_slots_for_large_miss_batch():
    import pytest
    from sparklab.moe.offload_cache import OffloadMoeCache

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU offload cache kernel")

    cache = OffloadMoeCache(
        num_layers=40,
        num_experts=256,
        cache_size=1664,
        device=torch.device("cuda"),
    )
    expert_ids = torch.arange(256, dtype=torch.int32, device="cuda").view(32, 8)

    cache.ensure_experts(0, expert_ids)
    torch.cuda.synchronize()

    assert int(cache.num_indices.item()) == 256
    assert expert_ids.min().item() >= 0
    assert expert_ids.max().item() < cache.cache_size
    evict_slots = cache.evict_slots[:256]
    assert evict_slots.min().item() >= 0
    assert evict_slots.max().item() < cache.cache_size
    assert torch.unique(evict_slots).numel() == evict_slots.numel()
    assert cache.src_indices[:256].tolist() == list(range(256))


def test_adjust_config_converts_moe_cache_rate_to_cache_size():
    from types import SimpleNamespace

    from sparklab.runtime.distributed import DistributedInfo
    from sparklab.runtime.engine.config import EngineConfig
    from sparklab.runtime.engine.engine import _adjust_config

    config = EngineConfig(
        model_path="/tmp/sparklab-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16,
        attention_backend="triton",
        moe_cache_rate=0.3,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="none",
            moe_backend="auto",
        ),
    )

    _adjust_config(config)

    from sparklab.moe import is_offload_moe_backend

    assert config.moe_cache_size == 24
    # Family, not member: a box with a benchbw profile resolves bf16 experts to hybrid.
    assert is_offload_moe_backend(config.moe_backend)


def test_graph_capture_reuses_warm_offload_cache_before_capture(monkeypatch):
    import sparklab.core as core
    from sparklab.core import Context, Req, get_global_ctx
    from sparklab.runtime.engine.graph import GraphRunner

    events = []
    _init_tp()
    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=1))

    class FakeGraph:
        def pool(self):
            return "pool"

    @contextmanager
    def fake_cuda_graph(graph, pool=None, stream=None):
        events.append("graph_enter")
        yield
        events.append("graph_exit")

    class FakeAttnBackend:
        def init_capture_graph(self, max_seq_len, bs_list):
            pass

        def prepare_for_capture(self, batch):
            pass

    class FakeModel:
        def forward(self):
            events.append("forward")
            batch = get_global_ctx().batch
            return torch.zeros(batch.size, 3)

    class FakeOffloadCache:
        def reset(self):
            events.append("reset")

    monkeypatch.setattr("torch.cuda.CUDAGraph", FakeGraph)
    monkeypatch.setattr("torch.cuda.graph", fake_cuda_graph)
    monkeypatch.setattr("torch.cuda.synchronize", lambda device=None: None)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr("sparklab.runtime.engine.graph.get_free_memory", lambda device: 1024)

    dummy_req = Req(
        input_ids=torch.tensor([0], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=-1,
        sampling_params=None,
        cache_handle=None,
    )
    GraphRunner(
        stream=None,
        device=torch.device("cpu"),
        model=FakeModel(),
        attn_backend=FakeAttnBackend(),
        cuda_graph_bs=[1],
        cuda_graph_max_bs=None,
        free_memory=1024,
        max_seq_len=1,
        vocab_size=3,
        dummy_req=dummy_req,
        moe_offload_cache=FakeOffloadCache(),
    )

    assert events == [
        "reset",
        "forward",
        "graph_enter",
        "forward",
        "graph_exit",
        "reset",
        "reset",
    ]


def test_nvfp4_materialize_keeps_bookkeeping_consistent_across_requests():
    """Regression: a full-layer prefill loads the layer's experts into slots [0, E).
    If that overwrite does not invalidate the previous owners' mappings, a later
    decode "hits" a stale slot_for_id entry and silently reads another expert's
    weights. materialize_layer must keep bookkeeping == slot contents."""
    import pytest
    from sparklab.moe.offload_cache import OffloadMoeCache

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU offload cache kernel")

    L, E, S = 2, 8, 8
    OUT, IN = 64, 512  # keep rows >= 128B so the fast_index_copy JIT has a kernel
    dev = torch.device("cuda")

    def bank(out, inner, dtype):
        # one independently allocated [E, out, inner] tensor per layer (the per-layer host
        # bank contract); each row keeps the old flat fingerprint layer*E+idx.
        layers = []
        for layer_index in range(L):
            t = torch.zeros(E, out, inner, dtype=dtype)
            for e in range(E):
                t[e].view(torch.uint8).fill_(layer_index * E + e)
            layers.append(t)
        return layers

    def pinned(layers):
        return [t.pin_memory() for t in layers]

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=S, device=dev, quant_format="nvfp4"
    )
    cache.set_bank_sources(
        {
            "gate_up_packed": pinned(bank(OUT, IN // 2, torch.uint8)),
            "gate_up_scale": pinned(bank(OUT, IN // 16, torch.float8_e4m3fn)),
            "gate_up_global": pinned([t.squeeze(-1).contiguous() for t in bank(OUT, 1, torch.float16)]),
            "down_packed": pinned(bank(OUT, IN // 2, torch.uint8)),
            "down_scale": pinned(bank(OUT, IN // 16, torch.float8_e4m3fn)),
            "down_global": pinned([t.squeeze(-1).contiguous() for t in bank(OUT, 1, torch.float16)]),
        }
    )
    cache.reset()

    def fingerprint(slot):  # which source row's bytes live in this slot?
        return int(cache.bank_caches["gate_up_packed"][slot].view(torch.uint8).flatten()[0].item())

    # Request A, decode: layer 0 loads experts 3 and 5 somewhere in the cache.
    ids = torch.tensor([3, 5], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids.tolist()] == [3, 5]

    # Request B, prefill: layer 1 is materialized into slots [0, E), overwriting
    # every slot (S == E), including the ones decode A used.
    cache.materialize_layer(1)
    cache.copy_missing()
    torch.cuda.synchronize()
    # The layer's experts fill slots [0, E) bijectively and the bookkeeping agrees.
    assert [fingerprint(s) for s in range(E)] == [E + e for e in range(E)]
    assert cache.slot_for_id[1].tolist() == list(range(E))

    # Request B, decode: layer 0 routes to experts 3/5 again. Their old slots were
    # overwritten, so this must be a miss + reload -- never a stale hit serving
    # layer-1 bytes.
    ids2 = torch.tensor([3, 5], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, ids2)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids2.tolist()] == [3, 5]

    # The prefilled layer's own experts still resolve to correct bytes (S == E, so
    # the layer-0 reload above evicted two layer-1 slots -- hit or miss, the
    # bookkeeping must never serve another expert's bytes).
    ids3 = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    cache.ensure_experts(1, ids3)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids3.tolist()] == [E + 1, E + 2]


def test_offload_cache_rebuild_resizes_and_preserves_sources():
    from sparklab.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"))
    gate_up = torch.randn(4, 32, 8)
    down = torch.randn(4, 8, 16)
    cache.set_bank_sources({"gate_up": [gate_up], "down": [down]})

    cache.rebuild(10)

    assert cache.cache_size == 10
    # host sources preserved (same objects, not reloaded)
    assert cache.bank_sources["gate_up"][0] is gate_up
    assert cache.bank_sources["down"][0] is down
    # GPU slot caches resized to the new cache_size, row shape unchanged
    assert cache.bank_caches["gate_up"].shape == (10, 32, 8)
    assert cache.bank_caches["down"].shape == (10, 8, 16)
    # bookkeeping resized + reset
    assert cache.id_of_slot.shape == (10,)
    assert cache.usage.shape == (10,)
    assert torch.all(cache.slot_for_id == -1)
    assert torch.all(cache.id_of_slot == -1)


def test_offload_cache_rebuild_disables_prefill_overlap_when_too_small():
    from sparklab.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    assert cache.prefill_overlap is True

    cache.rebuild(5)  # 5 < 2*num_experts (8) -> overlap must auto-disable

    assert cache.cache_size == 5
    assert cache.prefill_overlap is False
    assert cache.prefill_bank_buffers == []


def test_offload_cache_rebuild_keeps_overlap_at_boundary():
    from sparklab.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    cache.rebuild(8)  # exactly 2*num_experts -> overlap stays on
    assert cache.prefill_overlap is True
    assert cache.cache_size == 8


def test_offload_cache_validate_rebuild_enforces_marlin_cap_and_floor():
    # The constructor caps nvfp4_marlin slots at 992; a runtime rebuild must enforce the
    # same upper cap (and the num_experts floor), else marlin decode kernels later break.
    from sparklab.moe.offload_cache import MARLIN_MAX_CACHE_SIZE, OffloadMoeCache

    _init_tp()
    marlin = OffloadMoeCache(
        num_layers=1, num_experts=8, cache_size=16,
        device=torch.device("cpu"), quant_format="nvfp4_marlin",
    )
    with pytest.raises(ValueError, match="992"):
        marlin.validate_rebuild(MARLIN_MAX_CACHE_SIZE + 1)
    marlin.validate_rebuild(MARLIN_MAX_CACHE_SIZE)  # exactly at the cap: allowed

    bf16 = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="num_experts"):
        bf16.validate_rebuild(3)  # below the num_experts floor
