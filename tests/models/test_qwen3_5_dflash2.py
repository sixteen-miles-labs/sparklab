from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from safetensors.torch import save_file

from sparklab.models.qwen3_5_moe.dflash2 import _selector_topk, parse_dflash2_args
from sparklab.models.qwen3_5_moe.weight import iter_dflash2_weights


@pytest.mark.parametrize("accepted_prefix", [33, 34, 39])
def test_rejection_draft_excludes_rejected_suffix_and_restores_length(accepted_prefix):
    from sparklab.runtime.engine.engine import Engine

    engine = Engine.__new__(Engine)
    req = SimpleNamespace(device_len=40)
    batch = SimpleNamespace(reqs=[req])
    correction = torch.tensor([17], dtype=torch.int32)
    proposed = torch.tensor([23, 42], dtype=torch.int32)

    def propose(active_batch, anchor):
        assert active_batch is batch
        assert active_batch.reqs[0].device_len == accepted_prefix
        assert anchor is correction
        return proposed

    engine._propose_speculative = propose
    assert engine._propose_from_verified_prefix(batch, correction, accepted_prefix) is proposed
    assert req.device_len == 40


def test_rejection_draft_restores_length_on_failure():
    from sparklab.runtime.engine.engine import Engine

    engine = Engine.__new__(Engine)
    req = SimpleNamespace(device_len=40)
    batch = SimpleNamespace(reqs=[req])

    def fail(*_):
        raise RuntimeError("draft failed")

    engine._propose_speculative = fail
    with pytest.raises(RuntimeError, match="draft failed"):
        engine._propose_from_verified_prefix(batch, torch.tensor([17]), 33)
    assert req.device_len == 40


def _draft_config():
    return SimpleNamespace(
        architectures=["DFlash2DraftModel"],
        hidden_size=5120,
        intermediate_size=17408,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=5,
        vocab_size=248320,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        sliding_window=2048,
        rope_parameters={"rope_theta": 10_000_000},
        dflash_config={
            "block_size": 8,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61],
            "mask_token_id": 248070,
        },
    )


def test_parse_dflash2_checkpoint_geometry():
    args = parse_dflash2_args(_draft_config(), "draft")
    assert args.num_layers == 5
    assert args.target_layer_ids == (5, 19, 33, 47, 61)
    assert args.num_key_value_heads == 8
    assert args.head_dim == 128
    assert args.selector_top_k == 16


def test_dflash2_selector_topk_cpu_fallback():
    scores = torch.tensor([[1.0, 5.0, 3.0, 4.0]])
    values, indices = _selector_topk(scores, 2)
    assert values.tolist() == [[5.0, 4.0]]
    assert indices.tolist() == [[1, 3]]


def test_dflash2_weight_iterator_fuses_native_nvfp4(tmp_path):
    h, q, kv, intermediate = 16, 8, 4, 12
    tensors = {
        "fc.weight": torch.zeros((h, 5 * h), dtype=torch.bfloat16),
        "layers.0.self_attn.q_proj.weight": torch.full((q, h // 2), 1, dtype=torch.uint8),
        "layers.0.self_attn.k_proj.weight": torch.full((kv, h // 2), 2, dtype=torch.uint8),
        "layers.0.self_attn.v_proj.weight": torch.full((kv, h // 2), 3, dtype=torch.uint8),
        "layers.0.self_attn.q_proj.weight_scale": torch.ones((q, h // 16), dtype=torch.float8_e4m3fn),
        "layers.0.self_attn.k_proj.weight_scale": torch.ones((kv, h // 16), dtype=torch.float8_e4m3fn),
        "layers.0.self_attn.v_proj.weight_scale": torch.ones((kv, h // 16), dtype=torch.float8_e4m3fn),
        "layers.0.self_attn.q_proj.weight_scale_2": torch.tensor(1.0),
        "layers.0.self_attn.k_proj.weight_scale_2": torch.tensor(2.0),
        "layers.0.self_attn.v_proj.weight_scale_2": torch.tensor(3.0),
        "layers.0.mlp.gate_proj.weight": torch.zeros((intermediate, h // 2), dtype=torch.uint8),
        "layers.0.mlp.up_proj.weight": torch.zeros((intermediate, h // 2), dtype=torch.uint8),
        "layers.0.mlp.gate_proj.weight_scale": torch.ones((intermediate, h // 16), dtype=torch.float8_e4m3fn),
        "layers.0.mlp.up_proj.weight_scale": torch.ones((intermediate, h // 16), dtype=torch.float8_e4m3fn),
        "layers.0.mlp.gate_proj.weight_scale_2": torch.tensor(1.0),
        "layers.0.mlp.up_proj.weight_scale_2": torch.tensor(1.0),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    loaded = dict(iter_dflash2_weights(str(tmp_path), torch.device("cpu")))
    qkv = "layers.0.self_attn.qkv_proj"
    assert loaded[qkv + ".weight"].shape == (q + 2 * kv, h // 2)
    assert loaded[qkv + ".weight"][:, 0].tolist() == [1] * q + [2] * kv + [3] * kv
    assert loaded[qkv + ".weight_global"].tolist() == [1.0] * q + [2.0] * kv + [3.0] * kv
    assert loaded["layers.0.mlp.gate_up_proj.weight"].shape == (
        2 * intermediate,
        h // 2,
    )
    assert "fc.weight" in loaded


@pytest.mark.parametrize("graph_max", [0, 1])
def test_engine_injects_dflash2_kv_group(monkeypatch, graph_max):
    from sparklab.runtime.engine.engine import _adjust_speculative_config
    from tests.models.test_qwen3_5_dense import _inferact_qwen38_27b_config
    from sparklab.models.qwen3_5_moe.config import parse_config
    import sparklab.runtime.engine.engine as engine_module

    monkeypatch.setattr(engine_module, "cached_load_hf_config", lambda _: _draft_config())
    monkeypatch.delenv("SPARKLAB_DFLASH2_VERIFY_GRAPH", raising=False)
    model_config = parse_config(_inferact_qwen38_27b_config())
    config = SimpleNamespace(
        model_config=model_config,
        speculative_method="dflash2",
        speculative_tokens=16,
        speculative_draft_model="draft",
        draft_sample_method="greedy",
        max_running_req=4,
        cache_type="naive",
        cuda_graph_bs=[1, 2],
        cuda_graph_max_bs=graph_max,
        attention_backend="triton",
    )
    _adjust_speculative_config(config, lambda name, value: setattr(config, name, value))
    groups = {group.name: group for group in model_config.attention_groups}
    assert model_config.speculative_method == "dflash2"
    assert groups["dflash2"].layer_ids == (64, 65, 66, 67, 68)
    assert groups["dflash2"].num_kv_heads == 8
    assert groups["dflash2"].head_dim == 128
    assert config.max_running_req == 1
    assert config.cache_type == "radix"
    assert model_config.mtp_cuda_graph == (graph_max > 0)


def test_dflash2_uses_separate_target_and_draft_kv_geometry(monkeypatch):
    import sparklab.runtime.engine.engine as engine_module

    from sparklab.models.qwen3_5_moe.config import parse_config
    from sparklab.runtime.distributed import set_tp_info, try_get_tp_info
    from sparklab.runtime.engine.engine import _adjust_speculative_config
    from sparklab.runtime.kvcache import create_kvcache_pool, resolve_pool_class
    from sparklab.runtime.kvcache.grouped_mha_pool import GroupedMHAKVCache
    from tests.models.test_qwen3_5_dense import _inferact_qwen38_27b_config
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(engine_module, "cached_load_hf_config", lambda _: _draft_config())
    model_config = parse_config(_inferact_qwen38_27b_config())
    config = SimpleNamespace(
        model_config=model_config,
        speculative_method="dflash2",
        speculative_tokens=8,
        speculative_draft_model="draft",
        draft_sample_method="greedy",
        max_running_req=1,
        cache_type="radix",
        cuda_graph_bs=[],
        cuda_graph_max_bs=0,
    )
    _adjust_speculative_config(config, lambda name, value: setattr(config, name, value))
    assert resolve_pool_class(model_config) is GroupedMHAKVCache
    pool = create_kvcache_pool(
        model_config,
        num_pages=3,
        page_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    target = next(
        spec
        for spec in model_config.kv_cache_group_specs()
        if spec.name != "dflash2" and spec.layer_ids
    )
    draft = next(
        spec for spec in model_config.kv_cache_group_specs() if spec.name == "dflash2"
    )
    assert pool.k_cache(target.layer_ids[0]).shape[-2:] == (
        target.num_kv_heads,
        target.head_dim,
    )
    assert pool.k_cache(draft.layer_ids[0]).shape[-2:] == (8, 128)
