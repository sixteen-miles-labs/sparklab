from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from sparklab.models.qwen3_5_moe.config import parse_config
from sparklab.models.qwen3_5_moe.weight import iter_speculative_weights


def _config(*, mtp_layers: int = 1):
    text = SimpleNamespace(
        num_hidden_layers=8,
        layer_types=[
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ],
        hidden_size=64,
        vocab_size=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        norm_topk_prob=True,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        mtp_num_hidden_layers=mtp_layers,
    )
    return SimpleNamespace(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=text,
    )


def test_config_declares_native_mtp_and_engine_adds_full_kv_slot():
    from sparklab.runtime.engine.engine import _adjust_speculative_config

    model_config = parse_config(_config())
    config = SimpleNamespace(
        model_config=model_config,
        speculative_method="auto",
        speculative_tokens=3,
        draft_sample_method="greedy",
        max_running_req=8,
        cache_type="naive",
        cuda_graph_bs=[1, 2, 4],
        cuda_graph_max_bs=4,
    )

    def override(name, value):
        setattr(config, name, value)

    _adjust_speculative_config(config, override)
    _adjust_speculative_config(config, override)

    assert model_config.mtp_num_hidden_layers == 1
    assert model_config.speculative_method == "mtp"
    full = next(group for group in model_config.attention_groups if group.name == "full")
    assert full.layer_ids.count(model_config.num_layers) == 1
    assert full.num_index_layers == 0
    assert config.max_running_req == 1
    assert config.cache_type == "radix"
    assert config.cuda_graph_bs == [] and config.cuda_graph_max_bs == 0

    from sparklab.runtime.kvcache import create_kvcache_pool
    from sparklab.runtime.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    pool = create_kvcache_pool(
        model_config=model_config,
        num_pages=2,
        page_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert len(pool._layer_map) == model_config.num_layers + 1
    assert pool._layer_map[model_config.num_layers] >= 0


def test_speculative_weight_iterator_fuses_projections_and_bakes_norm(tmp_path):
    h, q, kv, i, experts = 4, 8, 2, 3, 2
    tensors = {
        "mtp.fc.weight": torch.arange(h * 2 * h, dtype=torch.bfloat16).view(h, 2 * h),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(h, dtype=torch.bfloat16),
        "mtp.layers.0.self_attn.q_proj.weight": torch.full((q, h), 1, dtype=torch.bfloat16),
        "mtp.layers.0.self_attn.k_proj.weight": torch.full((kv, h), 2, dtype=torch.bfloat16),
        "mtp.layers.0.self_attn.v_proj.weight": torch.full((kv, h), 3, dtype=torch.bfloat16),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": torch.full(
            (i, h), 4, dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": torch.full(
            (i, h), 5, dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.zeros(
            experts, 2 * i, h, dtype=torch.bfloat16
        ),
    }
    save_file(tensors, tmp_path / "model.safetensors")

    loaded = dict(iter_speculative_weights(str(tmp_path), torch.device("cpu")))

    assert loaded["layers.0.self_attn.qkv_proj.weight"].shape == (q + 2 * kv, h)
    assert loaded["layers.0.self_attn.qkv_proj.weight"][:, 0].tolist() == (
        [1] * q + [2] * kv + [3] * kv
    )
    assert loaded["layers.0.mlp.shared_expert.gate_up_proj.weight"].shape == (2 * i, h)
    torch.testing.assert_close(
        loaded["pre_fc_norm_hidden.weight"], torch.ones(h, dtype=torch.bfloat16)
    )
    assert "layers.0.mlp.experts.gate_up_proj" in loaded
