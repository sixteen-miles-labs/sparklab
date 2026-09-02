from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from sparklab.models.qwen3_5_moe.config import parse_config


def _inferact_qwen38_27b_config():
    text = SimpleNamespace(
        num_hidden_layers=64,
        layer_types=[
            "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
            for layer in range(64)
        ],
        hidden_size=5120,
        vocab_size=248320,
        intermediate_size=17408,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        max_position_embeddings=262144,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    return SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text,
        vision_config=SimpleNamespace(model_type="qwen3_5_vision"),
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "group_size": 16},
                }
            },
        },
    )


def test_inferact_qwen38_27b_uses_dense_text_only_nvfp4_path():
    config = parse_config(_inferact_qwen38_27b_config())

    assert config.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert config.moe_enabled is False
    assert config.num_experts == 0
    assert config.dense_quant == "nvfp4"
    assert config.attn_quant == "nvfp4"
    assert config.vision_config is None
    assert config.rotary_config.max_position == 262144


def test_inferact_qwen38_27b_reuses_prebuilt_dense_aot_shapes():
    from sparklab.kernels.aot_models import SUPPORTED_MODELS

    entry = next(
        model
        for model in SUPPORTED_MODELS
        if "Inferact/Qwen3.8-27B-NVFP4" in model.aliases
    )
    assert entry.architecture == "Qwen3_5ForConditionalGeneration"
    assert entry.hidden_size == 5120
    assert entry.kv_groups == ((4, 256),)
    assert entry.expert_formats == ()


def test_modelopt_nvfp4_weight_can_be_split_from_scales_across_shards(
    tmp_path, monkeypatch
):
    import sparklab.models.qwen3_5_moe.weight as weight_module

    base = "model.language_model.layers.0.linear_attn.out_proj"
    weight_name = base + ".weight"
    scale_name = base + ".weight_scale"
    global_name = base + ".weight_scale_2"
    save_file(
        {weight_name: torch.zeros((8, 4), dtype=torch.uint8)},
        tmp_path / "model-1.safetensors",
    )
    save_file(
        {
            scale_name: torch.ones((8, 1)),
            global_name: torch.ones(()),
        },
        tmp_path / "model-2.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    weight_name: "model-1.safetensors",
                    scale_name: "model-2.safetensors",
                    global_name: "model-2.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    config = _inferact_qwen38_27b_config()
    monkeypatch.setattr(weight_module, "cached_load_hf_config", lambda _path: config)
    monkeypatch.setattr(
        weight_module,
        "get_tp_info",
        lambda: SimpleNamespace(size=1, is_primary=lambda: True),
    )
    tensors = dict(
        weight_module.iter_weights(
            str(tmp_path),
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )

    prefix = "model.layers.0.linear_attn.out_proj"
    assert set(tensors) == {
        prefix + ".weight",
        prefix + ".weight_scale",
        prefix + ".weight_global",
    }
    assert tensors[prefix + ".weight"].shape == torch.Size([8, 4])
    assert tensors[prefix + ".weight_scale"].shape == torch.Size([8, 1])
    assert tensors[prefix + ".weight_global"].shape == torch.Size([8])
