from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from sparklab.attention.base import AttnType
from sparklab.models.kimi_k3.config import parse_config
from sparklab.models.kimi_k3.kda_reference import recurrent_kda
from sparklab.models.kimi_k3.ops import apply_attention_residual, situ_and_mul
from sparklab.models.kimi_k3.weight import (
    _MODEL_OPT_EXPERT_RE,
    _NVFP4_SOURCE_SPEC,
    _quant_fp8_per_row,
    _iter_modelopt_weights,
    map_weight_name,
    transform_ftw_weights,
)


def _config(layers: int = 8):
    full = list(range(4, layers + 1, 4))
    kda = [i for i in range(1, layers + 1) if i not in full]
    text = SimpleNamespace(
        hidden_size=64,
        vocab_size=256,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        hidden_act="situ",
        intermediate_size=128,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=64,
        v_head_dim=16,
        mla_use_output_gate=True,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=32,
        first_k_dense_replace=1,
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        num_expert_group=1,
        topk_group=1,
        routed_expert_hidden_size=32,
        latent_moe_use_norm=True,
        attn_res_block_size=4,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        tie_word_embeddings=False,
        linear_attn_config={
            "kda_layers": kda,
            "full_attn_layers": full,
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        quantization_config={
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 32}}
            },
        },
    )
    return SimpleNamespace(
        model_type="kimi_k3",
        architectures=["KimiK3ForConditionalGeneration"],
        text_config=text,
        media_placeholder_token_id=201,
    )


def test_package_exports_nvfp4_expert_loader_contract():
    import sparklab.models.kimi_k3 as package
    from sparklab.models.kimi_k3 import weight

    assert package.load_nvfp4_expert_sources is weight.load_nvfp4_expert_sources
    assert (
        package.load_nvfp4_expert_sources_parallel
        is weight.load_nvfp4_expert_sources_parallel
    )


def test_parse_config_builds_one_based_hybrid_partition():
    cfg = parse_config(_config())
    assert cfg.num_layers == 8
    assert tuple(i for i in range(8) if cfg.is_linear_layer(i)) == (0, 1, 2, 4, 5, 6)
    assert cfg.attn_type_for_layer(3) is AttnType.MLA
    assert cfg.attn_type_for_layer(7) is AttnType.MLA
    assert cfg.expert_quant == "mxfp4" and cfg.moe_weight_format == "mxfp4"
    assert cfg.num_moe_layers == 7
    assert cfg.linear_state_snapshots is False
    assert cfg.kimi_k3_args.routed_expert_hidden_size == 32


def test_parse_config_detects_nvidia_modelopt_mixed_layout():
    hf = _config()
    hf.text_config.quantization_config = None
    hf.quantization_config = {
        "quant_method": "modelopt_mixed",
        "producer": {"name": "modelopt", "version": "0.45.0"},
        "quantized_layers": {
            "language_model.model.layers.1.block_sparse_moe.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
            "language_model.model.layers.1.self_attn.q_proj": {
                "quant_algo": "FP8_PB_WO"
            },
        },
    }
    cfg = parse_config(hf)
    assert cfg.expert_quant == "nvfp4"
    assert cfg.expert_hidden_size == 32
    assert cfg.routed_expert_hidden_size == 32
    assert cfg.moe_weight_format == "nvfp4"
    assert cfg.attn_quant == "fp8_block"
    assert cfg.weight_block_size == (128, 128)
    assert cfg.fp8_block_scale_dtype == "float32"


def test_kimi_resident_mlp_fp8_is_explicit(monkeypatch):
    monkeypatch.delenv("SPARKLAB_KIMI_MLP_FP8", raising=False)
    cfg = parse_config(_config())
    assert cfg.dense_quant == cfg.lm_head_quant == "none"
    monkeypatch.setenv("SPARKLAB_KIMI_MLP_FP8", "1")
    cfg = parse_config(_config())
    assert cfg.dense_quant == cfg.lm_head_quant == "fp8_pertensor"


def test_ftw_transform_quantizes_only_resident_mlp():
    from types import SimpleNamespace

    shared = torch.linspace(-2, 2, 32, dtype=torch.float32).reshape(4, 8).bfloat16()
    attention = torch.ones(4, 8, dtype=torch.bfloat16)
    shared_name = (
        "model.layers.1.block_sparse_moe.shared_experts.gate_up_proj.weight"
    )
    attention_name = "model.layers.1.self_attn.q_proj.weight"
    embedding_name = "model.embed_tokens.weight"
    embedding = torch.linspace(-1, 1, 24, dtype=torch.float32).reshape(3, 8).bfloat16()
    got = dict(
        transform_ftw_weights(
            iter(
                (
                    (shared_name, shared),
                    (embedding_name, embedding),
                    (attention_name, attention),
                )
            ),
            SimpleNamespace(dense_quant="fp8_pertensor"),
        )
    )
    assert got[shared_name].dtype == torch.float8_e4m3fn
    scale_name = shared_name.removesuffix(".weight") + ".weight_scale"
    assert got[scale_name].shape == (4,)
    restored = got[shared_name].float() * got[scale_name][:, None]
    # E4M3 has four mantissa bits; per-row max scaling bounds relative rounding to
    # approximately half an FP8 bin (6.25%).
    torch.testing.assert_close(restored, shared.float(), rtol=0.07, atol=0.01)
    assert got[embedding_name].dtype == torch.float8_e4m3fn
    assert got["model.embed_tokens.weight_scale"].shape == (3,)
    assert got[attention_name] is attention


def test_ftw_transform_is_passthrough_when_disabled():
    from types import SimpleNamespace

    weight = torch.ones(4, 8, dtype=torch.bfloat16)
    items = [("model.layers.0.mlp.down_proj.weight", weight)]
    assert list(transform_ftw_weights(iter(items), SimpleNamespace(dense_quant="none"))) == items


def test_quant_fp8_per_row_handles_zero_rows():
    q, scale = _quant_fp8_per_row(torch.zeros(2, 8, dtype=torch.bfloat16))
    assert q.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.gt(0).all()


def test_nvidia_modelopt_expert_key_mapping_excludes_activation_scale():
    base = "language_model.model.layers.1.block_sparse_moe.experts.7.w1."
    for kind in ("weight", "weight_scale", "weight_scale_2"):
        match = _MODEL_OPT_EXPERT_RE.match(base + kind)
        assert match is not None
        assert match.groupdict() == {
            "layer": "1",
            "expert": "7",
            "proj": "w1",
            "kind": kind,
        }
    assert _MODEL_OPT_EXPERT_RE.match(base + "input_scale") is None


def test_nvidia_modelopt_resident_loader_keeps_only_kernel_aligned_fp8(tmp_path):
    from safetensors.torch import save_file

    tensors = {
        "language_model.model.layers.0.self_attn.q_proj.weight": torch.ones(
            128, 128, dtype=torch.float8_e4m3fn
        ),
        "language_model.model.layers.0.self_attn.q_proj.weight_scale": torch.ones(
            1, 1, 1, 1, dtype=torch.float32
        ),
        "language_model.model.layers.0.self_attn.b_proj.weight": torch.ones(
            96, 128, dtype=torch.float8_e4m3fn
        ),
        "language_model.model.layers.0.self_attn.b_proj.weight_scale": torch.ones(
            1, 1, 1, 1, dtype=torch.float32
        ),
        "language_model.model.layers.0.mlp.gate_proj.weight": torch.ones(
            8, 8, dtype=torch.bfloat16
        ),
        "language_model.model.layers.0.mlp.up_proj.weight": torch.full(
            (8, 8), 2, dtype=torch.bfloat16
        ),
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": torch.zeros(
            8, 4, dtype=torch.uint8
        ),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )

    got = dict(
        _iter_modelopt_weights(
            str(tmp_path),
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )
    assert got["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    assert got["model.layers.0.self_attn.q_proj.weight_scale_inv"].shape == (1, 1)
    assert got["model.layers.0.self_attn.b_proj.weight"].dtype == torch.bfloat16
    merged = got["model.layers.0.mlp.gate_up_proj.weight"]
    assert merged.shape == (16, 8)
    assert torch.equal(merged[:8], torch.ones_like(merged[:8]))
    assert torch.equal(merged[8:], torch.full_like(merged[8:], 2))
    assert not any("experts" in name for name in got)


def test_nvidia_modelopt_expert_loader_uses_latent_width(tmp_path):
    """The 7168-wide residual stream projects into 3584-wide routed experts."""
    from types import SimpleNamespace

    from sparklab.models.nvfp4_banks import load_nvfp4_expert_source_banks
    from safetensors.torch import save_file

    experts, latent, intermediate = 2, 32, 16
    tensors = {}
    expected_globals = {"w1": 0.5, "w3": 0.25, "w2": 0.75}
    for expert in range(experts):
        for projection, code in (("w1", 1), ("w3", 3), ("w2", 2)):
            prefix = (
                f"language_model.model.layers.1.block_sparse_moe.experts."
                f"{expert}.{projection}."
            )
            if projection in ("w1", "w3"):
                shape = (intermediate, latent // 2)
                scale_shape = (intermediate, latent // 16)
            else:
                shape = (latent, intermediate // 2)
                scale_shape = (latent, intermediate // 16)
            tensors[prefix + "weight"] = torch.full(shape, code, dtype=torch.uint8)
            tensors[prefix + "weight_scale"] = torch.full(
                scale_shape, 1.0, dtype=torch.float8_e4m3fn
            )
            tensors[prefix + "weight_scale_2"] = torch.tensor(
                expected_globals[projection], dtype=torch.float32
            )
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )
    config = SimpleNamespace(
        num_layers=2,
        first_k_dense_replace=1,
        num_experts=experts,
        hidden_size=64,
        expert_hidden_size=latent,
        moe_intermediate_size=intermediate,
    )
    seen = {}

    def sink(layer_id, banks):
        assert layer_id == 0
        seen.update({name: bank.tensor.clone() for name, bank in banks.items()})
        for bank in banks.values():
            bank.release()

    load_nvfp4_expert_source_banks(
        str(tmp_path),
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=lambda _path: None,
        primary=False,
        layer_sink=sink,
    )
    assert seen["gate_up_packed"].shape == (experts, 2 * intermediate, latent // 2)
    assert seen["gate_up_scale"].shape == (experts, 2 * intermediate, latent // 16)
    assert seen["down_packed"].shape == (experts, latent, intermediate // 2)
    assert seen["down_scale"].shape == (experts, latent, intermediate // 16)
    assert seen["gate_up_packed"][:, :intermediate].eq(1).all()
    assert seen["gate_up_packed"][:, intermediate:].eq(3).all()
    assert seen["down_packed"].eq(2).all()
    assert seen["gate_up_global"][:, :intermediate].eq(0.5).all()
    assert seen["gate_up_global"][:, intermediate:].eq(0.25).all()
    assert seen["down_global"].eq(0.75).all()


def test_hybrid_mla_pool_only_allocates_full_layers():
    from sparklab.runtime.kvcache import create_kvcache_pool

    cfg = parse_config(_config())
    pool = create_kvcache_pool(
        cfg, num_pages=3, page_size=1, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert pool.num_layers == 2
    assert pool.latent_rows(3).shape == (3, 80)
    assert pool.latent_rows(7).shape == (3, 80)
    with pytest.raises(KeyError):
        pool.latent_rows(0)


def test_parse_config_rejects_layer_holes():
    hf = _config()
    hf.text_config.linear_attn_config["kda_layers"].remove(1)
    with pytest.raises(ValueError, match="missing"):
        parse_config(hf)


def test_situ_matches_scalar_definition():
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    got = situ_and_mul(x, beta=4.0, linear_beta=25.0)
    gate, up = x.chunk(2, -1)
    expected = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected = expected * (25.0 * torch.tanh(up / 25.0))
    torch.testing.assert_close(got, expected)


def test_attention_residual_is_convex_mixture():
    prefix = torch.tensor([[3.0, 4.0]])
    blocks = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    # Zero projection makes all three scores equal.
    got = apply_attention_residual(
        prefix, blocks, torch.zeros(1, 2), torch.ones(2), 1e-5
    )
    torch.testing.assert_close(got, torch.tensor([[4.0 / 3.0, 2.0]]))


def test_kda_chunk_continuation_matches_whole_sequence():
    torch.manual_seed(7)
    shape = (2, 5, 3, 4)
    q, k, v = (torch.randn(shape) for _ in range(3))
    a = torch.randn(shape)
    beta = torch.randn(2, 5, 3)
    A_log = torch.randn(shape[-1])
    dt_bias = torch.randn(3 * 4)
    whole, final = recurrent_kda(q, k, v, a, beta, A_log, dt_bias, lower_bound=-5.0)
    first, state = recurrent_kda(
        q[:, :2], k[:, :2], v[:, :2], a[:, :2], beta[:, :2], A_log, dt_bias,
        lower_bound=-5.0,
    )
    second, continued = recurrent_kda(
        q[:, 2:], k[:, 2:], v[:, 2:], a[:, 2:], beta[:, 2:], A_log, dt_bias,
        initial_state=state, lower_bound=-5.0,
    )
    torch.testing.assert_close(torch.cat((first, second), dim=1), whole)
    torch.testing.assert_close(continued, final)


def test_kda_rejects_stale_per_head_a_log_layout():
    q = k = v = a = torch.zeros(1, 1, 3, 4)
    with pytest.raises(ValueError, match="head_dim=4"):
        recurrent_kda(
            q, k, v, a, torch.zeros(1, 1, 3), torch.zeros(3), torch.zeros(12)
        )


def test_text_checkpoint_name_mapping():
    assert map_weight_name("vision_tower.blocks.0.weight") is None
    assert map_weight_name("language_model.model.layers.1.self_attn.A_log") == (
        "model.layers.1.self_attn.A_log"
    )
    assert map_weight_name(
        "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias"
    ) == "model.layers.1.block_sparse_moe.e_score_correction_bias"
    assert map_weight_name(
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"
    ) is None


def test_registry_has_wrapper_and_text_architectures():
    from sparklab.models.register import get_model_spec

    for arch in ("KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"):
        spec = get_model_spec(arch)
        assert spec.module == "sparklab.models.kimi_k3"


def test_tokenizer_enables_remote_xtml_only_for_kimi(monkeypatch):
    import sparklab.utils.hf as hf

    seen = []
    fake = SimpleNamespace(chat_template="python-xtml")
    monkeypatch.setattr(hf, "_raw_config_json", lambda _: {"model_type": "kimi_k3"})
    monkeypatch.setattr(
        hf.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: seen.append((path, kwargs)) or fake,
    )
    assert hf.load_tokenizer("moonshotai/Kimi-K3") is fake
    assert seen == [("moonshotai/Kimi-K3", {"trust_remote_code": True})]


def test_kimi_xtml_reasoning_and_tool_call_parsers():
    from sparklab.serving.function_call_parser import FunctionCallParser
    from sparklab.serving.reasoning_parser import ReasoningParser

    transition = "<|close|>think<|sep|><|open|>response<|sep|>"
    tools_open = "<|open|>tools<|sep|>"
    wire = (
        f"check forecast{transition}It may rain."
        f"<|close|>response<|sep|>{tools_open}"
        '<|open|>call tool="weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Toronto'
        '<|close|>argument<|sep|>'
        '<|open|>argument key="days" type="number"<|sep|>2'
        '<|close|>argument<|sep|>'
        '<|close|>call<|sep|><|close|>tools<|sep|>'
    )
    reasoning, normal = ReasoningParser("kimi_k3", force_reasoning=True).parse_non_stream(wire)
    assert reasoning == "check forecast"
    parser = FunctionCallParser(
        [{"type": "function", "function": {"name": "weather", "parameters": {}}}],
        tool_call_parser="kimi_k3",
    )
    result = parser.parse_non_stream(normal)
    assert result.normal_text == "It may rain."
    assert result.calls[0].name == "weather"
    assert json.loads(result.calls[0].parameters) == {"city": "Toronto", "days": 2}
    assert parser.supports_streaming()

    streamed = FunctionCallParser(
        [{"type": "function", "function": {"name": "weather", "parameters": {}}}],
        tool_call_parser="kimi_k3",
    )
    text, calls = "", []
    for cut in (normal[:17], normal[17:53], normal[53:101], normal[101:]):
        delta, new_calls = streamed.parse_stream_chunk(cut)
        text += delta
        calls.extend(new_calls)
    text += streamed.finish_stream()
    assert text == "It may rain."
    assert len(calls) == 1 and json.loads(calls[0].parameters)["days"] == 2
