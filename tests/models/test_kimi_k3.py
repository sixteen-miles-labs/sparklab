from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.base import AttnType
from freetoken.models.kimi_k3.config import parse_config
from freetoken.models.kimi_k3.kda_reference import recurrent_kda
from freetoken.models.kimi_k3.ops import apply_attention_residual, situ_and_mul
from freetoken.models.kimi_k3.weight import map_weight_name


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


def test_hybrid_mla_pool_only_allocates_full_layers():
    from freetoken.kvcache import create_kvcache_pool

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
    from freetoken.models.register import get_model_spec

    for arch in ("KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"):
        spec = get_model_spec(arch)
        assert spec.module == "freetoken.models.kimi_k3"


def test_tokenizer_enables_remote_xtml_only_for_kimi(monkeypatch):
    import freetoken.utils.hf as hf

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
    from freetoken.server.function_call_parser import FunctionCallParser
    from freetoken.server.reasoning_parser import ReasoningParser

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
