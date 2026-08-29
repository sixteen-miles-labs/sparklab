from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sparklab.attention.base import AttnType
from sparklab.models.glm5_next.config import parse_config
from sparklab.models.glm5_next.hyper import Glm5NextHyperConnection
from sparklab.models.glm5_next.kpool import pool_index_states, select_kpool_tokens
from sparklab.models.glm5_next.weight import (
    _CT_NVFP4_SOURCE_SPEC,
    _is_kda_main_weight,
    _quant_fp8_per_row,
    load_nvfp4_expert_sources,
    map_weight_name,
)


def _config(layers: int = 8):
    kinds = [
        "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
        for i in range(layers)
    ]
    kda = [i for i, kind in enumerate(kinds) if kind == "linear_attention"]
    full = [i for i, kind in enumerate(kinds) if kind == "deepseek_sparse_attention"]
    text = SimpleNamespace(
        hidden_size=256,
        vocab_size=1024,
        num_hidden_layers=layers,
        num_attention_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        intermediate_size=512,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=0,
        v_head_dim=128,
        mla_use_nope=True,
        layer_types=kinds,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=128,
        first_k_dense_replace=3,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        swiglu_limit=10.0,
        tie_word_embeddings=False,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=8,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        hc_mult=4,
        hc_eps=1e-6,
        hc_sinkhorn_iters=5,
        linear_attn_config={
            "kda_layers": kda,
            "full_attn_layers": full,
            "num_heads": 4,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
    )
    return SimpleNamespace(
        model_type="glm5_next",
        architectures=["Glm5NextForConditionalGeneration"],
        text_config=text,
        image_token_id=999,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_shared_expert_disk_overlap_launches_before_routed_staging():
    from sparklab.models.glm5_next.moe import Glm5NextSparseMoe

    order = []
    cache = SimpleNamespace(
        shared_expert_overlap=True,
        disk_source=object(),
        shared_expert_stream=None,
        shared_expert_overlap_calls=0,
    )

    class Experts:
        offload_cache = cache

        @staticmethod
        def routed_forward(hidden, weights, ids):
            order.append("routed")
            return torch.ones_like(hidden)

    class Shared:
        @staticmethod
        def forward(hidden):
            order.append("shared")
            return hidden * 2

    block = object.__new__(Glm5NextSparseMoe)
    block.experts = Experts()
    block.shared_experts = Shared()
    block._route = lambda hidden: (
        torch.ones(hidden.size(0), 1, dtype=torch.float32, device=hidden.device),
        torch.zeros(hidden.size(0), 1, dtype=torch.int32, device=hidden.device),
    )
    hidden = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")

    out = block.forward(hidden)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.ones_like(hidden) + hidden * 2)
    assert order == ["shared", "routed"]
    assert cache.shared_expert_overlap_calls == 1


def test_parse_config_builds_glm53_hybrid_geometry():
    cfg = parse_config(_config())
    args = cfg.glm5_next_args
    assert cfg.expert_quant == cfg.attn_quant == cfg.dense_quant == "fp8_block"
    assert cfg.weight_block_size == (128, 128)
    assert cfg.fp8_block_scale_dtype == "float32"
    assert args.kda_layer_ids == (0, 1, 2, 4, 5, 6)
    assert args.dsa_layer_ids == (3, 7)
    assert args.kda_quant == "none"
    assert cfg.attn_type_for_layer(0) is AttnType.LINEAR
    assert cfg.attn_type_for_layer(3) is AttnType.DSA
    spec = cfg.kv_cache_group_specs()[0]
    assert spec.layer_ids == (3, 7)
    assert spec.index_head_dim == 8  # normalized key + per-channel KPool gate
    assert spec.num_index_layers == 2
    assert cfg.num_moe_layers == 5
    assert cfg.linear_state_snapshots is False


def test_parse_config_builds_opt_in_kda_fp8_projections():
    from sparklab.kernels.triton.fp8_pertensor_linear import Fp8PerTensorLinear
    from sparklab.layers import LinearReplicated
    from sparklab.models.glm5_next.kda import Glm5NextDeltaAttention

    hf = _config()
    hf.text_config.freetoken_kda_quant = "fp8_pertensor"
    cfg = parse_config(hf)
    attention = Glm5NextDeltaAttention(cfg, layer_id=0)

    assert cfg.glm5_next_args.kda_quant == "fp8_pertensor"
    assert isinstance(attention.q_proj, Fp8PerTensorLinear)
    assert isinstance(attention.k_proj, Fp8PerTensorLinear)
    assert isinstance(attention.v_proj, Fp8PerTensorLinear)
    assert isinstance(attention.o_proj, Fp8PerTensorLinear)
    assert isinstance(attention.f_a_proj, LinearReplicated)
    assert isinstance(attention.g_b_proj, LinearReplicated)


def test_parse_config_accepts_scoped_conversion_kda_override(monkeypatch):
    monkeypatch.setenv("_SPARKLAB_CONVERT_GLM5_KDA_QUANT", "fp8_pertensor")

    cfg = parse_config(_config())

    assert cfg.glm5_next_args.kda_quant == "fp8_pertensor"


def test_kda_fp8_per_row_quantization_matches_dequantized_reference():
    torch.manual_seed(23)
    source = torch.randn(16, 32, dtype=torch.bfloat16)
    quantized, scale = _quant_fp8_per_row(source)

    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.shape == source.shape
    assert scale.dtype == torch.float32
    assert scale.shape == (source.shape[0],)
    restored = quantized.float() * scale[:, None]
    torch.testing.assert_close(restored, source.float(), rtol=0.065, atol=0.025)


def test_kda_fp8_selector_excludes_sparse_mla_output_projection():
    cfg = parse_config(_config())

    assert _is_kda_main_weight("model.layers.0.self_attn.o_proj.weight", cfg)
    assert not _is_kda_main_weight("model.layers.3.self_attn.o_proj.weight", cfg)


def test_parse_config_accepts_redhat_expert_only_nvfp4():
    hf = _config()
    hf.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "format": "nvfp4-pack-quantized",
                "targets": ["re:.*mlp\\.experts\\..*(gate|up|down)_proj$"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "strategy": "tensor_group",
                },
            }
        },
    }
    cfg = parse_config(hf)
    assert cfg.expert_quant == cfg.moe_weight_format == "nvfp4"
    assert cfg.attn_quant == cfg.dense_quant == "none"
    assert cfg.weight_block_size is None


def test_glm53_compressed_tensors_expert_source_pattern():
    match = _CT_NVFP4_SOURCE_SPEC.key_pattern.match(
        "model.language_model.layers.3.mlp.experts.17.gate_proj.weight_packed"
    )
    assert match is not None
    assert match.groupdict() == {
        "layer": "3",
        "expert": "17",
        "proj": "gate_proj",
        "kind": "weight_packed",
    }


def test_glm53_compressed_tensors_experts_preserve_packed_rows(tmp_path, monkeypatch):
    import json

    from safetensors.torch import save_file

    tensors = {}
    for expert in range(2):
        for projection, base_value in (("gate_proj", 10), ("up_proj", 20), ("down_proj", 30)):
            base = (
                f"model.language_model.layers.0.mlp.experts.{expert}."
                f"{projection}"
            )
            value = base_value + expert
            tensors[base + ".weight_packed"] = torch.full(
                (16, 8), value, dtype=torch.uint8
            )
            tensors[base + ".weight_scale"] = torch.full(
                (16, 1), value, dtype=torch.float8_e4m3fn
            )
            tensors[base + ".weight_global_scale"] = torch.tensor(
                [2.0 + expert], dtype=torch.float32
            )
    shard = "model.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )
    config = SimpleNamespace(
        num_experts=2,
        hidden_size=16,
        moe_intermediate_size=16,
        num_moe_layers=1,
        first_k_dense_replace=0,
    )
    captured = {}
    monkeypatch.setattr(
        "sparklab.models.glm5_next.weight.get_tp_info",
        lambda: SimpleNamespace(is_primary=lambda: False),
    )

    def sink(layer, banks):
        assert layer == 0
        captured.update({name: bank.tensor.clone() for name, bank in banks.items()})
        for bank in banks.values():
            bank.release()

    load_nvfp4_expert_sources(str(tmp_path), config, layer_sink=sink)
    assert captured["gate_up_packed"][1, 0, 0].item() == 11
    assert captured["gate_up_packed"][1, 16, 0].item() == 21
    assert captured["down_packed"][1, 0, 0].item() == 31
    torch.testing.assert_close(
        captured["gate_up_global"][0, :16], torch.full((16,), 0.5, dtype=torch.float16)
    )
    torch.testing.assert_close(
        captured["down_global"][1],
        torch.full((16,), 1.0 / 3.0, dtype=torch.float16),
    )


def test_parse_config_rejects_disagreeing_layer_metadata():
    hf = _config()
    hf.text_config.linear_attn_config["kda_layers"].remove(0)
    with pytest.raises(ValueError, match="disagrees"):
        parse_config(hf)


def test_kpool_learned_dimensionwise_compression():
    # APE=0, gate logits make channel 0 choose token 0 and channel 1 choose token 3.
    keys = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    gates = torch.tensor([[20.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 20.0]])
    pooled = pool_index_states(torch.cat((keys, gates), -1), torch.zeros(4, 2), 4)
    torch.testing.assert_close(pooled, torch.tensor([[1.0, 40.0]]), atol=1e-6, rtol=1e-6)


def test_kpool_selection_expands_pools_and_keeps_incomplete_tail():
    torch.manual_seed(3)
    packed = torch.randn(14, 8)  # key dim 4 + gate dim 4
    q = torch.randn(2, 2, 4)
    weights = torch.randn(2, 2)
    selected, counts = select_kpool_tokens(
        q,
        weights,
        packed,
        torch.randn(4, 4),
        torch.tensor([7, 14]),
        token_topk=8,
        pool_size=4,
    )
    assert selected.shape == (2, 11)
    assert counts.tolist() == [7, 10]
    assert selected[0, :7].tolist() == list(range(7))  # short identity path
    # Long row: two complete selected pools (8 tokens) plus raw tail [12, 13].
    assert set(selected[1, :8].tolist()) in (
        set(range(0, 8)),
        set(range(4, 12)),
        set(range(0, 4)) | set(range(8, 12)),
    )
    assert selected[1, 8:10].tolist() == [12, 13]
    assert (selected[1, 10:] == -1).all()


def test_mhc_matches_explicit_mapping_and_sinkhorn_shape():
    torch.manual_seed(5)
    hc = Glm5NextHyperConnection(3, 4, 1e-6, 20, 1e-5)
    hc.fn.normal_()
    hc.base.normal_()
    hc.scale.copy_(torch.tensor([0.5, 0.75, 1.25]))
    streams = torch.randn(2, 4, 3)
    post, comb, collapsed = hc.forward(streams)
    assert post.shape == (2, 4)
    assert comb.shape == (2, 4, 4)
    assert collapsed.shape == (2, 3)
    # Sinkhorn projection is numerically doubly stochastic.
    torch.testing.assert_close(comb.sum(-1), torch.ones(2, 4), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(comb.sum(-2), torch.ones(2, 4), atol=2e-3, rtol=2e-3)
    output = torch.randn(2, 3)
    expanded = hc.expand(output, streams, post, comb)
    expected = post.unsqueeze(-1) * output.unsqueeze(1) + comb.transpose(-1, -2) @ streams
    torch.testing.assert_close(expanded, expected)


def test_glm53_checkpoint_name_mapping_and_registry():
    assert map_weight_name("model.visual.blocks.0.weight") is None
    assert map_weight_name("model.language_model.layers.3.hc_attn_fn") == (
        "model.layers.3.attn_hc.fn"
    )
    assert map_weight_name("model.language_model.layers.3.hc_ffn_scale") == (
        "model.layers.3.ffn_hc.scale"
    )
    assert map_weight_name("model.language_model.layers.3.self_attn.q_a_proj.weight") == (
        "model.layers.3.self_attn.q_a_proj.weight"
    )
    from sparklab.models.register import get_model_spec

    for arch in ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"):
        assert get_model_spec(arch).module == "sparklab.models.glm5_next"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_glm53_per_head_kda_kernel_matches_reference_and_continuation():
    from sparklab.kernels.fla import fused_sigmoid_gating_delta_rule_update

    torch.manual_seed(17)
    batch, steps, heads, dim = 1, 5, 3, 8
    q = torch.randn(batch, steps, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(steps, heads * dim, device="cuda", dtype=torch.bfloat16)
    beta_logits = torch.randn(steps, heads, device="cuda")
    a_log = torch.randn(heads, device="cuda")
    dt_bias = torch.randn(heads * dim, device="cuda")

    def run(q_part, k_part, v_part, a_part, b_part, state):
        n = q_part.shape[1]
        return fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a_part,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q_part,
            k=k_part,
            v=v_part,
            b=b_part,
            initial_state_source=state,
            initial_state_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
            scale=dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, n], device="cuda", dtype=torch.int32),
            is_kda=True,
            kda_a_log_per_head=True,
            lower_bound=-5.0,
        )

    whole_state = torch.zeros(1, heads, dim, dim, device="cuda")
    whole = run(q, k, v, a, beta_logits, whole_state)
    split_state = torch.zeros_like(whole_state)
    first = run(q[:, :2], k[:, :2], v[:, :2], a[:2], beta_logits[:2], split_state)
    second = run(q[:, 2:], k[:, 2:], v[:, 2:], a[2:], beta_logits[2:], split_state)
    torch.testing.assert_close(torch.cat((first, second), dim=1), whole, atol=0, rtol=0)

    # Independent explicit recurrence: A_log is one scalar per head and broadcasts
    # over that head's 128 (reduced to 8 here) key coordinates.
    q_ref = q.float()
    k_ref = k.float()
    q_ref = q_ref / torch.sqrt(q_ref.square().sum(-1, keepdim=True) + 1e-6) * dim**-0.5
    k_ref = k_ref / torch.sqrt(k_ref.square().sum(-1, keepdim=True) + 1e-6)
    # GLM-5.3's safe bounded forget gate is the checkpoint's native rule, not
    # the older ``clamp(-exp(A_log) * softplus(x), min=lower_bound)`` KDA rule.
    decay = -5.0 * torch.sigmoid(
        a_log.exp().view(1, heads, 1)
        * (a.float().view(steps, heads, dim) + dt_bias.view(heads, dim))
    )
    state = torch.zeros(heads, dim, dim, device="cuda")
    expected = []
    for step in range(steps):
        state *= decay[step].exp().unsqueeze(-1)
        memory = torch.einsum("hkv,hk->hv", state, k_ref[0, step])
        delta = (v[0, step].float() - memory) * beta_logits[step].sigmoid().unsqueeze(-1)
        state += torch.einsum("hk,hv->hkv", k_ref[0, step], delta)
        expected.append(torch.einsum("hkv,hk->hv", state, q_ref[0, step]))
    expected = torch.stack(expected).to(torch.bfloat16).unsqueeze(0)
    torch.testing.assert_close(whole, expected, atol=0, rtol=0)

    # GLM's causal-convolution prefill output is transposed from [B, C, T] to
    # [B, T, C], so its feature stride is T rather than 1. The fused kernel must
    # materialize that unsupported layout before addressing individual features.
    packed = torch.cat(
        (q.flatten(2), k.flatten(2), v.flatten(2)),
        dim=-1,
    )
    conv_layout = packed.transpose(1, 2).contiguous().transpose(1, 2)
    q_strided, k_strided, v_strided = [
        part.view(batch, steps, heads, dim)
        for part in conv_layout.split([heads * dim] * 3, dim=-1)
    ]
    assert q_strided.stride(-1) == steps
    assert q_strided.stride(-1) != 1
    strided_state = torch.zeros_like(whole_state)
    strided = run(
        q_strided,
        k_strided,
        v_strided,
        a,
        beta_logits,
        strided_state,
    )
    torch.testing.assert_close(strided, expected, atol=0, rtol=0)
