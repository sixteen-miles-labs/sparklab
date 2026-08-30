from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sparklab.attention import AttnType
from sparklab.attention.qsa import QSAAttnBackend
from sparklab.models.qwen4_exp.config import parse_config
from sparklab.models.qwen4_exp.hyper import GroupedPlusOneRMSNorm, Qwen4GatedResidual
from sparklab.models.qwen4_exp.ple import DiskNGramEmbedding, RawNGramStore
from sparklab.models.qwen4_exp.weight import copy_external_artifacts
from sparklab.models.qwen4_exp.weight import _iter_experts_layer_order
from sparklab.models.qwen4_exp.weight import iter_weights


def _config(layers: int = 8):
    kinds = ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(layers)]
    text = SimpleNamespace(
        num_hidden_layers=layers,
        layer_types=kinds,
        hidden_size=64,
        vocab_size=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5},
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
        hc_count=4,
        hc_lowrank=8,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=16,
        indexer_compress_ratio=4,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=100,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=2,
        seed=1234,
        eos_token_id=7,
        output_gate_type="sigmoid",
    )
    return SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text,
        image_token_id=200,
    )


def test_config_declares_qsa_separately_from_minimax_bsa():
    cfg = parse_config(_config())
    assert cfg.attn_type_for_layer(3) is AttnType.QSA
    assert cfg.attn_type_for_layer(7) is AttnType.QSA
    assert cfg.is_linear_layer(0)
    assert cfg.qwen4_exp_args.ple_layer_ids == (1,)
    specs = cfg.kv_cache_group_specs()
    assert len(specs) == 1 and specs[0].layer_ids == (3, 7)
    assert cfg.linear_state_snapshots is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qwen_shared_expert_overlap_launches_before_routed_staging():
    from sparklab.models.qwen3_5_moe.moe import Qwen3_5MoE

    order = []
    cache = SimpleNamespace(
        shared_expert_overlap=True,
        disk_source=object(),
        shared_expert_stream=None,
        shared_expert_overlap_calls=0,
    )

    class Gate:
        @staticmethod
        def forward(hidden):
            return torch.zeros(
                hidden.size(0), 8, device=hidden.device, dtype=hidden.dtype
            )

    class Experts:
        offload_cache = cache

        @staticmethod
        def forward(*, hidden_states, router_logits):
            order.append("routed")
            output = torch.ones_like(hidden_states)
            hidden_states.zero_()
            return output

    class Shared:
        @staticmethod
        def forward(hidden):
            order.append("shared")
            return hidden * 2

    block = object.__new__(Qwen3_5MoE)
    block.gate = Gate()
    block.experts = Experts()
    block.shared_expert = Shared()
    block.shared_expert_gate = Gate()
    hidden = torch.randn(3, 8, dtype=torch.bfloat16, device="cuda")
    original = hidden.clone()

    out = block.forward(hidden)
    torch.cuda.synchronize()

    expected = torch.ones_like(original) + original
    torch.testing.assert_close(out, expected)
    assert order == ["shared", "routed"]
    assert cache.shared_expert_overlap_calls == 1


def test_config_accepts_conversion_owned_nvfp4_experts():
    source = _config()
    source.text_config.sparklab_expert_quant = "nvfp4"
    assert parse_config(source).expert_quant == "nvfp4"


def test_config_detects_official_routed_only_block_fp8():
    source = _config()
    source.quantization_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    config = parse_config(source)
    assert config.expert_quant == "fp8_block"
    assert config.weight_block_size == (128, 128)
    assert config.fp8_block_scale_dtype == "bfloat16"
    assert config.shared_expert_quant == "none"


def test_config_detects_inferact_modelopt_nvfp4_experts():
    source = _config()
    source.quantization_config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "with_input_scale": True,
    }
    config = parse_config(source)
    assert config.expert_quant == "nvfp4"
    assert config.weight_block_size is None
    assert config.shared_expert_quant == "none"


def test_expert_wrapper_receives_conversion_streaming_controls(monkeypatch):
    import sparklab.models.qwen3_5_moe.weight as shared_weight
    from sparklab.moe.expert_banks import ExpertBanks, _build_expert_banks

    captured = {}
    expected = ExpertBanks("fp8_block", {})

    def fake_setup(*args, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(shared_weight, "setup_offload_expert_banks", fake_setup)
    sink = object()
    config = SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        expert_quant="fp8_block",
    )

    result = _build_expert_banks(
        "unused",
        config,
        torch.device("cpu"),
        torch.bfloat16,
        False,
        False,
        3,
        4096,
        decode_target="cpu",
        layer_sink=sink,
    )

    assert result is expected
    assert captured == {
        "device": torch.device("cpu"),
        "dtype": torch.bfloat16,
        "dummy": False,
        "parallel": False,
        "workers": 3,
        "chunk": 4096,
        "decode_target": "cpu",
        "layer_sink": sink,
    }


def test_hyper_connection_matches_reference_equations():
    torch.manual_seed(4)
    op = Qwen4GatedResidual(8, 4, 5, 1e-6)
    for tensor in op.state_dict().values():
        tensor.copy_(torch.randn_like(tensor) * 0.1)
    x = torch.randn(3, 32)
    mixed, residual, injection = op.forward(x)

    grouped = x.float().view(3, 4, 8)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
    normalized = normalized.flatten(-2) * (1 + op.hc_norm.weight.float())
    down = torch.nn.functional.linear(normalized, op.input_mix_weight_down.weight)
    weight = torch.sigmoid(torch.nn.functional.linear(
        torch.nn.functional.silu(down / 4), op.input_mix_weight_up.weight
    )).view(3, 4, 8)
    expected_mixed = (weight * normalized.view(3, 4, 8)).mean(1)
    expected_injection = 2 * torch.sigmoid(
        torch.nn.functional.linear(normalized, op.block_inject_weight.weight) / 4
    )
    torch.testing.assert_close(mixed, expected_mixed)
    torch.testing.assert_close(residual, x)
    torch.testing.assert_close(injection, expected_injection)


def test_grouped_norm_has_single_bfloat16_rounding():
    torch.manual_seed(23)
    op = GroupedPlusOneRMSNorm(16, 4, 1e-6)
    op.weight.copy_(torch.randn(16) * 0.7)
    x = torch.randn(5, 16).to(torch.bfloat16)
    grouped = x.float().view(5, 4, 4)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
    expected = (normalized.flatten(-2) * (1 + op.weight.float())).to(torch.bfloat16)
    torch.testing.assert_close(op.forward(x), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grouped_norm_cuda_kernel_is_reference_exact():
    torch.manual_seed(31)
    op = GroupedPlusOneRMSNorm(10240, 2560, 1e-6)
    op.weight = torch.randn(10240, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(2, 10240, dtype=torch.bfloat16, device="cuda")
    grouped = x.float().view(2, 4, 2560)
    normalized = grouped * torch.rsqrt(
        grouped.square().mean(-1, keepdim=True) + 1e-6
    )
    expected = (normalized.flatten(-2) * (1 + op.weight.float())).to(torch.bfloat16)
    torch.testing.assert_close(op.forward(x), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hyper_connection_cuda_kernels_are_reference_exact():
    from sparklab.kernels.triton.qwen4 import (
        hyper_injection_weights,
        hyper_mix,
        hyper_residual_inject,
        scaled_silu,
    )

    torch.manual_seed(35)
    rows, groups, hidden, lowrank = 2, 4, 2560, 320
    width = groups * hidden
    normed = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
    logits = torch.randn_like(normed)
    down = torch.randn(rows, lowrank, dtype=torch.bfloat16, device="cuda")
    injection_logits = torch.randn(rows, groups, dtype=torch.bfloat16, device="cuda")
    branch = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(rows, groups, dtype=torch.bfloat16, device="cuda")

    expected_mix = (
        torch.sigmoid(logits).view(rows, groups, hidden)
        * normed.view(rows, groups, hidden)
    ).mean(-2)
    expected_residual = normed + (
        branch.unsqueeze(-2) * injection.unsqueeze(-1)
    ).flatten(-2)
    torch.testing.assert_close(
        scaled_silu(down, groups),
        torch.nn.functional.silu(down / groups),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        hyper_mix(logits, normed, groups=groups, group_size=hidden),
        expected_mix,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        hyper_injection_weights(injection_logits, groups),
        2 * torch.sigmoid(injection_logits / groups),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        hyper_residual_inject(branch, normed, injection, groups=groups),
        expected_residual,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ple_decode_conv_fuses_state_update_reference_exactly():
    from sparklab.kernels.triton.qwen4 import ple_conv_decode

    torch.manual_seed(37)
    batch, width, state_len, kernel, dilation = 2, 512, 9, 4, 3
    x = torch.randn(batch, width, dtype=torch.bfloat16, device="cuda")
    initial = torch.randn(4, width, state_len, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(width, 1, kernel, dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    expected_state = initial.clone()
    expected = []
    for batch_index, slot in enumerate(indices.tolist()):
        current = x[batch_index:batch_index + 1].T.contiguous()
        history = torch.cat((expected_state[slot], current), -1)
        conv = torch.nn.functional.conv1d(
            history.unsqueeze(0), weight, groups=width, dilation=dilation
        ).squeeze(0).T
        expected_state[slot].copy_(history[:, -state_len:])
        expected.append(torch.nn.functional.silu(conv))
    expected = torch.cat(expected, 0)

    state = initial.clone()
    got = ple_conv_decode(x, state, weight, indices, dilation)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qsa_fused_index_scores_preserve_topk_selection():
    from sparklab.kernels.triton.qwen4 import qsa_index_scores

    torch.manual_seed(41)
    query = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    keys = torch.randn(777, 128, dtype=torch.bfloat16, device="cuda")
    expected = torch.relu(query.float() @ keys.float().T).sum(0) / (128**0.5)
    got = qsa_index_scores(query, keys)

    torch.testing.assert_close(got, expected, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(
        torch.topk(got, 512, sorted=False).indices.sort().values,
        torch.topk(expected, 512, sorted=False).indices.sort().values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qsa_fused_row_expansion_matches_chronological_reference():
    from sparklab.kernels.triton.qwen4 import qsa_expand_selected_rows

    torch.manual_seed(43)
    blocks = torch.randperm(777, device="cuda")[:512]
    physical = torch.randperm(4096, device="cuda", dtype=torch.int64).to(torch.int32)
    visible = 777 * 4 + 3
    offsets = torch.arange(4, device="cuda")
    logical = torch.sort(
        torch.cat(((blocks[:, None] * 4 + offsets).flatten(), torch.arange(3108, 3111, device="cuda")))
    ).values
    expected = physical.index_select(0, logical)

    got = qsa_expand_selected_rows(blocks, physical, ratio=4, visible=visible)

    torch.testing.assert_close(got, expected)


def test_external_ngram_artifact_streams_exact_rows(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    rows = [
        torch.arange(12, dtype=torch.bfloat16).view(3, 4),
        (100 + torch.arange(8, dtype=torch.bfloat16)).view(2, 4),
    ]
    weight_map = {}
    for index, tensor in enumerate(rows):
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{index}.weight"
        filename = f"model-{index}.safetensors"
        save_file({name: tensor}, source / filename)
        weight_map[name] = filename
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    args = SimpleNamespace(
        split_ngram_parts=2, ple_embed_dim=64, ngram_size=3, heads_per_ngram=8
    )
    artifacts = copy_external_artifacts(
        str(source), str(out), SimpleNamespace(qwen4_exp_args=args)
    )
    manifest = json.loads((out / "qwen4_ngram.json").read_text(encoding="utf-8"))
    assert artifacts[0]["nbytes"] == 40
    assert manifest["rows"] == 5 and manifest["dim"] == 4
    store = RawNGramStore(str(out), manifest, dim=4)
    try:
        reads = 0
        read_one = store._read_one

        def counted(row):
            nonlocal reads
            reads += 1
            return read_one(row)

        store._read_one = counted
        got = store.lookup(torch.tensor([[4, 0, 2]]))
        again = store.lookup(torch.tensor([[4, 0, 2]]))
    finally:
        store.close()
    expected = torch.stack((rows[1][1], rows[0][0], rows[0][2])).view(1, 3, 4)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(again, expected)
    assert reads == 3


def test_external_ngram_artifact_preserves_official_fp8_payload(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    rows = [
        torch.arange(12, dtype=torch.bfloat16).view(3, 4).to(torch.float8_e4m3fn),
        (16 + torch.arange(8, dtype=torch.bfloat16)).view(2, 4).to(torch.float8_e4m3fn),
    ]
    weight_map = {}
    source_payloads = []
    for index, tensor in enumerate(rows):
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{index}.weight"
        filename = f"model-{index}.safetensors"
        path = source / filename
        save_file({name: tensor}, path)
        weight_map[name] = filename
        data = path.read_bytes()
        header_size = struct.unpack("<Q", data[:8])[0]
        meta = json.loads(data[8:8 + header_size])[name]
        begin, end = meta["data_offsets"]
        source_payloads.append(data[8 + header_size + begin:8 + header_size + end])
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    args = SimpleNamespace(
        split_ngram_parts=2, ple_embed_dim=64, ngram_size=3, heads_per_ngram=8
    )

    copy_external_artifacts(str(source), str(out), SimpleNamespace(qwen4_exp_args=args))
    manifest = json.loads((out / "qwen4_ngram.json").read_text(encoding="utf-8"))
    assert manifest["dtype"] == "float8_e4m3fn"
    assert manifest["nbytes"] == 20
    assert (out / "qwen4_ngram.bin").read_bytes() == b"".join(source_payloads)
    store = RawNGramStore(str(out), manifest, dim=4)
    try:
        got = store.lookup(torch.tensor([[4, 0, 2]]))
    finally:
        store.close()
    expected = torch.stack((rows[1][1], rows[0][0], rows[0][2])).to(torch.bfloat16)
    torch.testing.assert_close(got, expected.view(1, 3, 4), rtol=0, atol=0)


def test_official_per_expert_fp8_tensors_do_not_enter_dense_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sparklab.models.qwen4_exp.weight.get_tp_info",
        lambda: SimpleNamespace(size=1),
    )
    expert = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    tensors = {
        "model.language_model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        expert + ".weight": torch.ones(4, 4, dtype=torch.float8_e4m3fn),
        expert + ".weight_scale_inv": torch.ones(1, 1, dtype=torch.bfloat16),
    }
    filename = "model.safetensors"
    save_file(tensors, tmp_path / filename)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: filename for name in tensors}}), encoding="utf-8"
    )
    got = list(iter_weights(
        str(tmp_path), torch.device("cpu"),
        include_moe_experts=False, include_non_moe=True,
    ))
    assert [name for name, _ in got] == ["model.norm.weight"]


def test_modelopt_nvfp4_expert_tensors_do_not_enter_dense_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sparklab.models.qwen4_exp.weight.get_tp_info",
        lambda: SimpleNamespace(size=1),
    )
    expert = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    tensors = {
        "model.language_model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        expert + ".weight": torch.ones(4, 2, dtype=torch.uint8),
        expert + ".weight_scale": torch.ones(4, 1, dtype=torch.float8_e4m3fn),
        expert + ".weight_scale_2": torch.ones(1, dtype=torch.float32),
        expert + ".input_scale": torch.ones(1, dtype=torch.float32),
    }
    filename = "model.safetensors"
    save_file(tensors, tmp_path / filename)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: filename for name in tensors}}), encoding="utf-8"
    )

    got = list(iter_weights(
        str(tmp_path), torch.device("cpu"),
        include_moe_experts=False, include_non_moe=True,
    ))

    assert [name for name, _ in got] == ["model.norm.weight"]


def test_inferact_nvfp4_source_spec_matches_served_experts_only():
    from sparklab.models.qwen4_exp.weight import _NVFP4_SOURCE_SPEC

    name = (
        "model.language_model.layers.47.mlp.experts.511.down_proj.weight_scale_2"
    )
    match = _NVFP4_SOURCE_SPEC.key_pattern.match(name)
    assert match is not None
    assert match.groupdict() == {
        "layer": "47",
        "expert": "511",
        "proj": "down_proj",
        "kind": "weight_scale_2",
    }
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(47, object()) == 47
    assert _NVFP4_SOURCE_SPEC.key_pattern.match(
        "mtp.layers.0.mlp.experts.0.down_proj.weight"
    ) is None


def test_expert_reader_pairs_layers_regardless_of_index_order(tmp_path):
    weight_map = {}
    expected = []
    # Deliberately publish layer 1 before layer 0 and down before gate-up.  The
    # streaming reader must still produce complete layer pairs in numeric order.
    for layer in (1, 0):
        for role in ("down_proj", "gate_up_proj"):
            raw = (
                f"model.language_model.layers.{layer}.mlp.experts.{role}"
            )
            filename = f"layer-{layer}-{role}.safetensors"
            tensor = torch.full((2, 3, 4), 10 * layer + (role == "down_proj"),
                                dtype=torch.bfloat16)
            save_file({raw: tensor}, tmp_path / filename)
            weight_map[raw] = filename
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    got = list(_iter_experts_layer_order(str(tmp_path), torch.device("cpu")))
    assert [name for name, _ in got] == [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
    ]
    assert [tensor.flatten()[0].item() for _, tensor in got] == [0, 1, 10, 11]


def test_ple_reconstructs_overlap_decode_token_from_current_batch():
    args = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=2,
        ple_embed_dim=16,
        ngram_vocab_size_base=101,
        ngram_vocab_divisor=8,
        seed=1234,
        eos_token_id=99,
    )
    embedding = DiskNGramEmbedding(args, vocab_size=128, layer_index=0)
    full = torch.tensor([5, 6, 7], dtype=torch.int32)
    expected = embedding.ids_for_request(full, 2, 3)
    # The host Req still ends at cached_len while Batch.input_ids already holds
    # the sampled token for the overlapping decode that is about to run.
    got = embedding.ids_for_request(full[:2], 2, 3, full[2:])
    torch.testing.assert_close(got, expected)


def test_ple_span_hash_matches_full_history_reference_across_eos_boundaries():
    args = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=2,
        ple_embed_dim=16,
        ngram_vocab_size_base=101,
        ngram_vocab_divisor=8,
        seed=1234,
        eos_token_id=99,
    )
    embedding = DiskNGramEmbedding(args, vocab_size=128, layer_index=0)
    ids = torch.tensor([5, 6, 99, 7, 8, 9, 10, 99, 11, 12], dtype=torch.int32)
    for start, end in ((0, 10), (1, 4), (3, 9), (9, 10)):
        shifted_full = [
            embedding._shift(ids, n).long() for n in range(args.ngram_size)
        ]
        blocks = []
        for ngram in range(2, args.ngram_size + 1):
            h0 = (ngram - 2) * args.heads_per_ngram
            h1 = h0 + args.heads_per_ngram
            mixed = shifted_full[0] * embedding._multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted_full[position] * embedding._multipliers[position]
                )
            sizes = embedding._head_vocab_sizes[h0:h1]
            offsets = embedding._head_offsets[h0:h1]
            blocks.append(torch.remainder(mixed[:, None], sizes) + offsets)
        expected = torch.cat(blocks, -1)[start:end]
        torch.testing.assert_close(
            embedding.ids_for_request(ids, start, end), expected
        )


def test_qsa_selects_complete_blocks_and_always_keeps_tail():
    backend = QSAAttnBackend.__new__(QSAAttnBackend)
    backend.args = SimpleNamespace(
        index_compress_ratio=4, index_block_topk=2, index_head_dim=4
    )
    backend.config = SimpleNamespace(rms_norm_eps=1e-6)

    raw = torch.arange(14 * 4, dtype=torch.float32).view(14, 4) / 10
    index_q = torch.tensor([[1.0, -0.5, 0.25, 0.1], [-0.2, 0.4, 0.8, -0.1]])
    physical = torch.arange(14, dtype=torch.int32) * 3 + 7

    pooled = raw[:12].view(3, 4, 4).mean(1)
    pooled = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-6)
    got = backend._selected_rows(index_q, pooled, physical, 14)
    score = torch.relu(index_q @ pooled.T).sum(0) / 2
    blocks = torch.topk(score, 2, sorted=False).indices
    logical = torch.cat((
        (blocks[:, None] * 4 + torch.arange(4)).flatten(),
        torch.tensor([12, 13]),
    )).sort().values
    torch.testing.assert_close(got, physical.index_select(0, logical))
    # The incomplete suffix bypasses top-k selection by definition.
    torch.testing.assert_close(got[-2:], physical[-2:])


def test_qsa_dense_budget_fast_path_does_not_need_index_keys():
    backend = QSAAttnBackend.__new__(QSAAttnBackend)
    backend.args = SimpleNamespace(index_compress_ratio=4, index_block_topk=512)
    physical = torch.arange(2051, dtype=torch.int32) * 2 + 1
    got = backend._selected_rows(torch.empty(4, 128), None, physical, 2051)
    torch.testing.assert_close(got, physical)


def test_qsa_materializes_each_completed_index_group_once(monkeypatch):
    physical = torch.tensor([9, 3, 12, 1, 8, 5, 14, 2], dtype=torch.int32)
    monkeypatch.setattr(
        "sparklab.attention.qsa.get_global_ctx",
        lambda: SimpleNamespace(page_table=physical.view(1, -1)),
    )
    backend = QSAAttnBackend.__new__(QSAAttnBackend)
    backend.args = SimpleNamespace(index_compress_ratio=4)
    backend.config = SimpleNamespace(rms_norm_eps=1e-6)
    backend.device = torch.device("cpu")
    raw = torch.arange(15 * 4, dtype=torch.float32).view(15, 4) / 10

    class FakeKVCache:
        @staticmethod
        def index_k_cache(_slot):
            return raw

    class IdentityRotary:
        @staticmethod
        def forward(positions, query, key):
            assert positions.tolist() == [0, 4]
            return query, key

    backend.kvcache = FakeKVCache()
    expected = raw.index_select(0, physical.long()).view(2, 4, 4).mean(1)
    expected = expected * torch.rsqrt(
        expected.square().mean(-1, keepdim=True) + 1e-6
    )
    untouched = raw.clone()
    backend._pool_completed_keys(
        0,
        [SimpleNamespace(cached_len=2, extend_len=6, device_len=8, table_idx=0)],
        torch.zeros(4),
        IdentityRotary(),
    )
    torch.testing.assert_close(raw[physical[[3, 7]].long()], expected)
    mask = torch.ones(raw.size(0), dtype=torch.bool)
    mask[physical[[3, 7]].long()] = False
    torch.testing.assert_close(raw[mask], untouched[mask])


def test_qsa_dense_metadata_is_built_once_and_supports_missing_index_query(monkeypatch):
    page_table = torch.tensor([[5, 6, 7, 8], [11, 12, 13, 14]], dtype=torch.int32)
    monkeypatch.setattr(
        "sparklab.attention.qsa.get_global_ctx",
        lambda: SimpleNamespace(page_table=page_table),
    )
    backend = QSAAttnBackend.__new__(QSAAttnBackend)
    backend.args = SimpleNamespace(index_compress_ratio=4, index_block_topk=512)
    backend.device = torch.device("cpu")
    backend._idx_slot = {3: 0}
    backend.config = SimpleNamespace(num_kv_heads=1, head_dim=2)
    backend.sm_scale = 0.5

    class FakeKVCache:
        def __init__(self):
            self.k = torch.zeros(16, 1, 2)
            self.v = torch.zeros_like(self.k)

        def store_kv(self, *args):
            pass

        def store_index_k(self, *args):
            pass

        def k_cache(self, _layer_id):
            return self.k

        def v_cache(self, _layer_id):
            return self.v

    backend.kvcache = FakeKVCache()
    reqs = [
        SimpleNamespace(extend_len=2, cached_len=1, device_len=3, table_idx=0),
        SimpleNamespace(extend_len=1, cached_len=0, device_len=1, table_idx=1),
    ]
    batch = SimpleNamespace(reqs=reqs, out_loc=torch.tensor([6, 7, 11]))
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert not backend.needs_index_query(batch)
    assert md.q_to_req.tolist() == [0, 1, 2]
    assert md.dense_indptr.tolist() == [0, 2, 5, 6]
    assert md.dense_indices.tolist() == [5, 6, 5, 6, 7, 11]
    assert md.dense_q_positions.tolist() == [1, 2, 0]

    captured = {}

    def fake_paged_attention(**kwargs):
        captured.update(kwargs)
        return torch.full((3, 1, 2), 9.0)

    monkeypatch.setattr(
        "sparklab.kernels.triton.attention.paged_attention", fake_paged_attention
    )
    result = backend.qsa_forward(
        q=torch.zeros(3, 1, 2),
        k=torch.zeros(3, 2),
        v=torch.zeros(3, 2),
        index_q=None,
        raw_index_k=torch.zeros(3, 2),
        k_norm_weight=torch.zeros(2),
        rotary=None,
        layer_id=3,
        batch=batch,
    )
    assert result.unique().item() == 9
    assert captured["indices"].data_ptr() == md.dense_indices.data_ptr()
    assert captured["q_to_req"].tolist() == [0, 1, 2]


def test_registry_has_qwen_wrapper_and_text_architectures():
    from sparklab.models.register import get_model_spec

    for architecture in ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"):
        assert get_model_spec(architecture).module == "sparklab.models.qwen4_exp"
