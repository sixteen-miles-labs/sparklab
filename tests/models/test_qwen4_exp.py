from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from freetoken.attention import AttnType
from freetoken.attention.qsa import QSAAttnBackend
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.hyper import GroupedPlusOneRMSNorm, Qwen4GatedResidual
from freetoken.models.qwen4_exp.ple import DiskNGramEmbedding, RawNGramStore
from freetoken.models.qwen4_exp.weight import copy_external_artifacts
from freetoken.models.qwen4_exp.weight import _iter_experts_layer_order
from freetoken.models.qwen4_exp.weight import iter_weights


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


def test_config_accepts_conversion_owned_nvfp4_experts():
    source = _config()
    source.text_config.freetoken_expert_quant = "nvfp4"
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
        "freetoken.models.qwen4_exp.weight.get_tp_info",
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


def test_qsa_selects_complete_blocks_and_always_keeps_tail():
    backend = QSAAttnBackend.__new__(QSAAttnBackend)
    backend.args = SimpleNamespace(
        index_compress_ratio=4, index_block_topk=2, index_head_dim=4
    )
    backend.config = SimpleNamespace(rms_norm_eps=1e-6)

    class IdentityRotary:
        @staticmethod
        def forward(_positions, query, key):
            return query, key

    raw = torch.arange(14 * 4, dtype=torch.float32).view(14, 4) / 10
    index_q = torch.tensor([[1.0, -0.5, 0.25, 0.1], [-0.2, 0.4, 0.8, -0.1]])
    physical = torch.arange(14, dtype=torch.int32) * 3 + 7
    got = backend._selected_rows(
        index_q, raw, physical, 14, torch.zeros(4), IdentityRotary()
    )

    pooled = raw[:12].view(3, 4, 4).mean(1)
    pooled = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-6)
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
    got = backend._selected_rows(
        torch.empty(4, 128), None, physical, 2051, torch.empty(128), None
    )
    torch.testing.assert_close(got, physical)


def test_registry_has_qwen_wrapper_and_text_architectures():
    from freetoken.models.register import get_model_spec

    for architecture in ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"):
        assert get_model_spec(architecture).module == "freetoken.models.qwen4_exp"
