from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sparklab.models.glm_moe_dsa.config import parse_config
from sparklab.runtime.engine.engine import _adjust_speculative_config


def tiny_config():
    return parse_config(SimpleNamespace(
        num_hidden_layers=4, hidden_size=64, vocab_size=256,
        num_attention_heads=4, q_lora_rank=16, kv_lora_rank=16,
        qk_nope_head_dim=32, qk_rope_head_dim=64, v_head_dim=32,
        rms_norm_eps=1e-6, max_position_embeddings=256,
        intermediate_size=64, moe_intermediate_size=32,
        hidden_act="silu", num_experts_per_tok=2, n_routed_experts=4,
        first_k_dense_replace=1, n_shared_experts=1,
        index_n_heads=16, index_head_dim=64, index_topk=32,
        indexer_types=["full", "shared", "full", "shared", "full"],
        num_nextn_predict_layers=1,
    ))


def enable_mtp(model, steps=3):
    config = SimpleNamespace(model_config=model, speculative_tokens=steps,
                             speculative_method="auto", draft_sample_method="greedy")
    _adjust_speculative_config(config, lambda name, value: setattr(config, name, value))
    return config


def test_full_glm_mtp_config_adds_only_draft_kv_and_is_idempotent():
    model = tiny_config()
    original_experts = model.num_moe_layers
    config = enable_mtp(model)
    _adjust_speculative_config(config, lambda name, value: setattr(config, name, value))
    assert model.num_layers == 4 and model.num_moe_layers == original_experts
    assert model.glm_dsa_args.indexer_types == ("full", "shared", "full", "shared", "full")
    assert model.attention_groups[0].layer_ids == (0, 1, 2, 3, 4)
    assert model.attention_groups[0].num_index_layers == 3
    assert config.cache_type == "radix" and config.max_running_req == 1
    assert config.cuda_graph_max_bs == 0 and config.cuda_graph_bs == []


def test_full_glm_draft_count_is_not_misclassified_as_qwen():
    model = tiny_config()
    object.__setattr__(model, "mtp_num_hidden_layers", 2)
    with pytest.raises(ValueError, match="no supported draft"):
        enable_mtp(model)


def test_mtp_shard_loader_fuses_experts_and_keeps_target_state_separate(tmp_path):
    from sparklab.models.glm_moe_dsa.mtp import GlmDsaMultiTokenPredictor
    from sparklab.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM
    from sparklab.runtime.distributed import set_tp_info, try_get_tp_info
    from sparklab.utils import torch_dtype
    from sparklab.layers import set_rope_device

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    model = tiny_config()
    object.__setattr__(model, "moe_backend", "offload")
    enable_mtp(model)
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        predictor = GlmDsaMultiTokenPredictor(model)
        target = GlmMoeDsaForCausalLM(model)
    assert not any("mtp" in name for name in target.state_dict())
    assert not hasattr(predictor.layer.mlp.experts, "offload_cache")
    expected = {name: torch.full(tuple(t.shape), 0.125 * (i + 1), dtype=t.dtype)
                for i, (name, t) in enumerate(predictor.state_dict().items())}
    raw = {}
    prefix = "model.layers.4."
    for name, value in expected.items():
        if name == "layer.mlp.experts.gate_up_proj":
            width = value.shape[1] // 2
            for expert in range(value.shape[0]):
                raw[f"{prefix}mlp.experts.{expert}.gate_proj.weight"] = value[expert, :width].clone()
                raw[f"{prefix}mlp.experts.{expert}.up_proj.weight"] = value[expert, width:].clone()
        elif name == "layer.mlp.experts.down_proj":
            for expert in range(value.shape[0]):
                raw[f"{prefix}mlp.experts.{expert}.down_proj.weight"] = value[expert].clone()
        else:
            suffix = name.removeprefix("layer.")
            suffix = suffix.replace("mlp.e_score_correction_bias", "mlp.gate.e_score_correction_bias")
            if name == "norm.weight":
                suffix = "shared_head.norm.weight"
            raw[prefix + suffix] = value.clone()
    save_file(raw, tmp_path / "mtp.safetensors")
    index = {"weight_map": {name: "mtp.safetensors" for name in raw}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    predictor.load_shards(str(tmp_path), torch.device("cpu"))
    for name, actual in predictor.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)


def test_mtp_reject_prefix_hides_suffix_and_uses_correction(monkeypatch):
    from contextlib import contextmanager
    from sparklab.core import Batch, Req
    from sparklab.models.glm_moe_dsa import mtp
    from sparklab.core import SamplingParams

    calls = []
    req = Req(input_ids=torch.tensor([1, 2, 3, 4, 5, 6]), table_idx=0,
              cached_len=2, output_len=10, uid=1, cache_handle=None,
              sampling_params=SamplingParams(temperature=0))
    batch = Batch(reqs=[req], phase="verify")
    batch.verify_cached_lens = (2,)
    batch.input_ids = torch.tensor([3, 4, 5, 6])
    batch.positions = torch.arange(2, 6).int()
    ctx = SimpleNamespace(page_table=torch.arange(32).reshape(1, -1),
                          attn_backend=SimpleNamespace(prepare_metadata=lambda b: None))

    @contextmanager
    def forward(current):
        ctx.batch = current
        yield

    ctx.forward_batch = forward
    monkeypatch.setattr(mtp, "get_global_ctx", lambda: ctx)

    def draft(embeddings, hidden):
        calls.append((ctx.batch.positions.clone(), embeddings.clone(), hidden.clone(),
                      ctx.batch.reqs[0].device_len))
        return hidden

    hidden = torch.arange(16).reshape(4, 4).float()
    model = SimpleNamespace(_mtp=SimpleNamespace(forward=draft), _mtp_steps=1,
                            _mtp_target_hidden=hidden,
                            model=SimpleNamespace(embed_tokens=SimpleNamespace(forward=lambda x:x)),
                            project_draft_logits=lambda h: h)
    result = mtp.propose_nextn(model, batch, torch.tensor([9]), prefix_tokens=2)
    assert result.tolist() == [3]
    assert calls[0][0].tolist() == [2, 3]
    assert calls[0][1].tolist() == [4, 9]
    torch.testing.assert_close(calls[0][2], hidden[:2])
    assert calls[0][3] == 4
    assert batch.input_ids.tolist() == [3, 4, 5, 6]
    assert req.device_len == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("prefix_length", [3, 40])
def test_native_mtp_gpu_prefix_recovery_and_recursive_decode(monkeypatch, prefix_length):
    import sparklab.core as core
    from sparklab.core import Batch, Context, Req, SamplingParams
    from sparklab.attention.dsa import DSAAttnBackend
    from sparklab.layers import set_rope_device
    from sparklab.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM
    from sparklab.runtime.distributed import set_tp_info, try_get_tp_info
    from sparklab.runtime.kvcache import create_kvcache_pool
    from sparklab.utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = tiny_config()
    for name in ("attn_quant", "dense_quant", "expert_quant", "lm_head_quant"):
        object.__setattr__(config, name, "none")
    enable_mtp(config, steps=3)
    device = torch.device("cuda")
    from sparklab.layers.rotary import get_rope
    get_rope.cache_clear()
    set_rope_device(device)
    ctx = Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    ctx.kv_cache = create_kvcache_pool(config, 65, 1, torch.bfloat16, device)
    ctx.page_table = torch.arange(65, device=device, dtype=torch.int32).reshape(1, -1)
    ctx.attn_backend = DSAAttnBackend(config)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = GlmMoeDsaForCausalLM(config)
    torch.manual_seed(7)
    for module in (model, model._mtp):
        module.load_state_dict({name: (
            torch.ones(tuple(t.shape), device=device, dtype=t.dtype)
            if "norm.weight" in name else
            torch.randn(tuple(t.shape), device=device, dtype=t.dtype) * 0.01
        ) for name, t in module.state_dict().items()})
    model.prepare_for_runtime()

    def forward(tokens, cached=0, phase="prefill"):
        req = Req(torch.tensor(tokens), 0, cached, 16, 1, SamplingParams(temperature=0), None)
        batch = Batch(reqs=[req], phase=phase)
        batch.padded_reqs = batch.reqs
        batch.input_ids = torch.tensor(tokens[cached:], device=device, dtype=torch.int32)
        batch.positions = torch.arange(cached, len(tokens), device=device, dtype=torch.int32)
        batch.out_loc = ctx.page_table[0, cached:len(tokens)]
        batch.active_table_idx = torch.tensor([0], device=device)
        if phase == "verify":
            batch.verify_cached_lens = (cached,)
        ctx.attn_backend.prepare_metadata(batch)
        with ctx.forward_batch(batch):
            logits = model.forward()
        assert torch.isfinite(logits).all()
        return batch

    prefix = list(range(1, prefix_length + 1))
    prompt = forward(prefix)
    first = model.propose_mtp(prompt, torch.tensor([4], device=device))
    assert first.shape == (3,)
    verified = forward(prefix + [4, 5, 6, 7], cached=prefix_length, phase="verify")
    proposals = model.propose_mtp_prefix(verified, torch.tensor([9], device=device), 2)
    recovered = ctx.kv_cache.latent_rows(4)[:prefix_length + 2].clone()
    # A fresh accepted-only target query must regenerate the same draft prefix,
    # regardless of rejected suffix rows still present in the physical cache.
    clean = forward(prefix + [4, 5], cached=prefix_length)
    expected = model.propose_mtp(clean, torch.tensor([9], device=device))
    torch.testing.assert_close(ctx.kv_cache.latent_rows(4)[:prefix_length + 2], recovered, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(proposals, expected, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["mtp", "dflash2"])
@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_engine_append_only_verification_never_replays_target(monkeypatch, method, accepted):
    from sparklab.core import Batch, Context, Req, SamplingParams
    from sparklab.runtime.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: engine.stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda _: None))
    engine.config = SimpleNamespace(speculative_tokens=3, speculative_method=method,
                                    model_config=SimpleNamespace(glm_dsa_args=object()))
    engine.linear_state_pool = None
    engine.mtp_graph_runner = None
    engine.cpu_moe_executor = None
    engine.ctx = Context(page_size=1)
    engine.mtp_stats = {key: 0 for key in ("target_forwards", "drafted", "accepted", "outputs",
                                          "replay_calls", "replay_tokens", "fast_carry_commits")}
    req = Req(torch.tensor([1, 2, 3, 4, 5, 6]), 0, 2, 10, 1,
              SamplingParams(temperature=0), None)
    batch = Batch(reqs=[req], phase="verify")
    batch.input_ids = torch.tensor([3, 4, 5, 6])
    batch.verify_cached_lens = (2,)
    logits = torch.full((4, 10), -100.)
    expected = [4, 5, 6, 7]
    if accepted < 3:
        expected[accepted] = 9
    logits[torch.arange(4), torch.tensor(expected)] = 100
    calls = []

    def propose(active, token):
        calls.append(("draft", active.reqs[0].device_len, token.item()))
        return torch.tensor([8])

    def prefix(active, token, length):
        calls.append(("prefix", length, token.item()))
        return torch.tensor([8])

    engine.model = SimpleNamespace(begin_external_inputs=lambda _: None, forward=lambda: logits,
                                    propose_mtp=propose, propose_mtp_prefix=prefix)
    engine._replay_verified_prefix = lambda *_: pytest.fail("append-only target was replayed")
    out = engine.forward_batch(batch, None)
    assert out.token_counts == (accepted + 1,)
    assert req.cached_len == 3 + accepted
    assert req.device_len == 4 + accepted
    assert engine.mtp_stats["accepted"] == accepted
    assert engine.mtp_stats["replay_calls"] == 0
    if accepted < 3:
        assert calls == ([("prefix", accepted + 1, 9)] if method == "mtp" else [("draft", accepted + 3, 9)])
    else:
        assert calls == [("draft", 6, 7)]
