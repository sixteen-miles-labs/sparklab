from __future__ import annotations

from types import SimpleNamespace

import pytest
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("steps,accepted", [(s, a) for s in (3, 4) for a in range(1, s + 1)])
@pytest.mark.parametrize("draft_graph", [False, True])
def test_mtp_prefix_recovery_matches_clean_draft_gpu(monkeypatch, steps, accepted, draft_graph):
    pytest.importorskip("flashinfer")
    import sparklab.core as core
    from sparklab.core import Batch, Context, Req, SamplingParams
    from sparklab.attention.fi import FlashInferBackend
    from sparklab.layers import set_rope_device
    from sparklab.layers.rotary import get_rope
    from sparklab.models.qwen3_5_moe.model import Qwen3_5MoEForCausalLM
    from sparklab.models.qwen3_5_moe.mtp import Qwen3_5MultiTokenPredictor
    from sparklab.runtime.engine.engine import _adjust_speculative_config
    from sparklab.runtime.distributed import set_tp_info, try_get_tp_info
    from sparklab.runtime.kvcache import create_kvcache_pool
    from sparklab.utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    hf = _config()
    hf.text_config.head_dim = 64
    config = parse_config(hf)
    monkeypatch.setenv("SPARKLAB_QWEN3_MTP4", "1")
    options = SimpleNamespace(model_config=config, speculative_tokens=steps,
                              cuda_graph_bs=[], cuda_graph_max_bs=0)
    _adjust_speculative_config(options, lambda n, v: setattr(options, n, v))
    device = torch.device("cuda")
    get_rope.cache_clear()
    set_rope_device(device)
    ctx = Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    ctx.kv_cache = create_kvcache_pool(config, 128, 1, torch.bfloat16, device)
    ctx.page_table = torch.arange(128, device=device, dtype=torch.int32)[None]
    ctx.attn_backend = FlashInferBackend(config)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        mtp = Qwen3_5MultiTokenPredictor(config)
    torch.manual_seed(31)
    mtp.load_state_dict({name: (
        torch.ones(tuple(t.shape), device=device, dtype=t.dtype)
        if "norm.weight" in name else
        torch.randn(tuple(t.shape), device=device, dtype=t.dtype) * 0.03
    ) for name, t in mtp.state_dict().items()})
    model = Qwen3_5MoEForCausalLM.__new__(Qwen3_5MoEForCausalLM)
    model._mtp, model._mtp_steps, model._dflash = mtp, steps, None
    embedding = torch.randn(256, 64, device=device, dtype=torch.bfloat16)
    head = torch.randn_like(embedding)
    model.model = SimpleNamespace(embed_tokens=SimpleNamespace(
        forward=lambda ids: torch.nn.functional.embedding(ids.long(), embedding)))
    model.lm_head = SimpleNamespace(forward=lambda h: torch.nn.functional.linear(h, head))
    lid = config.num_layers
    ctx.kv_cache.k_cache(lid).normal_()
    ctx.kv_cache.v_cache(lid).normal_()
    initial_k = ctx.kv_cache.k_cache(lid).clone()
    initial_v = ctx.kv_cache.v_cache(lid).clone()
    hidden = torch.randn(steps + 1, 64, device=device, dtype=torch.bfloat16)

    def batch_for(rows):
        req = Req(torch.arange(7 + rows), 0, 7, 16, 1, SamplingParams(), None)
        batch = Batch(reqs=[req], phase="verify")
        batch.padded_reqs = batch.reqs
        batch.input_ids = torch.arange(7, 7 + rows, dtype=torch.int32, device=device)
        batch.positions = batch.input_ids.clone()
        batch.out_loc = ctx.page_table[0, 7:7 + rows]
        ctx.attn_backend.prepare_metadata(batch)
        return batch

    model._mtp_target_hidden = hidden
    correction = torch.tensor([19], device=device, dtype=torch.int32)
    monkeypatch.setenv("SPARKLAB_QWEN3_MTP_DRAFT_GRAPH", "1" if draft_graph else "0")
    recovered = model.propose_mtp_prefix(batch_for(steps + 1), correction, accepted)
    recovered_k = ctx.kv_cache.k_cache(lid).clone()
    recovered_v = ctx.kv_cache.v_cache(lid).clone()
    ctx.kv_cache.k_cache(lid).copy_(initial_k)
    ctx.kv_cache.v_cache(lid).copy_(initial_v)
    model._mtp_target_hidden = hidden[:accepted]
    monkeypatch.setenv("SPARKLAB_QWEN3_MTP_DRAFT_GRAPH", "0")
    expected = model.propose_mtp(batch_for(accepted), correction)
    torch.testing.assert_close(recovered, expected, atol=0, rtol=0)
    torch.testing.assert_close(recovered_k, ctx.kv_cache.k_cache(lid), atol=0, rtol=0)
    torch.testing.assert_close(recovered_v, ctx.kv_cache.v_cache(lid), atol=0, rtol=0)
    model.destroy_mtp_draft_graphs()
    assert model._mtp_draft_graphs is None
    assert model._mtp_draft_graph_backend is None
    get_rope.cache_clear()
