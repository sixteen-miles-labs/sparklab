from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests.models.test_glm_dsa_mtp import tiny_config


def draft_config():
    return SimpleNamespace(
        architectures=["DFlash2DraftModel"], hidden_size=64, intermediate_size=128,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        num_hidden_layers=2, num_target_layers=4, vocab_size=256,
        max_position_embeddings=256, rms_norm_eps=1e-5, sliding_window=64,
        dflash_config=dict(block_size=8, conv_kernel_size=2, conv_group_size=16,
                          selector_rank=8, selector_top_k=4,
                          target_layer_ids=[1, 3], mask_token_id=255),
    )


def configure(monkeypatch, draft_path="draft"):
    import sparklab.runtime.engine.engine as engine

    monkeypatch.setattr(engine, "cached_load_hf_config", lambda _: draft_config())
    cfg = SimpleNamespace(model_config=tiny_config(), speculative_tokens=8,
                          speculative_method="dflash2", speculative_draft_model=str(draft_path),
                          draft_sample_method="greedy")
    engine._adjust_speculative_config(cfg, lambda k, v: setattr(cfg, k, v))
    return cfg


def test_glm_dflash_pool_isolation_budget_and_rebuild(monkeypatch):
    from sparklab.attention import AttnType
    from sparklab.runtime.engine.engine import _required_attn_types
    from sparklab.runtime.kvcache import create_kvcache_pool
    from sparklab.runtime.kvcache.dsa_pool import DSAKVCache
    from sparklab.runtime.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cfg = configure(monkeypatch)
    cfg.page_size = 1
    cfg.dtype = torch.bfloat16
    cfg.tp_info = SimpleNamespace(size=1)
    model = cfg.model_config
    assert model.num_layers == 4 and model.num_moe_layers == 3
    assert _required_attn_types(model) == frozenset({AttnType.DSA})
    pool = create_kvcache_pool(model, 9, 1, torch.bfloat16, torch.device("cpu"))
    assert isinstance(pool, DSAKVCache)
    assert pool.latent_rows(0).shape == (9, 80)
    assert pool.k_cache(4).shape[-2:] == (1, 64)
    assert pool.k_cache(4).data_ptr() != pool.v_cache(4).data_ptr()
    assert pool.k_cache(0).data_ptr() == pool.v_cache(0).data_ptr()
    assert pool.unit_bytes()[0] == pool.kv_cost(cfg)[0]
    pool.rebuild(17)
    assert pool.latent_rows(0).shape == (17, 80)
    assert pool.index_k_cache(0).shape[0] == 17
    assert pool.k_cache(4).shape[0] == 17
    assert pool.unit_bytes()[0] == pool.kv_cost(cfg)[0]


def test_glm_dflash_bf16_loader_exact_and_private(monkeypatch, tmp_path):
    from sparklab.layers import set_rope_device
    from sparklab.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM
    from sparklab.runtime.distributed import set_tp_info, try_get_tp_info
    from sparklab.utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    cfg = configure(monkeypatch, tmp_path)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = GlmMoeDsaForCausalLM(cfg.model_config)
    assert model._mtp is None
    assert not any("dflash" in name for name in model.state_dict())
    expected = {name: torch.full(tuple(t.shape), i * 0.125, dtype=t.dtype)
                for i, (name, t) in enumerate(model._dflash.state_dict().items())}
    raw = {}
    for name, value in expected.items():
        if name.endswith("self_attn.qkv_proj.weight"):
            suffix = "self_attn.qkv_proj.weight"
            for part, tensor in zip(("q", "k", "v"), value.split((128, 64, 64), dim=0)):
                raw[name[:-len(suffix)] + f"self_attn.{part}_proj.weight"] = tensor.clone()
        elif name.endswith("mlp.gate_up_proj.weight"):
            for part, tensor in zip(("gate", "up"), value.chunk(2, dim=0)):
                raw[name.replace("gate_up", part)] = tensor.clone()
        else:
            raw[name] = value
    save_file(raw, tmp_path / "model.safetensors")
    model.load_speculative_weights("unused-target", torch.device("cpu"))
    for name, actual in model._dflash.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    # Strict schema: unrecognized weights cannot be silently ignored.
    raw["unknown.weight"] = torch.zeros(1)
    save_file(raw, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="Unexpected DFlash2"):
        model.load_speculative_weights("unused-target", torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_glm_dflash_gpu_proposes_and_masks_rejected_suffix(monkeypatch):
    import sparklab.core as core
    from sparklab.core import Batch, Context, Req, SamplingParams
    from sparklab.layers import set_rope_device
    from sparklab.layers.rotary import get_rope
    from sparklab.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM
    from sparklab.runtime.distributed import set_tp_info, try_get_tp_info
    from sparklab.runtime.kvcache import create_kvcache_pool
    from sparklab.utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = configure(monkeypatch).model_config
    for name in ("attn_quant", "dense_quant", "expert_quant", "lm_head_quant"):
        object.__setattr__(config, name, "none")
    device = torch.device("cuda")
    get_rope.cache_clear()
    set_rope_device(device)
    ctx = Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    ctx.kv_cache = create_kvcache_pool(config, 65, 1, torch.bfloat16, device)
    ctx.page_table = torch.arange(65, device=device, dtype=torch.int32).reshape(1, -1)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = GlmMoeDsaForCausalLM(config)
    torch.manual_seed(19)
    for module in (model, model._dflash):
        module.load_state_dict({name: (
            torch.ones(tuple(t.shape), device=device, dtype=t.dtype)
            if "norm.weight" in name else
            torch.randn(tuple(t.shape), device=device, dtype=t.dtype) * 0.01
        ) for name, t in module.state_dict().items()})
    positions = torch.arange(12, device=device, dtype=torch.int32)
    captures = [torch.randn(12, 64, device=device, dtype=torch.bfloat16) for _ in range(2)]
    model._dflash.materialize_target_hidden(captures, positions, positions)
    req = Req(torch.arange(8), 0, 7, 16, 1, SamplingParams(temperature=0), None)
    batch = Batch(reqs=[req], phase="decode")
    proposal = model.propose_mtp(batch, torch.tensor([13], device=device))
    assert proposal.shape == (7,)
    assert proposal.min() >= 0 and proposal.max() < 256
    # Suffix target features exist in physical rows but are not part of the
    # accepted prefix. Poisoning them must not change the next proposed block.
    for lid in (4, 5):
        ctx.kv_cache.k_cache(lid)[8:].fill_(float("nan"))
        ctx.kv_cache.v_cache(lid)[8:].fill_(float("nan"))
    again = model.propose_mtp(batch, torch.tensor([13], device=device))
    torch.testing.assert_close(again, proposal, rtol=0, atol=0)
