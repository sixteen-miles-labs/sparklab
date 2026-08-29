"""BSAKVCache: paged GQA K/V + index-key slab (MiniMax-M3 block-sparse pool)."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from sparklab.runtime.kvcache.bsa_pool import BSAKVCache

DEV = torch.device("cuda")


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from sparklab.runtime.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "sparklab.runtime.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _pool(num_pages=8, page_size=128):
    return BSAKVCache(
        num_kv_heads=4,
        num_layers=5,
        head_dim=128,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=128,
        num_index_layers=2,
    )


def test_store_and_views():
    pool = _pool()
    assert pool.k_cache(0).shape == (8, 128, 4, 128)
    assert pool.index_k_cache(0).shape == (8 * 128, 128)
    assert pool.index_k_cache(1).shape == (8 * 128, 128)

    t = 6
    k = torch.randn(t, 4 * 128, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(t, 4 * 128, device=DEV, dtype=torch.bfloat16)
    ik = torch.randn(t, 128, device=DEV, dtype=torch.bfloat16)
    out_loc = torch.tensor([128, 129, 130, 200, 201, 202], device=DEV, dtype=torch.int64)
    pool.store_kv(k, v, out_loc.to(torch.int32), layer_id=3)
    pool.store_index_k(ik, out_loc, slot=1)

    k_rows = pool.k_cache(3).view(-1, 4, 128)
    assert torch.equal(k_rows[128], k[0].view(4, 128))
    assert torch.equal(k_rows[202], k[5].view(4, 128))
    assert torch.equal(pool.index_k_cache(1)[200], ik[3])
    # the other slot stays untouched (zero-initialized)
    assert pool.index_k_cache(0)[200].abs().sum().item() == 0.0


def test_rebuild_resizes_both_slabs_atomically():
    pool = _pool(num_pages=8)
    pool.rebuild(16)
    assert pool.k_cache(0).shape == (16, 128, 4, 128)
    assert pool.index_k_cache(0).shape == (16 * 128, 128)
    assert pool.index_k_cache(1).shape == (16 * 128, 128)


def test_unit_bytes_matches_cost_model():
    pool = _pool()
    kv_bytes, swa_bytes = pool.unit_bytes()
    # 5 layers x 2 slabs x 4 heads x 128 x 2 B + 2 index layers x 128 x 2 B
    assert kv_bytes == 5 * 2 * 4 * 128 * 2 + 2 * 128 * 2
    assert swa_bytes == 0


def test_factory_builds_bsa_pool():
    from sparklab.runtime.kvcache import create_kvcache_pool
    from sparklab.models.config import FullAttentionGroupConfig, ModelConfig, RotaryConfig

    rotary = RotaryConfig(head_dim=128, rotary_dim=64, max_position=4096, base=1e4, scaling=None)
    cfg = ModelConfig(
        num_layers=5,
        num_qo_heads=64,
        num_kv_heads=4,
        head_dim=128,
        hidden_size=6144,
        vocab_size=1000,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        rotary_config=rotary,
        hidden_act="swigluoai",
        tie_word_embeddings=False,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=512,
        norm_topk_prob=True,
        model_type="minimax_m3_vl",
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=tuple(range(5)),
                num_kv_heads=4,
                head_dim=128,
                rotary_config=rotary,
                mla=False,
                index_head_dim=128,
                num_index_layers=2,
            ),
        ),
    )
    pool = create_kvcache_pool(cfg, num_pages=4, page_size=128, dtype=torch.bfloat16, device=DEV)
    assert isinstance(pool, BSAKVCache)
    assert pool.index_k_cache(0).shape == (4 * 128, 128)
