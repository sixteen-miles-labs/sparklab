from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import torch

from sparklab.models import create_model
from sparklab.models.deepseek_v4.args import DeepseekV4Args
from sparklab.models.deepseek_v4.config import parse_config
from sparklab.models.deepseek_v4 import weight as dsv4_weight
from sparklab.models.deepseek_v4.compress import Compressor
from sparklab.models.deepseek_v4.dspark import draft_query_geometry
from sparklab.core import Context, SamplingParams
from sparklab.runtime.distributed import set_tp_info, try_get_tp_info
from sparklab.runtime.engine.engine import (
    Engine,
    _adjust_speculative_config,
    _verify_speculative_tokens,
)


def _fused_config(tmp_path):
    ratios = [0, 0, 4, 128, 4, 128, 4] + [0] * 39
    raw = asdict(
        DeepseekV4Args(
            n_layers=43,
            n_mtp_layers=3,
            compress_ratios=tuple(ratios),
            dspark_block_size=5,
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=(40, 41, 42),
            dspark_markov_rank=256,
        )
    )
    folder = tmp_path / "inference"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(raw))
    return parse_config(
        SimpleNamespace(_name_or_path=str(tmp_path), max_position_embeddings=1_048_576)
    )


def test_dspark_config_injects_three_draft_layers(tmp_path):
    model_config = _fused_config(tmp_path)
    config = SimpleNamespace(
        model_config=model_config,
        speculative_method="auto",
        speculative_tokens=7,
        draft_sample_method="probabilistic",
        max_running_req=8,
        cache_type="naive",
        cuda_graph_bs=[1, 2, 4],
        cuda_graph_max_bs=4,
    )

    def override(name, value):
        setattr(config, name, value)

    _adjust_speculative_config(config, override)
    _adjust_speculative_config(config, override)

    assert config.speculative_method == "dspark"
    assert model_config.speculative_method == "dspark"
    assert model_config.dsv4_args.dspark_enabled is True
    assert model_config.dsv4_args.runtime_n_layers == 46
    assert model_config.num_moe_layers == 46
    assert model_config.attention_groups[0].layer_ids[-3:] == (43, 44, 45)
    assert config.max_running_req == 1
    assert config.cache_type == "radix"
    assert config.cuda_graph_bs == [] and config.cuda_graph_max_bs == 0


def test_target_only_fused_config_keeps_43_layers(tmp_path):
    model_config = _fused_config(tmp_path)
    assert model_config.speculative_method == "none"
    assert model_config.dsv4_args.dspark_enabled is False
    assert model_config.dsv4_args.runtime_n_layers == 43
    assert model_config.num_moe_layers == 43


def test_probabilistic_verifier_accepts_and_emits_bonus():
    logits = torch.tensor(
        [[float("-inf"), 0.0, float("-inf")], [float("-inf"), float("-inf"), 0.0]]
    )
    drafts = torch.tensor([1])
    draft_probs = torch.tensor([[0.0, 1.0, 0.0]])
    accepted, chosen = _verify_speculative_tokens(
        logits, drafts, draft_probs, SamplingParams(temperature=1.0)
    )
    assert accepted == 1
    assert chosen.tolist() == [1, 2]


def test_probabilistic_verifier_rejects_from_positive_residual():
    logits = torch.tensor([[float("-inf"), 0.0, float("-inf")], [0.0, 0.0, 0.0]])
    drafts = torch.tensor([0])
    draft_probs = torch.tensor([[1.0, 0.0, 0.0]])
    accepted, chosen = _verify_speculative_tokens(
        logits, drafts, draft_probs, SamplingParams(temperature=1.0)
    )
    assert accepted == 0
    assert chosen.tolist() == [1]


def test_speculative_weight_names_match_draft_module(tmp_path, monkeypatch):
    model_config = _fused_config(tmp_path)
    config = SimpleNamespace(
        model_config=model_config,
        speculative_method="dspark",
        speculative_tokens=7,
        draft_sample_method="probabilistic",
        max_running_req=1,
        cache_type="radix",
        cuda_graph_bs=[],
        cuda_graph_max_bs=0,
    )
    _adjust_speculative_config(config, lambda name, value: setattr(config, name, value))
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    with torch.device("meta"):
        model = create_model(model_config)
    expected = set(model.speculative_state_dict())

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            pass

        def has(self, _name):
            return True

        def get(self, _name):
            return torch.zeros(1)

        def close(self):
            pass

    monkeypatch.setattr(dsv4_weight, "_weight_map", lambda _path: {})
    monkeypatch.setattr(dsv4_weight, "_ShardReader", FakeReader)
    monkeypatch.setattr(
        dsv4_weight, "_dequant_fp8_block", lambda _weight, _scale: torch.zeros(1)
    )
    names = {
        name
        for name, _ in dsv4_weight.iter_speculative_weights(
            str(tmp_path), torch.device("cpu")
        )
    }
    assert names == expected


def test_draft_proposal_runs_with_active_batch_context():
    engine = Engine.__new__(Engine)
    engine.ctx = Context(page_size=1)
    batch = object()

    class Model:
        def propose_mtp(self, received_batch, next_tokens):
            assert received_batch is batch
            assert engine.ctx.batch is batch
            return next_tokens + 1

    engine.model = Model()
    result = engine._propose_speculative(batch, torch.tensor([4]))

    assert result.tolist() == [5]
    assert engine.ctx._batch is None


def test_dspark_query_block_starts_at_sampled_anchor_and_stays_in_page():
    assert draft_query_geometry(device_len=127, page_size=128, max_steps=7) == (126, 2)
    assert draft_query_geometry(device_len=128, page_size=128, max_steps=7) == (127, 1)
    assert draft_query_geometry(device_len=129, page_size=128, max_steps=7) == (128, 7)


def test_speculative_verification_uses_unaligned_compressor_continuation(monkeypatch):
    compressor = Compressor(DeepseekV4Args(dim=8), compress_ratio=4, head_dim=2)
    compressor.P = 128
    seeded = []
    monkeypatch.setattr(
        Compressor, "cmp_pool", property(lambda _self: torch.empty(1))
    )
    monkeypatch.setattr(
        compressor, "_seed_carry_from_ring", lambda slot: seeded.append(slot)
    )
    sentinel = object()
    received = []

    def extend_unaligned(x, start_pos, window_slots, ti):
        received.append((x.shape, start_pos, window_slots.tolist(), ti))
        return sentinel

    monkeypatch.setattr(compressor, "_extend_unaligned", extend_unaligned)
    result = compressor.extend(
        torch.zeros(1, 2, 8, dtype=torch.bfloat16),
        start_pos=48,
        window_slots=torch.tensor([304, 305]),
        tail_window_slot=303,
        ti=7,
    )

    assert result is sentinel
    assert seeded == [303]
    assert received == [((1, 2, 8), 48, [304, 305], 7)]
