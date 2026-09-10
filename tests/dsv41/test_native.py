from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from sparklab.models.deepseek_v41.args import ModelArgs
from sparklab.models.deepseek_v41.config import parse_config
from sparklab.models.deepseek_v41.model import Decoder, DeepseekV41ForCausalLM
from sparklab.models.deepseek_v41.ops import dequant, fp4_roundtrip, fp8_roundtrip
from sparklab.models.deepseek_v41.weight import DTYPES, DiskWeights
from sparklab.models.register import get_model_spec
from sparklab.utils.hf import RawConfigShim

FIXTURE = Path(__file__).parent / "fixtures/tiny"


@pytest.fixture(scope="session", autouse=True)
def materialize_tiny_checkpoint(tmp_path_factory):
    import shutil
    from safetensors.torch import save_file

    global FIXTURE
    original = FIXTURE
    destination = tmp_path_factory.mktemp("dsv41-checkpoint")
    shutil.copytree(original, destination, dirs_exist_ok=True)
    shapes = json.loads((original / "tensor_shapes.json").read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        weights = {
            name: (torch.ones(spec["shape"], dtype=torch.bfloat16) if ".scale" in name
                   else torch.randn(spec["shape"], dtype=torch.bfloat16) * .1).to(DTYPES[spec["dtype"]])
            for name, spec in shapes.items()
        }
    save_file(weights, str(destination / "model.safetensors"))
    FIXTURE = destination
    yield
    FIXTURE = original


@pytest.fixture
def decoder():
    args = ModelArgs(**json.loads((FIXTURE / "inference/config.json").read_text()))
    store = DiskWeights(FIXTURE, cache_bytes=1 << 20)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model = Decoder(args, store, AutoTokenizer.from_pretrained(FIXTURE))
    yield model
    store.close()
    torch.set_num_threads(previous)


def test_complete_tiny_decoder_matches_pinned_reference_and_resets(decoder):
    golden = json.loads((FIXTURE / "expected.json").read_text())
    expected = torch.tensor(golden["logits"], dtype=torch.float32)
    # Includes Engram layers, both compression ratios, shared index results,
    # candidate filtering, window wraparound and shifted hyper-connections.
    for _ in range(2):
        actual = torch.cat([decoder.step(token) for token in golden["tokens"]])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        decoder.reset()


def test_context_and_image_inputs_fail_before_state_is_advanced(decoder):
    with pytest.raises(ValueError, match="text token"):
        decoder.step(decoder.args.image_token_id)
    assert decoder.position == 0
    decoder.position = decoder.args.max_seq_len
    with pytest.raises(ValueError, match="context limit"):
        decoder.step(1)


def test_disk_rows_are_bounded_and_table_materialization_is_forbidden():
    store = DiskWeights(FIXTURE, cache_bytes=4096)
    try:
        name = "layers.1.engram.embed.weight"
        ids = torch.tensor([[0, 3, 0]])
        rows = store.rows(name, ids)
        assert rows.shape == (1, 3, 32)
        torch.testing.assert_close(rows[0, 0].float(), rows[0, 2].float())
        assert store.cache_bytes == 0
        with pytest.raises(ValueError, match="read by row"):
            store.get(name)
        with pytest.raises(IndexError):
            store.rows(name, torch.tensor([-1]))
        for key in ("embed.weight", "head.weight", "layers.0.ffn.shared_experts.w1.weight"):
            store.get(key)
            assert store.cache_bytes <= 4096
    finally:
        store.close()


def test_missing_required_expert_fails_at_initialization(decoder):
    del decoder.store.metadata["layers.5.ffn.experts.3.w2.weight"]
    with pytest.raises(ValueError, match="missing DeepSeek V4.1 weight"):
        decoder.store.validate_geometry(decoder.args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gb10_decoder_matches_reference_on_the_same_device():
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("reference fixture was measured on GB10")
    args = ModelArgs(**json.loads((FIXTURE / "inference/config.json").read_text()))
    store = DiskWeights(FIXTURE, device="cuda", cache_bytes=1 << 20)
    try:
        model = Decoder(args, store, AutoTokenizer.from_pretrained(FIXTURE))
        golden = json.loads((FIXTURE / "expected-gb10.json").read_text())
        actual = torch.cat([model.step(token).cpu() for token in golden["tokens"]])
        torch.testing.assert_close(actual, torch.tensor(golden["logits"]), atol=1e-6, rtol=1e-6)
    finally:
        store.close()


def test_quantization_uses_e2m1_ties_and_e8m0_scales():
    values = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6] * 4)
    expected = torch.tensor([0, 1, 1, 2, 2, 4, 4, 6] * 4, dtype=torch.float32)
    torch.testing.assert_close(fp4_roundtrip(values), expected)
    torch.testing.assert_close(fp4_roundtrip(-values), -expected)
    assert torch.isfinite(fp4_roundtrip(torch.zeros(32))).all()
    assert torch.isfinite(fp4_roundtrip(torch.zeros(32), 16, True)).all()
    packed = torch.tensor([[0x21] * 16], dtype=torch.uint8)
    scales = torch.tensor([[128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    torch.testing.assert_close(dequant(packed, scales, True), torch.tensor([[1., 2.] * 16]))
    weight = torch.ones(32, 32).to(torch.float8_e4m3fn)
    torch.testing.assert_close(dequant(weight, scales), torch.full((32, 32), 2.))
    torch.testing.assert_close(fp8_roundtrip(torch.zeros(32)), torch.zeros(32))


def test_registration_and_engine_reconcile_native_constraints(monkeypatch):
    from sparklab.runtime.distributed import DistributedInfo
    from sparklab.runtime.engine.config import EngineConfig
    from sparklab.runtime.engine.engine import _adjust_config

    config = parse_config(RawConfigShim(_name_or_path=str(FIXTURE)))
    assert get_model_spec("DeepseekV41ForCausalLM").model_cls == "DeepseekV41ForCausalLM"
    assert config.dsv4_args is None and config.is_moe
    engine = EngineConfig(model_path=str(FIXTURE), tp_info=DistributedInfo(0, 1),
                          dtype=torch.bfloat16, attention_backend="triton")
    object.__setattr__(engine, "model_config", config)
    _adjust_config(engine)
    assert engine.cache_type == "naive"
    assert engine.cuda_graph_max_bs == 0
    assert engine.max_running_req == 1
    assert engine.num_token_override == 2048
    assert engine.moe_backend == "fused"
    multi = replace(engine, tp_info=DistributedInfo(0, 2))
    with pytest.raises(ValueError, match="TP=1"):
        _adjust_config(multi)


def test_engine_adapter_handles_chunk_continuation_and_new_requests(decoder, monkeypatch):
    model = DeepseekV41ForCausalLM(SimpleNamespace(dsv41_args=decoder.args))
    model.decoder = decoder
    req = SimpleNamespace(uid=1, mm_embeds=None)
    batch = SimpleNamespace(reqs=[req], positions=torch.tensor([0, 1, 2]),
                            input_ids=torch.tensor([3, 4, 5]), return_all_logits=True)
    monkeypatch.setattr("sparklab.models.deepseek_v41.model.get_global_ctx", lambda: SimpleNamespace(batch=batch))
    golden = torch.tensor(json.loads((FIXTURE / "expected.json").read_text())["logits"])
    torch.testing.assert_close(model.forward(), golden[:3], atol=1e-6, rtol=1e-6)
    batch.positions, batch.input_ids = torch.tensor([3]), torch.tensor([6])
    torch.testing.assert_close(model.forward(), golden[3:4], atol=1e-6, rtol=1e-6)
    batch.positions = torch.tensor([2])
    with pytest.raises(ValueError, match="consecutive"):
        model.forward()
    req.uid = 2
    batch.positions, batch.input_ids = torch.tensor([0]), torch.tensor([3])
    torch.testing.assert_close(model.forward(), golden[:1], atol=1e-6, rtol=1e-6)


def test_native_recipe_is_experimental_and_skips_ftw(tmp_path):
    from sparklab.acquire import AcquisitionError, acquire_recipe
    from sparklab.catalog import get_recipe

    recipe = get_recipe("deepseek-v4.1-flash")
    assert recipe.backend == "native" and recipe.intended_tier == "research"
    assert recipe.status == "experimental" and recipe.performance is None
    assert recipe.evidence == () and recipe.runtime_artifact is None
    assert recipe.deployment.runtime_format == "safetensors"
    with pytest.raises(AcquisitionError, match="omit --prepare"):
        acquire_recipe(recipe, root=str(tmp_path), prepare=True)
