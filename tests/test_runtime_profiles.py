"""Keep model format and paired launch settings explicit in GPU fixtures."""

import json

import pytest

from tests.e2e.runtime_profiles import (
    model_options, profile_arguments, server_model_options, target_only_server_profile,
)


@pytest.fixture
def artifact(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {
        "model_type": "qwen3_5_moe_text", "num_hidden_layers": 40, "num_experts": 256,
    }}))
    (tmp_path / "freetoken_weight.json").write_text('{"quant_format":"nvfp4"}')
    return tmp_path


def test_default_needs_no_artifact_and_preserves_existing_profiles():
    assert model_options("absent") == {}
    assert server_model_options("absent") == {}


def test_unknown_profile_is_an_error(artifact):
    with pytest.raises(ValueError, match="unknown real-model profile"):
        model_options(artifact, "typo")


@pytest.mark.parametrize("field,value", [
    ("quant_format", "nvfp4_marlin"), ("model_type", "qwen3_5_text"),
    ("num_hidden_layers", 48), ("num_experts", 128),
])
def test_other_format_or_architecture_cannot_use_this_profile(artifact, field, value):
    filename = "freetoken_weight.json" if field == "quant_format" else "config.json"
    path = artifact / filename
    data = json.loads(path.read_text())
    (data if field == "quant_format" else data["text_config"])[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="native Triton"):
        model_options(artifact, "qwen36-triton-resident")


def test_resident_gpu_uses_streamed_experts_and_flag_has_no_value(artifact):
    options = model_options(artifact, "qwen36-triton-resident")
    assert options["moe_storage"] == "disk"
    assert options["moe_preload_all"] is True
    assert options["moe_cache_rate"] == 1.0
    assert options["cuda_graph_max_bs"] == 0
    assert not any("speculative" in key for key in options)
    flags = server_model_options(artifact, "qwen36-triton-resident")
    args = profile_arguments(flags)
    i = args.index("--moe-preload-all")
    assert i + 1 == len(args) or args[i + 1].startswith("--")
    assert "True" not in args and "None" not in args
    # CLI and offline Engine receive the same model settings.
    assert len(args) == 2 * len(options) - 1
    assert args[args.index("--moe-cache-rate") + 1] == "1.0"


@pytest.mark.parametrize("name", ["default", "qwen36-triton-resident"])
@pytest.mark.parametrize("max_running", [1, 2])
def test_real_cli_accepts_full_fixture_profile(artifact, name, max_running):
    from sparklab.serving.args import parse_args

    options = model_options(artifact, name)
    profile = target_only_server_profile(
        artifact, name, context=4096, pages=512, max_running=max_running,
        graphs=max_running,
    )
    args, _ = parse_args([
        "--model", str(artifact), "--dtype", "bfloat16",
        "--tool-call-parser", "qwen", "--reasoning-parser", "off",
        *profile_arguments(profile),
    ])
    for key, value in options.items():
        assert getattr(args, key) == value
    assert args.speculative_tokens == 0
    assert not args.moe_cache_auto
    assert args.max_extend_tokens == 512
    assert args.max_running_req == max_running
    assert args.max_seq_len_override == 4096
    assert args.num_page_override == 512
