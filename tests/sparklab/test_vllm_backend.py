from dataclasses import replace
import json
import struct

import pytest

from sparklab.backends import BackendError, RuntimeRequest, get_backend
from sparklab.catalog import get_recipe


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "model_type": "diffusion_gemma",
        "quantization_config": {"quant_algo": "NVFP4", "quant_method": "modelopt"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"test.weight": "model.safetensors"},
        "metadata": {"total_size": 2},
    }))
    header = json.dumps({"test.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
    return tmp_path


def test_diffusion_recipe_uses_isolated_backend_and_pinned_weights(checkpoint):
    recipe = get_recipe("diffusiongemma-26b-a4b")
    backend = get_backend(recipe.backend)
    assert recipe.backend == "vllm"
    assert recipe.status == "experimental"
    assert len(recipe.revision) == 40
    validation = backend.validate_artifact(checkpoint, recipe.deployment)
    assert validation.details["shards"] == 1
    plan = backend.build_launch_plan(RuntimeRequest(
        recipe.slug, recipe.recipe_version, recipe.model, checkpoint, recipe.deployment,
        extra_args=("--port", "18081"),
    ))
    assert plan.command[:3] == ("docker", "run", "--rm")
    assert plan.command[plan.command.index("--memory") + 1] == "72g"
    assert plan.command[plan.command.index("--memory-swap") + 1] == "72g"
    assert f"{checkpoint.resolve()}:/artifact:ro" in plan.command
    assert plan.arguments[-2:] == ("--port", "18081")
    assert plan.metrics_path == "/metrics"
    assert "anthropic-messages" not in plan.capabilities
    assert json.loads(plan.arguments[plan.arguments.index("--diffusion-config") + 1])["canvas_length"] == 256


def test_vllm_rejects_other_architectures_and_truncated_shards(checkpoint):
    recipe = get_recipe("diffusiongemma-26b-a4b")
    backend = get_backend("vllm")
    shard = checkpoint / "model.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(BackendError, match="size mismatch"):
        backend.validate_artifact(checkpoint, recipe.deployment)
    config = json.loads((checkpoint / "config.json").read_text())
    config["architectures"] = ["Gemma4ForConditionalGeneration"]
    (checkpoint / "config.json").write_text(json.dumps(config))
    assert not backend.accepts_artifact(checkpoint, recipe.deployment)


def test_vllm_rejects_shard_path_escape(checkpoint):
    recipe = get_recipe("diffusiongemma-26b-a4b")
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"test.weight": "../model.safetensors"},
    }))
    assert not get_backend("vllm").accepts_artifact(checkpoint, recipe.deployment)


def test_vllm_requires_immutable_image_and_owned_model_identity(checkpoint):
    recipe = get_recipe("diffusiongemma-26b-a4b")
    backend = get_backend("vllm")
    options = dict(recipe.deployment.backend_options)
    with pytest.raises(BackendError, match="pinned"):
        backend.validate_deployment(replace(recipe.deployment, backend_options={**options, "image": "vllm/vllm-openai:latest"}))
    with pytest.raises(BackendError, match="identity"):
        backend.build_launch_plan(RuntimeRequest(
            recipe.slug, recipe.recipe_version, recipe.model, checkpoint, recipe.deployment,
            extra_args=("--model=some-other-model",),
        ))


def test_inkling_deployment_binds_checkpoint_architecture(checkpoint):
    recipe = get_recipe("inkling-2-mini")
    backend = get_backend("vllm")
    deployment = recipe.deployment
    assert recipe.status == "experimental"
    assert len(recipe.revision) == 40
    backend.validate_deployment(deployment)
    assert not backend.accepts_artifact(checkpoint, deployment)
    config = json.loads((checkpoint / "config.json").read_text())
    config.update(architectures=["Inkling2LoopForConditionalGeneration"], model_type="inkling2_loop")
    (checkpoint / "config.json").write_text(json.dumps(config))
    assert backend.validate_artifact(checkpoint, deployment).details["shards"] == 1
    assert not backend.accepts_artifact(checkpoint, get_recipe("diffusiongemma-26b-a4b").deployment)
    plan = backend.build_launch_plan(RuntimeRequest(
        "inkling-2-mini", "0.1.0", "thinkingmachines/Inkling-2-Mini", checkpoint, deployment,
    ))
    assert plan.backend_version == deployment.backend_options["image"]
    assert plan.served_model == "thinkingmachines/Inkling-2-Mini"
    with pytest.raises(BackendError, match="architecture"):
        backend.validate_deployment(replace(deployment, backend_options={
            **deployment.backend_options, "architecture": "unknown",
        }))


def test_diffusion_evidence_does_not_claim_autoregressive_speed_or_certification():
    from pathlib import Path
    from sparklab.certification import evaluate_tier

    recipe = get_recipe("diffusiongemma-26b-a4b")
    root = Path(__file__).resolve().parents[2]
    evidence = json.loads((root / "benchmarks/gb10/results/GB10-DIFFUSIONGEMMA-001.json").read_text())
    assert evidence["result_id"] in recipe.evidence
    assert evidence["recipe"]["revision"] == recipe.revision
    assert evidence["deployment"] == recipe.deployment.to_dict()
    assert evidence["metrics"]["decode_tokens_per_second"] is None
    assert recipe.performance is None
    assert not evaluate_tier(recipe, evidence, "research").passed
    raw = json.loads((root / evidence["source"]["raw_artifact"]).read_text())
    assert raw["status"] == "measured"
    assert len(raw["cases"]) == 6
    assert all(len(case["trials"]) == 3 for case in raw["cases"])


def test_inkling_smoke_evidence_does_not_claim_certification():
    from pathlib import Path
    from sparklab.certification import evaluate_tier

    recipe = get_recipe("inkling-2-mini")
    root = Path(__file__).resolve().parents[2]
    evidence = json.loads((root / "benchmarks/gb10/results/GB10-INKLING2-MINI-001.json").read_text())
    assert evidence["result_id"] in recipe.evidence
    assert evidence["recipe"]["revision"] == recipe.revision
    assert evidence["deployment"] == recipe.deployment.to_dict()
    assert not evaluate_tier(recipe, evidence, "research").passed
    assert recipe.performance is None
    raw = json.loads((root / evidence["source"]["raw_artifact"]).read_text())
    assert raw["status"] == "passed"
    assert len(raw["checks"]) == 5 and all(c["passed"] for c in raw["checks"].values())
    assert len(raw["trials"]) == 3
