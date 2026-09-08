import hashlib
import json
from dataclasses import replace

import pytest
import torch

from sparklab.backends import BackendError, RuntimeRequest, get_backend
from sparklab.backends import container
from sparklab.catalog import get_recipe
from sparklab.checkpoint.ftw import FTWWriter


def fixture_recipe():
    recipe = get_recipe("qwen3.6-35b-a3b")
    options = dict(recipe.deployment.backend_options)
    options["container"] = dict(options["container"], config_sha256=hashlib.sha256(b"{}").hexdigest())
    return replace(recipe, deployment=replace(recipe.deployment, backend_options=options))


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "model with spaces"
    path.mkdir()
    (path / "config.json").write_text("{}")
    writer = FTWWriter(str(path), shard_limit=4096)
    writer.add_tensor("weight", torch.ones(1))
    writer.finalize({"fingerprint": "9e48171e436f5ef5", "quant_format": "nvfp4_marlin", "counts": {"weight": 1}})
    return path


def test_container_rejects_previous_native_artifact(artifact):
    recipe = fixture_recipe()
    backend = get_backend("native")
    assert backend.validate_artifact(artifact, recipe.deployment).fingerprint == "9e48171e436f5ef5"
    p = artifact / "freetoken_weight.json"
    index = json.loads(p.read_text())
    index.update(fingerprint="bda7268c3e0afd7b", quant_format="nvfp4")
    p.write_text(json.dumps(index))
    assert not backend.accepts_artifact(artifact, recipe.deployment)
    with pytest.raises(BackendError, match="configuration/layout"):
        backend.validate_artifact(artifact, recipe.deployment)


def test_launch_executes_container_without_importing_host_serving(artifact, monkeypatch):
    recipe = fixture_recipe()
    backend = get_backend("native")
    plan = backend.build_launch_plan(RuntimeRequest(
        recipe.slug, recipe.recipe_version, recipe.model, artifact, recipe.deployment,
        ("--port", "1979"),
    ))
    calls = []
    monkeypatch.setattr("os.execvp", lambda *args: calls.append(args))
    backend.launch(plan, prog="test")
    assert calls == [("docker", list(plan.command))]
    assert f"{artifact.resolve()}:/artifact:ro" in plan.command
    assert plan.command[-2:] == ("--port", "1979")
    assert plan.command.count("SPARKLAB_QWEN3_MTP4=1") == 1


def test_prepare_uses_donor_image_and_read_only_source(tmp_path, monkeypatch):
    recipe = fixture_recipe()
    calls = []
    monkeypatch.setattr(container.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    container.prepare(tmp_path / "source", tmp_path / "destination", recipe.deployment)
    command, kwargs = calls[0]
    assert kwargs == {"check": True}
    assert f"{tmp_path}/source:/source:ro" in command
    assert f"{tmp_path}/destination:/artifact" in command
    assert "sparklab-qwen36:gb10-v1" in command
    assert "nvfp4_backend='marlin'" in command[-1]
    assert command[command.index("--memory-swap") + 1] == "96g"


@pytest.mark.parametrize("field,value", [
    ("image", "--privileged"), ("memory_gib", True), ("memory_gib", 0),
    ("environment", {"PATH": "bad"}), ("environment", {"SPARKLAB_X": 1}),
    ("config_sha256", ""),
])
def test_invalid_container_configuration_fails_closed(field, value):
    deployment = fixture_recipe().deployment
    config = dict(deployment.backend_options["container"], **{field: value})
    deployment = replace(deployment, backend_options=dict(deployment.backend_options, container=config))
    with pytest.raises(BackendError):
        get_backend("native").validate_deployment(deployment)


def test_fresh_conversion_mtime_fingerprint_is_not_a_model_identity(artifact):
    recipe = fixture_recipe()
    p = artifact / "freetoken_weight.json"
    data = json.loads(p.read_text())
    data["fingerprint"] = "different-source-mtimes"
    p.write_text(json.dumps(data))
    assert get_backend("native").accepts_artifact(artifact, recipe.deployment)
    (artifact / "config.json").write_text('{"different":true}')
    assert not get_backend("native").accepts_artifact(artifact, recipe.deployment)
