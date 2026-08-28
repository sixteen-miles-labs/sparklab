from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

from sparklab.backends import (
    ArtifactValidation,
    BackendCapabilities,
    BackendLaunchPlan,
    RuntimeBackend,
    RuntimeRequest,
    get_backend,
    list_backends,
)
from sparklab.catalog import DeploymentRecipe, ModelRecipe, get_recipe


class FakeBackend(RuntimeBackend):
    backend_id = "fake"

    def __init__(self) -> None:
        self.launched: BackendLaunchPlan | None = None

    @property
    def backend_version(self) -> str:
        return "test-1"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            artifact_formats=("fake-v1",),
            quantizations=("bf16",),
            execution_policies=("resident",),
            api_protocols=("openai-chat",),
        )

    def validate_deployment(self, deployment) -> None:
        assert deployment.backend == self.backend_id

    def migrate_v1_recipe(self, value):
        return dict(value)

    def accepts_artifact(self, path, deployment) -> bool:
        return (path / "ready").is_file()

    def validate_artifact(self, path, deployment) -> ArtifactValidation:
        assert self.accepts_artifact(path, deployment)
        return ArtifactValidation("fake-v1", "fake-fingerprint", {"ready": True})

    def prepare(self, source, destination, deployment, *, implementation=None):
        destination.mkdir(parents=True)
        (destination / "ready").write_text("ok", encoding="utf-8")
        return self.validate_artifact(destination, deployment)

    def build_launch_plan(self, request: RuntimeRequest) -> BackendLaunchPlan:
        return BackendLaunchPlan(
            backend=self.backend_id,
            backend_version=self.backend_version,
            recipe=request.recipe,
            checkpoint=str(request.checkpoint),
            served_model=request.model,
            arguments=("fake-server", str(request.checkpoint), *request.extra_args),
            capabilities=self.capabilities().api_protocols,
        )

    def launch(self, plan: BackendLaunchPlan, *, prog: str) -> None:
        self.launched = plan


def test_native_backend_compiles_qwen_recipe_options_in_stable_order(tmp_path):
    recipe = get_recipe("qwen3.8-flash-next")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    deployment = replace(recipe.deployment, runtime_format="safetensors")

    plan = get_backend("native").build_launch_plan(
        RuntimeRequest(
            recipe=recipe.slug,
            recipe_version=recipe.recipe_version,
            model=recipe.model,
            checkpoint=checkpoint,
            deployment=deployment,
            extra_args=("--port", "1919"),
        )
    )

    assert plan.backend == "native"
    assert plan.arguments == (
        "--model",
        str(checkpoint),
        "--served-model-name",
        "Qwen/Qwen3.8-Flash-Next-FP8",
        "--attention-backend",
        "qsa",
        "--moe-backend",
        "offload",
        "--moe-storage",
        "disk",
        "--moe-host-cache-gb",
        "3",
        "--moe-cache-auto",
        "--moe-prefill-sparse-max-tokens",
        "512",
        "--moe-prefill-hit-d2d",
        "--cuda-graph-max-bs",
        "0",
        "--cache-type",
        "naive",
        "--page-size",
        "16",
        "--max-running-req",
        "1",
        "--port",
        "1919",
    )


def test_native_prepare_preserves_qwen_source_precision(tmp_path, monkeypatch):
    recipe = get_recipe("qwen3.8-flash-next")
    backend = get_backend("native")
    calls = []

    def converter(*args, **kwargs):
        calls.append((args, kwargs))
        return {"fingerprint": "test"}

    expected = ArtifactValidation("ftw-fp8", "test", {})
    monkeypatch.setattr(backend, "validate_artifact", lambda *_args: expected)
    result = backend.prepare(
        tmp_path / "source",
        tmp_path / "runtime",
        recipe.deployment,
        implementation=converter,
    )

    assert result == expected
    assert calls[0][1]["expert_quantization"] is None
    assert calls[0][1]["moe_backend"] == "offload"


def test_schema_one_recipe_migrates_to_native_deployment():
    current = get_recipe("qwen3.8-flash-next").to_dict()
    current.pop("deployment")
    current.pop("parameters")
    current.update(
        schema_version="1.0",
        checkpoint_format="ftw",
        expert_quantization="nvfp4",
        execution_policy="nvme-moe",
        runtime_args=["--attention-backend", "qsa", "--moe-cache-auto"],
    )

    migrated = ModelRecipe.from_dict(current)

    assert migrated.schema_version == "2.0"
    assert migrated.deployment.backend == "native"
    assert migrated.deployment.runtime_format == "ftw"
    assert migrated.deployment.backend_options == {
        "attention_backend": "qsa",
        "moe_cache_auto": True,
        "convert_expert_quantization": "nvfp4",
    }
    assert migrated.parameters == "Unknown"


def test_only_explicit_builtin_backend_is_registered():
    assert [backend.backend_id for backend in list_backends()] == ["native"]


def test_fake_backend_satisfies_prepare_plan_launch_and_health_contract(tmp_path):
    backend = FakeBackend()
    deployment = DeploymentRecipe(
        backend="fake",
        backend_api="1.0",
        source_format="fake-v1",
        runtime_format="fake-v1",
        quantization="bf16",
        execution_policy="resident",
        backend_options={},
    )
    source = tmp_path / "source"
    source.mkdir()
    artifact = tmp_path / "runtime"

    validation = backend.prepare(source, artifact, deployment)
    plan = backend.build_launch_plan(
        RuntimeRequest(
            recipe="fake-recipe",
            recipe_version="1",
            model="fake/model",
            checkpoint=artifact,
            deployment=deployment,
            extra_args=("--port", "1919"),
        )
    )
    backend.launch(plan, prog="fake")

    assert validation.fingerprint == "fake-fingerprint"
    assert backend.launched == plan
    assert backend.normalize_health({"status": "ok", "model": "fake/model"}) == {
        "status": "ok",
        "ready": True,
        "model": "fake/model",
        "progress": None,
        "error": None,
    }


def test_product_modules_only_import_engine_inside_native_adapter():
    package = Path(__file__).resolve().parents[2] / "python" / "sparklab"
    violations = []
    allowed = {
        package / "backends" / "native.py",
        package / "backends" / "native_artifacts.py",
    }
    for source in package.rglob("*.py"):
        if source in allowed:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "freetoken" or name.startswith("freetoken.") for name in names):
                violations.append(str(source.relative_to(package)))
    assert violations == []
