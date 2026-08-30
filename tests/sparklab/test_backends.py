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
        "Inferact/Qwen3.8-Flash-Next-NVFP4",
        "--attention-backend",
        "qsa",
        "--moe-backend",
        "offload",
        "--moe-storage",
        "disk",
        "--moe-host-cache-gb",
        "0",
        "--moe-cache-auto",
        "--moe-preload-all",
        "--nvfp4-backend",
        "triton",
        "--cuda-graph-max-bs",
        "1",
        "--cache-type",
        "naive",
        "--page-size",
        "16",
        "--max-running-requests",
        "1",
        "--num-tokens",
        "131072",
        "--port",
        "1919",
    )


def test_native_backend_compiles_glm_residency_options(tmp_path):
    recipe = get_recipe("glm-5.3-flash")
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
        )
    )

    assert "--moe-host-cache-gb" in plan.arguments
    assert plan.arguments[plan.arguments.index("--moe-host-cache-gb") + 1] == "0"
    assert "--memory-ratio" in plan.arguments
    assert plan.arguments[plan.arguments.index("--memory-ratio") + 1] == "0.96"
    assert "--disable-moe-prefill-overlap" in plan.arguments


def test_native_backend_compiles_kimi_bounded_gb10_options(tmp_path):
    recipe = get_recipe("kimi-k3")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    deployment = replace(
        recipe.deployment,
        runtime_format="safetensors",
        backend_options={
            "moe_cache_size": 896,
            "moe_cache_policy": "layer_lru",
            "kimi_mlp_fp8": True,
            "disable_startup_prefill_warmup": True,
        },
    )

    plan = get_backend("native").build_launch_plan(
        RuntimeRequest(
            recipe=recipe.slug,
            recipe_version=recipe.recipe_version,
            model=recipe.model,
            checkpoint=checkpoint,
            deployment=deployment,
        )
    )

    assert plan.arguments[-6:] == (
        "--moe-cache-size",
        "896",
        "--moe-cache-policy",
        "layer_lru",
        "--kimi-mlp-fp8",
        "--disable-startup-prefill-warmup",
    )


def test_native_backend_compiles_deepseek_sparse_prefill_options(tmp_path):
    recipe = get_recipe("deepseek-v4")
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
        )
    )

    assert plan.arguments == (
        "--model",
        str(checkpoint),
        "--served-model-name",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "--moe-backend",
        "offload",
        "--moe-storage",
        "disk",
        "--moe-host-cache-gb",
        "4",
        "--memory-ratio",
        "0.9",
        "--moe-cache-auto",
        "--moe-prefill-sparse-max-tokens",
        "512",
        "--cuda-graph-max-bs",
        "0",
        "--cache-type",
        "radix",
        "--max-running-requests",
        "1",
    )


def test_native_prepare_preserves_qwen_nvfp4_source_precision(tmp_path, monkeypatch):
    recipe = get_recipe("qwen3.8-flash-next")
    backend = get_backend("native")
    calls = []

    def converter(*args, **kwargs):
        calls.append((args, kwargs))
        return {"fingerprint": "test"}

    expected = ArtifactValidation("ftw-nvfp4", "test", {})
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


def test_native_prepare_honors_resident_expert_bank_layout(tmp_path, monkeypatch):
    recipe = get_recipe("qwen3.6-35b-a3b")
    backend = get_backend("native")
    calls = []

    def converter(*args, **kwargs):
        calls.append((args, kwargs))
        return {"fingerprint": "test"}

    expected = ArtifactValidation("ftw-nvfp4", "test", {})
    monkeypatch.setattr(backend, "validate_artifact", lambda *_args: expected)
    result = backend.prepare(
        tmp_path / "source",
        tmp_path / "runtime",
        recipe.deployment,
        implementation=converter,
    )

    assert result == expected
    assert calls[0][1]["moe_backend"] == "offload"
    assert calls[0][1]["nvfp4_backend"] == "triton"


def test_native_prepare_applies_glm_kda_artifact_quantization(tmp_path, monkeypatch):
    recipe = get_recipe("glm-5.3-flash")
    backend = get_backend("native")
    calls = []

    def converter(*args, **kwargs):
        calls.append((args, kwargs))
        return {"fingerprint": "test"}

    expected = ArtifactValidation("ftw-nvfp4", "test", {})
    monkeypatch.setattr(backend, "validate_artifact", lambda *_args: expected)
    result = backend.prepare(
        tmp_path / "source",
        tmp_path / "runtime",
        recipe.deployment,
        implementation=converter,
    )

    assert result == expected
    assert calls[0][1]["kda_quantization"] == "fp8_pertensor"
    assert calls[0][1]["expert_quantization"] is None


def test_schema_one_recipe_migrates_to_native_deployment():
    current = get_recipe("qwen3.8-flash-next").to_dict()
    current.pop("deployment")
    current.pop("parameters")
    current.update(
        schema_version="1.0",
        checkpoint_format="ftw",
        expert_quantization="nvfp4",
        execution_policy="nvme-moe",
        runtime_args=[
            "--attention-backend",
            "qsa",
            "--moe-cache-auto",
            "--disable-moe-prefill-overlap",
        ],
    )

    migrated = ModelRecipe.from_dict(current)

    assert migrated.schema_version == "2.0"
    assert migrated.deployment.backend == "native"
    assert migrated.deployment.runtime_format == "ftw"
    assert migrated.deployment.backend_options == {
        "attention_backend": "qsa",
        "moe_cache_auto": True,
        "moe_prefill_overlap": False,
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


def test_native_runtime_does_not_import_the_product_control_plane():
    package = Path(__file__).resolve().parents[2] / "python" / "sparklab"
    runtime_roots = {
        "attention",
        "benchmark",
        "checkpoint",
        "daemon",
        "kernels",
        "layers",
        "llm",
        "message",
        "models",
        "moe",
        "runtime",
        "serving",
        "shell",
        "tokenizer",
        "utils",
    }
    product_modules = {
        "sparklab.acquire",
        "sparklab.backends",
        "sparklab.catalog",
        "sparklab.certification",
        "sparklab.cli",
        "sparklab.deployment",
        "sparklab.paths",
        "sparklab.planner",
        "sparklab.platform",
        "sparklab.recipes",
    }
    violations = []
    for source in package.rglob("*.py"):
        if source.relative_to(package).parts[0] not in runtime_roots:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(
                name == product or name.startswith(product + ".")
                for name in names
                for product in product_modules
            ):
                violations.append(str(source.relative_to(package)))
    assert violations == []
