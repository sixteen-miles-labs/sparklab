"""Isolated vLLM serving for the DiffusionGemma block-diffusion recipe.

vLLM's torch dependency differs from the native engine's. Keep it in Docker;
never import vLLM into SparkLab's Python environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

from .base import (
    ArtifactValidation, BackendCapabilities, BackendError, BackendLaunchPlan,
    RuntimeBackend, RuntimeRequest,
)


class VLLMBackend(RuntimeBackend):
    backend_id = "vllm"

    @property
    def backend_version(self):
        return "0.28.0"

    def capabilities(self):
        return BackendCapabilities(
            artifact_formats=("safetensors-nvfp4", "safetensors"), quantizations=("nvfp4",),
            execution_policies=("resident",), api_protocols=("openai-chat",),
            lifecycle=("launch", "health", "stop"),
        )

    def validate_deployment(self, deployment):
        if (deployment.backend, deployment.backend_api) != (self.backend_id, self.backend_api):
            raise BackendError("vLLM deployment requires backend API 1.0")
        if (deployment.source_format, deployment.runtime_format,
                deployment.quantization, deployment.execution_policy) != (
                "safetensors-nvfp4", "safetensors", "nvfp4", "resident"):
            raise BackendError("vLLM adapter requires resident ModelOpt NVFP4 safetensors")
        options = deployment.backend_options
        if set(options) != {"image", "memory_gib", "arguments"}:
            raise BackendError("vLLM options require image, memory_gib and arguments")
        if not isinstance(options["image"], str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._/-]*@sha256:[0-9a-f]{64}", options["image"]):
            raise BackendError("vLLM image must be pinned to a SHA256 digest")
        if type(options["memory_gib"]) is not int or options["memory_gib"] <= 0:
            raise BackendError("vLLM memory_gib must be a positive integer")
        arguments = options["arguments"]
        if not isinstance(arguments, list) or any(
                not isinstance(arg, str) or "\x00" in arg for arg in arguments):
            raise BackendError("vLLM arguments must be a list of strings")
        if any(arg.split("=", 1)[0] in {"--model", "--served-model-name"} for arg in arguments):
            raise BackendError("vLLM model identity is owned by the recipe")

    def migrate_v1_recipe(self, value):
        raise BackendError("vLLM recipes require schema 2.0")

    def accepts_artifact(self, path, deployment):
        try:
            config = json.loads((path / "config.json").read_text())
            index = json.loads((path / "model.safetensors.index.json").read_text())
            weights = index["weight_map"]
            shards = set(weights.values())
            return (
                config.get("architectures") == ["DiffusionGemmaForBlockDiffusion"]
                and config.get("model_type") == "diffusion_gemma"
                and config.get("quantization_config", {}).get("quant_algo") == "NVFP4"
                and config.get("quantization_config", {}).get("quant_method") == "modelopt"
                and bool(weights)
                and all(isinstance(shard, str) and Path(shard).name == shard
                        and (path / shard).is_file() for shard in shards)
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False

    def validate_artifact(self, path, deployment):
        self.validate_deployment(deployment)
        if not self.accepts_artifact(path, deployment):
            raise BackendError(f"{path} is not a DiffusionGemma ModelOpt NVFP4 checkpoint")
        from sparklab.acquire import AcquisitionError, validate_safetensors_snapshot
        try:
            details = validate_safetensors_snapshot(path)
        except AcquisitionError as exc:
            raise BackendError(str(exc)) from exc
        return ArtifactValidation(
            "safetensors", hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            details,
        )

    def prepare(self, source, destination, deployment, *, implementation=None):
        raise BackendError("vLLM loads source safetensors directly; omit --prepare")

    def build_launch_plan(self, request: RuntimeRequest):
        self.validate_deployment(request.deployment)
        if not self.accepts_artifact(request.checkpoint, request.deployment):
            raise BackendError("vLLM requires a DiffusionGemma NVFP4 checkpoint")
        if any(arg.split("=", 1)[0] in {"--model", "--served-model-name"}
               for arg in request.extra_args):
            raise BackendError("vLLM model identity cannot be overridden")
        options = request.deployment.backend_options
        arguments = (
            "--model", "/artifact", "--served-model-name", request.model,
            *options["arguments"], *request.extra_args,
        )
        limit = f"{options['memory_gib']}g"
        command = (
            "docker", "run", "--rm", "--gpus", "all", "--network", "host",
            "--shm-size", "4g", "--memory", limit, "--memory-swap", limit,
            "-v", f"{request.checkpoint.resolve()}:/artifact:ro",
            options["image"], *arguments,
        )
        return BackendLaunchPlan(
            backend=self.backend_id, backend_version=self.backend_version,
            recipe=request.recipe, checkpoint=str(request.checkpoint),
            served_model=request.model, arguments=arguments, command=command,
            capabilities=self.capabilities().api_protocols, metrics_path="/metrics",
        )

    def launch(self, plan, *, prog):
        try:
            subprocess.run(plan.command, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise BackendError(f"vLLM container failed: {exc}") from exc
