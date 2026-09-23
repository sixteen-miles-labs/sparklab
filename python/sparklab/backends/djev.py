"""Explicit, isolated vLLM backend for structured DiffusionGemma reads."""
from __future__ import annotations

import argparse
import hashlib
import os

from .base import ArtifactValidation, BackendCapabilities, BackendError, BackendLaunchPlan, RuntimeBackend

IMAGE = "sparklab-djev:gb10-v1"
CONFIG_SHA256 = "b4f650bd55f6c6ccd55656e27a7967c72ac28f1f3e315edaa29d6f5eebd05fde"
OPTIONS = {"image", "memory_gib", "canvas_length", "max_num_seqs", "max_model_len", "kv_cache_memory_bytes"}


class DjevBackend(RuntimeBackend):
    backend_id = "djev"

    @property
    def backend_version(self):
        return "1.0"

    def capabilities(self):
        return BackendCapabilities(
            artifact_formats=("safetensors",), quantizations=("nvfp4",),
            execution_policies=("resident",), api_protocols=("jev-systemone", "structured-chat"),
        )

    def validate_deployment(self, deployment):
        if (deployment.backend, deployment.backend_api, deployment.source_format,
            deployment.runtime_format, deployment.quantization, deployment.execution_policy) != (
                "djev", "1.0", "safetensors", "safetensors", "nvfp4", "resident"):
            raise BackendError("djev requires resident NVFP4 safetensors and backend API 1.0")
        opts = deployment.backend_options
        if set(opts) != OPTIONS or opts["image"] != IMAGE:
            raise BackendError("djev requires the pinned container image and all declared runtime options")
        bounds = {"memory_gib": (32, 80), "canvas_length": (16, 256),
                  "max_num_seqs": (1, 32), "max_model_len": (512, 16384),
                  "kv_cache_memory_bytes": (1 << 28, 8 << 30)}
        for key, (lo, hi) in bounds.items():
            if type(opts[key]) is not int or not lo <= opts[key] <= hi:
                raise BackendError(f"djev {key} must be an integer in [{lo}, {hi}]")
        if opts["canvas_length"] % 16:
            raise BackendError("djev canvas_length must be a multiple of 16")

    def migrate_v1_recipe(self, value):
        raise BackendError("djev requires schema 2.0 recipes")

    def accepts_artifact(self, path, deployment):
        try:
            self.validate_artifact(path, deployment)
            return True
        except BackendError:
            return False

    def validate_artifact(self, path, deployment):
        from sparklab.acquire import AcquisitionError, validate_safetensors_snapshot

        self.validate_deployment(deployment)
        try:
            if hashlib.sha256((path / "config.json").read_bytes()).hexdigest() != CONFIG_SHA256:
                raise ValueError("configuration differs from the pinned DiffusionGemma NVFP4 checkpoint")
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"):
                if not (path / name).is_file():
                    raise ValueError(f"missing {name}")
            details = validate_safetensors_snapshot(path)
            if details is None:
                raise ValueError("indexed safetensors are required")
            return ArtifactValidation("safetensors", CONFIG_SHA256, details)
        except (OSError, ValueError, AcquisitionError) as exc:
            raise BackendError(f"invalid djev checkpoint: {exc}") from exc

    def prepare(self, source, destination, deployment, *, implementation=None):
        raise BackendError("djev loads source weights directly; omit --prepare")

    def build_launch_plan(self, request):
        self.validate_artifact(request.checkpoint, request.deployment)
        parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
        parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8011)
        try:
            args, unknown = parser.parse_known_args(request.extra_args)
        except argparse.ArgumentError as exc:
            raise BackendError(str(exc)) from exc
        if unknown or not 1 <= args.port <= 65535:
            raise BackendError("djev accepts only --host 127.0.0.1 and --port 1..65535")
        checkpoint = request.checkpoint.resolve()
        if any(c in str(checkpoint) for c in (":", "\n", "\x00")):
            raise BackendError("checkpoint path contains an unsupported Docker mount character")
        opts = request.deployment.backend_options
        arguments = ["--model", "/model", "--served-model-name", request.model]
        for key in ("canvas_length", "max_num_seqs", "max_model_len", "kv_cache_memory_bytes"):
            arguments.extend(("--" + key.replace("_", "-"), str(opts[key])))
        # Only the gateway is published. The upstream port is private to this container.
        limit = f"{opts['memory_gib']}g"
        command = ["docker", "run", "--rm", "--init", "--gpus", "all", "--shm-size", "2g",
                   "--memory", limit, "--memory-swap", limit,
                   "-p", f"127.0.0.1:{args.port}:8011",
                   "--mount", "type=volume,src=sparklab-djev-gb10-v1-cache,dst=/root/.cache",
                   "-v", f"{checkpoint}:/model:ro", opts["image"], *arguments]
        return BackendLaunchPlan(
            backend=self.backend_id, backend_version=self.backend_version,
            recipe=request.recipe, checkpoint=str(checkpoint), served_model=request.model,
            arguments=tuple(arguments), command=tuple(command),
            capabilities=self.capabilities().api_protocols, metrics_path="/metrics",
        )

    def launch(self, plan, *, prog):
        os.execvp(plan.command[0], list(plan.command))
