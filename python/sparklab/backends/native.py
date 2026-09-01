"""Adapter for SparkLab's built-in native inference engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .base import (
    ArtifactValidation,
    BackendCapabilities,
    BackendError,
    BackendLaunchPlan,
    DeploymentConfig,
    RuntimeBackend,
    RuntimeRequest,
)
from .native_artifacts import validate_ftw

_VALUE_OPTIONS: dict[str, tuple[str, type | tuple[type, ...]]] = {
    "attention_backend": ("--attention-backend", str),
    "moe_backend": ("--moe-backend", str),
    "moe_storage": ("--moe-storage", str),
    "moe_host_cache_gb": ("--moe-host-cache-gb", (int, float)),
    "memory_ratio": ("--memory-ratio", (int, float)),
    "moe_cache_size": ("--moe-cache-size", int),
    "moe_cache_policy": ("--moe-cache-policy", str),
    "moe_cache_rate": ("--moe-cache-rate", (int, float)),
    "nvfp4_backend": ("--nvfp4-backend", str),
    "moe_prefill_sparse_max_tokens": ("--moe-prefill-sparse-max-tokens", int),
    "cuda_graph_max_bs": ("--cuda-graph-max-bs", int),
    "cache_type": ("--cache-type", str),
    "page_size": ("--page-size", int),
    "max_running_req": ("--max-running-requests", int),
    "num_tokens": ("--num-tokens", int),
    "speculative_method": ("--speculative-method", str),
    "speculative_tokens": ("--speculative-tokens", int),
    "draft_sample_method": ("--draft-sample-method", str),
}
_FLAG_OPTIONS: dict[str, str] = {
    "disable_startup_prefill_warmup": "--disable-startup-prefill-warmup",
    "kimi_mlp_fp8": "--kimi-mlp-fp8",
    "moe_cache_auto": "--moe-cache-auto",
    "moe_preload_all": "--moe-preload-all",
    "moe_prefill_hit_d2d": "--moe-prefill-hit-d2d",
    "moe_shared_expert_overlap": "--moe-shared-expert-overlap",
}
_FALSE_FLAG_OPTIONS: dict[str, str] = {
    "moe_prefill_overlap": "--disable-moe-prefill-overlap",
}
_OPTION_ORDER = (
    "attention_backend",
    "moe_backend",
    "moe_storage",
    "moe_host_cache_gb",
    "memory_ratio",
    "moe_cache_size",
    "moe_cache_policy",
    "moe_cache_rate",
    "moe_cache_auto",
    "moe_preload_all",
    "nvfp4_backend",
    "moe_prefill_sparse_max_tokens",
    "moe_prefill_overlap",
    "moe_prefill_hit_d2d",
    "moe_shared_expert_overlap",
    "cuda_graph_max_bs",
    "cache_type",
    "page_size",
    "max_running_req",
    "num_tokens",
    "speculative_method",
    "speculative_tokens",
    "draft_sample_method",
    "kimi_mlp_fp8",
    "disable_startup_prefill_warmup",
)
_SUPPORTED_FORMATS = (
    "safetensors",
    "safetensors-nvfp4",
    "ftw",
    "ftw-v1",
    "ftw-nvfp4",
    "ftw-fp8",
    "ftw-mxfp4",
    "ftw-ds-fp4",
    "gguf",
)
_SUPPORTED_QUANTIZATIONS = ("bf16", "fp8", "nvfp4", "mxfp4", "ds-fp4")


def _compile_options(options: Mapping[str, Any]) -> tuple[str, ...]:
    arguments: list[str] = []
    for key in _OPTION_ORDER:
        if key not in options:
            continue
        value = options[key]
        if key in _FLAG_OPTIONS:
            if value is True:
                arguments.append(_FLAG_OPTIONS[key])
            elif value is not False:
                raise BackendError(f"native backend option {key!r} must be boolean")
            continue
        if key in _FALSE_FLAG_OPTIONS:
            if value is False:
                arguments.append(_FALSE_FLAG_OPTIONS[key])
            elif value is not True:
                raise BackendError(f"native backend option {key!r} must be boolean")
            continue
        flag, expected = _VALUE_OPTIONS[key]
        if isinstance(value, bool) or not isinstance(value, expected):
            raise BackendError(
                f"native backend option {key!r} has invalid value {value!r}"
            )
        arguments.extend((flag, str(value)))
    return tuple(arguments)


def _parse_v1_arguments(arguments: Sequence[str]) -> dict[str, Any]:
    reverse_values = {flag: (key, expected) for key, (flag, expected) in _VALUE_OPTIONS.items()}
    reverse_flags = {flag: key for key, flag in _FLAG_OPTIONS.items()}
    reverse_false_flags = {flag: key for key, flag in _FALSE_FLAG_OPTIONS.items()}
    options: dict[str, Any] = {}
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        if flag in reverse_flags:
            options[reverse_flags[flag]] = True
            index += 1
            continue
        if flag in reverse_false_flags:
            options[reverse_false_flags[flag]] = False
            index += 1
            continue
        if flag not in reverse_values or index + 1 >= len(arguments):
            raise BackendError(f"cannot migrate unknown native runtime argument {flag!r}")
        key, expected = reverse_values[flag]
        raw = arguments[index + 1]
        if expected is int:
            value: Any = int(raw)
        elif expected == (int, float):
            value = float(raw)
            if value.is_integer():
                value = int(value)
        else:
            value = raw
        options[key] = value
        index += 2
    return options


class NativeBackend(RuntimeBackend):
    backend_id = "native"
    backend_api = "1.0"

    @property
    def backend_version(self) -> str:
        from sparklab.version import __version__

        return __version__

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            artifact_formats=_SUPPORTED_FORMATS,
            quantizations=_SUPPORTED_QUANTIZATIONS,
            execution_policies=("resident", "uma-moe", "nvme-moe"),
            api_protocols=("openai-chat", "openai-responses", "anthropic-messages"),
        )

    def validate_deployment(self, deployment: DeploymentConfig) -> None:
        if deployment.backend != self.backend_id:
            raise BackendError(
                f"native adapter cannot validate backend {deployment.backend!r}"
            )
        if deployment.backend_api != self.backend_api:
            raise BackendError(
                f"native backend API must be {self.backend_api!r}, "
                f"found {deployment.backend_api!r}"
            )
        if deployment.source_format not in _SUPPORTED_FORMATS:
            raise BackendError(
                f"unsupported native source format {deployment.source_format!r}"
            )
        if deployment.runtime_format not in _SUPPORTED_FORMATS:
            raise BackendError(
                f"unsupported native runtime format {deployment.runtime_format!r}"
            )
        if (
            deployment.quantization is not None
            and deployment.quantization not in _SUPPORTED_QUANTIZATIONS
        ):
            raise BackendError(
                f"unsupported native quantization {deployment.quantization!r}"
            )
        if deployment.execution_policy not in self.capabilities().execution_policies:
            raise BackendError(
                f"unsupported native execution policy {deployment.execution_policy!r}"
            )
        unknown = set(deployment.backend_options) - set(_OPTION_ORDER) - {
            "convert_expert_quantization",
            "convert_kda_quantization",
        }
        if unknown:
            raise BackendError(f"unknown native backend options: {sorted(unknown)}")
        convert_quant = deployment.backend_options.get("convert_expert_quantization")
        if convert_quant not in {None, "nvfp4"}:
            raise BackendError(
                "native convert_expert_quantization currently supports only 'nvfp4'"
            )
        kda_quant = deployment.backend_options.get("convert_kda_quantization")
        if kda_quant not in {None, "fp8_pertensor"}:
            raise BackendError(
                "native convert_kda_quantization currently supports only "
                "'fp8_pertensor'"
            )
        _compile_options(deployment.backend_options)

    def migrate_v1_recipe(self, value: Mapping[str, Any]) -> dict[str, Any]:
        migrated = dict(value)
        old_format = str(migrated.pop("checkpoint_format"))
        quantization = migrated.pop("expert_quantization", None)
        migrated.setdefault("parameters", "Unknown")
        migrated["schema_version"] = "2.0"
        migrated["deployment"] = {
            "backend": self.backend_id,
            "backend_api": self.backend_api,
            "source_format": "safetensors",
            "runtime_format": old_format,
            "quantization": quantization,
            "execution_policy": migrated.pop("execution_policy"),
            "backend_options": _parse_v1_arguments(migrated.pop("runtime_args", ())),
        }
        if quantization == "nvfp4":
            migrated["deployment"]["backend_options"][
                "convert_expert_quantization"
            ] = "nvfp4"
        return migrated

    def accepts_artifact(self, path: Path, deployment: DeploymentConfig) -> bool:
        if not path.is_dir() or not (path / "config.json").is_file():
            return False
        if deployment.runtime_format.startswith("ftw"):
            from sparklab.checkpoint import is_ftw_checkpoint

            return is_ftw_checkpoint(str(path))
        return True

    def validate_artifact(
        self, path: Path, deployment: DeploymentConfig
    ) -> ArtifactValidation:
        self.validate_deployment(deployment)
        if deployment.runtime_format.startswith("ftw"):
            return validate_ftw(path, runtime_format=deployment.runtime_format)
        if not self.accepts_artifact(path, deployment):
            raise BackendError(
                f"{path} is not a valid {deployment.runtime_format} native artifact"
            )
        return ArtifactValidation(
            format=deployment.runtime_format,
            fingerprint=None,
            details={"config": "config.json"},
        )

    def prepare(
        self,
        source: Path,
        destination: Path,
        deployment: DeploymentConfig,
        *,
        implementation: Callable[..., dict[str, Any]] | None = None,
    ) -> ArtifactValidation:
        self.validate_deployment(deployment)
        if not deployment.runtime_format.startswith("ftw"):
            raise BackendError(
                f"native preparation for {deployment.runtime_format!r} is not required"
            )
        if implementation is None:
            from sparklab.checkpoint import convert_checkpoint

            implementation = convert_checkpoint
        destination.parent.mkdir(parents=True, exist_ok=True)
        implementation(
            str(source),
            str(destination),
            moe_backend=deployment.backend_options.get(
                "moe_backend",
                "offload" if deployment.execution_policy == "nvme-moe" else "fused",
            ),
            nvfp4_backend=deployment.backend_options.get("nvfp4_backend", "auto"),
            expert_quantization=deployment.backend_options.get(
                "convert_expert_quantization"
            ),
            kda_quantization=deployment.backend_options.get(
                "convert_kda_quantization"
            ),
            device="cuda:0",
        )
        return self.validate_artifact(destination, deployment)

    def build_launch_plan(self, request: RuntimeRequest) -> BackendLaunchPlan:
        deployment = request.deployment
        self.validate_deployment(deployment)
        if not self.accepts_artifact(request.checkpoint, deployment):
            raise BackendError(
                f"{request.recipe} requires a valid {deployment.runtime_format} "
                f"artifact for backend {self.backend_id}"
            )
        arguments = [
            "--model",
            str(request.checkpoint),
            "--served-model-name",
            request.model,
        ]
        arguments.extend(_compile_options(deployment.backend_options))
        arguments.extend(request.extra_args)
        return BackendLaunchPlan(
            backend=self.backend_id,
            backend_version=self.backend_version,
            recipe=request.recipe,
            checkpoint=str(request.checkpoint),
            served_model=request.model,
            arguments=tuple(arguments),
            capabilities=self.capabilities().api_protocols,
        )

    def launch(self, plan: BackendLaunchPlan, *, prog: str) -> None:
        from sparklab.serving import launch_server

        launch_server(argv=list(plan.arguments), prog=prog)

    def run_command(
        self, command: str, argv: Sequence[str], *, prog: str
    ) -> int:
        arguments = list(argv)
        if command == "serve":
            from sparklab.serving import launch_server

            launch_server(argv=arguments, prog=prog)
            return 0
        if command == "shell":
            from sparklab.shell import main

            return main(arguments, prog=prog)
        if command == "ctl":
            from sparklab.control_cli import main

            return main(arguments, prog=prog)
        if command == "daemon":
            from sparklab.daemon import main

            return main(arguments, prog=prog)
        if command == "status":
            from sparklab.daemon.client import main

            return main(["status", *arguments], prog=prog)
        if command == "launch":
            from sparklab.launch import main

            return main(arguments, prog=prog)
        if command == "checkpoint":
            from sparklab.checkpoint.__main__ import main

            return main(arguments, prog=prog)
        if command == "bench-bw":
            from sparklab.moe.benchbw import main

            return main(arguments, prog=prog)
        return super().run_command(command, arguments, prog=prog)


__all__ = ["NativeBackend"]
