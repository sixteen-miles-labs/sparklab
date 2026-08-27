"""Backend-neutral runtime contracts owned by the Spark Lab product layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


class BackendError(RuntimeError):
    """A backend could not validate, prepare, plan, or launch a deployment."""


class DeploymentConfig(Protocol):
    """The recipe fields a runtime adapter is allowed to consume."""

    backend: str
    backend_api: str
    source_format: str
    runtime_format: str
    quantization: str | None
    execution_policy: str
    backend_options: Mapping[str, Any]


@dataclass(frozen=True)
class BackendCapabilities:
    artifact_formats: tuple[str, ...]
    quantizations: tuple[str, ...]
    execution_policies: tuple[str, ...]
    api_protocols: tuple[str, ...]
    lifecycle: tuple[str, ...] = ("launch", "health", "metrics", "stop")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactValidation:
    format: str
    fingerprint: str | None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "fingerprint": self.fingerprint,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class RuntimeRequest:
    recipe: str
    recipe_version: str
    model: str
    checkpoint: Path
    deployment: DeploymentConfig
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendLaunchPlan:
    backend: str
    backend_version: str
    recipe: str
    checkpoint: str
    served_model: str
    arguments: tuple[str, ...]
    environment: Mapping[str, str] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    health_path: str = "/health"
    metrics_path: str = "/v1/stats"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["environment"] = dict(self.environment)
        return value


class RuntimeBackend(ABC):
    """A runtime implementation behind Spark Lab's product contract.

    Preparation and planning are explicit and independently testable. ``launch`` may
    block for an in-process backend or supervise a subprocess for an external backend.
    """

    backend_id: str
    backend_api: str = "1.0"

    @property
    @abstractmethod
    def backend_version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    @abstractmethod
    def validate_deployment(self, deployment: DeploymentConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def migrate_v1_recipe(self, value: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def accepts_artifact(self, path: Path, deployment: DeploymentConfig) -> bool:
        raise NotImplementedError

    @abstractmethod
    def validate_artifact(
        self, path: Path, deployment: DeploymentConfig
    ) -> ArtifactValidation:
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        source: Path,
        destination: Path,
        deployment: DeploymentConfig,
        *,
        implementation: Callable[..., dict[str, Any]] | None = None,
    ) -> ArtifactValidation:
        raise NotImplementedError

    @abstractmethod
    def build_launch_plan(self, request: RuntimeRequest) -> BackendLaunchPlan:
        raise NotImplementedError

    @abstractmethod
    def launch(self, plan: BackendLaunchPlan, *, prog: str) -> None:
        raise NotImplementedError

    def normalize_health(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize backend health into the minimal Spark Lab lifecycle schema."""
        status = str(payload.get("status", "unknown"))
        return {
            "status": status,
            "ready": status == "ok",
            "model": payload.get("model"),
            "progress": payload.get("progress"),
            "error": payload.get("error") or payload.get("message"),
        }

    def normalize_metrics(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return dict(payload)

    def run_compat_command(
        self, command: str, argv: Sequence[str], *, prog: str
    ) -> int:
        raise BackendError(f"backend {self.backend_id!r} has no compatibility command {command!r}")


__all__ = [
    "ArtifactValidation",
    "BackendCapabilities",
    "BackendError",
    "BackendLaunchPlan",
    "DeploymentConfig",
    "RuntimeBackend",
    "RuntimeRequest",
]
