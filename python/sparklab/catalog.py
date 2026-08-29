"""Versioned, backend-qualified SparkLab model-recipe catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Iterable, Mapping

TIERS = ("fast", "frontier", "research")
STATUSES = ("certified", "preview", "experimental")
PORTFOLIO_ROLES = ("primary", "fallback")
RECIPE_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class RuntimeArtifact:
    """Immutable, prebuilt execution artifact published in a model repository."""

    repo_id: str
    revision: str
    bytes: int
    fingerprint: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeArtifact":
        artifact = cls(
            repo_id=str(value["repo_id"]),
            revision=str(value["revision"]),
            bytes=int(value["bytes"]),
            fingerprint=str(value["fingerprint"]),
        )
        artifact.validate()
        return artifact

    def validate(self) -> None:
        if not self.repo_id:
            raise ValueError("runtime_artifact.repo_id is required")
        if len(self.revision) != 40:
            raise ValueError(
                "runtime_artifact.revision must be a full 40-character commit"
            )
        if self.bytes <= 0:
            raise ValueError("runtime_artifact.bytes must be positive")
        if not self.fingerprint:
            raise ValueError("runtime_artifact.fingerprint is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "bytes": self.bytes,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class DeploymentRecipe:
    backend: str
    backend_api: str
    source_format: str
    runtime_format: str
    quantization: str | None
    execution_policy: str
    backend_options: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeploymentRecipe":
        options = value.get("backend_options", {})
        if not isinstance(options, dict):
            raise ValueError("deployment.backend_options must be an object")
        deployment = cls(
            backend=str(value["backend"]),
            backend_api=str(value["backend_api"]),
            source_format=str(value["source_format"]),
            runtime_format=str(value["runtime_format"]),
            quantization=(
                str(value["quantization"]) if value.get("quantization") else None
            ),
            execution_policy=str(value["execution_policy"]),
            backend_options={str(key): item for key, item in options.items()},
        )
        deployment.validate()
        return deployment

    def validate(self) -> None:
        for name in (
            "backend",
            "backend_api",
            "source_format",
            "runtime_format",
            "execution_policy",
        ):
            if not getattr(self, name):
                raise ValueError(f"deployment.{name} is required")
        from sparklab.backends import BackendError, get_backend

        try:
            get_backend(self.backend).validate_deployment(self)
        except BackendError as exc:
            raise ValueError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_api": self.backend_api,
            "source_format": self.source_format,
            "runtime_format": self.runtime_format,
            "quantization": self.quantization,
            "execution_policy": self.execution_policy,
            "backend_options": dict(self.backend_options),
        }


@dataclass(frozen=True)
class PerformanceSummary:
    decode_tokens_per_second: float
    warm_ttft_seconds: float
    evidence: str
    context_tokens: int | None = None
    endurance_minutes: float | None = None
    note: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PerformanceSummary":
        summary = cls(
            decode_tokens_per_second=float(value["decode_tokens_per_second"]),
            warm_ttft_seconds=float(value["warm_ttft_seconds"]),
            evidence=str(value["evidence"]),
            context_tokens=(
                int(value["context_tokens"]) if value.get("context_tokens") else None
            ),
            endurance_minutes=(
                float(value["endurance_minutes"])
                if value.get("endurance_minutes") is not None
                else None
            ),
            note=(str(value["note"]) if value.get("note") else None),
        )
        summary.validate()
        return summary

    def validate(self) -> None:
        if self.decode_tokens_per_second <= 0 or self.warm_ttft_seconds <= 0:
            raise ValueError("performance throughput and TTFT must be positive")
        if not self.evidence:
            raise ValueError("performance evidence is required")
        if self.context_tokens is not None and self.context_tokens <= 0:
            raise ValueError("performance context_tokens must be positive")
        if self.endurance_minutes is not None and self.endurance_minutes <= 0:
            raise ValueError("performance endurance_minutes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "warm_ttft_seconds": self.warm_ttft_seconds,
            "context_tokens": self.context_tokens,
            "endurance_minutes": self.endurance_minutes,
            "evidence": self.evidence,
            "note": self.note,
        }


@dataclass(frozen=True)
class ModelRecipe:
    schema_version: str
    recipe_version: str
    slug: str
    name: str
    model: str
    parameters: str
    intended_tier: str
    status: str
    implementation: str
    portfolio_role: str
    deployment: DeploymentRecipe
    profile: str
    description: str
    runtime_artifact: RuntimeArtifact | None = None
    revision: str | None = None
    source_bytes: int | None = None
    prepared_bytes: int | None = None
    minimum_free_bytes: int | None = None
    runtime_memory: dict[str, int] | None = None
    performance: PerformanceSummary | None = None
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw_value: Mapping[str, Any]) -> "ModelRecipe":
        value = dict(raw_value)
        schema_version = str(value.get("schema_version", ""))
        if schema_version == "1.0":
            from sparklab.backends import get_backend

            value = get_backend("native").migrate_v1_recipe(value)
            schema_version = str(value["schema_version"])
        if schema_version != RECIPE_SCHEMA_VERSION:
            raise ValueError(f"unsupported recipe schema {schema_version!r}")
        recipe = cls(
            schema_version=schema_version,
            recipe_version=str(value["recipe_version"]),
            slug=str(value["slug"]),
            name=str(value["name"]),
            model=str(value["model"]),
            parameters=str(value["parameters"]),
            intended_tier=str(value["intended_tier"]),
            status=str(value["status"]),
            implementation=str(value["implementation"]),
            portfolio_role=str(value.get("portfolio_role", "primary")),
            deployment=DeploymentRecipe.from_dict(value["deployment"]),
            profile=str(value["profile"]),
            description=str(value["description"]),
            runtime_artifact=(
                RuntimeArtifact.from_dict(value["runtime_artifact"])
                if value.get("runtime_artifact")
                else None
            ),
            revision=(str(value["revision"]) if value.get("revision") else None),
            source_bytes=(
                int(value["source_bytes"]) if value.get("source_bytes") else None
            ),
            prepared_bytes=(
                int(value["prepared_bytes"]) if value.get("prepared_bytes") else None
            ),
            minimum_free_bytes=(
                int(value["minimum_free_bytes"])
                if value.get("minimum_free_bytes")
                else None
            ),
            runtime_memory=(
                {str(key): int(size) for key, size in value["runtime_memory"].items()}
                if value.get("runtime_memory")
                else None
            ),
            performance=(
                PerformanceSummary.from_dict(value["performance"])
                if value.get("performance")
                else None
            ),
            evidence=tuple(str(item) for item in value.get("evidence", ())),
            limitations=tuple(str(item) for item in value.get("limitations", ())),
        )
        recipe.validate()
        return recipe

    @property
    def checkpoint_format(self) -> str:
        """Compatibility name for the backend-qualified runtime artifact format."""
        return self.deployment.runtime_format

    @property
    def expert_quantization(self) -> str | None:
        return self.deployment.quantization

    @property
    def execution_policy(self) -> str:
        return self.deployment.execution_policy

    @property
    def backend(self) -> str:
        return self.deployment.backend

    def validate(self) -> None:
        if self.schema_version != RECIPE_SCHEMA_VERSION:
            raise ValueError(f"unsupported recipe schema {self.schema_version!r}")
        if self.intended_tier not in TIERS:
            raise ValueError(f"invalid tier {self.intended_tier!r} for {self.slug}")
        if self.status not in STATUSES:
            raise ValueError(f"invalid status {self.status!r} for {self.slug}")
        if self.portfolio_role not in PORTFOLIO_ROLES:
            raise ValueError(
                f"invalid portfolio role {self.portfolio_role!r} for {self.slug}"
            )
        if (
            not self.slug
            or not self.model
            or not self.parameters
            or not self.recipe_version
        ):
            raise ValueError(
                "recipe slug, model, parameters, and recipe_version are required"
            )
        self.deployment.validate()
        for name in ("source_bytes", "prepared_bytes", "minimum_free_bytes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive for {self.slug}")
        if self.revision is not None and len(self.revision) != 40:
            raise ValueError(
                f"recipe revision must be a full 40-character commit for {self.slug}"
            )
        if self.runtime_artifact is not None:
            self.runtime_artifact.validate()
        if self.runtime_memory is not None:
            allowed = {
                "resident_weights_bytes",
                "expert_cache_bytes",
                "kv_cache_bytes",
                "workspace_bytes",
                "transient_bytes",
                "total_bytes",
            }
            unknown = set(self.runtime_memory) - allowed
            if unknown:
                raise ValueError(
                    f"unknown runtime-memory keys for {self.slug}: {sorted(unknown)}"
                )
            if any(value < 0 for value in self.runtime_memory.values()):
                raise ValueError(
                    f"runtime-memory values must be non-negative for {self.slug}"
                )
        if self.performance is not None:
            self.performance.validate()
            if self.performance.evidence not in self.evidence:
                raise ValueError(
                    f"performance evidence {self.performance.evidence!r} is not cited "
                    f"by {self.slug}"
                )
        if self.status in {"preview", "certified"} and not self.evidence:
            raise ValueError(
                f"{self.status} recipe {self.slug} must cite versioned evidence"
            )
        if self.status == "certified" and self.runtime_memory is None:
            raise ValueError(
                f"certified recipe {self.slug} must include a measured runtime-memory budget"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe_version": self.recipe_version,
            "slug": self.slug,
            "name": self.name,
            "model": self.model,
            "parameters": self.parameters,
            "intended_tier": self.intended_tier,
            "status": self.status,
            "implementation": self.implementation,
            "portfolio_role": self.portfolio_role,
            "deployment": self.deployment.to_dict(),
            "profile": self.profile,
            "description": self.description,
            "runtime_artifact": (
                self.runtime_artifact.to_dict()
                if self.runtime_artifact is not None
                else None
            ),
            "revision": self.revision,
            "source_bytes": self.source_bytes,
            "prepared_bytes": self.prepared_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
            "runtime_memory": self.runtime_memory,
            "performance": (
                self.performance.to_dict() if self.performance is not None else None
            ),
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
        }


def load_catalog() -> tuple[ModelRecipe, ...]:
    root = files("sparklab.recipes")
    recipes = []
    for resource in sorted(root.iterdir(), key=lambda item: item.name):
        if resource.name.endswith(".json"):
            recipes.append(
                ModelRecipe.from_dict(json.loads(resource.read_text(encoding="utf-8")))
            )
    slugs = [recipe.slug for recipe in recipes]
    if len(slugs) != len(set(slugs)):
        raise ValueError("duplicate SparkLab model recipe slug")
    order = {tier: index for index, tier in enumerate(TIERS)}
    return tuple(
        sorted(recipes, key=lambda item: (order[item.intended_tier], item.slug))
    )


def select_recipes(
    recipes: Iterable[ModelRecipe],
    *,
    tier: str | None = None,
    status: str | None = None,
    portfolio_role: str | None = None,
) -> tuple[ModelRecipe, ...]:
    return tuple(
        recipe
        for recipe in recipes
        if (tier is None or recipe.intended_tier == tier)
        and (status is None or recipe.status == status)
        and (portfolio_role is None or recipe.portfolio_role == portfolio_role)
    )


def get_recipe(slug: str) -> ModelRecipe:
    for recipe in load_catalog():
        if recipe.slug == slug:
            return recipe
    raise KeyError(slug)


__all__ = [
    "DeploymentRecipe",
    "ModelRecipe",
    "PORTFOLIO_ROLES",
    "PerformanceSummary",
    "RECIPE_SCHEMA_VERSION",
    "RuntimeArtifact",
    "STATUSES",
    "TIERS",
    "get_recipe",
    "load_catalog",
    "select_recipes",
]
