"""Versioned Spark Lab model-recipe catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Iterable

TIERS = ("fast", "frontier", "research")
STATUSES = ("certified", "preview", "experimental")


@dataclass(frozen=True)
class ModelRecipe:
    schema_version: str
    recipe_version: str
    slug: str
    name: str
    model: str
    intended_tier: str
    status: str
    implementation: str
    checkpoint_format: str
    execution_policy: str
    profile: str
    description: str
    revision: str | None = None
    source_bytes: int | None = None
    prepared_bytes: int | None = None
    minimum_free_bytes: int | None = None
    runtime_args: tuple[str, ...] = ()
    runtime_memory: dict[str, int] | None = None
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict) -> "ModelRecipe":
        recipe = cls(
            schema_version=str(value["schema_version"]),
            recipe_version=str(value["recipe_version"]),
            slug=str(value["slug"]),
            name=str(value["name"]),
            model=str(value["model"]),
            intended_tier=str(value["intended_tier"]),
            status=str(value["status"]),
            implementation=str(value["implementation"]),
            checkpoint_format=str(value["checkpoint_format"]),
            execution_policy=str(value["execution_policy"]),
            profile=str(value["profile"]),
            description=str(value["description"]),
            revision=(str(value["revision"]) if value.get("revision") else None),
            source_bytes=(int(value["source_bytes"]) if value.get("source_bytes") else None),
            prepared_bytes=(
                int(value["prepared_bytes"]) if value.get("prepared_bytes") else None
            ),
            minimum_free_bytes=(
                int(value["minimum_free_bytes"])
                if value.get("minimum_free_bytes")
                else None
            ),
            runtime_args=tuple(str(item) for item in value.get("runtime_args", ())),
            runtime_memory=(
                {str(key): int(size) for key, size in value["runtime_memory"].items()}
                if value.get("runtime_memory")
                else None
            ),
            evidence=tuple(value.get("evidence", ())),
            limitations=tuple(value.get("limitations", ())),
        )
        recipe.validate()
        return recipe

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported recipe schema {self.schema_version!r}")
        if self.intended_tier not in TIERS:
            raise ValueError(f"invalid tier {self.intended_tier!r} for {self.slug}")
        if self.status not in STATUSES:
            raise ValueError(f"invalid status {self.status!r} for {self.slug}")
        if not self.slug or not self.model or not self.recipe_version:
            raise ValueError("recipe slug, model, and recipe_version are required")
        for name in ("source_bytes", "prepared_bytes", "minimum_free_bytes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive for {self.slug}")
        if self.revision is not None and len(self.revision) != 40:
            raise ValueError(f"recipe revision must be a full 40-character commit for {self.slug}")
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
                raise ValueError(f"unknown runtime-memory keys for {self.slug}: {sorted(unknown)}")
            if any(value < 0 for value in self.runtime_memory.values()):
                raise ValueError(f"runtime-memory values must be non-negative for {self.slug}")
        if self.status in {"preview", "certified"} and not self.evidence:
            raise ValueError(f"{self.status} recipe {self.slug} must cite versioned evidence")
        if self.status == "certified" and self.runtime_memory is None:
            raise ValueError(
                f"certified recipe {self.slug} must include a measured runtime-memory budget"
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "recipe_version": self.recipe_version,
            "slug": self.slug,
            "name": self.name,
            "model": self.model,
            "intended_tier": self.intended_tier,
            "status": self.status,
            "implementation": self.implementation,
            "checkpoint_format": self.checkpoint_format,
            "execution_policy": self.execution_policy,
            "profile": self.profile,
            "description": self.description,
            "revision": self.revision,
            "source_bytes": self.source_bytes,
            "prepared_bytes": self.prepared_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
            "runtime_args": list(self.runtime_args),
            "runtime_memory": self.runtime_memory,
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
        }
        return result


def load_catalog() -> tuple[ModelRecipe, ...]:
    root = files("sparklab.recipes")
    recipes = []
    for resource in sorted(root.iterdir(), key=lambda item: item.name):
        if resource.name.endswith(".json"):
            recipes.append(ModelRecipe.from_dict(json.loads(resource.read_text(encoding="utf-8"))))
    slugs = [recipe.slug for recipe in recipes]
    if len(slugs) != len(set(slugs)):
        raise ValueError("duplicate Spark Lab model recipe slug")
    order = {tier: index for index, tier in enumerate(TIERS)}
    return tuple(sorted(recipes, key=lambda item: (order[item.intended_tier], item.slug)))


def select_recipes(
    recipes: Iterable[ModelRecipe],
    *,
    tier: str | None = None,
    status: str | None = None,
) -> tuple[ModelRecipe, ...]:
    return tuple(
        recipe
        for recipe in recipes
        if (tier is None or recipe.intended_tier == tier)
        and (status is None or recipe.status == status)
    )


def get_recipe(slug: str) -> ModelRecipe:
    for recipe in load_catalog():
        if recipe.slug == slug:
            return recipe
    raise KeyError(slug)


__all__ = [
    "ModelRecipe",
    "STATUSES",
    "TIERS",
    "get_recipe",
    "load_catalog",
    "select_recipes",
]
