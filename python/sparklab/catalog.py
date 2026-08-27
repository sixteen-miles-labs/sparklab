"""Versioned Spark Lab model-recipe catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Iterable

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

    def to_dict(self) -> dict:
        return {
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
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
        }


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
