"""Stable SparkLab state and artifact paths."""

from __future__ import annotations

import os
from pathlib import Path

from sparklab.catalog import ModelRecipe


def state_root(override: str | os.PathLike[str] | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    configured = os.getenv("SPARKLAB_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".sparklab"


def source_path(recipe: ModelRecipe, root: str | os.PathLike[str] | None = None) -> Path:
    revision = (recipe.revision or "unversioned")[:12]
    return state_root(root) / "models" / recipe.slug / "source" / revision


def prepared_path(recipe: ModelRecipe, root: str | os.PathLike[str] | None = None) -> Path:
    return state_root(root) / "models" / recipe.slug / "prepared" / recipe.recipe_version


def manifest_path(recipe: ModelRecipe, root: str | os.PathLike[str] | None = None) -> Path:
    return state_root(root) / "models" / recipe.slug / "manifest.json"


__all__ = ["manifest_path", "prepared_path", "source_path", "state_root"]
