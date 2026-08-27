"""Resolve and validate a recipe-backed Spark Lab server invocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from freetoken.checkpoint import is_ftw_checkpoint
from freetoken.platform.gb10 import GB10Snapshot
from sparklab.acquire import read_manifest
from sparklab.catalog import ModelRecipe
from sparklab.paths import prepared_path, source_path
from sparklab.planner import RuntimePlan, plan_runtime


class RuntimePlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecipeInvocation:
    recipe: str
    checkpoint: str
    arguments: tuple[str, ...]
    memory: RuntimePlan

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["memory"] = self.memory.to_dict()
        return result


def resolve_checkpoint(recipe: ModelRecipe, root: str | None = None) -> Path:
    manifest = read_manifest(recipe, root)
    candidates: list[Path] = []
    if manifest:
        if manifest.get("prepared_path"):
            candidates.append(Path(manifest["prepared_path"]))
        if manifest.get("source_path"):
            candidates.append(Path(manifest["source_path"]))
    candidates.extend((prepared_path(recipe, root), source_path(recipe, root)))
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    raise RuntimePlanError(
        f"no acquired checkpoint for {recipe.slug}; run `sparklab pull {recipe.slug}` first"
    )


def plan_invocation(
    recipe: ModelRecipe,
    snapshot: GB10Snapshot,
    *,
    root: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> RecipeInvocation:
    checkpoint = resolve_checkpoint(recipe, root)
    memory = plan_runtime(recipe, snapshot)
    if not memory.ready:
        raise RuntimePlanError("; ".join(memory.reasons))
    arguments = ["--model", str(checkpoint), "--served-model-name", recipe.model]
    arguments.extend(recipe.runtime_args)
    arguments.extend(extra_args)
    if recipe.execution_policy == "nvme-moe" and not is_ftw_checkpoint(str(checkpoint)):
        raise RuntimePlanError(
            f"{recipe.slug} requires a prepared FTW checkpoint for NVMe-backed experts"
        )
    return RecipeInvocation(
        recipe=recipe.slug,
        checkpoint=str(checkpoint),
        arguments=tuple(arguments),
        memory=memory,
    )


__all__ = [
    "RecipeInvocation",
    "RuntimePlanError",
    "plan_invocation",
    "resolve_checkpoint",
]
