"""Resolve and validate backend-qualified Spark Lab runtime invocations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sparklab.acquire import AcquisitionError, read_manifest
from sparklab.backends import BackendError, BackendLaunchPlan, RuntimeRequest, get_backend
from sparklab.catalog import ModelRecipe
from sparklab.paths import prepared_path, source_path
from sparklab.planner import RuntimePlan, plan_runtime
from sparklab.platform import GB10Snapshot


class RuntimePlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecipeInvocation:
    plan: BackendLaunchPlan
    memory: RuntimePlan

    @property
    def recipe(self) -> str:
        return self.plan.recipe

    @property
    def backend(self) -> str:
        return self.plan.backend

    @property
    def checkpoint(self) -> str:
        return self.plan.checkpoint

    @property
    def arguments(self) -> tuple[str, ...]:
        return self.plan.arguments

    def to_dict(self) -> dict[str, Any]:
        result = self.plan.to_dict()
        result["memory"] = self.memory.to_dict()
        return result


def _manifest_candidates(manifest: dict[str, Any] | None) -> list[Path]:
    if not manifest:
        return []
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return []
    candidates: list[Path] = []
    for role in ("runtime", "source"):
        artifact = artifacts.get(role)
        if isinstance(artifact, dict) and artifact.get("path"):
            candidates.append(Path(str(artifact["path"])))
    return candidates


def resolve_checkpoint(recipe: ModelRecipe, root: str | None = None) -> Path:
    """Resolve a runtime artifact accepted by the recipe-selected backend."""
    try:
        manifest = read_manifest(recipe, root)
    except AcquisitionError as exc:
        raise RuntimePlanError(str(exc)) from exc
    candidates = _manifest_candidates(manifest)
    candidates.extend((prepared_path(recipe, root), source_path(recipe, root)))
    backend = get_backend(recipe.backend)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if backend.accepts_artifact(resolved, recipe.deployment):
            return resolved
    raise RuntimePlanError(
        f"no {recipe.deployment.runtime_format} artifact accepted by backend "
        f"{recipe.backend!r} for {recipe.slug}; run "
        f"`sparklab pull {recipe.slug} --prepare` first"
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
    backend = get_backend(recipe.backend)
    try:
        backend.validate_artifact(checkpoint, recipe.deployment)
        plan = backend.build_launch_plan(
            RuntimeRequest(
                recipe=recipe.slug,
                recipe_version=recipe.recipe_version,
                model=recipe.model,
                checkpoint=checkpoint,
                deployment=recipe.deployment,
                extra_args=extra_args,
            )
        )
    except BackendError as exc:
        raise RuntimePlanError(str(exc)) from exc
    return RecipeInvocation(plan=plan, memory=memory)


__all__ = [
    "RecipeInvocation",
    "RuntimePlanError",
    "plan_invocation",
    "resolve_checkpoint",
]
