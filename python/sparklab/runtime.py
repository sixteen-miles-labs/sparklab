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
    if recipe.runtime_artifact is None:
        candidates.append(prepared_path(recipe, root))
    candidates.append(source_path(recipe, root))
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


def _hosted_runtime_record(
    recipe: ModelRecipe, checkpoint: Path, root: str | None
) -> dict[str, Any] | None:
    """Return provenance only when this path was acquired from a hosted runtime repo."""
    try:
        manifest = read_manifest(recipe, root)
    except AcquisitionError as exc:
        raise RuntimePlanError(str(exc)) from exc
    if not manifest:
        return None
    artifacts = manifest.get("artifacts")
    runtime = artifacts.get("runtime") if isinstance(artifacts, dict) else None
    if not isinstance(runtime, dict) or not runtime.get("repository"):
        return None
    path = runtime.get("path")
    if not path or Path(str(path)).expanduser().resolve() != checkpoint:
        return None
    return runtime


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
        validation = backend.validate_artifact(checkpoint, recipe.deployment)
        hosted = recipe.runtime_artifact
        record = _hosted_runtime_record(recipe, checkpoint, root)
        if hosted is not None and record is not None:
            if (
                record.get("repository") != hosted.repo_id
                or record.get("revision") != hosted.revision
            ):
                raise RuntimePlanError(
                    "runtime artifact provenance does not match the recipe's pinned "
                    "repository and revision"
                )
            if validation.fingerprint != hosted.fingerprint:
                raise RuntimePlanError(
                    "runtime artifact fingerprint mismatch: "
                    f"expected {hosted.fingerprint}, found {validation.fingerprint}"
                )
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
