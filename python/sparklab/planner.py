"""Fail-closed GB10 storage and unified-memory planning for model recipes."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sparklab.catalog import ModelRecipe
from sparklab.paths import prepared_path, source_path, state_root
from sparklab.platform import GB10_SAFETY_RESERVE_BYTES, GB10Snapshot


@dataclass(frozen=True)
class ArtifactPlan:
    root: str
    source_path: str
    prepared_path: str
    source_bytes: int | None
    prepared_bytes: int | None
    required_bytes: int | None
    free_bytes: int
    shortfall_bytes: int | None
    ready: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePlan:
    physical_bytes: int
    available_bytes: int
    safety_reserve_bytes: int
    usable_bytes: int
    required_bytes: int | None
    headroom_bytes: int | None
    swap_used_bytes: int
    components: dict[str, int]
    ready: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _existing_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except FileNotFoundError:
            continue
    return total


def plan_artifacts(
    recipe: ModelRecipe,
    *,
    root: str | None = None,
    include_prepared: bool = False,
) -> ArtifactPlan:
    base = state_root(root)
    base.mkdir(parents=True, exist_ok=True)
    source = source_path(recipe, base)
    prepared = prepared_path(recipe, base)
    free = int(shutil.disk_usage(base).free)
    reasons: list[str] = []
    if recipe.source_bytes is None:
        required = None
        reasons.append("recipe does not declare the pinned source artifact size")
    else:
        required = max(0, recipe.source_bytes - _existing_bytes(source))
        if include_prepared:
            if recipe.prepared_bytes is None:
                required = None
                reasons.append("recipe does not declare the prepared artifact size")
            else:
                required += max(0, recipe.prepared_bytes - _existing_bytes(prepared))
        if required is not None and recipe.minimum_free_bytes is not None:
            # minimum_free_bytes includes source + preparation scratch for a clean pull.
            # On a resume, retain only its safety margin above declared artifacts.
            declared = recipe.source_bytes + (recipe.prepared_bytes or 0)
            safety_margin = max(0, recipe.minimum_free_bytes - declared)
            required += safety_margin
        if required is not None and free < required:
            reasons.append(f"storage shortfall: need {required} bytes, have {free} bytes free")
    shortfall = None if required is None else max(0, required - free)
    return ArtifactPlan(
        root=str(base),
        source_path=str(source),
        prepared_path=str(prepared),
        source_bytes=recipe.source_bytes,
        prepared_bytes=recipe.prepared_bytes,
        required_bytes=required,
        free_bytes=free,
        shortfall_bytes=shortfall,
        ready=required is not None and shortfall == 0,
        reasons=tuple(reasons),
    )


def plan_runtime(recipe: ModelRecipe, snapshot: GB10Snapshot) -> RuntimePlan:
    swap_used = max(0, snapshot.swap_total_bytes - snapshot.swap_free_bytes)
    usable = max(
        0,
        min(snapshot.memory_total_bytes, snapshot.memory_available_bytes)
        - GB10_SAFETY_RESERVE_BYTES,
    )
    components = dict(recipe.runtime_memory or {})
    reasons: list[str] = []
    required: int | None
    if not components:
        required = None
        reasons.append("recipe has no measured GB10 runtime-memory budget")
    elif "total_bytes" in components:
        required = components["total_bytes"]
    else:
        required = sum(components.values())
    if swap_used:
        reasons.append(f"swap is in use ({swap_used} bytes); certified recipes require zero")
    if required is not None and required > usable:
        reasons.append(f"runtime memory shortfall: need {required} bytes, have {usable} usable")
    headroom = None if required is None else usable - required
    return RuntimePlan(
        physical_bytes=snapshot.memory_total_bytes,
        available_bytes=snapshot.memory_available_bytes,
        safety_reserve_bytes=GB10_SAFETY_RESERVE_BYTES,
        usable_bytes=usable,
        required_bytes=required,
        headroom_bytes=headroom,
        swap_used_bytes=swap_used,
        components=components,
        ready=required is not None and not reasons,
        reasons=tuple(reasons),
    )


__all__ = ["ArtifactPlan", "RuntimePlan", "plan_artifacts", "plan_runtime"]
