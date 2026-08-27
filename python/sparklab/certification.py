"""Fail-closed evaluation of Spark Lab GB10 model-certification evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sparklab.catalog import ModelRecipe, TIERS


@dataclass(frozen=True)
class TierGate:
    tier: str
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _at_least(value: Any, threshold: float, label: str, reasons: list[str]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < threshold:
        reasons.append(f"{label} must be >= {threshold}, found {value!r}")


def _at_most(value: Any, threshold: float, label: str, reasons: list[str]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value > threshold:
        reasons.append(f"{label} must be <= {threshold}, found {value!r}")


def evaluate_tier(recipe: ModelRecipe, evidence: dict[str, Any], tier: str) -> TierGate:
    """Evaluate one complete-checkpoint evidence record against a public tier gate."""
    if tier not in TIERS:
        raise ValueError(f"unknown Spark Lab tier: {tier!r}")
    reasons: list[str] = []
    if evidence.get("schema_version") != "1.0":
        reasons.append("evidence schema_version must be '1.0'")
    result_id = evidence.get("result_id")
    if not isinstance(result_id, str) or not result_id:
        reasons.append("evidence must have a versioned result_id")

    recorded_recipe = evidence.get("recipe") or {}
    for key, expected in (
        ("slug", recipe.slug),
        ("recipe_version", recipe.recipe_version),
        ("revision", recipe.revision),
    ):
        if recorded_recipe.get(key) != expected:
            reasons.append(
                f"recipe {key} mismatch: expected {expected!r}, "
                f"found {recorded_recipe.get(key)!r}"
            )
    engine = evidence.get("engine") or {}
    if not isinstance(engine.get("git_revision"), str) or not engine["git_revision"]:
        reasons.append("evidence must record the engine git_revision")
    checkpoint = evidence.get("checkpoint") or {}
    if checkpoint.get("full_model") is not True:
        reasons.append("checkpoint.full_model must be true")
    if not isinstance(checkpoint.get("fingerprint"), str) or not checkpoint["fingerprint"]:
        reasons.append("checkpoint fingerprint is required")

    validation = evidence.get("validation") or {}
    for key in (
        "output_correctness",
        "reasoning_parser",
        "tool_parser",
        "coding_agent_task",
        "memory_bounded",
    ):
        if validation.get(key) is not True:
            reasons.append(f"validation.{key} must be true")

    stability = evidence.get("stability") or {}
    _at_most(stability.get("swap_growth_bytes"), 0, "stability.swap_growth_bytes", reasons)

    # Research is the admission floor: complete, correct, bounded execution with
    # no swap growth, but no latency, context, or endurance promise.
    if tier == "research":
        return TierGate(tier=tier, passed=not reasons, reasons=tuple(reasons))

    metrics = evidence.get("metrics") or {}
    _at_least(stability.get("duration_minutes"), 60, "stability.duration_minutes", reasons)
    _at_most(stability.get("oom_count"), 0, "stability.oom_count", reasons)
    _at_most(stability.get("parser_failures"), 0, "stability.parser_failures", reasons)

    if tier == "fast":
        _at_least(metrics.get("decode_tokens_per_second"), 20, "decode_tokens_per_second", reasons)
        _at_most(metrics.get("warm_ttft_seconds"), 5, "warm_ttft_seconds", reasons)
        _at_least(validation.get("context_tokens"), 32_768, "validation.context_tokens", reasons)
        if validation.get("nvme_behavior") != "no_stalls":
            reasons.append("validation.nvme_behavior must be 'no_stalls' for Fast")
    else:
        _at_least(metrics.get("decode_tokens_per_second"), 5, "decode_tokens_per_second", reasons)
        _at_most(metrics.get("warm_ttft_seconds"), 20, "warm_ttft_seconds", reasons)
        _at_least(validation.get("context_tokens"), 65_536, "validation.context_tokens", reasons)
        if validation.get("nvme_behavior") not in {"no_stalls", "bounded"}:
            reasons.append("validation.nvme_behavior must be 'bounded' or 'no_stalls' for Frontier")

    return TierGate(tier=tier, passed=not reasons, reasons=tuple(reasons))


def evaluate_all_tiers(recipe: ModelRecipe, evidence: dict[str, Any]) -> tuple[TierGate, ...]:
    return tuple(evaluate_tier(recipe, evidence, tier) for tier in TIERS)


__all__ = ["TierGate", "evaluate_all_tiers", "evaluate_tier"]
