from __future__ import annotations

from sparklab.catalog import get_recipe
from sparklab.certification import evaluate_tier


def _evidence(recipe):
    return {
        "schema_version": "1.0",
        "result_id": "GB10-QWEN-001",
        "recipe": {
            "slug": recipe.slug,
            "recipe_version": recipe.recipe_version,
            "revision": recipe.revision,
        },
        "engine": {"git_revision": "abc123"},
        "checkpoint": {"full_model": True, "fingerprint": "deadbeef"},
        "metrics": {"decode_tokens_per_second": 21.0, "warm_ttft_seconds": 4.5},
        "validation": {
            "output_correctness": True,
            "reasoning_parser": True,
            "tool_parser": True,
            "coding_agent_task": True,
            "memory_bounded": True,
            "context_tokens": 65_536,
            "nvme_behavior": "no_stalls",
        },
        "stability": {
            "duration_minutes": 60,
            "oom_count": 0,
            "swap_growth_bytes": 0,
            "parser_failures": 0,
        },
    }


def test_complete_fast_evidence_passes_fast_and_research():
    recipe = get_recipe("qwen3.8-flash-next")
    evidence = _evidence(recipe)
    assert evaluate_tier(recipe, evidence, "fast").passed
    assert evaluate_tier(recipe, evidence, "research").passed


def test_frontier_requires_64k_even_when_fast_context_passes():
    recipe = get_recipe("qwen3.8-flash-next")
    evidence = _evidence(recipe)
    evidence["validation"]["context_tokens"] = 32_768
    assert evaluate_tier(recipe, evidence, "fast").passed
    frontier = evaluate_tier(recipe, evidence, "frontier")
    assert not frontier.passed
    assert any("context_tokens" in reason for reason in frontier.reasons)


def test_gate_rejects_reduced_model_and_recipe_mismatch():
    recipe = get_recipe("qwen3.8-flash-next")
    evidence = _evidence(recipe)
    evidence["checkpoint"]["full_model"] = False
    evidence["recipe"]["recipe_version"] = "0.1.0"
    result = evaluate_tier(recipe, evidence, "research")
    assert not result.passed
    assert any("full_model" in reason for reason in result.reasons)
    assert any("recipe_version mismatch" in reason for reason in result.reasons)


def test_frontier_allows_bounded_nvme_but_fast_does_not():
    recipe = get_recipe("qwen3.8-flash-next")
    evidence = _evidence(recipe)
    evidence["validation"]["nvme_behavior"] = "bounded"
    assert evaluate_tier(recipe, evidence, "frontier").passed
    assert not evaluate_tier(recipe, evidence, "fast").passed
