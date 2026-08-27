from __future__ import annotations

import json
from pathlib import Path

from sparklab.catalog import get_recipe, load_catalog, select_recipes


def test_catalog_has_unique_versioned_three_tier_recipes():
    recipes = load_catalog()
    assert {recipe.intended_tier for recipe in recipes} == {"fast", "frontier", "research"}
    assert len({recipe.slug for recipe in recipes}) == len(recipes)
    assert all(recipe.schema_version == "1.0" and recipe.recipe_version for recipe in recipes)
    assert not any(recipe.status == "certified" for recipe in recipes)


def test_catalog_contains_requested_portfolio_without_overclaiming_status():
    assert get_recipe("kimi-k3").model == "moonshotai/Kimi-K3"
    assert get_recipe("glm-5.3-flash").model == "zai-org/GLM-5.3-Flash"
    assert get_recipe("qwen3.8-flash-next").model == "Qwen/Qwen3.8-Flash-Next"
    assert get_recipe("deepseek-v4").status == "preview"
    assert get_recipe("kimi-k3").status == "experimental"
    assert {item.slug for item in select_recipes(load_catalog(), tier="fast")} == {
        "qwen3.6-35b-a3b",
        "qwen3.8-flash-next",
    }


def test_deepseek_recipe_points_to_checked_in_baseline():
    recipe = get_recipe("deepseek-v4")
    assert recipe.evidence == ("GB10-BASELINE-001",)
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-BASELINE-001.json").read_text()
    )
    assert result["result_id"] == recipe.evidence[0]
    assert result["status"] == "measured"
    assert result["metrics"]["decode_tokens_per_second"] == 9.217
    assert result["validation"]["output_hash"] == "fbf178b2bde5"
