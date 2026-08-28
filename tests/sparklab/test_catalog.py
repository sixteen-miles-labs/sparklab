from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparklab.catalog import (
    PerformanceSummary,
    RuntimeArtifact,
    get_recipe,
    load_catalog,
    select_recipes,
)


def test_catalog_has_unique_versioned_three_tier_recipes():
    recipes = load_catalog()
    assert {recipe.intended_tier for recipe in recipes} == {"fast", "frontier", "research"}
    assert len({recipe.slug for recipe in recipes}) == len(recipes)
    assert all(recipe.schema_version == "2.0" and recipe.recipe_version for recipe in recipes)
    assert all(recipe.parameters for recipe in recipes)
    assert not {recipe.slug for recipe in recipes if recipe.status == "certified"}


def test_catalog_contains_requested_portfolio_without_overclaiming_status():
    assert get_recipe("kimi-k3").model == "nvidia/Kimi-K3-NVFP4"
    assert get_recipe("glm-5.3-flash").model == "zai-org/GLM-5.3-Flash"
    assert get_recipe("qwen3.8-flash-next").model == "Qwen/Qwen3.8-Flash-Next-FP8"
    assert get_recipe("qwen3.6-35b-a3b").name == "Qwen3.6 35B A3B"
    assert get_recipe("glm-5.2").name == "GLM-5.2"
    assert get_recipe("glm-5.2").intended_tier == "research"
    assert {
        recipe.slug: recipe.parameters for recipe in load_catalog()
    } == {
        "qwen3.6-35b-a3b": "35B total / 3B active",
        "deepseek-v4": "284B total / 13B active",
        "glm-5.3-flash": "320B total / 18B active",
        "qwen3.8-flash-next": "125B LM + 55B aux / 6B active",
        "glm-5.2": "753B total / 40B active",
        "kimi-k3": "2.8T total / 16 of 896 experts",
    }
    assert get_recipe("deepseek-v4").status == "preview"
    assert get_recipe("kimi-k3").status == "experimental"
    assert {item.slug for item in select_recipes(load_catalog(), tier="fast")} == {
        "qwen3.6-35b-a3b",
    }
    qwen = get_recipe("qwen3.8-flash-next")
    assert qwen.recipe_version == "0.4.0"
    assert qwen.intended_tier == "frontier"
    assert qwen.status == "experimental"
    assert qwen.evidence == ("GB10-QWEN38-FP8-001",)
    assert qwen.backend == "native"
    assert qwen.deployment.runtime_format == "ftw-fp8"
    assert qwen.deployment.backend_options["attention_backend"] == "qsa"
    assert qwen.deployment.quantization == "fp8"
    assert "convert_expert_quantization" not in qwen.deployment.backend_options
    assert qwen.performance.decode_tokens_per_second == pytest.approx(4.987115832105288)
    assert qwen.performance.warm_ttft_seconds == pytest.approx(0.5798669513314962)
    assert qwen.deployment.backend_options["moe_host_cache_gb"] == 3
    kimi = get_recipe("kimi-k3")
    assert kimi.recipe_version == "0.2.0"
    assert kimi.deployment.runtime_format == "ftw-nvfp4"
    assert kimi.deployment.quantization == "nvfp4"
    assert kimi.performance is None
    glm52 = get_recipe("glm-5.2")
    assert glm52.recipe_version == "0.2.0"
    assert glm52.performance.decode_tokens_per_second == pytest.approx(0.802)
    assert glm52.performance.warm_ttft_seconds == pytest.approx(2.57)
    assert get_recipe("deepseek-v4").performance.warm_ttft_seconds == 14.045
    qwen36 = get_recipe("qwen3.6-35b-a3b")
    assert qwen36.performance.decode_tokens_per_second == pytest.approx(67.46036500779391)
    assert qwen36.performance.warm_ttft_seconds == pytest.approx(0.31954073812812567)
    assert qwen36.evidence == ("GB10-QWEN36-FAST-001",)
    primary = select_recipes(load_catalog(), portfolio_role="primary")
    assert {(item.intended_tier, item.slug) for item in primary} == {
        ("fast", "qwen3.6-35b-a3b"),
        ("frontier", "deepseek-v4"),
        ("frontier", "glm-5.3-flash"),
        ("frontier", "qwen3.8-flash-next"),
        ("research", "kimi-k3"),
    }
    assert {
        item.slug for item in select_recipes(load_catalog(), portfolio_role="fallback")
    } == {"glm-5.2"}


def test_next_model_recipes_are_immutable_and_capacity_plannable():
    qwen = get_recipe("qwen3.8-flash-next")
    glm = get_recipe("glm-5.3-flash")
    kimi = get_recipe("kimi-k3")
    glm52 = get_recipe("glm-5.2")
    assert qwen.revision == "970c569adaca6b35532111fd6b27351b2baefe50"
    assert glm.revision == "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a"
    assert kimi.revision == "f8c5234a0a880bcc6cbf779a315e7ee2f405b812"
    assert glm52.revision == "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
    assert qwen.source_bytes == 185553536918
    assert qwen.expert_quantization == "fp8"
    assert glm.source_bytes == 328337455672
    assert kimi.source_bytes == 1610038482254
    assert glm52.source_bytes == 464874323992
    assert qwen.execution_policy == glm.execution_policy == "nvme-moe"
    assert qwen.minimum_free_bytes > qwen.source_bytes + qwen.prepared_bytes
    assert glm.minimum_free_bytes > glm.source_bytes + glm.prepared_bytes
    assert kimi.minimum_free_bytes > kimi.source_bytes + kimi.prepared_bytes
    assert glm52.minimum_free_bytes > glm52.source_bytes + glm52.prepared_bytes


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


def test_glm52_recipe_points_to_measured_failed_research_evidence():
    from sparklab.certification import evaluate_tier

    recipe = get_recipe("glm-5.2")
    assert recipe.evidence == ("GB10-GLM52-RESEARCH-001",)
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-GLM52-RESEARCH-001.json").read_text()
    )
    assert result["result_id"] == recipe.evidence[0]
    assert result["status"] == "measured"
    assert result["metrics"]["decode_tokens_per_second"] == pytest.approx(0.802)
    assert result["metrics"]["warm_ttft_seconds"] == pytest.approx(2.57)
    assert result["stability"]["swap_growth_bytes"] == 696320
    evaluation = evaluate_tier(recipe, result, "research")
    assert not evaluation.passed
    assert any("reasoning_parser" in reason for reason in evaluation.reasons)


def test_historical_qwen_nvfp4_evidence_does_not_transfer_to_fp8_recipe():
    from sparklab.certification import evaluate_tier

    recipe = get_recipe("qwen3.8-flash-next")
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-QWEN38-FRONTIER-001.json").read_text()
    )
    assert result["result_id"] == "GB10-QWEN38-FRONTIER-001"
    evaluation = evaluate_tier(recipe, result, "frontier")
    assert not evaluation.passed
    assert any("recipe_version mismatch" in reason for reason in evaluation.reasons)


def test_qwen_fp8_recipe_points_to_measured_failed_frontier_evidence():
    from sparklab.certification import evaluate_tier

    recipe = get_recipe("qwen3.8-flash-next")
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-QWEN38-FP8-001.json").read_text()
    )
    assert result["result_id"] == recipe.evidence[0]
    assert result["metrics"]["decode_tokens_per_second"] == pytest.approx(4.987115832105288)
    assert result["metrics"]["warm_ttft_seconds"] == pytest.approx(0.5798669513314962)
    evaluation = evaluate_tier(recipe, result, "frontier")
    assert not evaluation.passed
    assert any("must be >= 5" in reason for reason in evaluation.reasons)


def test_preview_and_certified_statuses_fail_closed_without_evidence_or_memory():
    from dataclasses import replace

    base = get_recipe("qwen3.8-flash-next")
    with pytest.raises(ValueError, match="must cite versioned evidence"):
        replace(base, status="preview", evidence=(), performance=None).validate()
    with pytest.raises(ValueError, match="runtime-memory budget"):
        replace(
            base,
            status="certified",
            evidence=("GB10-QWEN-001",),
            runtime_memory=None,
            performance=None,
        ).validate()
    with pytest.raises(ValueError, match="performance evidence"):
        replace(
            base,
            performance=PerformanceSummary(
                decode_tokens_per_second=10,
                warm_ttft_seconds=1,
                evidence="GB10-UNKNOWN",
            ),
        ).validate()


def test_runtime_artifact_round_trips_and_requires_immutable_revision():
    from dataclasses import replace

    base = get_recipe("qwen3.8-flash-next")
    artifact = RuntimeArtifact(
        repo_id="freetoken/qwen-ftw",
        revision="d" * 40,
        bytes=123,
        fingerprint="source-fingerprint",
    )
    value = replace(base, runtime_artifact=artifact).to_dict()

    loaded = replace(base, runtime_artifact=artifact).from_dict(value)
    assert loaded.runtime_artifact == artifact
    with pytest.raises(ValueError, match="full 40-character commit"):
        RuntimeArtifact("freetoken/qwen-ftw", "main", 123, "fingerprint").validate()
    with pytest.raises(ValueError, match="fingerprint is required"):
        RuntimeArtifact("freetoken/qwen-ftw", "d" * 40, 123, "").validate()
