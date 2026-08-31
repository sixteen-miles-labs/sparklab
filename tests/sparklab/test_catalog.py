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
    assert {recipe.slug for recipe in recipes if recipe.status == "certified"} == {
        "qwen3.6-35b-a3b"
    }


def test_catalog_contains_requested_portfolio_without_overclaiming_status():
    assert get_recipe("kimi-k3").model == "nvidia/Kimi-K3-NVFP4"
    assert get_recipe("glm-5.3-flash").model == (
        "RedHatAI/GLM-5.3-Flash-NVFP4"
    )
    assert get_recipe("qwen3.8-flash-next").model == (
        "Inferact/Qwen3.8-Flash-Next-NVFP4"
    )
    assert get_recipe("qwen3.8-27b").model == "Inferact/Qwen3.8-27B-NVFP4"
    assert get_recipe("qwen3.6-35b-a3b").name == "Qwen3.6 35B A3B"
    assert get_recipe("glm-5.2").name == "GLM-5.2"
    assert get_recipe("glm-5.2").intended_tier == "research"
    assert get_recipe("glm-5.3").model == "Inferact/GLM-5.3-NVFP4"
    assert {
        recipe.slug: recipe.parameters for recipe in load_catalog()
    } == {
        "qwen3.6-35b-a3b": "35B total / 3B active",
        "qwen3.8-27b": "27B dense",
        "deepseek-v4": "284B total / 13B active",
        "glm-5.3-flash": "320B total / 18B active",
        "qwen3.8-flash-next": "125B LM + 55B aux / 6B active",
        "glm-5.2": "753B total / 40B active",
        "glm-5.3": "753B total / 40B active",
        "kimi-k3": "2.8T total / 16 of 896 experts",
    }
    assert get_recipe("deepseek-v4").status == "preview"
    assert get_recipe("kimi-k3").status == "experimental"
    assert {item.slug for item in select_recipes(load_catalog(), tier="fast")} == {
        "qwen3.6-35b-a3b",
    }
    qwen = get_recipe("qwen3.8-flash-next")
    assert qwen.recipe_version == "0.7.0"
    assert qwen.intended_tier == "frontier"
    assert qwen.status == "experimental"
    assert qwen.evidence == (
        "GB10-QWEN38-NVFP4-OPT-002",
        "GB10-QWEN38-NVFP4-OPT-001",
        "GB10-QWEN38-NVFP4-001",
    )
    assert qwen.backend == "native"
    assert qwen.deployment.source_format == "safetensors-nvfp4"
    assert qwen.deployment.runtime_format == "ftw-nvfp4"
    assert qwen.deployment.backend_options["attention_backend"] == "qsa"
    assert qwen.deployment.quantization == "nvfp4"
    assert "convert_expert_quantization" not in qwen.deployment.backend_options
    assert qwen.deployment.backend_options["nvfp4_backend"] == "triton"
    assert qwen.performance.decode_tokens_per_second == pytest.approx(16.607355237469154)
    assert qwen.performance.warm_ttft_seconds == pytest.approx(0.40297515969723463)
    assert qwen.deployment.backend_options["moe_host_cache_gb"] == 0
    assert qwen.deployment.backend_options["moe_preload_all"] is True
    assert qwen.deployment.backend_options["num_tokens"] == 131_072
    assert "moe_prefill_sparse_max_tokens" not in qwen.deployment.backend_options
    assert "moe_prefill_hit_d2d" not in qwen.deployment.backend_options
    assert qwen.runtime_memory == {"total_bytes": 107374182400}
    assert qwen.runtime_artifact is not None
    assert qwen.runtime_artifact.repo_id == "oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW"
    assert qwen.runtime_artifact.revision == "cbbcf69f52b9815b8a987fe839003fae12aa8050"
    assert qwen.runtime_artifact.fingerprint == "47e11ddb878adf4c"
    kimi = get_recipe("kimi-k3")
    assert kimi.recipe_version == "0.3.0"
    assert kimi.deployment.runtime_format == "ftw-nvfp4"
    assert kimi.deployment.quantization == "nvfp4"
    assert kimi.deployment.backend_options["moe_cache_size"] == 896
    assert kimi.deployment.backend_options["moe_cache_policy"] == "layer_lru"
    assert kimi.deployment.backend_options["kimi_mlp_fp8"] is True
    assert kimi.deployment.backend_options["disable_startup_prefill_warmup"] is True
    assert kimi.performance.decode_tokens_per_second == pytest.approx(
        0.16130749709147157
    )
    assert kimi.performance.warm_ttft_seconds == pytest.approx(395.4053193805739)
    assert kimi.evidence == ("GB10-KIMI-001",)
    assert kimi.runtime_memory == {"total_bytes": 105804767232}
    assert kimi.runtime_artifact is not None
    assert kimi.runtime_artifact.repo_id == "oakmindai/Kimi-K3-NVFP4-FTW"
    assert kimi.runtime_artifact.revision == "793f1f8436cd7de11e7912c41b3d49d4c9e4d11c"
    assert kimi.runtime_artifact.bytes == 1610936311808
    assert kimi.runtime_artifact.fingerprint == "534cbc4565d4279d"
    glm52 = get_recipe("glm-5.2")
    assert glm52.recipe_version == "0.2.0"
    assert glm52.performance.decode_tokens_per_second == pytest.approx(0.802)
    assert glm52.performance.warm_ttft_seconds == pytest.approx(2.57)
    deepseek = get_recipe("deepseek-v4")
    assert deepseek.recipe_version == "0.2.0"
    assert deepseek.revision == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    assert deepseek.performance.decode_tokens_per_second == pytest.approx(
        10.282266070445095
    )
    assert deepseek.performance.warm_ttft_seconds == pytest.approx(
        0.6035212678834796
    )
    assert deepseek.deployment.backend_options["moe_prefill_sparse_max_tokens"] == 512
    assert deepseek.deployment.backend_options["moe_cache_auto"] is True
    qwen36 = get_recipe("qwen3.6-35b-a3b")
    assert qwen36.status == "certified"
    assert qwen36.runtime_memory == {"total_bytes": 34359738368}
    assert qwen36.performance.decode_tokens_per_second == pytest.approx(67.7870954092827)
    assert qwen36.performance.warm_ttft_seconds == pytest.approx(0.32882933877408504)
    assert qwen36.evidence == ("GB10-QWEN36-FAST-001", "GB10-QWEN36-FAST-002")
    assert qwen36.runtime_artifact is not None
    assert qwen36.runtime_artifact.repo_id == "oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW"
    assert qwen36.runtime_artifact.revision == "fecab7acfd0590d2b268d8fb9ea1c88431471111"
    assert qwen36.runtime_artifact.fingerprint == "bda7268c3e0afd7b"
    primary = select_recipes(load_catalog(), portfolio_role="primary")
    assert {(item.intended_tier, item.slug) for item in primary} == {
        ("fast", "qwen3.6-35b-a3b"),
        ("frontier", "qwen3.8-27b"),
        ("frontier", "deepseek-v4"),
        ("frontier", "glm-5.3-flash"),
        ("frontier", "qwen3.8-flash-next"),
        ("research", "kimi-k3"),
    }
    assert {
        item.slug for item in select_recipes(load_catalog(), portfolio_role="fallback")
    } == {"glm-5.2", "glm-5.3"}


def test_next_model_recipes_are_immutable_and_capacity_plannable():
    qwen27 = get_recipe("qwen3.8-27b")
    qwen = get_recipe("qwen3.8-flash-next")
    glm = get_recipe("glm-5.3-flash")
    kimi = get_recipe("kimi-k3")
    glm52 = get_recipe("glm-5.2")
    glm53 = get_recipe("glm-5.3")
    deepseek = get_recipe("deepseek-v4")
    assert qwen.revision == "103a7608316173ca6edd49929544244de7ffda70"
    assert qwen27.revision == "6128240ebaf4eaa7bad2b3d1c72c37d677c5f462"
    assert qwen27.source_bytes == 26404418018
    assert qwen27.prepared_bytes == 24640689529
    assert qwen27.intended_tier == "frontier"
    assert qwen27.performance.decode_tokens_per_second == pytest.approx(8.832997654771269)
    assert qwen27.performance.warm_ttft_seconds == pytest.approx(0.14434478300245246)
    assert qwen27.performance.context_tokens == 65_536
    assert qwen27.evidence == ("GB10-QWEN38-27B-001",)
    assert qwen27.deployment.execution_policy == "resident"
    assert qwen27.deployment.backend_options["num_tokens"] == 65_536
    assert qwen27.deployment.backend_options["max_seq_len_override"] == 65_536
    assert qwen27.deployment.backend_options["nvfp4_backend"] == "triton"
    assert glm.recipe_version == "0.3.2"
    assert glm.revision == "9eaeadaf026871a90640e32c0604f6ab0b2d641d"
    assert kimi.revision == "f8c5234a0a880bcc6cbf779a315e7ee2f405b812"
    assert glm52.revision == "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
    assert glm53.revision == "ce67b36f3669192b5bb233819f0fda6c8a9837f8"
    assert deepseek.revision == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    assert qwen.source_bytes == 182838060595
    assert qwen.expert_quantization == "nvfp4"
    assert glm.source_bytes == 190262422658
    assert glm.deployment.source_format == "safetensors-nvfp4"
    assert glm.deployment.runtime_format == "ftw-nvfp4"
    assert glm.deployment.quantization == "nvfp4"
    assert glm.deployment.backend_options["nvfp4_backend"] == "triton"
    assert glm.deployment.backend_options["convert_kda_quantization"] == "fp8_pertensor"
    assert glm.deployment.backend_options["moe_host_cache_gb"] == 0
    assert glm.deployment.backend_options["memory_ratio"] == 0.96
    assert glm.deployment.backend_options["moe_prefill_overlap"] is False
    assert glm.runtime_artifact is not None
    assert glm.runtime_artifact.repo_id == "oakmindai/GLM-5.3-Flash-NVFP4-FTW"
    assert glm.runtime_artifact.revision == "f296cec0baceb2276121efe76f14d61b62c1e47d"
    assert glm.runtime_artifact.bytes == 184716947456
    assert glm.runtime_artifact.fingerprint == "4c021651a1e61802"
    assert kimi.source_bytes == 1610038482254
    assert kimi.prepared_bytes == 1610936311808
    assert glm52.source_bytes == 464874323992
    assert glm53.source_bytes == 464867183339
    assert glm53.deployment.quantization == "nvfp4"
    assert glm53.deployment.runtime_format == "ftw-nvfp4"
    assert glm53.performance.decode_tokens_per_second == pytest.approx(
        0.8126153861203544
    )
    assert glm53.performance.warm_ttft_seconds == pytest.approx(2.530337787233293)
    assert glm53.evidence == ("GB10-GLM53-RESEARCH-001",)
    assert deepseek.source_bytes == 166878536440
    assert deepseek.prepared_bytes == 157460918272
    assert qwen.execution_policy == glm.execution_policy == "nvme-moe"
    assert qwen.minimum_free_bytes > qwen.source_bytes + qwen.prepared_bytes
    assert glm.minimum_free_bytes > glm.source_bytes + glm.prepared_bytes
    assert kimi.minimum_free_bytes > kimi.source_bytes + kimi.prepared_bytes
    assert glm52.minimum_free_bytes > glm52.source_bytes + glm52.prepared_bytes
    assert glm53.minimum_free_bytes > glm53.source_bytes + glm53.prepared_bytes
    assert deepseek.minimum_free_bytes > deepseek.source_bytes + deepseek.prepared_bytes


def test_deepseek_recipe_points_to_checked_in_optimized_evidence():
    recipe = get_recipe("deepseek-v4")
    assert recipe.evidence == ("GB10-DSV4-SPARSE-001",)
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-DSV4-SPARSE-001.json").read_text()
    )
    assert result["result_id"] == recipe.evidence[0]
    assert result["status"] == "measured"
    assert result["recipe"]["recipe_version"] == recipe.recipe_version
    assert result["recipe"]["revision"] == recipe.revision
    assert result["metrics"]["decode_tokens_per_second"] == pytest.approx(
        10.282266070445095
    )
    assert result["metrics"]["warm_ttft_seconds"] == pytest.approx(
        0.6035212678834796
    )
    assert result["metrics"]["expert_cache_miss_rate_percent"] == 0
    assert result["metrics"]["physical_io_gib"] == 0
    assert result["validation"]["output_hash"] == "fbf178b2bde5"


def test_glm53_recipe_points_to_checked_in_fused_mhc_evidence():
    recipe = get_recipe("glm-5.3-flash")
    assert recipe.performance.evidence == "GB10-GLM53-MHC-003"
    assert recipe.performance.evidence in recipe.evidence
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-GLM53-MHC-003.json").read_text()
    )
    assert result["result_id"] == recipe.performance.evidence
    assert result["status"] == "measured"
    assert result["recipe"]["recipe_version"] == recipe.recipe_version
    assert result["recipe"]["revision"] == recipe.revision
    assert result["checkpoint"]["fingerprint"] == recipe.runtime_artifact.fingerprint
    assert result["metrics"]["decode_tokens_per_second"] == pytest.approx(
        recipe.performance.decode_tokens_per_second
    )
    assert result["metrics"]["warm_ttft_seconds"] == pytest.approx(
        recipe.performance.warm_ttft_seconds
    )
    assert result["comparison"]["decode_throughput_improvement_percent"] > 25
    assert result["admission"]["performance_gate_passed"] is True
    assert result["validation"]["output_hash_reproduced_runs"] == 2


def test_kimi_recipe_points_to_checked_in_bounded_capacity_evidence():
    recipe = get_recipe("kimi-k3")
    assert recipe.performance.evidence == "GB10-KIMI-001"
    assert recipe.performance.evidence in recipe.evidence
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-KIMI-001.json").read_text()
    )
    assert result["result_id"] == recipe.performance.evidence
    assert result["status"] == "measured"
    assert result["model"]["revision"] == recipe.revision
    assert result["model"]["fingerprint"] == "534cbc4565d4279d"
    assert result["model"]["checkpoint_bytes"] == recipe.prepared_bytes
    assert result["metrics"]["decode_tokens_per_second"] == pytest.approx(
        recipe.performance.decode_tokens_per_second
    )
    assert result["metrics"]["warm_ttft_seconds"] == pytest.approx(
        recipe.performance.warm_ttft_seconds
    )
    assert result["workload"]["completion_tokens"] == 256
    assert result["validation"]["oom_count"] == 0
    assert result["validation"]["runtime_swap_growth_bytes"] == 0
    assert result["validation"]["output_prefix_consistent_across_ladder"] is False
    assert result["validation"]["output_correctness_evaluated"] is False


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


def test_glm53_full_recipe_points_to_measured_failed_research_evidence():
    from sparklab.certification import evaluate_tier

    recipe = get_recipe("glm-5.3")
    assert recipe.runtime_artifact is not None
    assert recipe.runtime_artifact.repo_id == "oakmindai/GLM-5.3-NVFP4-FTW"
    assert recipe.runtime_artifact.revision == (
        "024b437e3f31f9cf9b8a936dffaf1302fbda1d30"
    )
    assert recipe.runtime_artifact.bytes == 428713099264
    assert recipe.runtime_artifact.fingerprint == "a0e799b03bceb4bf"
    assert recipe.performance.evidence == "GB10-GLM53-RESEARCH-001"
    assert recipe.performance.evidence in recipe.evidence
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-GLM53-RESEARCH-001.json").read_text()
    )
    assert result["result_id"] == recipe.performance.evidence
    assert result["status"] == "measured"
    assert result["recipe"]["recipe_version"] == recipe.recipe_version
    assert result["recipe"]["revision"] == recipe.revision
    assert result["checkpoint"]["fingerprint"] == "a0e799b03bceb4bf"
    assert result["checkpoint"]["bytes"] == recipe.prepared_bytes
    assert result["metrics"]["decode_tokens_per_second"] == pytest.approx(
        recipe.performance.decode_tokens_per_second
    )
    assert result["metrics"]["warm_ttft_seconds"] == pytest.approx(
        recipe.performance.warm_ttft_seconds
    )
    assert result["stability"]["oom_count"] == 0
    assert result["stability"]["swap_out_pages"] == 0
    assert result["validation"]["output_correctness_evaluated"] is False
    evaluation = evaluate_tier(recipe, result, "research")
    assert not evaluation.passed


def test_historical_qwen_nvfp4_evidence_does_not_transfer_to_new_checkpoint():
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


def test_historical_qwen_fp8_evidence_does_not_transfer_to_nvfp4_recipe():
    from sparklab.certification import evaluate_tier

    recipe = get_recipe("qwen3.8-flash-next")
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-QWEN38-FP8-001.json").read_text()
    )
    assert result["result_id"] == "GB10-QWEN38-FP8-001"
    assert result["metrics"]["decode_tokens_per_second"] == pytest.approx(4.987115832105288)
    assert result["metrics"]["warm_ttft_seconds"] == pytest.approx(0.5798669513314962)
    evaluation = evaluate_tier(recipe, result, "frontier")
    assert not evaluation.passed
    assert any("recipe_version mismatch" in reason for reason in evaluation.reasons)


def test_current_qwen_nvfp4_evidence_passes_all_non_endurance_frontier_gates():
    from sparklab.certification import evaluate_tier

    recipe = get_recipe("qwen3.8-flash-next")
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/gb10/results/GB10-QWEN38-NVFP4-OPT-002.json").read_text()
    )
    assert result["result_id"] == recipe.evidence[0]
    assert result["admission"]["performance_gate_passed"] is True
    assert result["metrics"]["decode_tokens_per_second"] == pytest.approx(
        16.607355237469154
    )
    assert result["metrics"]["warm_ttft_seconds"] == pytest.approx(
        0.40297515969723463
    )
    assert result["validation"]["context_tokens"] == 65_536
    assert result["validation"]["context_gate_run"] is True
    assert result["validation"]["capability_gate_run"] is True
    assert result["validation"]["quality_gate_run"] is True
    evaluation = evaluate_tier(recipe, result, "frontier")
    assert not evaluation.passed
    assert not any("recipe " in reason and "mismatch" in reason for reason in evaluation.reasons)
    assert not any("decode_tokens_per_second" in reason for reason in evaluation.reasons)
    assert not any("warm_ttft_seconds" in reason for reason in evaluation.reasons)
    assert not any("context_tokens" in reason for reason in evaluation.reasons)
    assert any("duration_minutes" in reason for reason in evaluation.reasons)


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
        repo_id="sparklab/qwen-ftw",
        revision="d" * 40,
        bytes=123,
        fingerprint="source-fingerprint",
    )
    value = replace(base, runtime_artifact=artifact).to_dict()

    loaded = replace(base, runtime_artifact=artifact).from_dict(value)
    assert loaded.runtime_artifact == artifact
    with pytest.raises(ValueError, match="full 40-character commit"):
        RuntimeArtifact("sparklab/qwen-ftw", "main", 123, "fingerprint").validate()
    with pytest.raises(ValueError, match="fingerprint is required"):
        RuntimeArtifact("sparklab/qwen-ftw", "d" * 40, 123, "").validate()
