from __future__ import annotations

import json

from sparklab import cli


def test_help_and_version_are_spark_lab_branded(capsys):
    assert cli.main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "usage: sparklab" in help_text
    assert "NVIDIA GB10" in help_text
    assert "legacy `ft`" in help_text
    assert "FreeToken" not in help_text

    assert cli.main(["--version"]) == 0
    version = capsys.readouterr().out
    assert "Spark Lab" in version
    assert "FreeToken" not in version


def test_models_json_exposes_tier_and_admission_status(capsys):
    assert cli.main(["models", "--tier", "research", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["product"] == "Spark Lab" and payload["platform"] == "gb10"
    assert [recipe["slug"] for recipe in payload["recipes"]] == ["kimi-k3"]
    assert payload["recipes"][0]["status"] == "experimental"


def test_models_can_select_primary_portfolio(capsys):
    assert cli.main(["models", "--role", "primary", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [recipe["slug"] for recipe in payload["recipes"]] == [
        "qwen3.6-35b-a3b",
        "deepseek-v4",
        "glm-5.3-flash",
        "qwen3.8-flash-next",
        "kimi-k3",
    ]
    assert all(recipe["portfolio_role"] == "primary" for recipe in payload["recipes"])


def test_models_human_table_groups_tiers_and_shows_performance_metrics(capsys):
    assert cli.main(["models", "--role", "primary"]) == 0
    output = capsys.readouterr().out
    assert "MODEL" in output and "QUANTIZATION" in output
    assert "TOK/S" in output and "TTFT" in output
    assert "FAST — Routine chat, editing, and short agent loops" in output
    assert "FRONTIER — Hard coding, reasoning, and long agent work" in output
    assert "RESEARCH — Complete or novel models" in output
    assert "12.51" in output and "0.870 s" in output
    assert "Qwen3.6 35B A3B" in output and "NVFP4" in output
    assert "Qwen3.6 35B A3B NVFP4" not in output
    qwen36_row = next(line for line in output.splitlines() if line.startswith("Qwen3.6"))
    assert qwen36_row.split()[-2:] == ["—", "—"]


def test_legacy_engine_command_is_delegated_unchanged(monkeypatch):
    seen = []
    monkeypatch.setitem(cli.COMMANDS, "serve", lambda args: seen.append(args) or 7)
    assert cli.main(["serve", "--model", "/checkpoint", "--port", "1919"]) == 7
    assert seen == [["--model", "/checkpoint", "--port", "1919"]]


def test_unknown_command_is_a_usage_error(capsys):
    assert cli.main(["spark"]) == 2
    captured = capsys.readouterr()
    assert "unknown sparklab command" in captured.err


def test_pull_dry_run_delegates_to_pinned_acquisition(monkeypatch, capsys):
    seen = {}

    def fake_acquire(recipe, **kwargs):
        seen.update(recipe=recipe, **kwargs)
        return {
            "artifact_plan": {
                "source_path": "/models/qwen",
                "prepared_path": "/models/qwen-ftw",
            }
        }

    monkeypatch.setattr("sparklab.acquire.acquire_recipe", fake_acquire)
    assert cli.main(["pull", "qwen3.8-flash-next", "--dry-run"]) == 0
    assert seen["recipe"].revision == "f5d08274bafd880402bd16f5e3e6c514136ec06c"
    assert seen["dry_run"] is True and seen["prepare"] is False
    assert "would acquire" in capsys.readouterr().out
