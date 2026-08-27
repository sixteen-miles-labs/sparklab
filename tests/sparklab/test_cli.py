from __future__ import annotations

import json

from sparklab import cli


def test_help_and_version_are_spark_lab_branded(capsys):
    assert cli.main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "usage: sparklab" in help_text
    assert "NVIDIA GB10" in help_text
    assert "legacy `ft`" in help_text

    assert cli.main(["--version"]) == 0
    assert "Spark Lab" in capsys.readouterr().out


def test_models_json_exposes_tier_and_admission_status(capsys):
    assert cli.main(["models", "--tier", "research", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["product"] == "Spark Lab" and payload["platform"] == "gb10"
    assert [recipe["slug"] for recipe in payload["recipes"]] == ["kimi-k3"]
    assert payload["recipes"][0]["status"] == "experimental"


def test_legacy_engine_command_is_delegated_unchanged(monkeypatch):
    seen = []
    monkeypatch.setitem(cli.COMMANDS, "serve", lambda args: seen.append(args) or 7)
    assert cli.main(["serve", "--model", "/checkpoint", "--port", "1919"]) == 7
    assert seen == [["--model", "/checkpoint", "--port", "1919"]]


def test_unknown_command_is_a_usage_error(capsys):
    assert cli.main(["spark"]) == 2
    captured = capsys.readouterr()
    assert "unknown sparklab command" in captured.err
