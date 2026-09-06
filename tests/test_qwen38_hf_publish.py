import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLISH_DIR = ROOT / "hf/models/qwen-3.8-Flash-Next-NVFP4-FTW"
spec = importlib.util.spec_from_file_location("qwen38_publish", PUBLISH_DIR / "push_weights.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def artifact(tmp_path, fingerprint):
    for name in publisher.REQUIRED_FILES:
        (tmp_path / name).touch()
    (tmp_path / "freetoken-00000.ftw").touch()
    (tmp_path / "freetoken_weight.json").write_text(json.dumps({
        "format": "freetoken_weight", "version": 1, "fingerprint": fingerprint,
    }))
    return argparse.Namespace(weights_dir=tmp_path, workers=None, private=False, create=False)


def test_nvidia_identity(tmp_path):
    args = artifact(tmp_path, publisher.EXPECTED_FINGERPRINT)
    assert publisher.validate_args(args) == tmp_path


def test_old_artifact_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        publisher.validate_args(artifact(tmp_path, "47e11ddb878adf4c"))


def test_missing_card_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "PUBLIC_CARD", tmp_path / "absent.md")
    with pytest.raises(ValueError, match="public model card is missing"):
        publisher.validate_args(artifact(tmp_path, publisher.EXPECTED_FINGERPRINT))


def test_card_structure_and_recipe_identity():
    from huggingface_hub import ModelCard

    card = ModelCard.load(str(publisher.PUBLIC_CARD))
    assert card.data.pipeline_tag == "text-generation"
    assert card.data.library_name == "sparklab"
    recipe = json.loads((ROOT / "python/sparklab/recipes/qwen3.8-flash-next.json").read_text())
    for value in (recipe["revision"], recipe["runtime_artifact"]["fingerprint"]):
        assert value in card.text
    for heading in ("What this repository contains", "Why use FTW?",
                    "Run with SparkLab on NVIDIA DGX Spark", "Credits and license",
                    "Performance and validation limits"):
        assert f"## {heading}" in card.text
    assert "README.md" in publisher.DEFAULT_IGNORE_PATTERNS


def test_uploader_publishes_canonical_card_after_weights(tmp_path, monkeypatch):
    args = artifact(tmp_path, publisher.EXPECTED_FINGERPRINT)
    args.repo_id = publisher.DEFAULT_REPO_ID
    args.revision = "test-branch"
    calls = []

    class FakeApi:
        def upload_large_folder(self, **kwargs):
            calls.append(("weights", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("card", kwargs))

    monkeypatch.setattr(publisher, "parse_args", lambda: args)
    monkeypatch.setattr(publisher, "HfApi", FakeApi)
    assert publisher.main() == 0
    assert [name for name, _ in calls] == ["weights", "card"]
    assert calls[1][1]["path_or_fileobj"] == str(publisher.PUBLIC_CARD)
    assert calls[1][1]["path_in_repo"] == "README.md"
    assert calls[1][1]["revision"] == "test-branch"
