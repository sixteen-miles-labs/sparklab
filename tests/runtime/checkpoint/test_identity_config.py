"""Portable state identity and bounded physical-memory admission contracts."""

from dataclasses import replace
import json
import shutil
from types import SimpleNamespace

import pytest
import torch

from sparklab.runtime.checkpoint.config import DiskSettings, runtime_reserve_bytes
from sparklab.runtime.checkpoint.identity import runtime_identity
from sparklab.runtime.checkpoint.state import Compatibility


def fixture_identity(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    for name, value in {
        "freetoken_weight.json": {"fingerprint": "weights-123"},
        "config.json": {"layers": 3},
        "tokenizer.json": {"tokens": [1, 2]},
        "tokenizer_config.json": {"eos": 2},
    }.items():
        (root / name).write_text(json.dumps(value))
    (root / "chat_template.jinja").write_text("{{ messages }}")
    adapter = SimpleNamespace(
        kv=SimpleNamespace(
            _kv_buffer=torch.empty(2, 2, 16, 4, 1, 8), _layer_map=[0, 2]
        ),
        linear=None,
        page_size=4,
    )
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        attention_backend="triton",
        model_config=SimpleNamespace(
            rotary_config=None, attn_quant=None, expert_quant=None
        ),
    )
    return root, adapter, config


def test_portable_identity_ignores_location_and_capacity_but_covers_semantics(tmp_path):
    root, adapter, config = fixture_identity(tmp_path)

    def identity(path=root, scope="instance"):
        return runtime_identity(path, adapter, config, scope=scope)

    original = identity()
    relocated = tmp_path / "elsewhere"
    shutil.copytree(root, relocated)
    assert identity(relocated) == original
    adapter.kv._kv_buffer = torch.empty(2, 2, 32, 4, 1, 8)
    assert identity() == original
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
        "freetoken_weight.json",
    ]:
        before = (root / name).read_bytes()
        (root / name).write_bytes(before + b" ")
        assert identity() != original, name
        (root / name).write_bytes(before)
    assert identity(scope="other") != original
    adapter.page_size = 8
    assert identity() != original
    adapter.page_size = 4
    config.dtype = torch.float32
    assert identity() != original


def test_unidentified_artifact_and_excess_templates_rejected(tmp_path):
    root, adapter, config = fixture_identity(tmp_path)
    (root / "freetoken_weight.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable FTW"):
        runtime_identity(root, adapter, config, scope="instance")
    (root / "freetoken_weight.json").write_text('{"fingerprint":"w"}')
    extras = root / "additional_chat_templates"
    extras.mkdir()
    for i in range(64):
        (extras / f"{i}.jinja").touch()
    with pytest.raises(ValueError, match="template"):
        runtime_identity(root, adapter, config, scope="instance")


def test_one_pass_prefix_keys_match_independent_canonical_digest():
    config = {
        key: key
        for key in [
            "artifact",
            "precision",
            "model_config",
            "tokenizer",
            "template",
            "state_layout",
            "cache_geometry",
            "continuation",
            "scope",
        ]
    }
    identity = Compatibility.from_config(config)
    tokens = [0, 32, 991, -1, 248319, 17]
    lengths = [1, 2, 4, 6]
    assert identity.prefix_keys(tokens, lengths) == {
        n: identity.prefix_key(tokens[:n]) for n in lengths
    }
    for lengths in [[0], [7], [True]]:
        with pytest.raises(ValueError):
            identity.prefix_keys(tokens, lengths)


def test_off_by_default_and_explicit_bounded_reservations(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith(("SPARKLAB_PREFIX_", "SPARKLAB_EXACT_PREFIX")):
            monkeypatch.delenv(key)
    assert DiskSettings.from_env() is None
    assert runtime_reserve_bytes() == 0
    monkeypatch.setenv("SPARKLAB_PREFIX_STORE", "/unused")
    with pytest.raises(ValueError, match="requires"):
        DiskSettings.from_env()
    monkeypatch.setenv("SPARKLAB_EXACT_PREFIX", "1")
    settings = DiskSettings.from_env()
    assert settings.reserve_bytes >= (settings.tickets + 5) * settings.staging_bytes
    assert runtime_reserve_bytes() == settings.reserve_bytes + (48 << 20)
    assert replace(settings, tickets=8).reserve_bytes >= settings.reserve_bytes
    assert (
        replace(settings, staging_bytes=64 << 20).reserve_bytes > settings.reserve_bytes
    )
    for key, value in [
        ("SPARKLAB_PREFIX_IO_TICKETS", "9"),
        ("SPARKLAB_PREFIX_RESTORE_TIMEOUT", "nan"),
        ("SPARKLAB_PREFIX_RESTORE_POLICY", "fast"),
        ("SPARKLAB_PREFIX_SCOPE", ""),
    ]:
        with monkeypatch.context() as trial:
            trial.setenv(key, value)
            with pytest.raises(ValueError):
                DiskSettings.from_env()


def test_named_huggingface_templates_change_compatibility_and_share_a_bound(tmp_path):
    root, adapter, config = fixture_identity(tmp_path)

    def current():
        return runtime_identity(root, adapter, config, scope="instance")

    baseline = current()
    templates = root / "chat_templates"
    templates.mkdir()
    template = templates / "tool_use.jinja"
    template.write_text("original named template")
    first = current()
    assert first != baseline
    template.write_text("changed named template")
    assert current() != first
    template.unlink()
    assert current() == baseline
    legacy = root / "additional_chat_templates"
    legacy.mkdir()
    for i in range(32):
        (templates / f"{i}.jinja").touch()
        (legacy / f"{i}.jinja").touch()
    with pytest.raises(ValueError, match="too many template"):
        current()
