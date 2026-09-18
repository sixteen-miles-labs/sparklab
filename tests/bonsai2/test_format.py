import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_prism_reader_extensions_and_truncation(tmp_path):
    from sparklab.models.gguf.prism_reader import PrismGGUFReader
    from sparklab.models.gguf.reader import iter_gguf_tensors

    def string(s):
        return struct.pack("<Q", len(s)) + s

    for kind, size in ((142, 34), (143, 28)):
        p = tmp_path / f"{kind}.gguf"
        header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 0)
        header += string(b"weight") + struct.pack("<IQQIQ", 2, 128, 1, kind, 0)
        header += b"\0" * (-len(header) % 32)
        p.write_bytes(header + bytes(range(size)))
        r = PrismGGUFReader(p)
        assert r.tensors[0].n_bytes == size
        t = next(iter_gguf_tensors(str(p)))
        assert t.shape == (1, 128) and t.ggml_type == kind
        np.testing.assert_array_equal(t._raw[0], np.arange(size))
        p.write_bytes(header + bytes(size - 1))
        with pytest.raises(ValueError, match="exceeds"):
            PrismGGUFReader(p)


def test_qwen_gguf_control_and_user_defined_tokens(monkeypatch):
    import sparklab.models.gguf.tokenizer as mod
    from tokenizers import Tokenizer, models
    from transformers.integrations import ggml

    tokens = ["a", "<|im_start|>", "<|im_end|>", "<|image_pad|>", "<think>", "</think>"]
    meta = {
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [1, 3, 3, 3, 4, 4],
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.padding_token_id": 2,
        "tokenizer.ggml.bos_token_id": 1,
    }
    monkeypatch.setattr(mod, "load_gguf_metadata", lambda _: meta)
    monkeypatch.setattr(mod, "gguf_architecture", lambda _: "qwen35")
    monkeypatch.setattr(
        ggml,
        "convert_gguf_tokenizer",
        lambda *_: (
            Tokenizer(models.BPE({s: i for i, s in enumerate(tokens)}, [])),
            {},
        ),
    )
    tok = mod.load_gguf_tokenizer("unused")
    assert tok.encode("<think></think><|image_pad|>", add_special_tokens=False) == [
        4,
        5,
        3,
    ]
    assert (
        tok.decode([4, 5, 3], skip_special_tokens=True).replace(" ", "")
        == "<think></think>"
    )
    assert len(tok) == len(tokens)


def test_gdn_tiled_rows_restore_grouped_order():
    from sparklab.models.bonsai2.weights import ungroup

    m = {"qwen35.ssm.group_count": 2, "qwen35.ssm.time_step_rank": 6}
    grouped = torch.arange(24).reshape(12, 2)
    tiled = grouped.reshape(2, 3, 2, 2).transpose(0, 1).reshape(12, 2)
    torch.testing.assert_close(ungroup(tiled, m, 2), grouped)


def test_recipe_selects_only_requested_gguf_files():
    from sparklab.backends import BackendError, get_backend
    from sparklab.backends.gguf_artifacts import filenames
    from sparklab.catalog import get_recipe

    recipe = get_recipe("bonsai2-27b")
    get_backend("native").validate_deployment(recipe.deployment)
    assert set(filenames(recipe.deployment)) == {"gguf_file", "vision_file"}
    for name in ("../bad.gguf", "/bad.gguf", "other/file.gguf", "bad.bin"):
        d = replace(recipe.deployment, backend_options={"gguf_file": name})
        with pytest.raises(BackendError):
            filenames(d)


def test_gguf_acquisition_is_selective_and_needs_no_conversion(tmp_path, monkeypatch):
    from sparklab.acquire import AcquisitionError, acquire_recipe
    from sparklab.backends import gguf_artifacts
    from sparklab.catalog import get_recipe

    r = get_recipe("bonsai2-27b")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr(
        gguf_artifacts, "validate", lambda *_: {"total_bytes": r.source_bytes}
    )
    result = acquire_recipe(r, root=str(tmp_path), downloader=download)
    assert result["prepare"] is False
    assert len(calls) == 1
    assert calls[0]["revision"] == r.revision
    assert set(calls[0]["allow_patterns"]) == {
        "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
        "README.md",
        "LICENSE",
        "NOTICE.txt",
    }
    with pytest.raises(AcquisitionError, match="omit --prepare"):
        acquire_recipe(r, root=str(tmp_path), prepare=True, downloader=download)
    assert len(calls) == 1


def test_gguf_launch_includes_text_and_projector(tmp_path, monkeypatch):
    from sparklab.backends import get_backend, gguf_artifacts
    from sparklab.backends.base import RuntimeRequest
    from sparklab.catalog import get_recipe

    r = get_recipe("bonsai2-27b")
    monkeypatch.setattr(gguf_artifacts, "validate", lambda *_: {})
    plan = get_backend("native").build_launch_plan(
        RuntimeRequest(r.slug, r.recipe_version, r.model, tmp_path, r.deployment)
    )
    args = list(plan.arguments)
    assert args[args.index("--model") + 1] == str(
        tmp_path / r.deployment.backend_options["gguf_file"]
    )
    assert args[args.index("--vision-model") + 1] == str(
        tmp_path / r.deployment.backend_options["vision_file"]
    )
    # Exercise the real CLI parser too: a launch plan can contain a string that
    # the backend accepts but the serving CLI rejects (e.g. speculative "none").
    import sparklab.utils
    from sparklab.serving.args import parse_args

    monkeypatch.setattr(
        sparklab.utils,
        "cached_load_hf_config",
        lambda _: SimpleNamespace(to_dict=lambda: {"model_type": "qwen35"}),
    )
    parsed, _ = parse_args(args)
    assert parsed.speculative_tokens == 0
    assert parsed.vision_model == str(
        tmp_path / r.deployment.backend_options["vision_file"]
    )


def test_rotation_metadata_rejects_unknown_conventions():
    from sparklab.models.bonsai2.weights import validate_rotation

    prefix = "prism.hadamard."
    metadata = {
        prefix + k: v
        for k, v in {
            "version": 1,
            "block_size": 1024,
            "transform": "normalized-sylvester-walsh-hadamard",
            "axis": "input-last-dimension",
            "sign_mode": "explicit",
            "gdn_v_grouped": True,
            "inverse_weight_names": ["token_embd.weight"],
            "weight_names": ["output.weight"],
            "sign_widths": [1024],
            "sign_values": [1] * 1024,
        }.items()
    }
    validate_rotation(metadata)
    for key, value in [
        ("version", 2),
        ("block_size", 512),
        ("sign_mode", "random"),
        ("gdn_v_grouped", False),
        ("weight_names", []),
        ("sign_values", [0] * 1024),
        ("sign_widths", [1024, 1024]),
        ("inverse_weight_names", []),
    ]:
        with pytest.raises(ValueError):
            validate_rotation(dict(metadata, **{prefix + key: value}))
