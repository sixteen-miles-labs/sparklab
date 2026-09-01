from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from sparklab.acquire import acquire_recipe
from sparklab.backends.native import _compile_options
from sparklab.catalog import RuntimeArtifact, SupplementalArtifact, get_recipe
from sparklab.checkpoint.ftw import FTWWriter


def _minimal_ftw(path: Path) -> None:
    writer = FTWWriter(str(path), shard_limit=4096)
    writer.add_tensor("model.a", torch.arange(8, dtype=torch.bfloat16))
    writer.finalize({
        "fingerprint": "0123456789abcdef",
        "counts": {"weight": 1},
        "external_artifacts": [],
    })


def test_prebuilt_acquisition_adds_pinned_mtp_sidecar(tmp_path):
    supplemental = SupplementalArtifact(
        repo_id="publisher/qwen",
        revision="b" * 40,
        filename="nvfp4_experts_mtp.safetensors",
        bytes=3,
    )
    recipe = replace(
        get_recipe("qwen3.8-flash-next"),
        source_bytes=1,
        prepared_bytes=4,
        minimum_free_bytes=5,
        runtime_artifact=RuntimeArtifact(
            repo_id="sparklab/qwen-ftw",
            revision="a" * 40,
            bytes=1,
            fingerprint="0123456789abcdef",
            supplemental_files=(supplemental,),
        ),
    )
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        if kwargs.get("allow_patterns"):
            (destination / supplemental.filename).write_bytes(b"mtp")
        else:
            (destination / "config.json").write_text("{}", encoding="utf-8")
            _minimal_ftw(destination)
        return str(destination)

    result = acquire_recipe(
        recipe, root=str(tmp_path), prepare=True, downloader=downloader
    )

    assert calls[1]["allow_patterns"] == [supplemental.filename]
    runtime = result["manifest"]["artifacts"]["runtime"]
    assert runtime["supplemental_files"][0]["bytes"] == 3
    assert Path(runtime["supplemental_files"][0]["path"]).read_bytes() == b"mtp"


def test_native_backend_compiles_speculative_token_option():
    assert _compile_options({"speculative_tokens": 2}) == (
        "--speculative-tokens",
        "2",
    )


def test_native_backend_compiles_dspark_options():
    assert _compile_options({
        "speculative_method": "dspark",
        "speculative_tokens": 7,
        "draft_sample_method": "probabilistic",
    }) == (
        "--speculative-method", "dspark",
        "--speculative-tokens", "7",
        "--draft-sample-method", "probabilistic",
    )
