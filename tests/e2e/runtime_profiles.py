"""Explicit offline model profiles shared by real-model FLARE fixtures."""

import json
from pathlib import Path


def model_options(model, name="default"):
    if name == "default":
        return {}
    if name != "qwen36-triton-resident":
        raise ValueError(f"unknown real-model profile: {name}")
    root = Path(model)
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    index = json.loads((root / "freetoken_weight.json").read_text())
    if (
        text.get("model_type") != "qwen3_5_moe_text"
        or text.get("num_hidden_layers") != 40
        or text.get("num_experts") != 256
        or index.get("quant_format") != "nvfp4"
    ):
        raise ValueError("Qwen3.6 profile requires its native Triton NVFP4 MoE artifact")
    # GB10 shares host/device memory. Stream each bank once into the immutable
    # GPU cache rather than retaining a second complete pinned host bank.
    # Speculation is deliberately absent; all callers fix speculative_tokens=0.
    return {
        "attention_backend": "triton",
        "nvfp4_backend": "triton",
        "nvfp4_prefill_backend": "w4a16",
        "moe_backend": "offload",
        "moe_storage": "disk",
        "moe_preload_all": True,
        "moe_cache_rate": 1.0,
        "cuda_graph_max_bs": 0,
        "memory_ratio": 0.85,
    }


def server_model_options(model, name="default"):
    return {
        "--" + key.replace("_", "-"): None if value is True else value
        for key, value in model_options(model, name).items()
    }


def target_only_server_profile(
    model, name="default", *, context=4096, pages=512, max_running=2, graphs=2,
):
    profile = {
        "--num-pages": pages,
        "--page-size": 16,
        "--max-seq-len-override": context,
        "--max-running-requests": max_running,
        "--cuda-graph-max-bs": graphs,
        "--max-prefill-length": 512,
        "--memory-ratio": 0.85,
        "--attention-backend": "triton",
        "--nvfp4-backend": "triton",
        "--nvfp4-prefill-backend": "w4a16",
        "--speculative-tokens": 0,
    }
    profile.update(server_model_options(model, name))
    return profile


def profile_arguments(profile):
    """Render actual flags, retaining their JSON identity in the launch profile."""
    return [
        part
        for key, value in profile.items()
        for part in ((key,) if value is None else (key, str(value)))
    ]
