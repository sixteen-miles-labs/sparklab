"""Container execution for native recipes with a separate dependency environment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .base import BackendError


def settings(deployment):
    value = deployment.backend_options.get("container")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "image", "environment", "memory_gib", "config_sha256", "quant_format"
    }:
        raise BackendError("container requires image, environment, memory_gib, config_sha256 and quant_format")
    if not isinstance(value["image"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", value["image"]):
        raise BackendError("invalid container image")
    if type(value["memory_gib"]) is not int or value["memory_gib"] <= 0:
        raise BackendError("container memory_gib must be a positive integer")
    env = value["environment"]
    if not isinstance(env, dict) or any(
        not isinstance(k, str) or not re.fullmatch(r"SPARKLAB_[A-Z0-9_]+", k)
        or not isinstance(v, str) or "\x00" in v for k, v in env.items()
    ):
        raise BackendError("container environment must contain string SPARKLAB_* settings")
    if any(not isinstance(value[k], str) or not value[k] for k in ("config_sha256", "quant_format")):
        raise BackendError("container artifact identity must be nonempty strings")
    return value


def accepts(path: Path, config) -> bool:
    try:
        index = json.loads((path / "freetoken_weight.json").read_text())
        # FTW fingerprints include source mtimes and differ after a fresh download.
        # Pin the model configuration and layout instead; acquisition pins source revision.
        return (isinstance(index, dict) and index.get("quant_format") == config["quant_format"]
                and hashlib.sha256((path / "config.json").read_bytes()).hexdigest() == config["config_sha256"])
    except (OSError, ValueError):
        return False


def prefix(config):
    limit = f"{config['memory_gib']}g"
    return ["docker", "run", "--rm", "--gpus", "all", "--network", "host",
            "--shm-size", "16g", "--memory", limit, "--memory-swap", limit]


def serve_command(config, checkpoint: Path, arguments):
    command = prefix(config)
    for key, value in sorted(config["environment"].items()):
        command.extend(("-e", f"{key}={value}"))
    command.extend(("-v", f"{checkpoint.resolve()}:/artifact:ro", config["image"], "serve"))
    return tuple(command) + tuple(arguments)


def prepare(source: Path, destination: Path, deployment):
    config = settings(deployment)
    destination.mkdir(parents=True, exist_ok=True)
    # Paths are Docker arguments, never interpolated into Python or shell code.
    program = (
        "from sparklab.checkpoint.convert import convert_checkpoint; "
        "convert_checkpoint('/source', '/artifact', "
        f"moe_backend={deployment.backend_options.get('moe_backend', 'offload')!r}, "
        f"nvfp4_backend={deployment.backend_options.get('nvfp4_backend', 'auto')!r}, "
        "device='cuda:0')"
    )
    command = prefix(config) + [
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "TRITON_CACHE_DIR=/tmp/triton", "-e", "XDG_CACHE_HOME=/tmp/cache",
        "-v", f"{source.resolve()}:/source:ro",
        "-v", f"{destination.resolve()}:/artifact",
        "--entrypoint", "python3", config["image"], "-c", program,
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BackendError(f"container preparation failed; build {config['image']} using benchmarks/qwen36_marlin/Dockerfile: {exc}") from exc
