#!/usr/bin/env python3
"""Build a hard-linked Qwen3.8 NVFP4 + FP8-PLE FTW artifact.

The routed experts and resident tensors remain byte-identical to the source FTW.
Only the external BF16 PLE table is streamed through an E4M3 cast. Both paths must
live on the same filesystem so the roughly 78 GB of FTW shards can be hard-linked.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from sparklab.models.qwen4_exp.weight import requantize_raw_ngram_artifact


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source Qwen3.8 FTW")
    parser.add_argument("--output", type=Path, required=True, help="new hybrid FTW path")
    parser.add_argument("--chunk-rows", type=int, default=65_536)
    return parser.parse_args()


def _link_tree(source: Path, destination: Path) -> None:
    destination.mkdir()
    for entry in source.iterdir():
        if entry.name in {"qwen4_ngram.bin", "qwen4_ngram.json", "freetoken_weight.json"}:
            continue
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, copy_function=os.link)
        elif entry.is_file():
            os.link(entry, target)


def main() -> int:
    args = _args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / "freetoken_weight.json").is_file():
        raise SystemExit(f"not an FTW checkpoint: {source}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    if source.stat().st_dev != output.parent.stat().st_dev:
        raise SystemExit("source and output must be on the same filesystem for hard links")

    building = output.with_name(f".{output.name}.building-{os.getpid()}")
    try:
        _link_tree(source, building)
        ple = requantize_raw_ngram_artifact(
            str(source), str(building), chunk_rows=args.chunk_rows
        )
        index = json.loads((source / "freetoken_weight.json").read_text(encoding="utf-8"))
        artifacts = [
            artifact for artifact in index.get("external_artifacts", [])
            if artifact.get("kind") != "qwen4_ngram"
        ]
        artifacts.append({"kind": "qwen4_ngram", **ple})
        index["external_artifacts"] = artifacts
        index["derived_from_fingerprint"] = index.get("fingerprint")
        index["artifact_variant"] = "nvfp4-experts-fp8-ple"
        index_path = building / "freetoken_weight.json"
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        os.rename(building, output)
    except BaseException:
        # Keep an interrupted 51 GB conversion available for diagnosis; its hidden,
        # PID-qualified name cannot be mistaken for a runnable artifact.
        if building.exists():
            print(f"incomplete build retained at {building}")
        raise

    print(json.dumps({
        "output": str(output),
        "variant": "nvfp4-experts-fp8-ple",
        "ple_dtype": ple["dtype"],
        "ple_bytes": ple["nbytes"],
        "hardlinked_ftw_bytes": index["total_bytes"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
