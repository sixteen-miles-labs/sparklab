#!/usr/bin/env python3
"""Upload Qwen3.8 Flash Next NVFP4 FTW weights to Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_REPO_ID = "oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW"
DEFAULT_WEIGHTS_DIR = Path(
    "~/.sparklab/models/qwen3.8-flash-next/prepared/0.9.0"
).expanduser()
PUBLIC_CARD = Path(__file__).with_name("MODEL_CARD.md")
EXPECTED_FINGERPRINT = "94e1ee0daa442357"
REQUIRED_FILES = (
    "config.json",
    "freetoken_weight.json",
    "qwen4_ngram.bin",
    "qwen4_ngram.json",
    "nvfp4_experts_mtp.safetensors",
)
DEFAULT_IGNORE_PATTERNS = [
    ".git/**",
    ".cache/**",
    "__pycache__/**",
    "*.pyc",
    "*.tmp",
    "README.md",  # Publish the version-controlled card after the weight transfer.
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the prepared Qwen3.8 Flash Next NVFP4 FTW artifact with "
            "Hugging Face's resumable large-folder uploader."
        )
    )
    parser.add_argument(
        "weights_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help=f"FTW artifact directory (default: {DEFAULT_WEIGHTS_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Destination model repository (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument("--revision", help="Destination branch")
    parser.add_argument("--workers", type=int, help="Number of upload workers")
    parser.add_argument(
        "--create", action="store_true", help="Create the repository if needed"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository; requires --create",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Path:
    if not PUBLIC_CARD.is_file():
        raise ValueError(f"public model card is missing: {PUBLIC_CARD}")
    weights_dir = args.weights_dir.expanduser().resolve()
    if not weights_dir.is_dir():
        raise ValueError(f"weights directory does not exist: {weights_dir}")
    missing = [name for name in REQUIRED_FILES if not (weights_dir / name).is_file()]
    if missing:
        raise ValueError(f"FTW artifact is missing: {', '.join(missing)}")
    if not list(weights_dir.glob("freetoken-*.ftw")):
        raise ValueError("FTW artifact has no freetoken-*.ftw shards")
    index = json.loads((weights_dir / "freetoken_weight.json").read_text())
    if (index.get("format") != "freetoken_weight" or index.get("version") != 1
            or index.get("fingerprint") != EXPECTED_FINGERPRINT):
        raise ValueError("artifact does not match the NVIDIA model card")
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.private and not args.create:
        raise ValueError("--private requires --create")
    return weights_dir


def main() -> int:
    args = parse_args()
    try:
        weights_dir = validate_args(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    api = HfApi()
    try:
        if args.create:
            api.create_repo(
                repo_id=args.repo_id,
                repo_type="model",
                private=args.private,
                exist_ok=True,
            )
            # Some organizations default newly created repositories to private even
            # when `private=False` is supplied to create_repo.
            if not args.private:
                api.update_repo_settings(
                    repo_id=args.repo_id, repo_type="model", private=False
                )
        print(f"Uploading {weights_dir} to https://huggingface.co/{args.repo_id}")
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=weights_dir,
            revision=args.revision,
            ignore_patterns=DEFAULT_IGNORE_PATTERNS,
            num_workers=args.workers,
            print_report=True,
        )
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(PUBLIC_CARD),
            path_in_repo="README.md",
            revision=args.revision,
            commit_message="Publish canonical SparkLab Qwen3.8 NVIDIA FTW model card",
        )
    except HfHubHTTPError as error:
        print(f"Hugging Face upload failed: {error}", file=sys.stderr)
        return 1

    print(f"Upload complete: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
