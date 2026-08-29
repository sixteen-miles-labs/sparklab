#!/usr/bin/env python3
"""Upload the validated Kimi K3 NVFP4 FTW artifact to Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_REPO_ID = "oakmindai/Kimi-K3-NVFP4-FTW"
DEFAULT_WEIGHTS_DIR = Path(
    "~/.sparklab/models/kimi-k3/prepared/0.2.0"
).expanduser()
PUBLIC_CARD = Path(__file__).with_name("MODEL_CARD.md")
SOURCE_REPOSITORY = "nvidia/Kimi-K3-NVFP4"
SOURCE_REVISION = "f8c5234a0a880bcc6cbf779a315e7ee2f405b812"
EXPECTED_FINGERPRINT = "534cbc4565d4279d"
EXPECTED_TOTAL_BYTES = 1_610_936_311_808
EXPECTED_SHARDS = 194
REQUIRED_FILES = (
    "config.json",
    "freetoken_weight.json",
    "tokenizer_config.json",
    "tiktoken.model",
    "LICENSE",
)
DEFAULT_IGNORE_PATTERNS = [
    ".git/**",
    ".cache/**",
    "__pycache__/**",
    "*.pyc",
    "*.tmp",
    # These are committed after the large transfer so the public card replaces the
    # upstream card and the published index contains no workstation-local path.
    "README.md",
    "freetoken_weight.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload SparkLab's validated Kimi K3 NVFP4 FTW artifact with "
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
        "--create", action="store_true", help="Create the public repository if needed"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the local artifact without contacting Hugging Face",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, dict]:
    weights_dir = args.weights_dir.expanduser().resolve()
    if not weights_dir.is_dir():
        raise ValueError(f"weights directory does not exist: {weights_dir}")
    missing = [name for name in REQUIRED_FILES if not (weights_dir / name).is_file()]
    if missing:
        raise ValueError(f"FTW artifact is missing: {', '.join(missing)}")
    if not PUBLIC_CARD.is_file():
        raise ValueError(f"public model card is missing: {PUBLIC_CARD}")

    try:
        index = json.loads((weights_dir / "freetoken_weight.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read FTW index: {error}") from error

    if index.get("format") != "freetoken_weight" or index.get("version") != 1:
        raise ValueError("unsupported FTW identity")
    if index.get("fingerprint") != EXPECTED_FINGERPRINT:
        raise ValueError(
            "unexpected FTW fingerprint: "
            f"{index.get('fingerprint')!r} (expected {EXPECTED_FINGERPRINT!r})"
        )
    if index.get("total_bytes") != EXPECTED_TOTAL_BYTES:
        raise ValueError(
            "unexpected FTW byte count: "
            f"{index.get('total_bytes')!r} (expected {EXPECTED_TOTAL_BYTES})"
        )

    shards = index.get("shards") or []
    if len(shards) != EXPECTED_SHARDS:
        raise ValueError(f"expected {EXPECTED_SHARDS} FTW shards, found {len(shards)}")
    indexed_names: set[str] = set()
    for shard in shards:
        name = shard.get("file")
        if not isinstance(name, str) or name in indexed_names:
            raise ValueError(f"invalid or duplicate FTW shard entry: {name!r}")
        indexed_names.add(name)
        path = weights_dir / name
        if not path.is_file():
            raise ValueError(f"FTW shard is missing: {name}")
        if path.stat().st_size != shard.get("nbytes"):
            raise ValueError(
                f"FTW shard size mismatch for {name}: "
                f"{path.stat().st_size} != {shard.get('nbytes')}"
            )
    actual_names = {path.name for path in weights_dir.glob("freetoken-*.ftw")}
    if actual_names != indexed_names:
        raise ValueError(
            "FTW shard set mismatch: "
            f"missing={sorted(indexed_names - actual_names)}, "
            f"stale={sorted(actual_names - indexed_names)}"
        )
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1")
    return weights_dir, index


def public_index(index: dict) -> dict:
    value = dict(index)
    value["source_model_path"] = f"{SOURCE_REPOSITORY}@{SOURCE_REVISION}"
    return value


def verify_remote(api: HfApi, repo_id: str, index: dict, revision: str | None) -> str:
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    siblings = {item.rfilename: item for item in info.siblings or []}
    required = {"README.md", "freetoken_weight.json", *REQUIRED_FILES}
    missing = sorted(name for name in required if name not in siblings)
    if missing:
        raise ValueError(f"published repository is missing: {', '.join(missing)}")
    for shard in index["shards"]:
        remote = siblings.get(shard["file"])
        if remote is None:
            raise ValueError(f"published repository is missing {shard['file']}")
        if remote.size != shard["nbytes"]:
            raise ValueError(
                f"published shard size mismatch for {shard['file']}: "
                f"{remote.size} != {shard['nbytes']}"
            )
    remote_shards = [
        name
        for name in siblings
        if name.startswith("freetoken-") and name.endswith(".ftw")
    ]
    if len(remote_shards) != EXPECTED_SHARDS:
        raise ValueError("published repository has an unexpected FTW shard count")
    return info.sha


def main() -> int:
    args = parse_args()
    try:
        weights_dir, index = validate_args(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"Validated {EXPECTED_SHARDS} shards, {EXPECTED_TOTAL_BYTES} bytes, "
        f"fingerprint {EXPECTED_FINGERPRINT}"
    )
    if args.validate_only:
        return 0

    api = HfApi()
    try:
        if args.create:
            api.create_repo(
                repo_id=args.repo_id,
                repo_type="model",
                private=False,
                exist_ok=True,
            )
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
        with tempfile.TemporaryDirectory(prefix="kimi-k3-ftw-publish-") as tmp_name:
            index_path = Path(tmp_name) / "freetoken_weight.json"
            index_path.write_text(
                json.dumps(public_index(index), indent=2) + "\n", encoding="utf-8"
            )
            api.create_commit(
                repo_id=args.repo_id,
                repo_type="model",
                revision=args.revision,
                commit_message="Publish validated SparkLab FTW metadata",
                operations=[
                    CommitOperationAdd(
                        path_in_repo="README.md", path_or_fileobj=str(PUBLIC_CARD)
                    ),
                    CommitOperationAdd(
                        path_in_repo="freetoken_weight.json",
                        path_or_fileobj=str(index_path),
                    ),
                ],
            )
        sha = verify_remote(api, args.repo_id, index, args.revision)
    except (HfHubHTTPError, OSError, ValueError) as error:
        print(f"Hugging Face upload failed: {error}", file=sys.stderr)
        return 1

    print(f"Upload complete: https://huggingface.co/{args.repo_id}/tree/{sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
