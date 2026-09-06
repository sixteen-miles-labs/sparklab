"""Validate and atomically publish the pinned Qwen3.8-27B FTW and canonical card."""
import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, ModelCard

REPO = "oakmindai/Qwen3.8-27B-NVFP4-FTW"
CARD = Path(__file__).with_name("MODEL_CARD.md")
METADATA = (
    "LICENSE", "config.json", "chat_template.jinja", "generation_config.json",
    "hf_quant_config.json", "merges.txt", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
)


def validate(root):
    index = json.loads((root / "freetoken_weight.json").read_text())
    assert index["format"] == "freetoken_weight" and index["version"] == 1
    assert index["fingerprint"] == "af78f7411817bb73"
    assert index["total_bytes"] == 24_617_562_112
    assert len(index["shards"]) == 3
    offset = 0
    for number, shard in enumerate(index["shards"]):
        assert shard["file"] == f"freetoken-{number:05d}.ftw"
        assert shard["global_off"] == offset
        assert (root / shard["file"]).stat().st_size == shard["nbytes"]
        offset += shard["nbytes"]
    assert offset == index["total_bytes"]
    names = set()
    for tensor in index["tensors"]:
        assert tensor["name"] not in names
        names.add(tensor["name"])
        assert tensor["global_off"] >= 0 and tensor["nbytes"] > 0
        assert tensor["global_off"] % index["align"] == 0
        assert tensor["global_off"] + tensor["nbytes"] <= offset
    for name in METADATA:
        assert (root / name).is_file(), name
    card = ModelCard.load(str(CARD))
    assert card.data.pipeline_tag == "text-generation"
    assert index["fingerprint"] in card.text
    # Remove workstation paths and obsolete checksums for the source safetensors.
    index["source_model_path"] = "Inferact/Qwen3.8-27B-NVFP4"
    index["source_revision"] = "6128240ebaf4eaa7bad2b3d1c72c37d677c5f462"
    index["copied_metadata"] = list(METADATA) + ["README.md"]
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.weights_dir.resolve()
    index = validate(root)
    print(f"Validated {len(index['tensors'])} tensors, three shards, {index['total_bytes']} weight bytes", flush=True)
    if args.validate_only:
        return
    api = HfApi()
    before = api.model_info(REPO)
    assert {s.rfilename for s in before.siblings} <= {".gitattributes"}, "Destination not empty; review before overwriting"
    files = list(METADATA) + [s["file"] for s in index["shards"]]
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name)) for name in files]
    operations += [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=CARD.read_bytes()),
        CommitOperationAdd(path_in_repo="freetoken_weight.json", path_or_fileobj=(json.dumps(index, indent=2) + "\n").encode()),
    ]
    commit = api.create_commit(repo_id=REPO, parent_commit=before.sha, operations=operations,
        commit_message="Publish pinned Qwen3.8-27B NVFP4 FTW weights and SparkLab model card")
    print(commit.commit_url, flush=True)
    after = api.model_info(REPO, revision=commit.oid, files_metadata=True)
    remote = {s.rfilename: s for s in after.siblings}
    assert set(remote) == set(files) | {".gitattributes", "README.md", "freetoken_weight.json"}
    for op in operations:
        entry = remote[op.path_in_repo]
        assert entry.size == op.upload_info.size
        if entry.lfs:
            assert entry.lfs.sha256 == op.upload_info.sha256.hex()
    print("Verified remote file inventory, sizes, and LFS SHA-256 hashes.", flush=True)


if __name__ == "__main__":
    main()
