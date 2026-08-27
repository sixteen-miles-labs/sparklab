"""Pinned, resumable model acquisition and optional FTW preparation."""

from __future__ import annotations

import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import snapshot_download

from sparklab.catalog import ModelRecipe
from sparklab.paths import manifest_path, prepared_path, source_path
from sparklab.planner import ArtifactPlan, plan_artifacts


class AcquisitionError(RuntimeError):
    pass


def validate_safetensors_snapshot(directory: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Validate an indexed safetensors snapshot using headers only.

    Hugging Face already verifies each downloaded blob, but this second gate
    proves that the local index is internally complete before a many-hour FTW
    conversion begins: every referenced shard exists, each mapped tensor occurs
    exactly once in that shard, all byte ranges are valid, and the logical tensor
    byte total matches the publisher's index metadata.

    Unindexed/non-safetensors checkpoints return ``None`` and retain their
    existing model-loader validation path.
    """
    folder = Path(directory)
    index_path = folder / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"invalid safetensors index {index_path}: {exc}") from exc
    if not isinstance(weight_map, dict) or not weight_map:
        raise AcquisitionError(f"empty safetensors weight map: {index_path}")

    by_file: dict[str, set[str]] = {}
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise AcquisitionError("safetensors weight_map entries must be strings")
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise AcquisitionError(f"unsafe safetensors shard path: {filename!r}")
        by_file.setdefault(filename, set()).add(name)

    logical_bytes = physical_bytes = 0
    for filename, expected_names in sorted(by_file.items()):
        path = folder / filename
        if not path.is_file():
            raise AcquisitionError(f"safetensors snapshot is missing shard: {path}")
        fd = os.open(path, os.O_RDONLY)
        try:
            prefix = os.pread(fd, 8, 0)
            if len(prefix) != 8:
                raise AcquisitionError(f"truncated safetensors prefix: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size <= 1 or header_size > path.stat().st_size - 8:
                raise AcquisitionError(f"invalid safetensors header size in {path}")
            raw_header = os.pread(fd, header_size, 8)
            if len(raw_header) != header_size:
                raise AcquisitionError(f"truncated safetensors header: {path}")
            header = json.loads(raw_header)
        except (OSError, json.JSONDecodeError) as exc:
            raise AcquisitionError(f"cannot read safetensors header {path}: {exc}") from exc
        finally:
            os.close(fd)

        actual_names = set(header) - {"__metadata__"}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)[:5]
            extra = sorted(actual_names - expected_names)[:5]
            raise AcquisitionError(
                f"safetensors index/header mismatch in {path.name}: "
                f"missing={missing}, extra={extra}"
            )
        ranges: list[tuple[int, int, str]] = []
        for name in actual_names:
            meta = header[name]
            try:
                begin, end = (int(value) for value in meta["data_offsets"])
                shape = [int(value) for value in meta["shape"]]
                dtype = str(meta["dtype"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AcquisitionError(f"invalid tensor metadata for {name}: {meta}") from exc
            if begin < 0 or end < begin or any(dim < 0 for dim in shape) or not dtype:
                raise AcquisitionError(f"invalid tensor range for {name}: {meta}")
            logical_bytes += end - begin
            ranges.append((begin, end, name))
        cursor = 0
        for begin, end, name in sorted(ranges):
            if begin != cursor:
                raise AcquisitionError(
                    f"non-contiguous/overlapping safetensors range in {path.name} "
                    f"at {name}: expected offset {cursor}, found {begin}"
                )
            cursor = end
        max_end = cursor
        expected_size = 8 + header_size + max_end
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise AcquisitionError(
                f"safetensors shard size mismatch for {path.name}: "
                f"expected {expected_size}, found {actual_size}"
            )
        physical_bytes += actual_size

    published = int((index.get("metadata") or {}).get("total_size", logical_bytes))
    if published != logical_bytes:
        raise AcquisitionError(
            f"safetensors logical byte total mismatch: index={published}, headers={logical_bytes}"
        )
    return {
        "index": index_path.name,
        "shards": len(by_file),
        "tensors": len(weight_map),
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
    }


def validate_ftw_checkpoint(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a complete FTW checkpoint without reading its weight payloads.

    Conversion streams hundreds of GiB and publishes the index last.  This gate
    independently proves that every indexed shard and external artifact exists at
    its exact recorded size, and that all tensor metadata describes one contiguous,
    aligned logical stream.  It catches interrupted conversions, stale shard sets,
    and malformed indexes before their fingerprint can be used as certification
    evidence.
    """
    from freetoken.checkpoint.ftw import ALIGN, FORMAT_TAG, FORMAT_VERSION, INDEX_NAME

    folder = Path(directory)
    index_path = folder / INDEX_NAME
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"cannot read FTW index {index_path}: {exc}") from exc
    if index.get("format") != FORMAT_TAG or index.get("version") != FORMAT_VERSION:
        raise AcquisitionError(
            f"unsupported FTW identity in {index_path}: "
            f"format={index.get('format')!r}, version={index.get('version')!r}"
        )
    if index.get("align") != ALIGN:
        raise AcquisitionError(
            f"FTW alignment mismatch: expected {ALIGN}, found {index.get('align')!r}"
        )

    shards = index.get("shards")
    tensors = index.get("tensors")
    total_bytes = index.get("total_bytes")
    if not isinstance(shards, list) or not shards:
        raise AcquisitionError("FTW index has no shards")
    if not isinstance(tensors, list) or not tensors:
        raise AcquisitionError("FTW index has no tensors")
    if not isinstance(total_bytes, int) or total_bytes <= 0:
        raise AcquisitionError(f"invalid FTW total_bytes: {total_bytes!r}")

    shard_cursor = 0
    indexed_shard_names: set[str] = set()
    for shard in shards:
        try:
            filename = shard["file"]
            offset = int(shard["global_off"])
            nbytes = int(shard["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AcquisitionError(f"invalid FTW shard metadata: {shard!r}") from exc
        if (
            not isinstance(filename, str)
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
        ):
            raise AcquisitionError(f"unsafe FTW shard path: {filename!r}")
        if filename in indexed_shard_names:
            raise AcquisitionError(f"duplicate FTW shard: {filename}")
        indexed_shard_names.add(filename)
        if offset != shard_cursor or nbytes <= 0:
            raise AcquisitionError(
                f"non-contiguous FTW shard {filename}: expected offset "
                f"{shard_cursor}, found offset={offset}, nbytes={nbytes}"
            )
        path = folder / filename
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise AcquisitionError(f"FTW checkpoint is missing shard: {path}") from exc
        if actual_size != nbytes:
            raise AcquisitionError(
                f"FTW shard size mismatch for {filename}: expected {nbytes}, "
                f"found {actual_size}"
            )
        shard_cursor += nbytes
    if shard_cursor != total_bytes:
        raise AcquisitionError(
            f"FTW shard total mismatch: index={total_bytes}, shards={shard_cursor}"
        )
    actual_shard_names = {path.name for path in folder.glob("*.ftw") if path.is_file()}
    if actual_shard_names != indexed_shard_names:
        raise AcquisitionError(
            "FTW shard set mismatch: "
            f"missing={sorted(indexed_shard_names - actual_shard_names)}, "
            f"stale={sorted(actual_shard_names - indexed_shard_names)}"
        )

    import torch

    tensor_cursor = 0
    names: set[str] = set()
    kind_counts: dict[str, int] = {}
    logical_bytes = 0
    for tensor in tensors:
        try:
            name = tensor["name"]
            kind = tensor["kind"]
            dtype_name = tensor["dtype"]
            shape = [int(dim) for dim in tensor["shape"]]
            offset = int(tensor["global_off"])
            nbytes = int(tensor["nbytes"])
            dtype = getattr(torch, dtype_name)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise AcquisitionError(f"invalid FTW tensor metadata: {tensor!r}") from exc
        if not isinstance(name, str) or not name or name in names:
            raise AcquisitionError(f"invalid or duplicate FTW tensor name: {name!r}")
        if not isinstance(kind, str) or not kind:
            raise AcquisitionError(f"invalid FTW tensor kind for {name}: {kind!r}")
        if (
            offset != tensor_cursor
            or offset % ALIGN
            or nbytes < 0
            or any(dim < 0 for dim in shape)
        ):
            raise AcquisitionError(
                f"invalid FTW tensor range for {name}: expected aligned offset "
                f"{tensor_cursor}, found offset={offset}, nbytes={nbytes}"
            )
        expected_nbytes = int(torch.empty((), dtype=dtype).element_size())
        for dim in shape:
            expected_nbytes *= dim
        if nbytes != expected_nbytes:
            raise AcquisitionError(
                f"FTW tensor size mismatch for {name}: expected {expected_nbytes}, "
                f"found {nbytes}"
            )
        names.add(name)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        logical_bytes += nbytes
        tensor_cursor = ((offset + nbytes + ALIGN - 1) // ALIGN) * ALIGN
    if tensor_cursor != total_bytes:
        raise AcquisitionError(
            f"FTW tensor stream mismatch: index={total_bytes}, tensors={tensor_cursor}"
        )
    recorded_counts = index.get("counts") or {}
    if recorded_counts != kind_counts:
        raise AcquisitionError(
            f"FTW kind count mismatch: index={recorded_counts!r}, "
            f"tensors={kind_counts!r}"
        )

    external_bytes = 0
    external = index.get("external_artifacts") or []
    if not isinstance(external, list):
        raise AcquisitionError("FTW external_artifacts must be a list")
    for artifact in external:
        try:
            filename = artifact["file"]
            nbytes = int(artifact["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AcquisitionError(f"invalid FTW external artifact: {artifact!r}") from exc
        if (
            not isinstance(filename, str)
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
            or nbytes < 0
        ):
            raise AcquisitionError(f"unsafe FTW external artifact: {artifact!r}")
        path = folder / filename
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise AcquisitionError(f"FTW checkpoint is missing external artifact: {path}") from exc
        if actual_size != nbytes:
            raise AcquisitionError(
                f"FTW external artifact size mismatch for {filename}: expected "
                f"{nbytes}, found {actual_size}"
            )
        external_bytes += nbytes

    fingerprint = index.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise AcquisitionError("FTW checkpoint has no source fingerprint")
    return {
        "index": INDEX_NAME,
        "fingerprint": fingerprint,
        "shards": len(shards),
        "tensors": len(tensors),
        "logical_bytes": logical_bytes,
        "physical_bytes": total_bytes,
        "external_artifacts": len(external),
        "external_bytes": external_bytes,
        "kind_counts": kind_counts,
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def acquire_recipe(
    recipe: ModelRecipe,
    *,
    root: str | None = None,
    prepare: bool = False,
    dry_run: bool = False,
    downloader: Callable[..., str] = snapshot_download,
    converter: Callable[..., dict] | None = None,
) -> dict[str, Any]:
    """Acquire an exact recipe revision and optionally convert it to FTW.

    Hugging Face's local-dir downloader is resumable. The manifest is written only
    after the pinned snapshot completes, so a partial directory is never advertised
    as ready.
    """
    if recipe.revision is None:
        raise AcquisitionError(f"{recipe.slug} is not pinned to an immutable revision")
    plan = plan_artifacts(recipe, root=root, include_prepared=prepare)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "recipe": recipe.slug,
        "recipe_version": recipe.recipe_version,
        "model": recipe.model,
        "revision": recipe.revision,
        "prepare": prepare,
        "dry_run": dry_run,
        "artifact_plan": plan.to_dict(),
    }
    if not plan.ready:
        raise AcquisitionError("; ".join(plan.reasons) or "artifact plan is not ready")
    if dry_run:
        return result

    source = source_path(recipe, root)
    source.mkdir(parents=True, exist_ok=True)
    resolved = Path(
        downloader(
            repo_id=recipe.model,
            revision=recipe.revision,
            local_dir=str(source),
        )
    ).resolve()
    config = resolved / "config.json"
    if not config.is_file():
        raise AcquisitionError(f"pinned snapshot completed without config.json: {resolved}")
    source_validation = validate_safetensors_snapshot(resolved)

    prepared: Path | None = None
    index: dict[str, Any] | None = None
    prepared_validation: dict[str, Any] | None = None
    if prepare:
        if recipe.implementation == "planned":
            raise AcquisitionError(
                f"{recipe.slug} cannot be prepared until its text architecture is implemented"
            )
        if converter is None:
            from freetoken.checkpoint import convert_checkpoint

            converter = convert_checkpoint
        prepared = prepared_path(recipe, root)
        prepared.parent.mkdir(parents=True, exist_ok=True)
        index = converter(
            str(resolved),
            str(prepared),
            moe_backend="offload" if recipe.execution_policy == "nvme-moe" else "fused",
            nvfp4_backend="auto",
            device="cuda:0",
        )
        prepared_validation = validate_ftw_checkpoint(prepared)

    payload = {
        "schema_version": "1.0",
        "recipe": recipe.slug,
        "recipe_version": recipe.recipe_version,
        "model": recipe.model,
        "revision": recipe.revision,
        "source_path": str(resolved),
        "prepared_path": str(prepared) if prepared else None,
        "prepared_fingerprint": index.get("fingerprint") if index else None,
        "prepared_validation": prepared_validation,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_validation": source_validation,
    }
    _write_manifest(manifest_path(recipe, root), payload)
    result["manifest"] = payload
    return result


def read_manifest(recipe: ModelRecipe, root: str | None = None) -> dict[str, Any] | None:
    path = manifest_path(recipe, root)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


__all__ = [
    "AcquisitionError", "acquire_recipe", "read_manifest",
    "validate_ftw_checkpoint", "validate_safetensors_snapshot",
]
