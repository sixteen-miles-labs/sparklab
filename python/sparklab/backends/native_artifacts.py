"""Prepared-artifact validation for Spark Lab's built-in native backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import ArtifactValidation, BackendError


def validate_ftw(path: Path, *, runtime_format: str) -> ArtifactValidation:
    """Validate a complete FTW checkpoint without reading weight payloads."""
    from freetoken.checkpoint.ftw import ALIGN, FORMAT_TAG, FORMAT_VERSION, INDEX_NAME

    index_path = path / INDEX_NAME
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError(f"cannot read FTW index {index_path}: {exc}") from exc
    if index.get("format") != FORMAT_TAG or index.get("version") != FORMAT_VERSION:
        raise BackendError(
            f"unsupported FTW identity in {index_path}: "
            f"format={index.get('format')!r}, version={index.get('version')!r}"
        )
    if index.get("align") != ALIGN:
        raise BackendError(
            f"FTW alignment mismatch: expected {ALIGN}, found {index.get('align')!r}"
        )

    shards = index.get("shards")
    tensors = index.get("tensors")
    total_bytes = index.get("total_bytes")
    if not isinstance(shards, list) or not shards:
        raise BackendError("FTW index has no shards")
    if not isinstance(tensors, list) or not tensors:
        raise BackendError("FTW index has no tensors")
    if not isinstance(total_bytes, int) or total_bytes <= 0:
        raise BackendError(f"invalid FTW total_bytes: {total_bytes!r}")

    shard_cursor = 0
    indexed_shard_names: set[str] = set()
    for shard in shards:
        try:
            filename = shard["file"]
            offset = int(shard["global_off"])
            nbytes = int(shard["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"invalid FTW shard metadata: {shard!r}") from exc
        if (
            not isinstance(filename, str)
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
        ):
            raise BackendError(f"unsafe FTW shard path: {filename!r}")
        if filename in indexed_shard_names:
            raise BackendError(f"duplicate FTW shard: {filename}")
        indexed_shard_names.add(filename)
        if offset != shard_cursor or nbytes <= 0:
            raise BackendError(
                f"non-contiguous FTW shard {filename}: expected offset "
                f"{shard_cursor}, found offset={offset}, nbytes={nbytes}"
            )
        shard_path = path / filename
        try:
            actual_size = shard_path.stat().st_size
        except OSError as exc:
            raise BackendError(f"FTW checkpoint is missing shard: {shard_path}") from exc
        if actual_size != nbytes:
            raise BackendError(
                f"FTW shard size mismatch for {filename}: expected {nbytes}, "
                f"found {actual_size}"
            )
        shard_cursor += nbytes
    if shard_cursor != total_bytes:
        raise BackendError(
            f"FTW shard total mismatch: index={total_bytes}, shards={shard_cursor}"
        )
    actual_shard_names = {
        shard_path.name for shard_path in path.glob("*.ftw") if shard_path.is_file()
    }
    if actual_shard_names != indexed_shard_names:
        raise BackendError(
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
            raise BackendError(f"invalid FTW tensor metadata: {tensor!r}") from exc
        if not isinstance(name, str) or not name or name in names:
            raise BackendError(f"invalid or duplicate FTW tensor name: {name!r}")
        if not isinstance(kind, str) or not kind:
            raise BackendError(f"invalid FTW tensor kind for {name}: {kind!r}")
        if (
            offset != tensor_cursor
            or offset % ALIGN
            or nbytes < 0
            or any(dim < 0 for dim in shape)
        ):
            raise BackendError(
                f"invalid FTW tensor range for {name}: expected aligned offset "
                f"{tensor_cursor}, found offset={offset}, nbytes={nbytes}"
            )
        expected_nbytes = int(torch.empty((), dtype=dtype).element_size())
        for dim in shape:
            expected_nbytes *= dim
        if nbytes != expected_nbytes:
            raise BackendError(
                f"FTW tensor size mismatch for {name}: expected {expected_nbytes}, "
                f"found {nbytes}"
            )
        names.add(name)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        logical_bytes += nbytes
        tensor_cursor = ((offset + nbytes + ALIGN - 1) // ALIGN) * ALIGN
    if tensor_cursor != total_bytes:
        raise BackendError(
            f"FTW tensor stream mismatch: index={total_bytes}, tensors={tensor_cursor}"
        )
    recorded_counts = index.get("counts") or {}
    if recorded_counts != kind_counts:
        raise BackendError(
            f"FTW kind count mismatch: index={recorded_counts!r}, tensors={kind_counts!r}"
        )

    external_bytes = 0
    external = index.get("external_artifacts") or []
    if not isinstance(external, list):
        raise BackendError("FTW external_artifacts must be a list")
    for artifact in external:
        try:
            filename = artifact["file"]
            nbytes = int(artifact["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"invalid FTW external artifact: {artifact!r}") from exc
        if (
            not isinstance(filename, str)
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
            or nbytes < 0
        ):
            raise BackendError(f"unsafe FTW external artifact: {artifact!r}")
        external_path = path / filename
        try:
            actual_size = external_path.stat().st_size
        except OSError as exc:
            raise BackendError(
                f"FTW checkpoint is missing external artifact: {external_path}"
            ) from exc
        if actual_size != nbytes:
            raise BackendError(
                f"FTW external artifact size mismatch for {filename}: expected "
                f"{nbytes}, found {actual_size}"
            )
        external_bytes += nbytes

    fingerprint = index.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise BackendError("FTW checkpoint has no source fingerprint")
    return ArtifactValidation(
        format=runtime_format,
        fingerprint=fingerprint,
        details={
            "index": INDEX_NAME,
            "shards": len(shards),
            "tensors": len(tensors),
            "logical_bytes": logical_bytes,
            "physical_bytes": total_bytes,
            "external_artifacts": len(external),
            "external_bytes": external_bytes,
            "kind_counts": kind_counts,
        },
    )


__all__ = ["validate_ftw"]
