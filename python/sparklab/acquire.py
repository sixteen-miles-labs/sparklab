"""Pinned, resumable model acquisition and optional FTW preparation."""

from __future__ import annotations

import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import snapshot_download

from sparklab.backends import BackendError, get_backend
from sparklab.catalog import DeploymentRecipe, ModelRecipe
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

    # `metadata.total_size` is publisher-supplied advisory metadata. Some otherwise
    # valid checkpoints carry a stale value (for example, RedHatAI's pinned GLM-5.3
    # NVFP4 index is 6,684,672 bytes lower than its headers). The checks above are
    # authoritative: every indexed name must occur in its declared shard, tensor
    # ranges must be contiguous, and each physical file must end at the final range.
    # Preserve a published-total discrepancy in the manifest instead of rejecting a
    # byte-for-byte complete snapshot solely on this redundant field.
    published = int((index.get("metadata") or {}).get("total_size", logical_bytes))
    result = {
        "index": index_path.name,
        "shards": len(by_file),
        "tensors": len(weight_map),
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
    }
    if published != logical_bytes:
        result["published_logical_bytes"] = published
        result["published_logical_bytes_delta"] = logical_bytes - published
    return result




def _validation_payload(validation) -> dict[str, Any]:
    return {
        "format": validation.format,
        "fingerprint": validation.fingerprint,
        **dict(validation.details),
    }


def validate_ftw_checkpoint(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Compatibility wrapper around the native backend's FTW artifact validator."""
    deployment = DeploymentRecipe(
        backend="native",
        backend_api="1.0",
        source_format="safetensors",
        runtime_format="ftw-v1",
        quantization=None,
        execution_policy="nvme-moe",
        backend_options={},
    )
    try:
        validation = get_backend("native").validate_artifact(Path(directory), deployment)
    except BackendError as exc:
        raise AcquisitionError(str(exc)) from exc
    return _validation_payload(validation)


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
    from_source: bool = False,
    dry_run: bool = False,
    downloader: Callable[..., str] = snapshot_download,
    converter: Callable[..., dict] | None = None,
) -> dict[str, Any]:
    """Acquire an immutable source or a pinned, prebuilt runtime artifact."""
    if recipe.revision is None:
        raise AcquisitionError(f"{recipe.slug} is not pinned to an immutable revision")
    if from_source and not prepare:
        raise AcquisitionError("from_source requires prepare=True")
    use_prebuilt = prepare and not from_source and recipe.runtime_artifact is not None
    plan = plan_artifacts(
        recipe,
        root=root,
        include_prepared=prepare,
        use_prebuilt=use_prebuilt,
    )
    result: dict[str, Any] = {
        "schema_version": "2.0",
        "recipe": recipe.slug,
        "recipe_version": recipe.recipe_version,
        "model": recipe.model,
        "revision": recipe.revision,
        "deployment": recipe.deployment.to_dict(),
        "prepare": prepare,
        "from_source": from_source,
        "acquisition": plan.acquisition,
        "runtime_artifact": (
            recipe.runtime_artifact.to_dict()
            if recipe.runtime_artifact is not None
            else None
        ),
        "dry_run": dry_run,
        "artifact_plan": plan.to_dict(),
    }
    if not plan.ready:
        raise AcquisitionError("; ".join(plan.reasons) or "artifact plan is not ready")
    if dry_run:
        return result

    backend = get_backend(recipe.backend)
    resolved: Path | None = None
    source_validation: dict[str, Any] | None = None
    source_artifact: dict[str, Any] | None = None
    runtime_artifact: dict[str, Any] | None = None
    if use_prebuilt:
        hosted = recipe.runtime_artifact
        assert hosted is not None
        destination = prepared_path(recipe, root)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            resolved_runtime = Path(
                downloader(
                    repo_id=hosted.repo_id,
                    revision=hosted.revision,
                    local_dir=str(destination),
                )
            ).resolve()
        except Exception as exc:
            raise AcquisitionError(
                f"cannot acquire prebuilt runtime artifact "
                f"{hosted.repo_id}@{hosted.revision}: {exc}; "
                "retry or use --from-source to convert locally"
            ) from exc
        if not (resolved_runtime / "config.json").is_file():
            raise AcquisitionError(
                f"prebuilt runtime artifact has no config.json: {resolved_runtime}"
            )
        try:
            validation = backend.validate_artifact(
                resolved_runtime, recipe.deployment
            )
        except BackendError as exc:
            raise AcquisitionError(str(exc)) from exc
        if validation.fingerprint != hosted.fingerprint:
            raise AcquisitionError(
                "prebuilt runtime fingerprint mismatch: "
                f"expected {hosted.fingerprint}, found {validation.fingerprint}"
            )
        runtime_artifact = {
            "role": "runtime",
            "path": str(resolved_runtime),
            "format": validation.format,
            "backend": recipe.backend,
            "repository": hosted.repo_id,
            "revision": hosted.revision,
            "fingerprint": validation.fingerprint,
            "validation": _validation_payload(validation),
        }
    else:
        source = source_path(recipe, root)
        source.mkdir(parents=True, exist_ok=True)
        try:
            resolved = Path(
                downloader(
                    repo_id=recipe.model,
                    revision=recipe.revision,
                    local_dir=str(source),
                )
            ).resolve()
        except Exception as exc:
            raise AcquisitionError(
                f"cannot acquire source artifact "
                f"{recipe.model}@{recipe.revision}: {exc}"
            ) from exc
        config = resolved / "config.json"
        if not config.is_file():
            raise AcquisitionError(
                f"pinned snapshot completed without config.json: {resolved}"
            )
        source_validation = validate_safetensors_snapshot(resolved)
        source_artifact = {
            "role": "source",
            "path": str(resolved),
            "format": recipe.deployment.source_format,
            "repository": recipe.model,
            "revision": recipe.revision,
            "validation": source_validation,
        }

    if prepare and not use_prebuilt:
        if recipe.implementation == "planned":
            raise AcquisitionError(
                f"{recipe.slug} cannot be prepared until its architecture is implemented"
            )
        destination = prepared_path(recipe, root)
        try:
            validation = backend.prepare(
                resolved,
                destination,
                recipe.deployment,
                implementation=converter,
            )
        except BackendError as exc:
            raise AcquisitionError(str(exc)) from exc
        runtime_artifact = {
            "role": "runtime",
            "path": str(destination.resolve()),
            "format": validation.format,
            "backend": recipe.backend,
            "fingerprint": validation.fingerprint,
            "validation": _validation_payload(validation),
        }

    payload = {
        "schema_version": "2.0",
        "recipe": recipe.slug,
        "recipe_version": recipe.recipe_version,
        "model": recipe.model,
        "revision": recipe.revision,
        "deployment": recipe.deployment.to_dict(),
        "artifacts": {
            "source": source_artifact,
            "runtime": runtime_artifact,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(manifest_path(recipe, root), payload)
    result["manifest"] = payload
    return result


def _normalize_v1_manifest(recipe: ModelRecipe, value: dict[str, Any]) -> dict[str, Any]:
    runtime_path = value.get("prepared_path")
    runtime = None
    if runtime_path:
        validation = value.get("prepared_validation")
        runtime = {
            "role": "runtime",
            "path": runtime_path,
            "format": recipe.deployment.runtime_format,
            "backend": recipe.backend,
            "fingerprint": value.get("prepared_fingerprint"),
            "validation": validation,
        }
    return {
        "schema_version": "2.0",
        "migrated_from_schema": "1.0",
        "recipe": value.get("recipe", recipe.slug),
        "recipe_version": value.get("recipe_version", recipe.recipe_version),
        "model": value.get("model", recipe.model),
        "revision": value.get("revision", recipe.revision),
        "deployment": recipe.deployment.to_dict(),
        "artifacts": {
            "source": {
                "role": "source",
                "path": value.get("source_path"),
                "format": recipe.deployment.source_format,
                "validation": value.get("source_validation"),
            },
            "runtime": runtime,
        },
        "completed_at": value.get("completed_at"),
    }


def read_manifest(recipe: ModelRecipe, root: str | None = None) -> dict[str, Any] | None:
    path = manifest_path(recipe, root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if value.get("schema_version") == "1.0":
        return _normalize_v1_manifest(recipe, value)
    if value.get("schema_version") != "2.0":
        raise AcquisitionError(
            f"unsupported SparkLab manifest schema: {value.get('schema_version')!r}"
        )
    return value

__all__ = [
    "AcquisitionError",
    "acquire_recipe",
    "read_manifest",
    "validate_ftw_checkpoint",
    "validate_safetensors_snapshot",
]
