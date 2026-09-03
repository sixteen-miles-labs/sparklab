from __future__ import annotations

import json
import os
import re
import struct
from typing import Iterator

import safetensors
import torch
from sparklab.runtime.distributed import get_tp_info
from sparklab.models.loader import drop_page_cache, iter_weight_files
from sparklab.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)

_EXPERT = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$")
_PER_EXPERT = re.compile(
    r"^(?:model\.language_model\.|language_model\.|model\.)"
    r"layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\."
    r"(weight|weight_scale_inv|weight_scale|weight_scale_2|input_scale)$"
)
_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="Qwen4-Exp NVFP4 experts",
)
_EXPERT_LAYER = re.compile(
    r"^(?:model\.language_model\.|language_model\.|model\.)"
    r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
_SHARD_RE = re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")
_MTP_FILE = "nvfp4_experts_mtp.safetensors"
_NGRAM_MANIFEST = "qwen4_ngram.json"
_NGRAM_PAYLOAD = "qwen4_ngram.bin"
_NGRAM_DTYPES = {
    "bfloat16": (torch.bfloat16, 2),
    "float8_e4m3fn": (torch.float8_e4m3fn, 1),
}
_SKIP = (
    "model.visual.", "visual.", "mtp.",
)
_FUSIONS = {
    ".self_attn.qkv_proj.weight": (
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ),
    ".linear_attn.in_proj.weight": (
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ),
}
_FP8_MAX = 448.0
_FP8_PROJECTION_SUFFIXES = (
    ".self_attn.qkv_proj.weight",
    ".self_attn.o_proj.weight",
    ".self_attn.index_qk_proj.weight",
    ".linear_attn.in_proj_qkvz.weight",
    ".linear_attn.out_proj.weight",
    ".mlp.shared_expert.gate_up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
    ".ple.key_proj.weight",
    ".ple.value_proj.weight",
)


def _quant_fp8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream one BF16 projection into the W8A16 resident representation."""
    values = weight.float()
    scale = (values.abs().amax(dim=1) / _FP8_MAX).clamp(min=1e-12)
    quantized = (values / scale[:, None]).clamp(
        -_FP8_MAX, _FP8_MAX
    ).to(torch.float8_e4m3fn)
    return quantized, scale.to(torch.float32)


def _is_fp8_resident_projection(name: str, weight: torch.Tensor) -> bool:
    if weight.ndim != 2 or not name.endswith(".weight"):
        return False
    return name.endswith(_FP8_PROJECTION_SUFFIXES) or "hyper_connection" in name


def _transform_resident_weights(weights, config):
    if getattr(config, "attn_quant", "none") != "fp8_pertensor":
        yield from weights
        return
    linear = config.linear_attention_group()
    qkvz_rows = None
    if linear is not None:
        key_dim = linear.num_key_heads * linear.key_head_dim
        value_dim = linear.num_value_heads * linear.value_head_dim
        qkvz_rows = 2 * key_dim + 2 * value_dim

    for name, weight in weights:
        if name.endswith(".linear_attn.in_proj.weight"):
            if qkvz_rows is None or qkvz_rows >= weight.shape[0]:
                raise ValueError(
                    f"invalid Qwen4 GDN fusion {name}: shape={tuple(weight.shape)} "
                    f"qkvz_rows={qkvz_rows}"
                )
            base = name.removesuffix("in_proj.weight")
            qkvz, ba = weight[:qkvz_rows], weight[qkvz_rows:]
            quantized, scale = _quant_fp8_per_row(qkvz)
            yield base + "in_proj_qkvz.weight", quantized
            yield base + "in_proj_qkvz.weight_scale", scale
            yield base + "in_proj_ba.weight", ba
            continue
        if _is_fp8_resident_projection(name, weight):
            if weight.dtype == torch.float8_e4m3fn:
                # Idempotent for a future FTW conversion that stores native FP8.
                yield name, weight
            else:
                quantized, scale = _quant_fp8_per_row(weight)
                yield name, quantized
                yield name.removesuffix(".weight") + ".weight_scale", scale
            continue
        yield name, weight


def transform_runtime_weights(weights, config):
    """Adapt legacy Qwen4 source/FTW tensors to the selected resident format."""
    yield from _transform_resident_weights(weights, config)


def transform_ftw_weights(weights, config):
    """FTW counterpart used by the generic checkpoint replay path."""
    yield from _transform_resident_weights(weights, config)


def _rename(raw: str) -> str | None:
    if raw.startswith(_SKIP) or ".ngram_embedding." in raw:
        return None
    name = raw
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model."):]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model."):]
    # The indexer is flattened in the runtime attention op.
    name = name.replace(".self_attn.indexer.index_qk_proj.", ".self_attn.index_qk_proj.")
    name = name.replace(".self_attn.indexer.q_layernorm.", ".self_attn.index_q_norm.")
    name = name.replace(".self_attn.indexer.k_layernorm.", ".self_attn.index_k_norm.")
    # Runtime naming keeps the disk provider directly under PLE.
    if ".ple.ple_embedding." in name:
        return None
    return name


def _fuse(name: str, tensor: torch.Tensor, pending: dict):
    for fused, parts in _FUSIONS.items():
        for index, suffix in enumerate(parts):
            if name.endswith(suffix):
                out_name = name[:-len(suffix)] + fused
                slots = pending.setdefault(out_name, {})
                slots[index] = tensor
                if len(slots) == len(parts):
                    del pending[out_name]
                    return out_name, torch.cat([slots[i] for i in range(len(parts))], 0)
                return ()
    return None


def _iter_experts_layer_order(
    model_path: str, device: torch.device,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read packed experts as adjacent gate-up/down pairs for each layer.

    The published checkpoint puts each multi-GiB packed expert tensor in its own
    shard.  Files downloaded concurrently do not necessarily appear in numeric
    order from ``glob()``, and the generic streaming bank loader retains a layer
    until both tensors arrive.  Following the HF index in layer order therefore
    bounds conversion memory to one completed/partial layer instead of, in the
    worst case, the entire expert pool.

    The caller checks for an HF index before selecting this path.
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(index_path)
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    layers: dict[int, dict[str, tuple[str, str]]] = {}
    for raw, filename in weight_map.items():
        match = _EXPERT_LAYER.match(raw)
        if match is not None:
            layer, role = int(match.group(1)), match.group(2)
            layers.setdefault(layer, {})[role] = (raw, filename)
    if not layers:
        raise ValueError("Qwen4 checkpoint index contains no packed expert tensors")

    expected_layers = list(range(max(layers) + 1))
    if sorted(layers) != expected_layers:
        raise ValueError(f"Qwen4 checkpoint has non-contiguous expert layers: {sorted(layers)}")
    for layer in expected_layers:
        parts = layers[layer]
        missing = {"gate_up_proj", "down_proj"} - parts.keys()
        if missing:
            raise ValueError(f"Qwen4 expert layer {layer} is missing {sorted(missing)}")
        for role in ("gate_up_proj", "down_proj"):
            raw, filename = parts[role]
            path = os.path.join(model_path, filename)
            try:
                with safetensors.safe_open(path, framework="pt", device=str(device)) as handle:
                    name = _rename(raw)
                    assert name is not None
                    yield name, handle.get_tensor(raw)
            finally:
                # The consumer has copied this multi-GiB tensor into its bounded
                # per-layer bank before requesting the next item. Do not let the
                # original safetensors pages accumulate beside those banks.
                drop_page_cache(path)


def iter_weights(
    model_path: str, device: torch.device, *,
    include_moe_experts: bool, include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size != 1:
        raise NotImplementedError("Qwen4-Exp weight loading currently supports TP=1")
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    packed_index = False
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index_names = json.load(handle)["weight_map"]
        packed_index = any(_EXPERT_LAYER.match(name) is not None for name in index_names)
        per_expert_index = any(_PER_EXPERT.match(name) is not None for name in index_names)
    else:
        per_expert_index = False
    if include_moe_experts and per_expert_index:
        raise ValueError(
            "Qwen4 per-expert FP8/NVFP4 checkpoints require --moe-backend offload; "
            "resident expert loading is not supported"
        )
    if (
        include_moe_experts
        and not include_non_moe
        and packed_index
    ):
        yield from _iter_experts_layer_order(model_path, device)
        return
    pending: dict = {}
    shared: dict[str, dict[str, torch.Tensor]] = {}
    for filename in iter_weight_files(model_path):
        with safetensors.safe_open(filename, framework="pt", device=str(device)) as handle:
            for raw in handle.keys():
                name = _rename(raw)
                if name is None:
                    continue
                # Converted checkpoints pack experts by layer; source FP8/NVFP4 checkpoints
                # store per-expert weights plus quantization metadata. Both belong exclusively
                # to expert banks.
                is_expert = _EXPERT.match(name) is not None or _PER_EXPERT.match(raw) is not None
                if is_expert != include_moe_experts and not (
                    include_moe_experts and include_non_moe
                ):
                    continue
                tensor = handle.get_tensor(raw)

                if name.endswith((
                    ".mlp.shared_expert.gate_proj.weight",
                    ".mlp.shared_expert.up_proj.weight",
                )):
                    prefix, role = name.rsplit(".", 2)[0], name.rsplit(".", 2)[1]
                    slots = shared.setdefault(prefix, {})
                    slots[role] = tensor
                    if set(slots) == {"gate_proj", "up_proj"}:
                        yield prefix + ".gate_up_proj.weight", torch.cat(
                            [slots["gate_proj"], slots["up_proj"]], 0
                        )
                        del shared[prefix]
                    continue

                merged = _fuse(name, tensor, pending)
                if merged is not None:
                    if merged:
                        yield merged
                    continue
                yield name, tensor
    if pending or shared:
        raise ValueError(
            f"incomplete Qwen4 projection groups: fusions={list(pending)}, shared={list(shared)}"
        )


def setup_offload_expert_banks(*args, **kwargs):
    """Use the shared precision-preserving per-expert FP8/NVFP4 bank provider."""
    from sparklab.models.qwen3_5_moe.weight import setup_offload_expert_banks as setup

    return setup(*args, **kwargs)


def load_nvfp4_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Load Inferact's per-expert ModelOpt NVFP4 rows into native offload banks."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str,
    config,
    *,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
):
    """Parallel counterpart using the common chunked expert reader."""
    from sparklab.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


def _copy_range(src_fd: int, dst_fd: int, offset: int, length: int) -> None:
    remaining = length
    source = offset
    while remaining:
        try:
            copied = os.copy_file_range(src_fd, dst_fd, min(remaining, 1 << 30), source, None)
        except TypeError:  # older Python positional offset signature
            copied = os.copy_file_range(src_fd, dst_fd, min(remaining, 1 << 30), source)
        if copied == 0:
            raise OSError(f"short copy_file_range at {source}, {remaining} bytes remain")
        source += copied
        remaining -= copied


def find_mtp_sidecar(model_path: str) -> str | None:
    """Return the publisher's standalone Qwen4 MTP checkpoint, if present."""
    override = os.getenv("SPARKLAB_QWEN4_MTP_PATH")
    candidates = [override, os.path.join(model_path, _MTP_FILE)]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.realpath(path)
    return None


def _copy_file_atomic(source: str, destination: str) -> int:
    size = os.path.getsize(source)
    temporary = destination + ".tmp"
    src_fd = os.open(source, os.O_RDONLY)
    try:
        with open(temporary, "wb", buffering=0) as out:
            _copy_range(src_fd, out.fileno(), 0, size)
            os.fsync(out.fileno())
    finally:
        os.close(src_fd)
    os.replace(temporary, destination)
    return size


def _write_bf16_as_fp8(
    src_fd: int,
    dst_fd: int,
    *,
    source_offset: int,
    rows: int,
    dim: int,
    chunk_rows: int,
) -> int:
    """Stream one contiguous BF16 matrix into an E4M3 payload."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    input_row_bytes = dim * 2
    output_bytes = 0
    for row_start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - row_start)
        length = count * input_row_bytes
        payload = os.pread(
            src_fd,
            length,
            source_offset + row_start * input_row_bytes,
        )
        if len(payload) != length:
            raise OSError(f"short Qwen4 n-gram read: {len(payload)}/{length}")
        source = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16)
        quantized = source.to(torch.float8_e4m3fn).view(torch.uint8)
        data = quantized.numpy().tobytes()
        offset = 0
        while offset < len(data):
            written = os.write(dst_fd, data[offset:])
            if written <= 0:
                raise OSError(f"short Qwen4 n-gram write: {offset}/{len(data)}")
            offset += written
        output_bytes += offset
        try:
            os.posix_fadvise(
                src_fd,
                source_offset + row_start * input_row_bytes,
                length,
                os.POSIX_FADV_DONTNEED,
            )
        except OSError:
            pass
    return output_bytes


def requantize_raw_ngram_artifact(
    model_path: str,
    out_dir: str,
    *,
    storage_dtype: str = "float8_e4m3fn",
    chunk_rows: int = 65_536,
) -> dict:
    """Build a precision-reduced PLE overlay from an existing FTW side artifact.

    The target directory contains only the replacement manifest and payload. It can
    be copied over a hard-linked FTW clone without rewriting the model shards.
    """
    if storage_dtype not in _NGRAM_DTYPES:
        raise ValueError(f"unsupported Qwen4 n-gram target dtype: {storage_dtype!r}")
    source_dir = os.path.realpath(model_path)
    destination = os.path.realpath(out_dir)
    os.makedirs(destination, exist_ok=True)
    manifest_path = os.path.join(source_dir, _NGRAM_MANIFEST)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    source_dtype = str(manifest.get("dtype", ""))
    if source_dtype not in _NGRAM_DTYPES:
        raise ValueError(f"unsupported Qwen4 n-gram source dtype: {source_dtype!r}")
    rows, dim = int(manifest["rows"]), int(manifest["dim"])
    source_item_size = _NGRAM_DTYPES[source_dtype][1]
    expected = rows * dim * source_item_size
    if int(manifest.get("nbytes", -1)) != expected:
        raise ValueError(f"invalid Qwen4 n-gram manifest: {manifest}")
    source_file = os.path.join(source_dir, str(manifest["file"]))
    if os.path.getsize(source_file) != expected:
        raise ValueError(
            f"Qwen4 n-gram payload size mismatch: {os.path.getsize(source_file)}/{expected}"
        )

    final_path = os.path.join(destination, _NGRAM_PAYLOAD)
    temporary = final_path + ".tmp"
    if source_dtype == storage_dtype:
        _copy_file_atomic(source_file, final_path)
        output_bytes = expected
    elif source_dtype == "bfloat16" and storage_dtype == "float8_e4m3fn":
        src_fd = os.open(source_file, os.O_RDONLY)
        try:
            with open(temporary, "wb", buffering=0) as out:
                output_bytes = _write_bf16_as_fp8(
                    src_fd,
                    out.fileno(),
                    source_offset=0,
                    rows=rows,
                    dim=dim,
                    chunk_rows=chunk_rows,
                )
                os.fsync(out.fileno())
                try:
                    os.posix_fadvise(
                        out.fileno(), 0, output_bytes, os.POSIX_FADV_DONTNEED
                    )
                except OSError:
                    pass
        finally:
            os.close(src_fd)
        os.replace(temporary, final_path)
    else:
        raise ValueError(
            f"unsupported Qwen4 n-gram conversion: {source_dtype} -> {storage_dtype}"
        )

    result = {
        **manifest,
        "file": _NGRAM_PAYLOAD,
        "dtype": storage_dtype,
        "nbytes": output_bytes,
        "source_dtype": source_dtype,
        "quantization": "cast_rne" if source_dtype != storage_dtype else "preserve",
    }
    temp_manifest = os.path.join(destination, _NGRAM_MANIFEST + ".tmp")
    with open(temp_manifest, "w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
    os.replace(temp_manifest, os.path.join(destination, _NGRAM_MANIFEST))
    return result


def copy_external_artifacts(
    model_path: str,
    out_dir: str,
    model_config,
    *,
    ngram_dtype: str = "preserve",
) -> list[dict]:
    """Extract the PLE table into one precision-preserving random-read file.

    This streams exact safetensors data ranges in kernel space: no tensor is
    materialized and page cache pressure stays bounded. The final manifest is
    published atomically only after every pinned shard is present and validated.
    """
    args = model_config.qwen4_exp_args
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    parts = []
    for name, filename in weight_map.items():
        match = _SHARD_RE.search(name)
        if match:
            parts.append((int(match.group(1)), name, filename))
    parts.sort()
    if [p[0] for p in parts] != list(range(args.split_ngram_parts)):
        raise ValueError("Qwen4 external artifact is missing split n-gram tensors")

    if ngram_dtype not in {"preserve", "float8_e4m3fn"}:
        raise ValueError(f"unsupported Qwen4 n-gram target dtype: {ngram_dtype!r}")
    final_path = os.path.join(out_dir, _NGRAM_PAYLOAD)
    temporary = final_path + ".tmp"
    total_rows = total_bytes = 0
    storage_dtype = None
    dtype_bytes = {"BF16": ("bfloat16", 2), "F8_E4M3": ("float8_e4m3fn", 1)}
    with open(temporary, "wb", buffering=0) as out:
        for _, name, filename in parts:
            path = os.path.join(model_path, filename)
            fd = os.open(path, os.O_RDONLY)
            try:
                header_size = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
                header = json.loads(os.pread(fd, header_size, 8))
                meta = header[name]
                if meta["dtype"] not in dtype_bytes or len(meta["shape"]) != 2:
                    raise ValueError(f"unexpected Qwen4 n-gram tensor {name}: {meta}")
                source_dtype, item_size = dtype_bytes[meta["dtype"]]
                manifest_dtype = (
                    "float8_e4m3fn" if ngram_dtype == "float8_e4m3fn"
                    else source_dtype
                )
                if storage_dtype is None:
                    storage_dtype = manifest_dtype
                elif storage_dtype != manifest_dtype:
                    raise ValueError("Qwen4 n-gram shards use inconsistent storage dtypes")
                begin, end = meta["data_offsets"]
                length = end - begin
                expected = int(meta["shape"][0]) * int(meta["shape"][1]) * item_size
                if length != expected:
                    raise ValueError(f"invalid Qwen4 n-gram byte length for {name}")
                rows = int(meta["shape"][0])
                dim = int(meta["shape"][1])
                if source_dtype == manifest_dtype:
                    _copy_range(fd, out.fileno(), 8 + header_size + begin, length)
                    written = length
                elif source_dtype == "bfloat16" and manifest_dtype == "float8_e4m3fn":
                    written = _write_bf16_as_fp8(
                        fd,
                        out.fileno(),
                        source_offset=8 + header_size + begin,
                        rows=rows,
                        dim=dim,
                        chunk_rows=65_536,
                    )
                else:
                    raise ValueError(
                        f"unsupported Qwen4 n-gram conversion: "
                        f"{source_dtype} -> {manifest_dtype}"
                    )
                # Keep the destination from becoming dirty page-cache pressure on unified
                # memory. Commit and evict each part before copying the next one; source
                # pages are evicted below.
                os.fdatasync(out.fileno())
                try:
                    os.posix_fadvise(
                        out.fileno(), total_bytes, written, os.POSIX_FADV_DONTNEED
                    )
                except OSError:
                    pass
                total_rows += rows
                total_bytes += written
                try:
                    os.posix_fadvise(fd, 8 + header_size + begin, length, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            finally:
                os.close(fd)
        os.fsync(out.fileno())
    os.replace(temporary, final_path)
    manifest = {
        "schema_version": "1.0",
        "file": "qwen4_ngram.bin",
        "dtype": storage_dtype,
        "rows": total_rows,
        "dim": args.ple_embed_dim // ((args.ngram_size - 1) * args.heads_per_ngram),
        "nbytes": total_bytes,
        "parts": args.split_ngram_parts,
    }
    manifest_path = os.path.join(out_dir, _NGRAM_MANIFEST)
    temp_manifest = manifest_path + ".tmp"
    with open(temp_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
    os.replace(temp_manifest, manifest_path)
    artifacts = [{"kind": "qwen4_ngram", **manifest}]
    mtp_source = find_mtp_sidecar(model_path)
    if mtp_source is not None:
        mtp_file = _MTP_FILE
        mtp_bytes = _copy_file_atomic(mtp_source, os.path.join(out_dir, mtp_file))
        artifacts.append({
            "kind": "qwen4_mtp",
            "file": mtp_file,
            "nbytes": mtp_bytes,
            "format": "safetensors-nvfp4",
        })
    return artifacts


__all__ = [
    "copy_external_artifacts",
    "find_mtp_sidecar",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "requantize_raw_ngram_artifact",
    "setup_offload_expert_banks",
]
