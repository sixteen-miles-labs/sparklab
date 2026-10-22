"""Bounded nonexecutable prefix record framing and streaming checksums.

All filesystem functions are for the background I/O worker. No pickle, pointers,
physical page IDs, or model-worker disk operations belong to this format.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
import struct

from .state import StateLayout

MAGIC = b"SPKPX001"
PREFIX = struct.Struct("<8sI32s")
MAX_HEADER = 1 << 20
MAX_TOKENS = 131072
DTYPES = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e8m0fnu": 1,
    "torch.bool": 1,
}
HEX = re.compile(r"^[0-9a-f]{64}$")
NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class RecordError(ValueError):
    pass


@dataclass(frozen=True)
class RecordLimits:
    max_record_bytes: int = 4 << 30
    chunk_bytes: int = 8 << 20

    def __post_init__(self):
        if (
            not 1 <= self.max_record_bytes <= 64 << 30
            or not 4096 <= self.chunk_bytes <= 64 << 20
            or self.chunk_bytes % 4096
        ):
            raise ValueError("invalid record/staging bounds")


class TensorSource:
    """CPU image source. GPU transfer uses an explicitly staged source instead."""

    def __init__(self, image):
        if any(
            t.device.type != "cpu" or not t.is_contiguous()
            for t in image._tensors.values()
        ):
            raise ValueError("TensorSource requires owned contiguous CPU components")
        self.image = image
        self.compatibility = image.compatibility
        self.tokens = image.tokens
        self.layout = image.layout

    def chunks(self, name, chunk_bytes):
        data = memoryview(
            self.image._tensors[name]
            .view(__import__("torch").uint8)
            .reshape(-1)
            .numpy()
        )
        for pos in range(0, len(data), chunk_bytes):
            yield data[pos : pos + chunk_bytes]


def _json(header):
    return json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def make_header(source):
    entries = []
    offset = 0
    for name, shape, dtype in source.layout.components:
        nbytes = math.prod(shape) * DTYPES[dtype]
        entries.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": dtype,
                "offset": offset,
                "nbytes": nbytes,
                "sha256": "0" * 64,
            }
        )
        offset += nbytes
    return {
        "version": 1,
        "compatibility": source.compatibility.fingerprint,
        "scope": source.compatibility.scope,
        "tokens": list(source.tokens),
        "key": source.compatibility.prefix_key(source.tokens),
        "page_size": source.layout.page_size,
        "layer_map": list(source.layout.layer_map)
        if source.layout.layer_map is not None
        else None,
        "layout_digest": source.layout.fingerprint,
        "components": entries,
        "payload_bytes": offset,
    }


def validate_header(header, compatibility, limits):
    if (
        not isinstance(header, dict)
        or type(header.get("version")) is not int
        or header.get("version") != 1
    ):
        raise RecordError("unsupported prefix record version")
    required = {
        "version",
        "compatibility",
        "scope",
        "tokens",
        "key",
        "page_size",
        "layer_map",
        "layout_digest",
        "components",
        "payload_bytes",
    }
    if set(header) != required:
        raise RecordError("unsupported header fields")
    if (
        header.get("compatibility") != compatibility.fingerprint
        or header.get("scope") != compatibility.scope
    ):
        raise RecordError("incompatible prefix record identity/scope")
    tokens = header.get("tokens")
    if (
        not isinstance(tokens, list)
        or not 1 <= len(tokens) <= MAX_TOKENS
        or any(type(t) is not int or not 0 <= t < 2**31 for t in tokens)
    ):
        raise RecordError("invalid consumed prefix")
    key = header.get("key")
    if (
        not isinstance(key, str)
        or not HEX.fullmatch(key)
        or key != compatibility.prefix_key(tokens)
    ):
        raise RecordError("consumed prefix checksum mismatch")
    page_size = header.get("page_size")
    if type(page_size) is not int or not 1 <= page_size <= 65536:
        raise RecordError("invalid page geometry")
    layers = header.get("layer_map")
    if layers is not None and (
        not isinstance(layers, list)
        or len(layers) > 4096
        or any(type(i) is not int or not -1 <= i < 4096 for i in layers)
    ):
        raise RecordError("invalid layer mapping")
    components = header.get("components")
    if not isinstance(components, list) or not 1 <= len(components) <= 64:
        raise RecordError("invalid component count")
    offset = 0
    seen = set()
    layout = []
    for part in components:
        if not isinstance(part, dict):
            raise RecordError("invalid component")
        name = part.get("name")
        shape = part.get("shape")
        dtype = part.get("dtype")
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in seen:
            raise RecordError("invalid/duplicate component name")
        seen.add(name)
        if (
            not isinstance(shape, list)
            or not 1 <= len(shape) <= 8
            or any(type(d) is not int or not 0 <= d <= 2**31 for d in shape)
            or not isinstance(dtype, str)
            or dtype not in DTYPES
        ):
            raise RecordError("invalid component dimensions/dtype")
        if set(part) != {"name", "shape", "dtype", "offset", "nbytes", "sha256"}:
            raise RecordError("unsupported component fields")
        if name == "kv" and (
            len(shape) != 5
            or shape[0] != 2
            or shape[2] != len(tokens)
            or any(d == 0 for d in shape)
        ):
            raise RecordError("KV geometry does not describe the consumed prefix")
        nbytes = math.prod(shape) * DTYPES[dtype]
        if type(part.get("nbytes")) is not int or type(part.get("offset")) is not int:
            raise RecordError("invalid length/offset type")
        if (
            nbytes > limits.max_record_bytes
            or part.get("nbytes") != nbytes
            or part.get("offset") != offset
        ):
            raise RecordError("invalid component length/offset")
        if not isinstance(part.get("sha256"), str) or not HEX.fullmatch(part["sha256"]):
            raise RecordError("invalid component checksum")
        offset += nbytes
        if offset > limits.max_record_bytes:
            raise RecordError("oversized record")
        layout.append((name, tuple(shape), dtype))
    if (
        "kv" not in seen
        or type(header.get("payload_bytes")) is not int
        or header.get("payload_bytes") != offset
    ):
        raise RecordError("incomplete prefix record")
    actual = StateLayout(
        tuple(sorted(layout)), page_size, tuple(layers) if layers is not None else None
    )
    if actual.fingerprint != header.get("layout_digest"):
        raise RecordError("state layout checksum mismatch")
    if len(_json(header)) > MAX_HEADER:
        raise RecordError("oversized header")
    return actual


def write_all(fd, data):
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0 or count > len(view):
            raise OSError("invalid/short record write")
        view = view[count:]


def read_exact(fd, size):
    chunks = bytearray()
    while len(chunks) < size:
        data = os.read(fd, size - len(chunks))
        if not data:
            raise RecordError("truncated prefix record")
        if len(data) > size - len(chunks):
            raise RecordError("oversized read result")
        chunks.extend(data)
    return chunks


def write_record(fd, source, limits):
    header = make_header(source)
    validate_header(header, source.compatibility, limits)
    encoded = _json(header)
    total = PREFIX.size + len(encoded) + header["payload_bytes"]
    if total > limits.max_record_bytes:
        raise RecordError("oversized framed record")
    write_all(fd, PREFIX.pack(MAGIC, len(encoded), bytes(32)))
    write_all(fd, encoded)
    for part in header["components"]:
        checksum = hashlib.sha256()
        written = 0
        for chunk in source.chunks(part["name"], limits.chunk_bytes):
            if (
                not 0 < len(chunk) <= limits.chunk_bytes
                or written + len(chunk) > part["nbytes"]
            ):
                raise RecordError("invalid/oversized source chunk")
            write_all(fd, chunk)
            checksum.update(chunk)
            written += len(chunk)
        if written != part["nbytes"]:
            raise RecordError("short component source")
        part["sha256"] = checksum.hexdigest()
    final = _json(header)
    if len(final) != len(encoded):
        raise AssertionError("fixed-width checksum framing changed")
    os.lseek(fd, 0, os.SEEK_SET)
    write_all(fd, PREFIX.pack(MAGIC, len(final), hashlib.sha256(final).digest()))
    write_all(fd, final)
    if os.fstat(fd).st_size != total:
        raise RecordError("record byte count mismatch")
    return header, total


def read_header(fd, compatibility, limits):
    size = os.fstat(fd).st_size
    if size < PREFIX.size or size > limits.max_record_bytes:
        raise RecordError("invalid record file size")
    os.lseek(fd, 0, os.SEEK_SET)
    magic, length, checksum = PREFIX.unpack(read_exact(fd, PREFIX.size))
    if magic != MAGIC or not 1 <= length <= MAX_HEADER or PREFIX.size + length > size:
        raise RecordError("invalid record framing/header length")
    raw = read_exact(fd, length)
    if hashlib.sha256(raw).digest() != checksum:
        raise RecordError("header checksum mismatch")
    try:
        header = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RecordError("nonfinite header")
            ),
        )
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise RecordError("invalid record JSON") from exc
    layout = validate_header(header, compatibility, limits)
    if PREFIX.size + length + header["payload_bytes"] != size:
        raise RecordError("declared file length mismatch")
    return header, layout


def read_record(fd, compatibility, limits, sink=None):
    """Validate framing before any sink allocation; copy only into private state.

    A sink is never published by this function. The caller may publish only on a
    successful return, after its own GPU completion fence. Corruption after some
    private writes is a failed restore, not a partially visible shared checkpoint.
    """
    header, layout = read_header(fd, compatibility, limits)
    if sink is not None:
        sink.begin(header, layout)
    for part in header["components"]:
        checksum = hashlib.sha256()
        offset = 0
        while offset < part["nbytes"]:
            size = min(limits.chunk_bytes, part["nbytes"] - offset)
            preferred = getattr(sink, "preferred_size", None)
            if preferred is not None:
                size = preferred(part["name"], offset, size)
                if type(size) is not int or not 0 < size <= min(
                    limits.chunk_bytes, part["nbytes"] - offset
                ):
                    raise RecordError("invalid staged read size")
            chunk = read_exact(fd, size)
            checksum.update(chunk)
            if sink is not None:
                sink.chunk(part["name"], offset, chunk)
            offset += len(chunk)
        if checksum.hexdigest() != part["sha256"]:
            raise RecordError("component checksum mismatch")
    return header, layout
