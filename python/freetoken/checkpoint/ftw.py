"""FreeToken Weight (FTW) checkpoint: one O_DIRECT-friendly on-disk format for a whole model.

The format is a single *logical contiguous byte region* of all tensors, sliced *physically*
into shard files of at most ``shard_limit`` bytes (default 8 GiB, for HF/filesystem
friendliness). It exists because reading the original safetensors back fast is awkward:
tensors are packed with no alignment, so an individual tensor can't be O_DIRECT-read at an
arbitrary offset; and the earlier per-bank cache prototype worked around that by giving every expert bank
its own file -- which doesn't cover dense weights and turns a model's long tail of tiny
tensors (norms, biases, router) into hundreds of tiny I/Os.

FTW fixes both:

* **Aligned.** Every tensor starts at a 4096-aligned region offset and is padded to 4096;
  shards are cut at 4096-aligned boundaries. So any tensor (or any shard-local slice of one)
  is read with offset, length (rounded up to 4096), and destination all block-aligned --
  exactly what O_DIRECT requires. A tensor larger than a shard simply spans shards; because
  both its start and the shard boundary are aligned, each piece stays aligned.
* **Unified.** It holds dense weights as ``kind="weight"`` (exactly what a model's
  ``iter_weights`` yields -- post fusion/TP-shard, fed straight to ``load_state_dict``) and
  the offload expert state as ``kind="experts_bank"`` (post backend-repack -- the per-expert
  weight banks plus, distinguished only by their reserved names, the alpha scale vectors;
  the FTW content). The converter runs the per-model loaders once; this reader is
  model-agnostic.

Layout on disk::

    <dir>/freetoken_weight.json        # index: tensors[] + shards[] + meta
    <dir>/freetoken-00000.ftw         # the byte region, sliced <= shard_limit
    <dir>/freetoken-00001.ftw
    <dir>/config.json, tokenizer*, ...# copied so the dir is a self-contained checkpoint
"""

from __future__ import annotations

import json
import logging
import math
import mmap
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

INDEX_NAME = "freetoken_weight.json"
FORMAT_TAG = "freetoken_weight"
FORMAT_VERSION = 1
ALIGN = 4096  # O_DIRECT block alignment (== page size on this platform)
DEFAULT_SHARD_LIMIT = 8 << 30  # 8 GiB; must be a multiple of ALIGN
_SHARD_FMT = "freetoken-{:05d}.ftw"
_DEFAULT_CHUNK = 8 << 20
_BANK_CONCURRENCY = 4
_DISK_EXPERT_READ_WORKERS = 16
_ALPHA_NAMES = ("gate_up_alpha", "down_alpha")
# Per-layer expert-bank entry name (converter streaming path, see checkpoint/convert.py):
# each layer of a bank is its own FTW tensor instead of one flat [num_layers*E, ...] region.
_LAYER_ENTRY_RE = re.compile(r"^(?P<base>.+)#L(?P<layer>\d{5})$")


def layer_bank_entry_name(bank_name: str, layer_id: int) -> str:
    """Name of one per-layer ``experts_bank`` FTW entry; :func:`load_ftw_banks` groups
    entries matching ``_LAYER_ENTRY_RE`` back into a per-layer bank list by base name."""
    return f"{bank_name}#L{layer_id:05d}"


@dataclass(frozen=True)
class ExpertRowDescriptor:
    """Byte address of one expert row inside an FTW expert-bank entry.

    ``read_off``/``read_nbytes`` describe the aligned enclosing window required by
    ``O_DIRECT``. The logical row begins at ``head_pad`` within that window.
    """

    bank_name: str
    layer_id: int
    expert_id: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    global_off: int
    nbytes: int
    read_off: int
    read_nbytes: int
    head_pad: int


def _expert_row_descriptor(
    entry: dict, *, bank_name: str, layer_id: int, expert_id: int
) -> ExpertRowDescriptor:
    num_experts, *row_shape = entry["shape"]
    if not 0 <= expert_id < num_experts:
        raise IndexError(
            f"expert_id {expert_id} out of range for {bank_name!r} layer {layer_id} "
            f"with {num_experts} experts"
        )
    dtype = _dtype_of(entry["dtype"])
    row_bytes = (math.prod(row_shape) if row_shape else 1) * _elsize(dtype)
    assert row_bytes * num_experts == entry["nbytes"], (
        entry["name"], row_bytes, num_experts, entry["nbytes"]
    )
    off = entry["global_off"] + expert_id * row_bytes
    read_off = (off // ALIGN) * ALIGN
    read_end = _align_up(off + row_bytes)
    return ExpertRowDescriptor(
        bank_name=bank_name,
        layer_id=layer_id,
        expert_id=expert_id,
        dtype=dtype,
        shape=tuple(row_shape),
        global_off=off,
        nbytes=row_bytes,
        read_off=read_off,
        read_nbytes=read_end - read_off,
        head_pad=off - read_off,
    )


def _pread_into(fd: int, mv: memoryview, offset: int) -> None:
    """POSIX positional read into ``mv`` at ``offset``, looping over any short preadv.

    preadv may return short (a signal, or the EOF-adjacent tail); the bare call this
    replaces ignored the return value, so a short read silently left the tail of the
    destination unfilled -- garbage bytes in the middle of a weight tensor with no
    error anywhere. Resuming at the running offset stays O_DIRECT-legal: the writer
    pads every tensor to ALIGN and cuts shards at ALIGN boundaries, so direct-IO
    short reads land on block boundaries."""
    done = 0
    total = len(mv)
    while done < total:
        n = os.preadv(fd, [mv[done:]], offset + done)
        if n == 0:
            break  # EOF
        done += n


def _align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _dtype_str(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def _dtype_of(s: str) -> torch.dtype:
    return getattr(torch, s)


def _elsize(dt: torch.dtype) -> int:
    return torch.empty((), dtype=dt).element_size()


def is_ftw_checkpoint(path: str) -> bool:
    """True if ``path`` is a directory holding a FreeToken Weight (FTW) index."""
    return os.path.isfile(os.path.join(path, INDEX_NAME))


# ============================== writer ==============================
class FTWWriter:
    """Stream tensors into the FTW, rolling shard files at ``shard_limit``.

    Tensors are written in call order into one logical byte stream; each is padded to
    ``ALIGN`` so the next starts aligned. A tensor that doesn't fit the current shard's
    remaining room is split across shards (the split point is the shard boundary, which is
    aligned). Call :meth:`add_tensor` for each tensor, then :meth:`finalize`.
    """

    def __init__(self, out_dir: str, *, shard_limit: int = DEFAULT_SHARD_LIMIT):
        assert shard_limit % ALIGN == 0, "shard_limit must be a multiple of ALIGN"
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_limit = shard_limit
        self._tensors: list[dict] = []
        self._shards: list[dict] = []
        self._global = 0  # running FTW offset (incl. padding)
        self._f = None  # current shard file handle
        self._shard_idx = -1
        self._shard_start = 0  # FTW offset where the current shard began
        self._cur = 0  # bytes written to the current shard

    def _finish_shard(self) -> None:
        """Commit and evict the completed shard's buffered pages.

        Conversion can write checkpoints larger than host RAM.  Merely closing a file
        leaves its dirty pages in Linux's page cache, so a fast producer can accumulate
        almost the entire checkpoint in memory and drive the machine into global OOM
        reclaim.  Sync before POSIX_FADV_DONTNEED: the latter is only a hint and cannot
        discard dirty pages reliably.
        """
        if self._f is None:
            return
        f = self._f
        self._f = None
        try:
            f.flush()
            os.fsync(f.fileno())
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            f.close()

    def _roll(self) -> None:
        if self._f is not None:
            self._shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                 "global_off": self._shard_start, "nbytes": self._cur})
            self._finish_shard()
        self._shard_idx += 1
        self._shard_start = self._global
        self._cur = 0
        self._f = open(os.path.join(self.out_dir, _SHARD_FMT.format(self._shard_idx)), "wb")

    def _write_raw(self, data: memoryview) -> None:
        """Write ``data`` into the FTW byte stream, splitting across shards at the limit."""
        if self._f is None:
            self._roll()
        off = 0
        n = len(data)
        while off < n:
            if self._cur == self.shard_limit:
                self._roll()
            take = min(n - off, self.shard_limit - self._cur)
            self._f.write(data[off:off + take])
            off += take
            self._cur += take
            self._global += take

    def add_tensor(self, name: str, tensor: torch.Tensor, kind: str = "weight") -> None:
        t = tensor.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8)
        nbytes = int(raw.numel())
        # A small tensor (<= shard) never splits: roll early so it lands whole in one shard.
        if self._f is None or (nbytes <= self.shard_limit
                               and self._cur + nbytes > self.shard_limit):
            self._roll()
        global_off = self._global
        assert global_off % ALIGN == 0, "tensor start must be aligned (invariant)"
        self._write_raw(memoryview(raw.numpy()))
        self._tensors.append({"name": name, "kind": kind, "dtype": _dtype_str(t.dtype),
                              "shape": list(t.shape), "global_off": global_off, "nbytes": nbytes})
        # pad to ALIGN so the next tensor starts aligned
        pad = _align_up(self._global) - self._global
        if pad:
            self._write_raw(memoryview(bytes(pad)))

    def finalize(self, meta: dict) -> dict:
        if self._f is not None:
            self._shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                 "global_off": self._shard_start, "nbytes": self._cur})
            self._finish_shard()
        index = {"format": FORMAT_TAG, "version": FORMAT_VERSION, "align": ALIGN,
                 "shard_limit": self.shard_limit, "total_bytes": self._global,
                 "tensors": self._tensors, "shards": self._shards, **meta}
        tmp = os.path.join(self.out_dir, INDEX_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(index, f)
        os.replace(tmp, os.path.join(self.out_dir, INDEX_NAME))
        return index


# ============================== reader ==============================
class FTWReader:
    """Random-access reader over an FTW checkpoint.

    Maps a tensor's logical byte range to one-or-more shard-file ranges (split at shard
    boundaries) and reads each piece with chunked multi-threaded O_DIRECT directly into the
    destination buffer. Offsets/lengths are all 4096-aligned (lengths rounded up into the
    rounded-up destination), so O_DIRECT is always legal -- including the tail of a tensor
    (the rounding reads into the region's padding, which is discarded by the tensor view)."""

    def __init__(self, path: str):
        with open(os.path.join(path, INDEX_NAME)) as f:
            self.index = json.load(f)
        assert self.index.get("format") == FORMAT_TAG, f"not a {FORMAT_TAG}: {path}"
        self.dir = path
        self.shards = sorted(self.index["shards"], key=lambda s: s["global_off"])
        self.tensors = {t["name"]: t for t in self.index["tensors"]}
        self._fds: dict[str, int] = {}
        self._maps: dict[str, tuple[mmap.mmap, memoryview]] = {}
        # O_DIRECT (DMA straight from disk, bypassing the page cache) is the fast path but a
        # perf choice, not a correctness one. Some filesystems reject it at open with EINVAL
        # (tmpfs, many overlay/network mounts) and the flag is Linux-only; when it's absent
        # we fall back to mmap (below), NOT to chunked buffered preadv -- a whole-shard
        # mapping + kernel readahead copies far faster than per-chunk page-cache reads.
        # 0 here means "O_DIRECT unavailable -> use the mmap path".
        self._direct = getattr(os, "O_DIRECT", 0)
        self._probed = False
        self._lock = threading.Lock()  # load_ftw_banks calls read_into concurrently

    def meta(self, key: str, default=None):
        return self.index.get(key, default)

    def entries(self, *kinds: str) -> list[dict]:
        keep = set(kinds)
        return [t for t in self.index["tensors"] if not keep or t["kind"] in keep]

    def expert_row_descriptors(
        self, *, num_layers: int
    ) -> dict[tuple[int, int, str], ExpertRowDescriptor]:
        """Index every routed-expert row without reading weight data.

        Supports both FTW expert layouts accepted by :func:`load_ftw_banks`: one flat
        ``[layers * experts, ...]`` entry or independently aligned per-layer entries.
        Alpha vectors are fixed-resident metadata and intentionally excluded.
        """
        entries = [
            e for e in self.entries("experts_bank")
            if e["name"] not in _ALPHA_NAMES
        ]
        flat: list[dict] = []
        layered: dict[str, dict[int, dict]] = {}
        for entry in entries:
            match = _LAYER_ENTRY_RE.match(entry["name"])
            if match is None:
                flat.append(entry)
            else:
                layered.setdefault(match.group("base"), {})[
                    int(match.group("layer"))
                ] = entry

        mixed = {e["name"] for e in flat} & layered.keys()
        if mixed:
            raise ValueError(f"FTW bank(s) mix flat and per-layer layouts: {sorted(mixed)}")

        result: dict[tuple[int, int, str], ExpertRowDescriptor] = {}
        for entry in flat:
            total, *row_shape = entry["shape"]
            if total % num_layers:
                raise ValueError(
                    f"FTW bank {entry['name']!r} has {total} rows, not divisible by "
                    f"num_layers={num_layers}"
                )
            num_experts = total // num_layers
            dtype = _dtype_of(entry["dtype"])
            row_bytes = (math.prod(row_shape) if row_shape else 1) * _elsize(dtype)
            for layer_id in range(num_layers):
                layer_entry = {
                    **entry,
                    "shape": [num_experts, *row_shape],
                    "global_off": entry["global_off"] + layer_id * num_experts * row_bytes,
                    "nbytes": num_experts * row_bytes,
                }
                for expert_id in range(num_experts):
                    desc = _expert_row_descriptor(
                        layer_entry,
                        bank_name=entry["name"],
                        layer_id=layer_id,
                        expert_id=expert_id,
                    )
                    result[(layer_id, expert_id, entry["name"])] = desc

        for bank_name, by_layer in layered.items():
            expected = list(range(num_layers))
            if sorted(by_layer) != expected:
                raise ValueError(
                    f"FTW bank {bank_name!r} has layers {sorted(by_layer)}, expected {expected}"
                )
            for layer_id, entry in by_layer.items():
                num_experts = entry["shape"][0]
                for expert_id in range(num_experts):
                    desc = _expert_row_descriptor(
                        entry,
                        bank_name=bank_name,
                        layer_id=layer_id,
                        expert_id=expert_id,
                    )
                    result[(layer_id, expert_id, bank_name)] = desc
        return result

    def read_expert_row(self, descriptor: ExpertRowDescriptor) -> torch.Tensor:
        """Read one descriptor into an owning CPU tensor.

        This correctness-oriented API performs one aligned FTW read and clones the logical
        slice. The bounded host cache will reuse the descriptor with persistent pinned
        staging buffers instead of allocating once per miss.
        """
        buf = _transient_buffer(descriptor.read_nbytes)
        try:
            self.read_into(
                memoryview(buf),
                {"global_off": descriptor.read_off, "nbytes": descriptor.read_nbytes},
            )
            raw = torch.frombuffer(
                buf,
                dtype=torch.uint8,
                count=descriptor.nbytes,
                offset=descriptor.head_pad,
            ).clone()
        finally:
            buf.close()
        row = raw.view(descriptor.dtype)
        return row.view(*descriptor.shape) if descriptor.shape else row.view(())

    def _ensure_mode(self) -> None:
        """Resolve the read backend once: keep O_DIRECT if the filesystem accepts it, else
        drop to the mmap fallback. Thread-safe -- ``_probed`` is published only after
        ``_direct`` is final, so a concurrent reader never races onto a stale direct path."""
        if self._probed:
            return
        with self._lock:
            if self._probed:
                return
            if self._direct and self.shards:
                try:
                    os.close(os.open(os.path.join(self.dir, self.shards[0]["file"]),
                                     os.O_RDONLY | self._direct))
                except OSError:
                    self._direct = 0
                    logger.warning("O_DIRECT unsupported on %s; using mmap fallback for "
                                   "FTW load", self.dir)
            self._probed = True

    def _fd(self, file: str) -> int:
        fd = self._fds.get(file)
        if fd is None:
            with self._lock:  # first-open only; chunk reads reuse the cached fd lock-free
                fd = self._fds.get(file)
                if fd is None:
                    fd = os.open(os.path.join(self.dir, file), os.O_RDONLY | self._direct)
                    self._fds[file] = fd
        return fd

    def _map(self, file: str) -> memoryview:
        entry = self._maps.get(file)
        if entry is None:
            with self._lock:
                entry = self._maps.get(file)
                if entry is None:
                    fd = os.open(os.path.join(self.dir, file), os.O_RDONLY)
                    try:
                        m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                    finally:
                        os.close(fd)  # the mapping keeps its own reference to the file
                    try:
                        m.madvise(mmap.MADV_SEQUENTIAL)  # kernel readahead for streaming
                    except (AttributeError, OSError):
                        pass
                    entry = (m, memoryview(m))
                    self._maps[file] = entry
        return entry[1]

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        for m, mv in self._maps.values():
            mv.release()
            m.close()
        self._maps.clear()

    def _pieces(self, global_off: int, nbytes: int):
        """Yield (file, file_off, dest_off, length) covering [global_off, +nbytes),
        split at shard boundaries. All file_off/dest_off are ALIGN-aligned."""
        dest_off = 0
        remaining = nbytes
        pos = global_off
        for sh in self.shards:
            s0, s1 = sh["global_off"], sh["global_off"] + sh["nbytes"]
            if pos >= s1 or remaining <= 0:
                continue
            if pos < s0:  # regions are contiguous; a gap means a corrupt index
                raise ValueError("FTW gap / misordered shards")
            take = min(remaining, s1 - pos)
            yield sh["file"], pos - s0, dest_off, take
            pos += take
            dest_off += take
            remaining -= take
        if remaining:
            raise ValueError("tensor range exceeds FTW shards")

    def read_into(self, dest: memoryview, entry: dict, *, workers: int = 8,
                  chunk: int = _DEFAULT_CHUNK) -> None:
        """Read one tensor's bytes into ``dest`` (length >= entry nbytes rounded to ALIGN)."""
        self._ensure_mode()
        jobs = []  # (file, file_off, dest_off, length) all ALIGN-aligned
        for file, file_off, dest_off, length in self._pieces(entry["global_off"], entry["nbytes"]):
            rlen = _align_up(length)  # round the tail up; padding is in-region, harmless
            for c in range(0, rlen, chunk):
                jobs.append((file, file_off + c, dest_off + c, min(chunk, rlen - c)))

        # Open/map each distinct shard once, single-threaded, so the pool only reuses handles.
        touch = self._fd if self._direct else self._map
        for file in {j[0] for j in jobs}:
            touch(file)

        if self._direct:
            def rd(job):
                file, fo, do, ln = job
                _pread_into(self._fd(file), dest[do:do + ln], fo)
        else:
            def rd(job):
                file, fo, do, ln = job
                dest[do:do + ln] = self._map(file)[fo:fo + ln]  # memcpy from the mapping

        if len(jobs) <= 1:
            for j in jobs:
                rd(j)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, jobs))


def _transient_buffer(nbytes: int) -> mmap.mmap:
    return mmap.mmap(-1, _align_up(nbytes))


class FTWDiskExpertSource:
    """Bounded disk source with synchronous decode and optional prefill lookahead.

    Decode stages only requested rows into one parity-selected pinned layer. Prefill may
    use two pinned layer buffers so layer N+1's O_DIRECT read overlaps layer N's GPU work.
    The second buffer is charged against the pageable host-LRU budget by
    :func:`open_ftw_disk_banks`, keeping total host allocation bounded.
    """

    def __init__(self, reader: FTWReader, descriptors, staging, cache_banks=None,
                 *, read_workers: int | None = None,
                 cache_policy: str | None = None,
                 staging_buffers=None) -> None:
        self.reader = reader
        self.descriptors = descriptors
        self.staging_buffers = list(staging_buffers or [staging])
        self.staging = self.staging_buffers[0]
        self.num_staging_buffers = len(self.staging_buffers)
        self.cache_banks = cache_banks or {}
        self.cache_capacity = next(iter(self.cache_banks.values())).tensor.shape[0] if self.cache_banks else 0
        self.cache_row_bytes = sum(bank.tensor[0].numel() * bank.tensor.element_size()
                                   for bank in self.staging.values())
        self.cache_allocated_bytes = sum(
            _align_up(getattr(bank, "nbytes", bank.tensor.numel() * bank.tensor.element_size()))
            for bank in self.cache_banks.values()
        )
        self.staging_allocated_bytes = sum(
            _align_up(getattr(bank, "nbytes", bank.tensor.numel() * bank.tensor.element_size()))
            for buffers in self.staging_buffers
            for bank in buffers.values()
        )
        self.host_allocated_bytes = self.staging_allocated_bytes + self.cache_allocated_bytes
        self._cache: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._free_slots = list(range(self.cache_capacity - 1, -1, -1))
        if cache_policy is None:
            cache_policy = os.getenv("FREETOKEN_DISK_CACHE_POLICY", "lru")
        self.cache_policy = cache_policy.strip().lower()
        if self.cache_policy not in {"lru", "layer_lru"}:
            raise ValueError(
                "FREETOKEN_DISK_CACHE_POLICY must be 'lru' or 'layer_lru', "
                f"got {self.cache_policy!r}"
            )
        self._num_layers = max((key[0] for key in descriptors), default=-1) + 1
        self._layer_caches = [OrderedDict() for _ in range(self._num_layers)]
        base, extra = divmod(self.cache_capacity, max(1, self._num_layers))
        self._layer_cache_quotas = [
            base + (layer_id < extra) for layer_id in range(self._num_layers)
        ]
        self._cache_recency: dict[tuple[int, int], int] = {}
        self._cache_step = 0
        self._lock = threading.Lock()
        if read_workers is None:
            read_workers = int(os.getenv(
                "FREETOKEN_DISK_READ_WORKERS", str(_DISK_EXPERT_READ_WORKERS)
            ))
        self.read_workers = max(1, int(read_workers))
        # A persistent pool avoids creating threads for every MoE layer. Each task owns
        # only one aligned expert-bank row, so peak transient memory stays bounded by
        # read_workers rather than the number of rows in a full-layer prefill.
        self._read_pool = ThreadPoolExecutor(max_workers=self.read_workers,
                                             thread_name_prefix="ftw-expert")
        self._prefill_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="ftw-prefill")
            if self.num_staging_buffers > 1 else None
        )
        self._prefill_futures = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.cache_bypasses = 0
        self.read_ops = 0
        self.logical_bytes = 0
        self.physical_bytes = 0
        self.read_seconds = 0.0

    def _stage_bank_row(self, job) -> tuple[int, int]:
        layer_id, expert_id, bank_name, slot, buffer_id = job
        bank = self.staging_buffers[buffer_id][bank_name]
        desc = self.descriptors[(layer_id, expert_id, bank_name)]
        # InferenceMode is thread-local. Engine construction creates the host banks under
        # inference mode, so pool workers must enter it explicitly before mutating them.
        with torch.inference_mode():
            bank.tensor[expert_id].copy_(self.reader.read_expert_row(desc))
            if slot is not None:
                self.cache_banks[bank_name].tensor[slot].copy_(bank.tensor[expert_id])
        return desc.nbytes, desc.read_nbytes

    def _stage_cached_row(self, job) -> None:
        expert_id, bank_name, slot, buffer_id = job
        with torch.inference_mode():
            self.staging_buffers[buffer_id][bank_name].tensor[expert_id].copy_(
                self.cache_banks[bank_name].tensor[slot]
            )

    def _cache_staged_row(self, job) -> None:
        expert_id, bank_name, slot, buffer_id = job
        with torch.inference_mode():
            self.cache_banks[bank_name].tensor[slot].copy_(
                self.staging_buffers[buffer_id][bank_name].tensor[expert_id]
            )

    def _read_staging_extent(self, job) -> tuple[int, int, int]:
        layer_id, bank_name, first_expert, count, row_bytes, buffer_id = job
        bank = self.staging_buffers[buffer_id][bank_name]
        desc = self.descriptors[(layer_id, first_expert, bank_name)]
        whole = bank.memoryview()
        dest = whole[first_expert * row_bytes:(first_expert + count) * row_bytes]
        try:
            self.reader.read_into(
                dest,
                {"global_off": desc.global_off, "nbytes": count * row_bytes},
                workers=1,
            )
        finally:
            dest.release()
            whole.release()
        return count * row_bytes, count * row_bytes, count

    def _direct_extent_jobs(self, layer_id: int, misses, buffer_id: int) -> list | None:
        """Coalesce aligned consecutive rows directly into the pinned staging banks.

        Returns ``None`` when a bank/row layout cannot satisfy O_DIRECT alignment; the
        caller then keeps the general transient-buffer row path.
        """
        expert_ids = sorted(expert_id for expert_id, _ in misses)
        jobs = []
        for bank_name, bank in self.staging_buffers[buffer_id].items():
            if not callable(getattr(bank, "memoryview", None)):
                return None
            rows = [self.descriptors[(layer_id, expert_id, bank_name)]
                    for expert_id in expert_ids]
            if any(
                row.head_pad or row.nbytes != row.read_nbytes
                or row.global_off % ALIGN or row.nbytes % ALIGN
                for row in rows
            ):
                return None
            start = previous = expert_ids[0]
            row_bytes = rows[0].nbytes
            if any(row.nbytes != row_bytes for row in rows):
                return None
            for expert_id in expert_ids[1:]:
                if expert_id != previous + 1:
                    jobs.append((layer_id, bank_name, start,
                                 previous - start + 1, row_bytes, buffer_id))
                    start = expert_id
                previous = expert_id
            jobs.append((layer_id, bank_name, start, previous - start + 1,
                         row_bytes, buffer_id))
        return jobs

    def stage(self, layer_id: int, expert_ids: list[int], *, admit: bool = True,
              buffer_id: int | None = None) -> None:
        import time

        if not expert_ids:
            return
        if buffer_id is None:
            buffer_id = layer_id % self.num_staging_buffers
        staging = self.staging_buffers[buffer_id]
        # Cache kernels already emit unique source indices, but CPU-overflow routing can
        # repeat an expert. Coalesce it before reserving cache slots or issuing I/O.
        expert_ids = list(dict.fromkeys(expert_ids))
        start = time.perf_counter()
        # The current disk path is synchronous. Holding one lock makes duplicate concurrent
        # requests coalesce naturally and keeps LRU state deterministic; async I/O will
        # replace this with explicit loading/ready entry states in the next phase.
        with self._lock:
            read_jobs = []
            misses = []
            cached_jobs = []
            admitted = []
            for expert_id in expert_ids:
                key = (layer_id, expert_id)
                slot = self._cache.get(key)
                if slot is not None:
                    self.cache_hits += 1
                    if admit:
                        self._cache.move_to_end(key)
                        if self.cache_policy == "layer_lru":
                            self._layer_caches[layer_id].move_to_end(expert_id)
                            self._cache_step += 1
                            self._cache_recency[key] = self._cache_step
                    cached_jobs.extend(
                        (expert_id, bank_name, slot, buffer_id) for bank_name in staging
                    )
                    continue

                self.cache_misses += 1
                slot = None
                if admit and self.cache_capacity:
                    if self.cache_policy == "layer_lru":
                        layer_cache = self._layer_caches[layer_id]
                        quota = self._layer_cache_quotas[layer_id]
                        if self._free_slots:
                            slot = self._free_slots.pop()
                        else:
                            # Borrow unused capacity freely, but once full evict from a
                            # layer above its fair share before touching protected rows.
                            over_quota = [
                                lid for lid, entries in enumerate(self._layer_caches)
                                if len(entries) > self._layer_cache_quotas[lid]
                            ]
                            if over_quota:
                                victim_layer = min(
                                    over_quota,
                                    key=lambda lid: self._cache_recency[
                                        (lid, next(iter(self._layer_caches[lid])))
                                    ],
                                )
                            elif quota and layer_cache:
                                # Every layer is exactly at its floor. Replace within
                                # this layer so another layer cannot fall below its floor.
                                victim_layer = layer_id
                            else:
                                # More layers than slots: a zero-quota layer cannot
                                # displace a layer that owns one of the protected slots.
                                victim_layer = None
                            if victim_layer is not None:
                                victim_expert, slot = self._layer_caches[
                                    victim_layer
                                ].popitem(last=False)
                                victim_key = (victim_layer, victim_expert)
                                del self._cache[victim_key]
                                del self._cache_recency[victim_key]
                                self.cache_evictions += 1
                    elif self._free_slots:
                        slot = self._free_slots.pop()
                    else:
                        _, slot = self._cache.popitem(last=False)
                        self.cache_evictions += 1
                elif not admit:
                    self.cache_bypasses += 1

                if slot is not None:
                    self._cache[key] = slot
                    if self.cache_policy == "layer_lru":
                        self._layer_caches[layer_id][expert_id] = slot
                        self._cache_step += 1
                        self._cache_recency[key] = self._cache_step
                    admitted.append((key, slot))
                misses.append((expert_id, slot))

            try:
                # Use one bounded pool for cache copies and reads. For DSV4 this turns four
                # serial bank reads per expert into enough independent O_DIRECT work to
                # reach the NVMe queue-depth plateau.
                if cached_jobs:
                    list(self._read_pool.map(self._stage_cached_row, cached_jobs))
                direct_jobs = (
                    self._direct_extent_jobs(layer_id, misses, buffer_id) if misses else []
                )
                if direct_jobs is None:
                    read_jobs = [
                        (layer_id, expert_id, bank_name, slot, buffer_id)
                        for expert_id, slot in misses
                        for bank_name in staging
                    ]
                if direct_jobs:
                    sizes = list(self._read_pool.map(self._read_staging_extent, direct_jobs))
                    self.read_ops += sum(ops for _, _, ops in sizes)
                    self.logical_bytes += sum(logical for logical, _, _ in sizes)
                    self.physical_bytes += sum(physical for _, physical, _ in sizes)
                    cache_jobs = [
                        (expert_id, bank_name, slot, buffer_id)
                        for expert_id, slot in misses if slot is not None
                        for bank_name in staging
                    ]
                    if cache_jobs:
                        list(self._read_pool.map(self._cache_staged_row, cache_jobs))
                elif read_jobs:
                    sizes = list(self._read_pool.map(self._stage_bank_row, read_jobs))
                    self.read_ops += len(sizes)
                    self.logical_bytes += sum(logical for logical, _ in sizes)
                    self.physical_bytes += sum(physical for _, physical in sizes)
            except BaseException:
                # Do not advertise partially-filled cache slots after a failed batch.
                for key, slot in admitted:
                    if self._cache.get(key) == slot:
                        del self._cache[key]
                        if self.cache_policy == "layer_lru":
                            self._layer_caches[key[0]].pop(key[1], None)
                            self._cache_recency.pop(key, None)
                        self._free_slots.append(slot)
                raise
        self.read_seconds += time.perf_counter() - start

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        """Start a full-layer bypass read into the layer's parity staging buffer."""
        if self._prefill_pool is None or layer_id in self._prefill_futures:
            return
        expert_count = next(iter(self.staging.values())).tensor.shape[0]
        self._prefill_futures[layer_id] = self._prefill_pool.submit(
            self.stage, layer_id, list(range(expert_count)), admit=False
        )

    def wait_prefill_layer(self, layer_id: int) -> None:
        if self._prefill_pool is None:
            self.stage(
                layer_id,
                list(range(next(iter(self.staging.values())).tensor.shape[0])),
                admit=False,
            )
            return
        self.prefetch_prefill_layer(layer_id)
        self._prefill_futures.pop(layer_id).result()

    def stats(self) -> dict:
        return {
            "cache_capacity_entries": self.cache_capacity,
            "cache_capacity_bytes": self.cache_capacity * self.cache_row_bytes,
            "cache_allocated_bytes": self.cache_allocated_bytes,
            "staging_allocated_bytes": self.staging_allocated_bytes,
            "host_allocated_bytes": self.host_allocated_bytes,
            "cache_occupancy_entries": len(self._cache),
            "cache_occupancy_bytes": len(self._cache) * self.cache_row_bytes,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "cache_bypasses": self.cache_bypasses,
            "read_ops": self.read_ops,
            "logical_bytes": self.logical_bytes,
            "physical_bytes": self.physical_bytes,
            "read_seconds": self.read_seconds,
            "read_workers": self.read_workers,
            "cache_policy": self.cache_policy,
            "staging_buffers": self.num_staging_buffers,
        }

    def close(self) -> None:
        if self._prefill_pool is not None:
            self._prefill_pool.shutdown(wait=True)
        self._read_pool.shutdown(wait=True)


def open_ftw_disk_banks(path: str, *, num_layers: int, num_experts: int,
                        host_cache_bytes: int = 0,
                        prefill_overlap: bool = False,
                        cache_policy: str = "lru"):
    """Open FTW routed experts with one or two pinned staging layers.

    When overlap is enabled, the second staging layer is charged against the requested
    host-cache budget so total host allocation does not grow relative to single-buffer
    disk mode.
    """
    from freetoken.moe.host_banks import HostBank

    reader = FTWReader(path)
    descriptors = reader.expert_row_descriptors(num_layers=num_layers)
    bank_names = sorted({key[2] for key in descriptors})
    if not bank_names:
        reader.close()
        return None, None

    # The original one-layer staging footprint sits outside ``host_cache_bytes``.
    # A second pinned layer is safe only when it can replace an equal amount of the
    # requested pageable LRU. Reject before allocating/faulting/pinning that layer;
    # otherwise a small cache setting could transiently exceed the stated bound.
    extra_staging_bytes = 0
    if prefill_overlap:
        extra_staging_bytes = sum(
            _align_up(
                num_experts
                * math.prod(descriptors[(0, 0, bank_name)].shape)
                * torch.empty(
                    (), dtype=descriptors[(0, 0, bank_name)].dtype
                ).element_size()
            )
            for bank_name in bank_names
        )
        if host_cache_bytes < extra_staging_bytes:
            reader.close()
            raise ValueError(
                "disk prefill overlap needs a host expert-cache budget of at least "
                f"{extra_staging_bytes / 2**30:.2f} GiB for its second staging layer; "
                "increase --moe-host-cache-gb or disable prefill overlap"
            )

    staging_buffers = []
    for _ in range(2 if prefill_overlap else 1):
        staging = {}
        for bank_name in bank_names:
            desc = descriptors[(0, 0, bank_name)]
            bank = HostBank((num_experts, *desc.shape), desc.dtype)
            bank.tensor.zero_()  # fault pages before cudaHostRegister
            bank.pin()
            staging[bank_name] = bank
        staging_buffers.append(staging)
    staging = staging_buffers[0]

    row_bytes = sum(bank.tensor[0].numel() * bank.tensor.element_size()
                    for bank in staging.values())
    bank_row_bytes = [bank.tensor[0].numel() * bank.tensor.element_size()
                      for bank in staging.values()]
    cache_budget_bytes = max(0, host_cache_bytes - extra_staging_bytes)
    total_experts = num_layers * num_experts
    cache_capacity = min(total_experts, cache_budget_bytes // row_bytes) if row_bytes else 0
    while cache_capacity and sum(_align_up(cache_capacity * nbytes)
                                 for nbytes in bank_row_bytes) > cache_budget_bytes:
        cache_capacity -= 1
    if cache_budget_bytes and cache_capacity < 1:
        reader.close()
        raise ValueError(
            f"host expert cache budget {host_cache_bytes} bytes cannot hold one "
            f"{row_bytes}-byte expert"
        )
    cache_banks = {}
    if cache_capacity:
        for bank_name, staging_bank in staging.items():
            shape = (cache_capacity, *staging_bank.tensor.shape[1:])
            bank = HostBank(shape, staging_bank.tensor.dtype)
            bank.tensor.zero_()
            # The host LRU is copied into the one-layer staging banks by the CPU; neither
            # CUDA DMA nor the CPU MoE executor holds pointers to it. Pinning tens of GiB
            # here needlessly forces unrelated system pages into swap. Only ``staging``
            # above must remain cudaHostRegister'd.
            cache_banks[bank_name] = bank

    alpha_tensors = {}
    for entry in reader.entries("experts_bank"):
        if entry["name"] not in _ALPHA_NAMES:
            continue
        bank = HostBank(tuple(entry["shape"]), _dtype_of(entry["dtype"]))
        reader.read_into(bank.memoryview(), entry)
        bank.pin()
        alpha_tensors[entry["name"]] = bank.tensor

    from freetoken.moe.expert_banks import ExpertBanks

    source = FTWDiskExpertSource(
        reader,
        descriptors,
        staging,
        cache_banks,
        cache_policy=cache_policy,
        staging_buffers=staging_buffers,
    )
    sources = {
        name: [staging_buffers[layer_id % len(staging_buffers)][name].tensor
               for layer_id in range(num_layers)]
        for name in staging
    }
    banks = ExpertBanks(
        reader.meta("quant_format"),
        sources,
        gate_up_alpha=alpha_tensors.get("gate_up_alpha"),
        down_alpha=alpha_tensors.get("down_alpha"),
    )
    return banks, source


def iter_ftw_weights(path: str, *, kinds=("weight",), workers: int = 8,
                       chunk: int = _DEFAULT_CHUNK, prefetch: int = 2):
    """Yield ``(name, host_tensor)`` for the requested kinds, reading each tensor via
    chunked O_DIRECT. A background thread prefetches the next ``prefetch`` tensors so the
    disk stays busy while the consumer copies the current one to the GPU. Transient buffers
    are freed as the consumer advances (peak host mem ~ prefetch+1 tensors)."""
    import queue
    import threading

    from freetoken.utils.progress import byte_bar

    reader = FTWReader(path)
    entries = reader.entries(*kinds)
    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    _DONE = object()
    err: list[BaseException] = []
    cancel = threading.Event()

    def _put(item) -> bool:
        # A plain q.put would deadlock teardown: if the consumer stops with the queue
        # full (early break out of the generator, or an exception mid-load), close()
        # runs the finally below, which joins this thread while it waits for queue
        # space forever. Poll the cancel flag instead of blocking indefinitely.
        while not cancel.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _producer():
        try:
            for e in entries:
                buf = _transient_buffer(e["nbytes"])
                reader.read_into(memoryview(buf), e, workers=workers, chunk=chunk)
                dt = _dtype_of(e["dtype"])
                t = torch.frombuffer(buf, dtype=dt, count=e["nbytes"] // _elsize(dt))
                if not _put((e["name"], t.view(*e["shape"]) if e["shape"] else t, buf, e["nbytes"])):
                    return
        except BaseException as ex:  # surface to consumer
            err.append(ex)
        finally:
            _put(_DONE)

    th = threading.Thread(target=_producer, name="FTW-prefetch", daemon=True)
    th.start()
    bar = byte_bar(sum(e["nbytes"] for e in entries), "Loading weights (FTW)")
    try:
        while True:
            item = q.get()
            if item is _DONE:
                break
            name, tensor, buf, nbytes = item
            yield name, tensor
            bar.update(nbytes)
            del tensor, buf  # buffer reclaimable once the consumer drops the tensor
    finally:
        bar.close()
        cancel.set()
        th.join()
        reader.close()
    if err:
        raise err[0]


def load_ftw_banks(
    path: str, *, num_layers: int, workers: int = 8, chunk: int = _DEFAULT_CHUNK
):
    """Reconstruct the offload :class:`ExpertBanks` from the FTW's ``experts_bank``
    entries, on the per-layer host bank contract (one pinned ``[num_experts, ...]``
    HostBank per layer per bank; see ``moe.offload_cache.set_bank_sources``).

    Two on-disk row layouts, distinguished per bank name (a file never mixes them for
    the same name -- checked below):

    * **Flat region** (pre-existing files, and non-streamable formats): one entry per
      bank, ONE contiguous ``[num_layers * num_experts, ...]`` region. ``num_layers``
      isn't part of that region's shape, so the caller passes it
      (``ModelConfig.num_moe_layers`` -- FTW checkpoints carry the model's config.json);
      the ``expert_bank_num_layers`` index meta the converter records is used as a
      cross-check when present. A layer's byte range within the region generally is
      NOT 4096-aligned (only the whole region's start is guaranteed aligned) -- read it
      via its ALIGNED enclosing window ``[align_down(off), align_up(off+len))`` into a
      page-aligned scratch HostBank, and view the real per-layer tensor as a
      head-offset slice.
    * **Per-layer** (streamable-format conversion, see :mod:`freetoken.checkpoint.convert`):
      one entry per ``(bank, layer)``, name ``f"{bank_name}#L{layer_id:05d}"``. Each was
      written by its own ``add_tensor`` call, so its start is already ALIGN-aligned --
      no windowing/head-pad needed, read straight into a HostBank shaped like the entry.

    Alphas (``gate_up_alpha``/``down_alpha``) stay flat ``[num_layers*num_experts]``
    vectors, unaffected by the row split (fixed GPU residency; see
    ``cache_budget.expert_bytes_per_slot``).
    """
    from freetoken.moe.host_banks import HostBank, PinPipeline, alloc_banks
    from freetoken.utils.progress import byte_bar

    reader = FTWReader(path)
    bank_entries = reader.entries("experts_bank")
    if not bank_entries:
        reader.close()
        return None

    alpha_entries = [e for e in bank_entries if e["name"] in _ALPHA_NAMES]
    row_entries = [e for e in bank_entries if e["name"] not in _ALPHA_NAMES]

    meta_layers = reader.meta("expert_bank_num_layers")
    if meta_layers is not None and meta_layers != num_layers:
        reader.close()
        raise RuntimeError(
            f"{path!r} was converted with {meta_layers} expert-bank layers but the "
            f"model config says num_moe_layers={num_layers}; the checkpoint does not "
            "match its config"
        )

    # Alphas: unchanged, one flat HostBank per entry.
    alpha_specs = {e["name"]: (tuple(e["shape"]), _dtype_of(e["dtype"])) for e in alpha_entries}
    alpha_hb = alloc_banks(alpha_specs)

    # Split row entries into the two layouts by name.
    flat_entries: list[dict] = []
    per_layer_groups: dict[str, dict[int, dict]] = {}
    for e in row_entries:
        m = _LAYER_ENTRY_RE.match(e["name"])
        if m is None:
            flat_entries.append(e)
            continue
        per_layer_groups.setdefault(m.group("base"), {})[int(m.group("layer"))] = e

    mixed = {e["name"] for e in flat_entries} & per_layer_groups.keys()
    assert not mixed, f"FTW bank(s) mix flat and per-layer row layouts: {sorted(mixed)}"

    # Row banks: one padded-window HostBank per (name, layer_id) for the flat layout, plus
    # how to carve the real [num_experts, *row_shape] tensor out of its head; ``None`` marks
    # a per-layer entry (direct view, no carving needed).
    row_hb: dict[str, list] = {}
    row_view_args: dict[str, list] = {}
    row_jobs = []  # (name, HostBank, window_off, window_len, layer_bytes) -- flat layout
    layer_jobs = []  # (name, HostBank, entry) -- per-layer layout, direct aligned read

    for e in flat_entries:
        name = e["name"]
        total, *row_shape = e["shape"]
        assert total % num_layers == 0, (name, total, num_layers)
        num_experts = total // num_layers
        dtype = _dtype_of(e["dtype"])
        row_bytes = (math.prod(row_shape) if row_shape else 1) * _elsize(dtype)
        layer_bytes = num_experts * row_bytes
        assert layer_bytes * num_layers == e["nbytes"], (name, layer_bytes, num_layers, e["nbytes"])
        row_hb[name] = []
        row_view_args[name] = []
        for layer_id in range(num_layers):
            off = e["global_off"] + layer_id * layer_bytes
            win_off = (off // ALIGN) * ALIGN
            win_end = _align_up(off + layer_bytes)
            head_pad = off - win_off
            bank = HostBank((win_end - win_off,), torch.uint8)
            row_hb[name].append(bank)
            row_view_args[name].append((head_pad, layer_bytes, num_experts, tuple(row_shape), dtype))
            row_jobs.append((name, bank, win_off, win_end - win_off, layer_bytes))

    for base, by_layer in per_layer_groups.items():
        assert sorted(by_layer) == list(range(num_layers)), (
            f"FTW bank {base!r} has per-layer entries for layers {sorted(by_layer)}, "
            f"expected exactly range({num_layers})"
        )
        row_hb[base] = []
        row_view_args[base] = []
        for layer_id in range(num_layers):
            e = by_layer[layer_id]
            assert e["global_off"] % ALIGN == 0, (base, layer_id, e["global_off"])  # writer invariant
            bank = HostBank(tuple(e["shape"]), _dtype_of(e["dtype"]))
            row_hb[base].append(bank)
            row_view_args[base].append(None)
            layer_jobs.append((base, bank, e))

    total_bytes = sum(e["nbytes"] for e in bank_entries)
    bar = byte_bar(total_bytes, "Loading expert banks (FTW)")

    # Jobs are per (bank, layer) -- many small reads, so a wider pool; each bank pins
    # as its read completes, overlapping cudaHostRegister with the remaining reads.
    n_jobs = len(alpha_entries) + len(row_jobs) + len(layer_jobs)
    try:
        with PinPipeline() as pins:

            def _read_alpha(e):
                bank = alpha_hb[e["name"]]
                reader.read_into(bank.memoryview(), e, workers=workers, chunk=chunk)
                pins.submit(bank)
                bar.update(e["nbytes"])

            def _read_row(job):
                _name, bank, win_off, win_len, layer_bytes = job
                reader.read_into(bank.memoryview(), {"global_off": win_off, "nbytes": win_len},
                                 workers=workers, chunk=chunk)
                pins.submit(bank)
                bar.update(layer_bytes)

            def _read_layer(job):
                _name, bank, entry = job
                reader.read_into(bank.memoryview(), entry, workers=workers, chunk=chunk)
                pins.submit(bank)
                bar.update(entry["nbytes"])

            with ThreadPoolExecutor(min(max(_BANK_CONCURRENCY, 16), max(n_jobs, 1))) as ex:
                futures = [ex.submit(_read_alpha, e) for e in alpha_entries]
                futures += [ex.submit(_read_row, job) for job in row_jobs]
                futures += [ex.submit(_read_layer, job) for job in layer_jobs]
                for f in futures:
                    f.result()
    finally:
        bar.close()
        reader.close()

    sources: dict[str, list] = {}
    for name, banks in row_hb.items():
        views = []
        for bank, view_args in zip(banks, row_view_args[name]):
            if view_args is None:  # per-layer entry: already shaped [num_experts, ...]
                views.append(bank.tensor)
                continue
            head_pad, layer_bytes, num_experts, row_shape, dtype = view_args
            raw = bank.tensor[head_pad:head_pad + layer_bytes].view(dtype)
            views.append(raw.view(num_experts, *row_shape) if row_shape else raw.view(num_experts))
        sources[name] = views

    from freetoken.moe.expert_banks import ExpertBanks

    # alphas are the small per-expert scale vectors, distinguished by their reserved names
    # (not a separate kind); everything else under experts_bank is a weight source.
    alpha_kw = {n: alpha_hb[n].tensor for n in alpha_hb}
    return ExpertBanks(reader.meta("quant_format"), sources, **alpha_kw)


__all__ = [
    "INDEX_NAME", "FORMAT_TAG", "FORMAT_VERSION", "ALIGN", "DEFAULT_SHARD_LIMIT",
    "is_ftw_checkpoint", "FTWWriter", "FTWReader", "ExpertRowDescriptor",
    "iter_ftw_weights", "load_ftw_banks", "layer_bank_entry_name", "open_ftw_disk_banks",
]
