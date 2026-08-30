from __future__ import annotations

import json
import math
import os
import re
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from sparklab.core import get_global_ctx
from sparklab.layers import BaseOP, LinearReplicated

from .hyper import GroupedPlusOneRMSNorm

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_SHARD_RE = re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")
_DEFAULT_ROW_CACHE_MB = 256
_STORAGE_DTYPES = {
    "BF16": (torch.bfloat16, 2),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "bfloat16": (torch.bfloat16, 2),
    "float8_e4m3fn": (torch.float8_e4m3fn, 1),
}


def _splitmix64(value: int) -> int:
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _M1) & _MASK64
    value = ((value ^ (value >> 27)) * _M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _multipliers(vocab: int, ngram: int, layer_index: int, seed: int) -> torch.Tensor:
    bound = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    base = seed + 10007 * layer_index
    return torch.tensor([
        2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % bound) + 1
        for i in range(ngram)
    ], dtype=torch.long, device="cpu")


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    value = start
    for _ in range(count):
        value += 1
        while not _is_prime(value):
            value += 1
    return value


class _ZeroNGramStore:
    def __init__(self, dim: int):
        self.dim = dim

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*ids.shape, self.dim), dtype=torch.bfloat16)


class _CachedRowStore:
    """Bounded decoded-row cache shared by source and FTW PLE stores.

    A warm 32K prompt can revisit roughly 160 MiB of hashed rows. Caching row
    payloads avoids hundreds of thousands of repeat ``pread`` syscalls while
    keeping the 95 GiB table itself disk-resident. Capacity is byte-derived and
    overrideable for experiments; it is not counted as correctness state.
    """

    def _init_row_cache(self) -> None:
        raw = os.getenv("SPARKLAB_QWEN4_NGRAM_CACHE_MB", str(_DEFAULT_ROW_CACHE_MB))
        try:
            cache_mb = int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid SPARKLAB_QWEN4_NGRAM_CACHE_MB={raw!r}") from exc
        if cache_mb < 0:
            raise ValueError("SPARKLAB_QWEN4_NGRAM_CACHE_MB must be non-negative")
        self._row_cache_capacity = (cache_mb << 20) // self.row_bytes
        self._row_cache: OrderedDict[int, bytes] = OrderedDict()

    def _lookup_rows(self, keys: list[int]) -> list[bytes]:
        rows: list[bytes | None] = [None] * len(keys)
        missing_positions, missing_keys = [], []
        for position, key in enumerate(keys):
            value = self._row_cache.get(key)
            if value is None:
                missing_positions.append(position)
                missing_keys.append(key)
            else:
                self._row_cache.move_to_end(key)
                rows[position] = value
        if missing_keys:
            loaded = list(self._executor.map(self._read_one, missing_keys))
            for position, key, value in zip(missing_positions, missing_keys, loaded):
                rows[position] = value
                if self._row_cache_capacity:
                    self._row_cache[key] = value
                    self._row_cache.move_to_end(key)
            while len(self._row_cache) > self._row_cache_capacity:
                self._row_cache.popitem(last=False)
        return [value for value in rows if value is not None]

    def _lookup(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.detach().to(device="cpu", dtype=torch.long).flatten()
        if flat.numel() <= 32:
            # Decode is normally one request x 16 n-gram heads. Avoid torch.unique
            # and index_select for that tiny case: the row cache already accepts
            # repeated keys and preserving their order gives the final table directly.
            rows = self._lookup_rows(flat.tolist())
            payload = bytearray().join(rows)
            table = torch.frombuffer(payload, dtype=self.storage_dtype).clone()
            table = table.view(*ids.shape, self.dim)
            return table if table.dtype == torch.bfloat16 else table.to(torch.bfloat16)
        unique, inverse = torch.unique(flat, sorted=False, return_inverse=True)
        rows = self._lookup_rows(unique.tolist())
        if len(rows) != unique.numel():
            raise RuntimeError("Qwen4 n-gram row cache returned an incomplete lookup")
        payload = bytearray().join(rows)
        table = torch.frombuffer(payload, dtype=self.storage_dtype).clone().view(-1, self.dim)
        if table.dtype != torch.bfloat16:
            table = table.to(torch.bfloat16)
        return table.index_select(0, inverse).view(*ids.shape, self.dim)


class SafetensorNGramStore(_CachedRowStore):
    """Random-row reader over the split n-gram tensors without mmap page retention."""

    def __init__(self, model_path: str, expected_parts: int, dim: int):
        folder = Path(model_path)
        index_path = folder / "model.safetensors.index.json"
        with index_path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        parts: list[tuple[int, str, Path]] = []
        for name, filename in weight_map.items():
            match = _SHARD_RE.search(name)
            if match:
                parts.append((int(match.group(1)), name, folder / filename))
        parts.sort()
        if [i for i, _, _ in parts] != list(range(expected_parts)):
            raise ValueError(
                f"Qwen4 n-gram table requires shards 0..{expected_parts - 1}, "
                f"found {[i for i, _, _ in parts]}"
            )
        self.dim = dim
        storage_dtype = None
        item_size = None
        self._parts: list[tuple[int, int, int]] = []  # fd, absolute data offset, rows
        self._fds: list[int] = []
        for _, name, path in parts:
            fd = os.open(path, os.O_RDONLY)
            self._fds.append(fd)
            header_size = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            header = json.loads(os.pread(fd, header_size, 8))
            meta = header[name]
            if meta["dtype"] not in _STORAGE_DTYPES or meta["shape"][1] != dim:
                raise ValueError(f"unexpected Qwen4 n-gram tensor {name}: {meta}")
            part_dtype, part_item_size = _STORAGE_DTYPES[meta["dtype"]]
            if storage_dtype is None:
                storage_dtype, item_size = part_dtype, part_item_size
            elif storage_dtype != part_dtype:
                raise ValueError("Qwen4 n-gram shards use inconsistent storage dtypes")
            begin, end = meta["data_offsets"]
            rows = int(meta["shape"][0])
            if end - begin != rows * dim * part_item_size:
                raise ValueError(f"non-contiguous Qwen4 n-gram tensor: {name}")
            self._parts.append((fd, 8 + header_size + begin, rows))
        assert storage_dtype is not None and item_size is not None
        self.storage_dtype = storage_dtype
        self.row_bytes = dim * item_size
        self._starts = []
        total = 0
        for _, _, rows in self._parts:
            self._starts.append(total)
            total += rows
        self.total_rows = total
        self._executor = ThreadPoolExecutor(max_workers=min(16, expected_parts))
        self._init_row_cache()

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for fd in self._fds:
            os.close(fd)
        self._fds.clear()
        self._row_cache.clear()

    def _read_one(self, global_row: int) -> bytes:
        import bisect

        part = bisect.bisect_right(self._starts, global_row) - 1
        if part < 0 or part >= len(self._parts):
            raise IndexError(f"n-gram row {global_row} outside [0, {self.total_rows})")
        fd, base, rows = self._parts[part]
        local = global_row - self._starts[part]
        if local >= rows:
            raise IndexError(f"n-gram row {global_row} crosses shard boundary")
        data = os.pread(fd, self.row_bytes, base + local * self.row_bytes)
        if len(data) != self.row_bytes:
            raise OSError(f"short n-gram row read: {len(data)}/{self.row_bytes}")
        return data

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return self._lookup(ids)


class RawNGramStore(_CachedRowStore):
    """Random-row reader for the contiguous PLE artifact embedded beside FTW."""

    def __init__(self, model_path: str, manifest: dict, dim: int):
        self.dim = dim
        dtype_name = str(manifest.get("dtype", ""))
        if dtype_name not in _STORAGE_DTYPES:
            raise ValueError(f"unsupported Qwen4 FTW n-gram dtype: {dtype_name!r}")
        self.storage_dtype, item_size = _STORAGE_DTYPES[dtype_name]
        self.row_bytes = dim * item_size
        self.total_rows = int(manifest["rows"])
        if int(manifest["dim"]) != dim or int(manifest["nbytes"]) != self.total_rows * self.row_bytes:
            raise ValueError(f"invalid Qwen4 FTW n-gram manifest: {manifest}")
        self._fd = os.open(Path(model_path) / manifest["file"], os.O_RDONLY)
        self._executor = ThreadPoolExecutor(max_workers=16)
        self._init_row_cache()

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        os.close(self._fd)
        self._row_cache.clear()

    def _read_one(self, row: int) -> bytes:
        if row < 0 or row >= self.total_rows:
            raise IndexError(f"n-gram row {row} outside [0, {self.total_rows})")
        data = os.pread(self._fd, self.row_bytes, row * self.row_bytes)
        if len(data) != self.row_bytes:
            raise OSError(f"short n-gram row read: {len(data)}/{self.row_bytes}")
        return data

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return self._lookup(ids)


class DiskNGramEmbedding(BaseOP):
    def __init__(self, args, vocab_size: int, layer_index: int):
        self.args = args
        self.layer_index = layer_index
        self.ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
        self.head_dim = args.ple_embed_dim // self.ngram_heads
        sizes, offsets, total = [], [], 0
        for head in range(self.ngram_heads):
            global_head = layer_index * self.ngram_heads + head
            size = _nth_prime_after(args.ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        # Deterministic metadata, intentionally not part of BaseOP.state_dict.
        self._head_vocab_sizes = torch.tensor(sizes, dtype=torch.long, device="cpu")
        self._head_offsets = torch.tensor(offsets, dtype=torch.long, device="cpu")
        self._multipliers = _multipliers(vocab_size, args.ngram_size, layer_index, args.seed)
        # Scalar mirrors used by the one-token decode hash. Python integer arithmetic
        # avoids launching roughly twenty tiny CPU tensor operations for a 1x16 result.
        self._head_vocab_sizes_py = self._head_vocab_sizes.tolist()
        self._head_offsets_py = self._head_offsets.tolist()
        self._multipliers_py = self._multipliers.tolist()
        self.padded_rows = math.ceil(total / args.ngram_vocab_divisor) * args.ngram_vocab_divisor
        self._store = None

    def bind(self, model_path: str, *, dummy: bool = False) -> None:
        manifest_path = Path(model_path) / "qwen4_ngram.json"
        if dummy:
            self._store = _ZeroNGramStore(self.head_dim)
        elif manifest_path.is_file():
            self._store = RawNGramStore(
                model_path, json.loads(manifest_path.read_text(encoding="utf-8")), self.head_dim
            )
        else:
            self._store = SafetensorNGramStore(
                model_path, self.args.split_ngram_parts, self.head_dim
            )
        if not dummy and self._store.total_rows != self.padded_rows:
            raise ValueError(
                f"Qwen4 n-gram rows {self._store.total_rows} != config {self.padded_rows}"
            )

    def _shift(self, ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return ids
        positions = torch.arange(ids.numel(), dtype=torch.long)
        eos_pos = torch.where(ids == self.args.eos_token_id, positions, -1)
        inclusive = torch.cummax(eos_pos, 0).values
        previous_eos = torch.cat((eos_pos.new_full((1,), -1), inclusive[:-1]))
        source = positions - shift
        shifted = ids[source.clamp_min(0)]
        valid = (positions - previous_eos - 1 >= shift) & (source >= 0)
        return torch.where(valid, shifted, ids.new_full((), self.args.eos_token_id))

    def _shift_span(
        self, ids: torch.Tensor, start: int, end: int, shift: int
    ) -> torch.Tensor:
        """Return one shifted output span without scanning the preceding context.

        A shift is valid exactly when its source exists and none of the ``shift``
        intervening tokens is EOS. Qwen3.8 uses shifts 0..2, so decode now touches at
        most the current token and its two predecessors instead of rebuilding arange,
        where and cummax tensors over the complete request on every generated token.
        """
        if shift == 0:
            return ids[start:end]
        positions = torch.arange(start, end, dtype=torch.long)
        source = positions - shift
        valid = source >= 0
        shifted = ids[source.clamp_min(0)]
        for offset in range(shift):
            check = (source + offset).clamp_min(0)
            valid &= ids[check] != self.args.eos_token_id
        return torch.where(valid, shifted, ids.new_full((), self.args.eos_token_id))

    def _ids_for_one(self, ids: torch.Tensor, position: int) -> torch.Tensor:
        """Scalar-equivalent n-gram hash for the common one-token decode span."""
        shifted = []
        for shift in range(self.args.ngram_size):
            source = position - shift
            valid = source >= 0
            if valid:
                valid = all(
                    int(ids[index]) != self.args.eos_token_id
                    for index in range(source, position)
                )
            shifted.append(int(ids[source]) if valid else self.args.eos_token_id)

        output = []
        for ngram in range(2, self.args.ngram_size + 1):
            h0 = (ngram - 2) * self.args.heads_per_ngram
            h1 = h0 + self.args.heads_per_ngram
            mixed = (shifted[0] * self._multipliers_py[0]) & _MASK64
            for token_index in range(1, ngram):
                mixed ^= (
                    shifted[token_index] * self._multipliers_py[token_index]
                ) & _MASK64
            # Match signed int64 overflow before torch.remainder.
            if mixed >= 1 << 63:
                mixed -= 1 << 64
            output.extend(
                mixed % self._head_vocab_sizes_py[head]
                + self._head_offsets_py[head]
                for head in range(h0, h1)
            )
        return torch.tensor([output], dtype=torch.long)

    def ids_for_request(
        self, ids: torch.Tensor, start: int, end: int,
        current_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # With overlap scheduling, decode N+1 can launch before decode N's sampled
        # token is appended to Req.input_ids on the host.  The current token is
        # already present in Batch.input_ids on the GPU, so reconstruct that one
        # in-flight tail explicitly. Prefill and non-overlap decode take the cheap
        # host-only branch.
        # Keep the history in its compact request dtype. Converting the whole prefix
        # to int64 here would reintroduce an O(context) copy before the span-only hash.
        ids = ids[:end]
        if ids.numel() < end:
            if ids.numel() != start or current_ids is None:
                raise RuntimeError(
                    "Qwen4 PLE token history is incomplete: "
                    f"host={ids.numel()}, start={start}, end={end}"
                )
            current = current_ids.detach().to(device="cpu", dtype=ids.dtype)
            if current.numel() != end - start:
                raise RuntimeError(
                    "Qwen4 PLE current-token span mismatch: "
                    f"got={current.numel()}, expected={end - start}"
                )
            ids = torch.cat((ids, current))
        if end - start == 1:
            return self._ids_for_one(ids, start)
        shifted = [
            self._shift_span(ids, start, end, n).long()
            for n in range(self.args.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.args.ngram_size + 1):
            h0 = (ngram - 2) * self.args.heads_per_ngram
            h1 = h0 + self.args.heads_per_ngram
            mixed = shifted[0] * self._multipliers[0]
            for pos in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[pos] * self._multipliers[pos])
            sizes = self._head_vocab_sizes[h0:h1]
            offsets = self._head_offsets[h0:h1]
            blocks.append(torch.remainder(mixed[:, None], sizes) + offsets)
        return torch.cat(blocks, -1)

    def forward(self, batch) -> torch.Tensor:
        if self._store is None:
            raise RuntimeError("Qwen4 disk n-gram embedding was not bound before weight load")
        all_ids, offset = [], 0
        for req in (batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs):
            length = req.extend_len
            current = batch.input_ids[offset:offset + length]
            all_ids.append(self.ids_for_request(
                req.input_ids, req.cached_len, req.device_len, current
            ))
            offset += length
        rows = self._store.lookup(torch.cat(all_ids, 0))
        return rows.flatten(-2)


class _DepthwiseDilatedConv(BaseOP):
    def __init__(self, width: int, kernel: int):
        self.weight = torch.empty(width, 1, kernel)


class Qwen4PLE(BaseOP):
    def __init__(self, config, layer_id: int, ple_index: int):
        args = config.qwen4_exp_args
        self.layer_id = layer_id
        self.ple_index = ple_index
        self.hidden_size = config.hidden_size
        self.hc_count = args.hc_count
        self.width = self.hidden_size * self.hc_count
        self.embedding = DiskNGramEmbedding(args, config.vocab_size, ple_index)
        self.key_proj = LinearReplicated(args.ple_embed_dim, self.width, has_bias=False)
        self.value_proj = LinearReplicated(args.ple_embed_dim, self.hidden_size, has_bias=False)
        self.norm_key = GroupedPlusOneRMSNorm(self.width, self.hidden_size, config.rms_norm_eps)
        self.norm_query = GroupedPlusOneRMSNorm(self.width, self.hidden_size, config.rms_norm_eps)
        self.norm_conv = GroupedPlusOneRMSNorm(self.width, self.hidden_size, config.rms_norm_eps)
        self.conv1d = _DepthwiseDilatedConv(self.width, args.ple_conv_kernel_size)
        self.dilation = args.ngram_size
        self.state_len = (args.ple_conv_kernel_size - 1) * self.dilation

    def bind(self, model_path: str, *, dummy: bool = False) -> None:
        self.embedding.bind(model_path, dummy=dummy)

    def _conv(self, x: torch.Tensor, batch) -> torch.Tensor:
        pool = get_global_ctx().linear_state_pool
        states = pool.ensure_aux_state(
            f"qwen4_ple_{self.layer_id}_conv", (self.width, self.state_len), x.dtype
        )
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        if batch.is_decode:
            from sparklab.kernels.triton.qwen4 import ple_conv_decode

            return ple_conv_decode(
                x,
                states,
                self.conv1d.weight,
                batch.fla_metadata.cache_indices,
                self.dilation,
            )
        outputs, offset = [], 0
        for req in reqs:
            length = req.extend_len
            current = x[offset:offset + length].T.contiguous()
            slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
            history = torch.cat((states[slot], current), -1)
            out = F.conv1d(
                history.unsqueeze(0), self.conv1d.weight,
                groups=self.width, dilation=self.dilation,
            ).squeeze(0).T
            states[slot].copy_(history[:, -self.state_len:])
            outputs.append(F.silu(out))
            offset += length
        return torch.cat(outputs, 0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        embed = self.embedding.forward(batch).to(hidden.device, dtype=hidden.dtype)
        key = self.norm_key.forward(self.key_proj.forward(embed)).view(
            -1, self.hc_count, self.hidden_size
        )
        query = self.norm_query.forward(hidden).view(-1, self.hc_count, self.hidden_size)
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.sign() * gate.abs().clamp_min(1e-6).sqrt()
        value = self.value_proj.forward(embed).unsqueeze(-2)
        gated = (torch.sigmoid(gate) * value).flatten(-2)
        return gated + self._conv(self.norm_conv.forward(gated), batch)


__all__ = [
    "DiskNGramEmbedding", "Qwen4PLE", "RawNGramStore", "SafetensorNGramStore"
]
