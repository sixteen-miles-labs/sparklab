"""Bounded safetensors reads; Engram tables are never materialized in memory."""
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import struct

import torch

DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32,
          "F16": torch.float16, "F8_E4M3": torch.float8_e4m3fn,
          "F8_E8M0": torch.float8_e8m0fnu, "U8": torch.uint8,
          "I8": torch.int8, "I32": torch.int32, "I64": torch.int64,
          "F4": torch.uint8}


class DiskWeights:
    def __init__(self, path, device="cpu", cache_bytes=0):
        self.path = Path(path)
        self.device = torch.device(device)
        self.cache_limit = cache_bytes
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.metadata = {}
        self.fds = {}
        try:
            index = json.loads((self.path / "model.safetensors.index.json").read_text())
            weight_map = index["weight_map"]
            for shard in sorted(set(weight_map.values())):
                if Path(shard).is_absolute() or ".." in Path(shard).parts:
                    raise ValueError(f"unsafe weight shard: {shard}")
                fd = os.open(self.path / shard, os.O_RDONLY)
                self.fds[shard] = fd
                prefix = os.pread(fd, 8, 0)
                if len(prefix) != 8:
                    raise ValueError(f"truncated safetensors header: {shard}")
                size = struct.unpack("<Q", prefix)[0]
                if size > 64 << 20:
                    raise ValueError(f"oversized safetensors header: {shard}")
                header = json.loads(os.pread(fd, size, 8))
                data_size = os.fstat(fd).st_size - size - 8
                for name, entry in header.items():
                    if name == "__metadata__":
                        continue
                    if weight_map.get(name) != shard:
                        raise ValueError(f"safetensors index mismatch: {name}")
                    start, end = entry["data_offsets"]
                    shape = tuple(entry["shape"])
                    dtype = DTYPES[entry["dtype"]]
                    # DeepSeek's packed F4 tensors use byte-shaped storage, as
                    # does torch.float4_e2m1fn_x2; two logical values per byte.
                    expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                    if start < 0 or end > data_size or end - start != expected:
                        raise ValueError(f"invalid tensor byte range: {name}")
                    self.metadata[name] = (fd, size + 8 + start, shape, dtype)
            if self.metadata.keys() != weight_map.keys():
                raise ValueError("incomplete safetensors snapshot")
        except Exception:
            self.close()
            raise

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
        self.cache.clear()
        self.cache_bytes = 0

    def validate_geometry(self, args):
        """Check the entire text tower using metadata before reading weight data."""
        a = args
        shapes = {"embed.weight": (a.vocab_size, a.dim), "head.weight": (a.vocab_size, a.dim),
                  "norm.weight": (a.dim,)}
        for layer in range(a.n_layers):
            root = f"layers.{layer}"
            attn = root + ".attn"
            shapes.update({
                attn + ".wq_a.weight": (a.q_lora_rank, a.dim),
                attn + ".wq_b.weight": (a.n_heads * a.head_dim, a.q_lora_rank),
                attn + ".q_norm.weight": (a.q_lora_rank,),
                attn + ".wkv.weight": (a.head_dim, a.dim),
                attn + ".kv_norm.weight": (a.head_dim,),
                attn + ".wo_a.weight": (a.o_groups * a.o_lora_rank, a.n_heads * a.head_dim // a.o_groups),
                attn + ".wo_b.weight": (a.dim, a.o_groups * a.o_lora_rank),
                attn + ".attn_sink": (a.n_heads,),
                root + ".ffn.gate.weight": (a.n_routed_experts, a.dim),
                root + ".ffn.gate.bias": (a.n_routed_experts,),
            })
            for kind in ("attn", "ffn"):
                shapes[root + f".{kind}_norm.weight"] = (a.dim,)
                shapes[root + f".hc_{kind}_fn"] = ((2 + a.hc_mult) * a.hc_mult, a.hc_mult * a.dim)
                shapes[root + f".hc_{kind}_base"] = ((2 + a.hc_mult) * a.hc_mult,)
                shapes[root + f".hc_{kind}_scale"] = (3,)
            for expert in (*range(a.n_routed_experts), "shared"):
                prefix = root + (".ffn.shared_experts" if expert == "shared" else f".ffn.experts.{expert}")
                width = a.moe_inter_dim * (a.n_shared_experts if expert == "shared" else 1)
                for projection in ("w1", "w3"):
                    shapes[prefix + f".{projection}.weight"] = (width, a.dim)
                shapes[prefix + ".w2.weight"] = (a.dim, width)
            if layer in a.kv_source_layers:
                shapes[attn + ".compressor.wkv.weight"] = (a.head_dim, a.dim)
                shapes[attn + ".compressor.norm.weight"] = (a.head_dim,)
                if a.compress_ratios[layer] > 1:
                    shapes[attn + ".compressor.wgate.weight"] = (a.head_dim, a.dim)
                shapes[attn + ".indexer.wk.weight"] = (a.index_head_dim, a.head_dim)
                shapes[attn + ".indexer.k_norm.weight"] = (a.index_head_dim,)
            if layer in a.index_source_layers:
                shapes[attn + ".indexer.wq_b.weight"] = (a.index_n_heads * a.index_head_dim, a.q_lora_rank)
                shapes[attn + ".indexer.weights_proj.weight"] = (a.index_n_heads, a.dim)
            if layer in a.engram_layer_ids:
                rows = a.engram_num_embeddings[a.engram_layer_ids.index(layer)]
                prefix = root + ".engram"
                shapes[prefix + ".embed.weight"] = (rows, a.engram_head_dim)
                shapes[prefix + ".q_weight"] = shapes[prefix + ".k_weight"] = (a.hc_mult, a.dim)
                shapes[prefix + ".wkv.weight"] = (
                    a.dim * (a.hc_mult + 1),
                    (a.engram_max_ngram_size - 1) * a.engram_n_heads * a.engram_head_dim,
                )
        for name, logical in shapes.items():
            if name not in self.metadata:
                raise ValueError(f"missing DeepSeek V4.1 weight: {name}")
            _, _, shape, dtype = self.metadata[name]
            packed = dtype in (torch.uint8, torch.int8) and name.endswith(".weight")
            expected = (logical[0], logical[1] // 2) if packed else logical
            if shape != expected:
                raise ValueError(f"DeepSeek V4.1 weight shape mismatch: {name}: {shape} != {expected}")
            if packed or dtype == torch.float8_e4m3fn:
                if packed or ".engram.embed." in name:
                    scale_shape = (logical[0], logical[1] // 32)
                else:
                    scale_shape = tuple((dim + 31) // 32 for dim in logical)
                scale = self.metadata.get(name.removesuffix(".weight") + ".scale")
                if scale is None or scale[2:] != (scale_shape, torch.float8_e8m0fnu):
                    raise ValueError(f"DeepSeek V4.1 requires correctly shaped E8M0 scales: {name}")

    def __del__(self):
        self.close()

    def _read(self, name, first=0, rows=None):
        fd, offset, shape, dtype = self.metadata[name]
        count = shape[0] if rows is None else rows
        if first < 0 or count < 0 or first + count > shape[0]:
            raise IndexError(f"tensor row range out of bounds: {name}")
        stride = math.prod(shape[1:]) * torch.empty((), dtype=dtype).element_size()
        size = count * stride
        data = bytearray(os.pread(fd, size, offset + first * stride))
        if len(data) != size:
            raise ValueError(f"truncated tensor: {name}")
        tensor = torch.frombuffer(data, dtype=dtype).reshape(count, *shape[1:]).to(self.device)
        # pread avoids mmap retaining a 100 GB table; evict clean read pages too.
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, offset + first * stride, size, os.POSIX_FADV_DONTNEED)
        return tensor

    def get(self, name):
        if ".engram.embed." in name:
            raise ValueError("Engram tables must be read by row")
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        fd, offset, shape, dtype = self.metadata[name]
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        while self.cache and self.cache_bytes + size > self.cache_limit:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= old.numel() * old.element_size()
        tensor = self._read(name)
        if size <= self.cache_limit:
            self.cache[name] = tensor
            self.cache_bytes += size
        return tensor

    def rows(self, name, indices):
        # Hashing stays on CPU. Only the requested rows cross into device memory.
        ids = indices.reshape(-1).tolist()
        unique = {index: self._read(name, index, 1) for index in set(ids)}
        return torch.cat([unique[index] for index in ids]).reshape(
            *indices.shape, *self.metadata[name][2][1:]
        )


def iter_weights(model_path, device, *, include_moe_experts=True, include_non_moe=True):
    # The model owns a bounded on-demand disk store, including resident projections.
    # No hundreds-of-GB tensor is handed to the generic eager materializer.
    return iter(())
