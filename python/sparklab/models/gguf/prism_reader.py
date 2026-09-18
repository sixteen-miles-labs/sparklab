"""Read Prism's private GGML types without changing gguf-py's global type enum.

Type IDs/layouts follow PrismML-Eng/llama.cpp at 1a07bfa5f4144274c8f1c9963821dd9d9a51854b.
The upstream reader still parses the header and all standard tensors.
"""

import math

import gguf
import numpy as np

PRISM_QUANT_SIZES = {142: (128, 34), 143: (128, 28)}


class PrismGGUFReader(gguf.GGUFReader):
    def _build_tensors(self, start_offs, fields):
        names = [f.name for f in fields]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate GGUF tensor names")
        special = [f for f in fields if int(f.parts[4][0]) in PRISM_QUANT_SIZES]
        super()._build_tensors(
            start_offs,
            [f for f in fields if int(f.parts[4][0]) not in PRISM_QUANT_SIZES],
        )
        if special and self.endianess != gguf.GGUFEndian.LITTLE:
            raise ValueError("Prism ternary GGUF requires little-endian storage")
        from gguf.gguf_reader import ReaderTensor

        for f in special:
            _, _, _, dims, raw_type, offset = f.parts
            kind = int(raw_type[0])
            block, size = PRISM_QUANT_SIZES[kind]
            shape = tuple(int(d) for d in dims)
            if not shape or any(d <= 0 for d in shape) or shape[0] % block:
                raise ValueError(f"{f.name}: invalid ternary tensor shape {shape}")
            count = math.prod(shape)
            nbytes = count // block * size
            start = int(start_offs) + int(offset[0])
            if start < start_offs or start + nbytes > self.data.size:
                raise ValueError(f"{f.name}: ternary tensor exceeds GGUF file")
            data = self._get(start, np.uint8, nbytes)
            self.tensors.append(
                ReaderTensor(f.name, kind, dims, count, nbytes, start, data, f)
            )
        self.tensors.sort(key=lambda t: t.field.offset)
