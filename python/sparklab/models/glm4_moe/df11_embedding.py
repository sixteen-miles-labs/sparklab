from __future__ import annotations

import torch
from sparklab.kernels.triton.df11 import DF11_LMAX, df11_compress_rows
from sparklab.kernels.triton.df11_decode import df11_gather_decode
from sparklab.layers import BaseOP
from sparklab.layers.base import _concat_prefix
from sparklab.utils import nvtx_annotate


def compress_df11_embedding(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compress a BF16 ``[vocab, hidden]`` table into the row-contiguous DF11 buffers."""
    c = df11_compress_rows(weight)
    return {k: c[k] for k in EmbeddingDF11._DF11_KEYS}


class EmbeddingDF11(BaseOP):
    """Token embedding whose table is stored losslessly as row-contiguous DF11 (compressed BF16).

    The lookup decodes **only the gathered rows** (one per token) bit-for-bit, so it adds
    negligible work while shrinking the 1.5GB bf16 table by ~30%. Single-device only (the
    offload backend never shards vocab), so there is no TP/all-reduce path.
    """

    _DF11_KEYS = ("low8", "bitstream", "chunk_start", "lut")

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.n = num_embeddings * embedding_dim
        # Placeholders; real (entropy-sized) buffers are installed by load_state_dict.
        self.low8 = torch.empty(0, dtype=torch.uint8)
        self.bitstream = torch.empty(0, dtype=torch.int32)
        self.chunk_start = torch.empty(0, dtype=torch.int64)
        self.lut = torch.empty(0, dtype=torch.int32)

    def _bundle(self) -> dict:
        return {
            "low8": self.low8,
            "bitstream": self.bitstream,
            "chunk_start": self.chunk_start,
            "lut": self.lut,
            "meta": (self.num_embeddings, self.embedding_dim, self.n, DF11_LMAX),
        }

    @nvtx_annotate("EmbeddingDF11")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = df11_gather_decode(self._bundle(), x)  # [T, hidden] bf16
        return out.view(*x.shape, self.embedding_dim)

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        for name in self._DF11_KEYS:
            setattr(self, name, state_dict.pop(_concat_prefix(prefix, name)))
        self.n = self.low8.numel()
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


__all__ = ["EmbeddingDF11", "compress_df11_embedding"]
