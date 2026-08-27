"""Pinned Qwen4-Exp text-runtime geometry.

Qwen3.8-Flash-Next combines three linear GatedDeltaNet layers with one QSA
layer, four Hyper-Connection streams, and a disk-sized hashed n-gram embedding
used by PLE. Keeping those knobs together prevents the engine, model and weight
loader from independently interpreting the experimental checkpoint format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class Qwen4ExpArgs:
    hc_count: int
    hc_lowrank: int
    qsa_layer_ids: Tuple[int, ...]
    index_n_heads: int
    index_kv_heads: int
    index_head_dim: int
    index_token_budget: int
    index_compress_ratio: int
    ple_layer_ids: Tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ngram_vocab_divisor: int
    split_ngram_parts: int
    seed: int
    eos_token_id: int
    output_gate_activation: str

    @property
    def index_block_topk(self) -> int:
        return self.index_token_budget // self.index_compress_ratio

    def qsa_slot(self, layer_id: int) -> int:
        return self.qsa_layer_ids.index(layer_id)


def load_args(text: Any, layer_types: tuple[str, ...]) -> Qwen4ExpArgs:
    qsa_ids = tuple(
        i for i, value in enumerate(layer_types)
        if value in {"full_attention", "qwen_sparse_attention"}
    )
    ple_one_based = tuple(sorted(set(getattr(text, "ple_layer_ids", ()) or ())))
    ple_ids = tuple(i - 1 for i in ple_one_based)
    ngram_size = int(getattr(text, "ngram_size", 3))
    heads_per_ngram = int(getattr(text, "heads_per_ngram", 8))
    ple_embed_dim = int(getattr(text, "ple_embed_dim", text.hidden_size))
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    if ngram_heads <= 0 or ple_embed_dim % ngram_heads:
        raise ValueError(
            "Qwen4 PLE embed dim must divide evenly across its n-gram heads: "
            f"{ple_embed_dim} % {ngram_heads}"
        )
    if any(i < 0 or i >= len(layer_types) for i in ple_ids):
        raise ValueError(f"Qwen4 PLE layer ids are one-based and out of range: {ple_one_based}")
    if any(layer_types[i] != "linear_attention" for i in ple_ids):
        raise ValueError("Qwen4 PLE is supported only on linear-attention layers")

    qsa_values = {
        "index_n_heads": int(getattr(text, "indexer_n_heads", 0) or 0),
        "index_kv_heads": int(getattr(text, "indexer_kv_heads", 0) or 0),
        "index_head_dim": int(getattr(text, "indexer_head_dim", 0) or 0),
        "index_token_budget": int(getattr(text, "indexer_budget", 0) or 0),
        "index_compress_ratio": int(getattr(text, "indexer_compress_ratio", 0) or 0),
    }
    if qsa_ids and any(v <= 0 for v in qsa_values.values()):
        raise ValueError(f"Qwen4 QSA requires positive indexer geometry: {qsa_values}")
    if qsa_values["index_kv_heads"] != 1:
        raise ValueError("Qwen4 QSA requires exactly one index key head")
    if qsa_values["index_token_budget"] % qsa_values["index_compress_ratio"]:
        raise ValueError("Qwen4 QSA token budget must be divisible by compression ratio")
    if qsa_values["index_token_budget"] + qsa_values["index_compress_ratio"] - 1 > 4096:
        raise ValueError("Qwen4 QSA selected-token kernel supports at most 4096 indices")

    hc_count = int(getattr(text, "hc_count", 0))
    hc_lowrank = int(getattr(text, "hc_lowrank", 0))
    if hc_count <= 1 or hc_lowrank <= 0:
        raise ValueError(f"Qwen4 requires hc_count>1 and hc_lowrank>0, got {hc_count}/{hc_lowrank}")
    activation = str(getattr(text, "output_gate_type", None) or text.hidden_act)
    if activation not in {"sigmoid", "silu"}:
        raise ValueError(f"unsupported Qwen4 GDN output gate: {activation}")
    eos = getattr(text, "eos_token_id", None)
    eos = eos[0] if isinstance(eos, list) and eos else eos
    if ple_ids and eos is None:
        raise ValueError("Qwen4 PLE requires eos_token_id")

    return Qwen4ExpArgs(
        hc_count=hc_count,
        hc_lowrank=hc_lowrank,
        qsa_layer_ids=qsa_ids,
        ple_layer_ids=ple_ids,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=int(getattr(text, "ple_conv_kernel_size", 4)),
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=int(getattr(text, "ngram_vocab_size_base", 20_000_000)),
        ngram_vocab_divisor=int(getattr(text, "make_ngram_vocab_size_divisible_by", 128)),
        split_ngram_parts=int(getattr(text, "split_ngram_parts", 128)),
        seed=int(getattr(text, "seed", 1234)),
        eos_token_id=int(eos or 0),
        output_gate_activation=activation,
        **qsa_values,
    )


__all__ = ["Qwen4ExpArgs", "load_args"]
