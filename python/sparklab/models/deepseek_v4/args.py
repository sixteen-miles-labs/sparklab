"""DeepSeek-V4-Flash hyperparameters.

Field names mirror the authors' ``inference/config.json`` (consumed by the
reference ``ModelArgs``) so the port stays 1:1 with the reference. ``load_args``
reads that file from the checkpoint directory (it ships alongside the weights);
the few runtime knobs (batch / sequence length) are overlaid by the runner.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Literal, Tuple


@dataclass
class DeepseekV4Args:
    # ----- runtime -----
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = "fp4"
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    # Runtime cache representation. FP8 is true one-byte storage; BF16 keeps
    # the historical quantize-then-dequantize representation.
    kv_storage_dtype: Literal["bf16", "fp8"] = "bf16"
    index_storage_dtype: Literal["bf16", "fp4"] = "bf16"

    # ----- shape -----
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_hash_layers: int = 3
    n_mtp_layers: int = 1
    n_heads: int = 64

    # ----- moe -----
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    # ----- mla -----
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int, ...] = (0, 0, 4, 128, 4, 128, 4, 0)

    # ----- rope / yarn -----
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1

    # ----- lightning indexer -----
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512

    # ----- hyper-connections -----
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ----- fused DSpark draft -----
    # The 0731 checkpoint ships three draft decoder blocks under ``mtp.*``.
    # They are optional at runtime: ordinary target-only serving keeps the
    # original 43-layer cache/expert geometry byte-for-byte unchanged.
    dspark_block_size: int = 0
    dspark_noise_token_id: int = -1
    dspark_target_layer_ids: Tuple[int, ...] = ()
    dspark_markov_rank: int = 0
    dspark_enabled: bool = False

    def __post_init__(self) -> None:
        # JSON lists -> tuple so the dataclass stays hashable / immutable-ish.
        if isinstance(self.compress_ratios, list):
            self.compress_ratios = tuple(self.compress_ratios)
        if isinstance(self.dspark_target_layer_ids, list):
            self.dspark_target_layer_ids = tuple(self.dspark_target_layer_ids)

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def runtime_n_layers(self) -> int:
        return self.n_layers + (self.n_mtp_layers if self.dspark_enabled else 0)

    @property
    def runtime_compress_ratios(self) -> Tuple[int, ...]:
        ratios = tuple(self.compress_ratios)[: self.runtime_n_layers]
        if self.dspark_enabled and len(ratios) != self.runtime_n_layers:
            raise ValueError(
                f"DeepSeek-V4 config has {len(ratios)} compression ratios for "
                f"{self.runtime_n_layers} runtime layers"
            )
        return ratios


def _config_path(model_path: str) -> str:
    """Locate the authors' ModelArgs JSON inside the checkpoint directory."""
    candidates = [
        os.path.join(model_path, "inference", "config.json"),
        os.path.join(model_path, "model_args.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No DeepSeek-V4 ModelArgs JSON found under {model_path} "
        f"(looked for inference/config.json)"
    )


def load_args(model_path: str, **overrides) -> DeepseekV4Args:
    """Build :class:`DeepseekV4Args` from the checkpoint's ``inference/config.json``.

    ``overrides`` (e.g. ``max_seq_len``, ``max_batch_size``) take precedence over the
    file, letting the runner size the per-request caches.
    """
    with open(_config_path(model_path)) as f:
        raw = json.load(f)
    valid = {f.name for f in fields(DeepseekV4Args)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    kwargs.update(overrides)
    return DeepseekV4Args(**kwargs)


__all__ = ["DeepseekV4Args", "load_args"]
