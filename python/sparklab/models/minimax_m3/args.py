"""MiniMax-M3 (``minimax_m3``) hyperparameters.

MiniMax-M3 is a 428B-A23B multimodal MoE (SparkLab serves the text tower; the
``vision_tower.`` stack is skipped like every other checkpoint's). The text tower is
GQA (64 q heads / 4 kv heads / head_dim 128) with per-head Gemma-style (1+w) q/k
norms and partial NeoX RoPE (rotary_dim 64); the first ``sparse_attention_freq==0``
layers run dense attention + a dense SwiGLU-OAI MLP, every later layer runs
**block-sparse attention** (a lightning indexer scores 128-token blocks and each
query attends only its top-``topk_blocks`` blocks, per KV head) + a sigmoid-routed
MoE (128 NVFP4 experts, top-4, + 1 MXFP8 shared expert).

This payload carries the sparse-indexer geometry and the swigluoai/dense-MLP scalars
the model module and the ``m3_sparse`` attention backend need; it is stashed on
``ModelConfig.m3_args`` (opaque to the engine). The MoE/router knobs live on
``ModelConfig`` directly (shared with the glm4_moe/glm_moe_dsa path).

Semantics pinned against the vLLM reference implementation
(the vLLM tree's ``vllm/models/minimax_m3/``):

* index heads == KV heads (4): each index head selects top-k blocks for ITS kv
  head's whole GQA group -- there is no cross-head score reduction.
* one shared index KEY head (``index_k_proj`` is ``[index_dim, hidden]``).
* block score = max over the block's 128 positions of ``dot(index_q, index_k)``
  (``sparse_score_type == "max"``), causal-masked, NO softmax scale (only the
  ordering is consumed).
* the newest ``local_blocks`` (1) blocks are force-selected; ``init_blocks`` (0)
  leading blocks would be too. Selection always includes the query's own block.
* index q/k get the same per-head Gemma norm + partial NeoX RoPE as the main q/k.
* ``sparse_disable_index_value`` is 1 on every sparse layer: the indexer is
  score-only (no index value/output projections exist in the checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class MiniMaxM3Args:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    norm_eps: float
    # RoPE (partial NeoX; shared by the main q/k and the indexer q/k)
    rotary_dim: int
    rope_theta: float
    max_position: int
    # SwiGLU-OAI scalars (routed experts, shared experts and the dense MLPs)
    swiglu_alpha: float
    swiglu_limit: float
    dense_intermediate_size: int
    shared_intermediate_size: int
    # Block-sparse attention (all zeros / empty when serving the dense ablation)
    index_dim: int
    num_index_heads: int
    topk_blocks: int
    block_size: int
    init_blocks: int
    local_blocks: int
    sparse_layer_ids: Tuple[int, ...]
    # Layers whose FFN is the sparse MoE block (the rest run the dense MLP)
    moe_layer_ids: Tuple[int, ...]

    @property
    def use_sparse(self) -> bool:
        return bool(self.sparse_layer_ids) and self.index_dim > 0 and self.topk_blocks > 0

    def is_sparse_layer(self, layer_id: int) -> bool:
        return self.use_sparse and layer_id in self.sparse_layer_ids

    def sparse_slot(self, layer_id: int) -> int:
        """Index-key slab slot for a sparse layer (slot order = sparse-layer order)."""
        return self.sparse_layer_ids.index(layer_id)


def _freq_ids(freq, num_layers: int) -> Tuple[int, ...]:
    if not freq:
        return tuple(range(num_layers))
    return tuple(i for i, f in enumerate(freq[:num_layers]) if f)


# Config-shape normalization. Two shapes reach load_args: raw checkpoint dicts
# (the auto_map remote-code path keeps rope_theta / sparse_attention_config /
# moe_layer_freq verbatim), and transformers >= 5's native minimax_m3_vl
# TextConfig, whose __post_init__ moves rope_theta into rope_parameters, pops the
# sparse config and moe_layer_freq into flat index_* keys + layer_types /
# mlp_layer_types, and force-sets hidden_act to "silu". Every sibling family
# carries the same dual-shape shim.
def _rope_theta(text_config: Any) -> float:
    theta = getattr(text_config, "rope_theta", None)
    if theta is None:
        params = getattr(text_config, "rope_parameters", None) or {}
        theta = params.get("rope_theta") if isinstance(params, dict) else None
    return float(theta) if theta is not None else 10000.0


def _sparse_geometry(text_config: Any) -> dict:
    """The sparse-attention geometry in the raw ``sparse_attention_config`` dict
    shape, whichever config shape arrived. The native class has no flat keys for
    init_block / score_type / disable_index_value -- it hardcodes those M3
    semantics, so the translation pins the same values."""
    raw = getattr(text_config, "sparse_attention_config", None)
    if raw:
        return dict(raw)
    layer_types = getattr(text_config, "layer_types", None) or []
    if "minimax_m3_sparse" not in layer_types:
        return {}
    return {
        "use_sparse_attention": True,
        "sparse_attention_freq": [
            1 if t == "minimax_m3_sparse" else 0 for t in layer_types
        ],
        "sparse_num_index_heads": int(getattr(text_config, "index_n_heads", 4)),
        "sparse_index_dim": int(getattr(text_config, "index_head_dim", 128)),
        "sparse_block_size": int(getattr(text_config, "index_block_size", 128)),
        "sparse_topk_blocks": int(getattr(text_config, "index_topk_blocks", 16)),
        "sparse_local_block": int(getattr(text_config, "index_local_blocks", 1)),
        "sparse_init_block": 0,
        "sparse_score_type": "max",
    }


def _moe_layer_freq(text_config: Any):
    freq = getattr(text_config, "moe_layer_freq", None)
    if freq is not None:
        return freq
    mlp_types = getattr(text_config, "mlp_layer_types", None)
    if mlp_types is not None:
        return [1 if t == "sparse" else 0 for t in mlp_types]
    return None


def load_args(text_config: Any, num_layers: int, *, sparse_enabled: bool) -> MiniMaxM3Args:
    """Build the payload from the HF ``text_config`` (already unwrapped), resolving
    the sparse-attention switch ONCE here (``sparse_enabled`` folds SPARKLAB_M3_SPARSE
    in parse_config): the pool factory, the KV cost model, the backend and the model
    modules all read the resolved payload / group spec, never the env."""
    sparse_cfg = _sparse_geometry(text_config)
    use_sparse = sparse_enabled and bool(sparse_cfg.get("use_sparse_attention", False))

    head_dim_early = getattr(text_config, "head_dim", None) or (
        text_config.hidden_size // text_config.num_attention_heads
    )
    if use_sparse:
        score_type = sparse_cfg.get("sparse_score_type", "max")
        assert score_type == "max", (
            f"MiniMax-M3 sparse attention only implements sparse_score_type='max', "
            f"got {score_type!r}"
        )
        sparse_layer_ids = _freq_ids(sparse_cfg.get("sparse_attention_freq"), num_layers)
        # Score-only indexer: the checkpoint ships no index value/output projections.
        disable_value = sparse_cfg.get("sparse_disable_index_value")
        if disable_value is not None:
            assert all(
                bool(disable_value[i]) for i in sparse_layer_ids if i < len(disable_value)
            ), "MiniMax-M3 sparse layers must be score-only (sparse_disable_index_value)"
        # ---- geometry invariants the kernels/backend hardcode; fail a variant ----
        # checkpoint at PARSE time instead of mis-addressing / mis-compiling at
        # serve time.
        blk = int(sparse_cfg.get("sparse_block_size", 128))
        assert blk == 128, (
            f"MiniMax-M3 kernels and the pinned KV page size assume 128-token "
            f"sparse blocks, got sparse_block_size={blk}"
        )
        idx_dim = int(sparse_cfg.get("sparse_index_dim", 0))
        assert idx_dim == head_dim_early, (
            f"the indexer reuses the main rotary (rotary slices by head_dim); "
            f"sparse_index_dim={idx_dim} != head_dim={head_dim_early}"
        )
        topk = int(sparse_cfg.get("sparse_topk_blocks", 0))
        assert 0 < topk < 64, (
            f"the decode top-k kernel's merge stage supports topk_blocks < 64, "
            f"got {topk} (would fail Triton compile at serve time)"
        )
        local = int(sparse_cfg.get("sparse_local_block", 0))
        assert local >= 1, (
            "the attend kernels rely on the forced local block guaranteeing every "
            f"query at least one visible position; got sparse_local_block={local}"
        )
    else:
        sparse_layer_ids = ()

    # swigluoai is implemented in four places (triton *_and_mul, fused_nvfp4, the
    # CPU GEMV epilogue, eager) with the (up + 1) bias hardcoded; a checkpoint
    # that changes swiglu_beta needs all four updated.
    beta = float(getattr(text_config, "swiglu_beta", 1.0))
    assert beta == 1.0, f"swigluoai implementations hardcode beta=1.0, got {beta}"
    # Plain-rope only: silently ignoring a variant checkpoint's rope_scaling would
    # mis-position every token past the scaling boundary. Newer HF configs spell
    # the key `rope_parameters`; check both so a renamed config still trips this.
    scaling = (getattr(text_config, "rope_scaling", None)
               or getattr(text_config, "rope_parameters", None) or {})
    rope_type = scaling.get("rope_type", scaling.get("type", "default")) if scaling else "default"
    assert rope_type in (None, "default"), (
        f"MiniMax-M3 support implements plain rope only, got rope_scaling={scaling!r}"
    )
    scoring = getattr(text_config, "scoring_func", "sigmoid")
    assert scoring == "sigmoid", (
        f"MiniMax-M3 routing implements sigmoid scoring only, got {scoring!r}"
    )

    assert not bool(getattr(text_config, "attention_output_gate", False)), (
        "MiniMax-M3 attention_output_gate is not supported (M3 ships it disabled)"
    )
    assert getattr(text_config, "qk_norm_type", "per_head") == "per_head"
    assert bool(getattr(text_config, "use_gemma_norm", True)), (
        "MiniMax-M3 support assumes Gemma-style (1+w) RMSNorm (use_gemma_norm)"
    )

    head_dim = getattr(text_config, "head_dim", None) or (
        text_config.hidden_size // text_config.num_attention_heads
    )
    rotary_dim = getattr(text_config, "rotary_dim", None)
    if rotary_dim is None:
        rotary_dim = int(head_dim * getattr(text_config, "partial_rotary_factor", 1.0))

    num_index_heads = int(sparse_cfg.get("sparse_num_index_heads", 0)) if use_sparse else 0
    if use_sparse:
        # Per-KV-head selection: each index head feeds one kv head's GQA group.
        assert num_index_heads == text_config.num_key_value_heads, (
            f"expected sparse_num_index_heads == num_key_value_heads, got "
            f"{num_index_heads} != {text_config.num_key_value_heads}"
        )

    return MiniMaxM3Args(
        hidden_size=text_config.hidden_size,
        num_heads=text_config.num_attention_heads,
        num_kv_heads=text_config.num_key_value_heads,
        head_dim=head_dim,
        norm_eps=text_config.rms_norm_eps,
        rotary_dim=rotary_dim,
        rope_theta=_rope_theta(text_config),
        max_position=text_config.max_position_embeddings,
        swiglu_alpha=float(getattr(text_config, "swiglu_alpha", 1.702)),
        swiglu_limit=float(getattr(text_config, "swiglu_limit", 7.0)),
        dense_intermediate_size=int(
            getattr(text_config, "dense_intermediate_size", 0)
            or text_config.intermediate_size
        ),
        shared_intermediate_size=int(
            getattr(text_config, "shared_intermediate_size", 0)
            or text_config.intermediate_size
        ),
        index_dim=int(sparse_cfg.get("sparse_index_dim", 0)) if use_sparse else 0,
        num_index_heads=num_index_heads,
        topk_blocks=int(sparse_cfg.get("sparse_topk_blocks", 0)) if use_sparse else 0,
        block_size=int(sparse_cfg.get("sparse_block_size", 128)) if use_sparse else 0,
        init_blocks=int(sparse_cfg.get("sparse_init_block", 0)) if use_sparse else 0,
        local_blocks=int(sparse_cfg.get("sparse_local_block", 0)) if use_sparse else 0,
        sparse_layer_ids=sparse_layer_ids,
        moe_layer_ids=_freq_ids(_moe_layer_freq(text_config), num_layers),
    )


__all__ = ["MiniMaxM3Args", "load_args"]
