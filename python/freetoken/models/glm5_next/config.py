"""Engine-facing configuration for the GLM-5.3-Flash text tower.

The public checkpoint is a multimodal wrapper, but Spark Lab serves and certifies
its language tower.  The tower is a 3:1 KDA/NoPE-MLA hybrid and uses four manifold-
constrained Hyper-Connection streams.  Spark Lab supports both the publisher's
dynamic 128x128 block-FP8 checkpoint and Red Hat AI's routed-expert NVFP4 derivative.
"""

from __future__ import annotations

import os
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_compressed_tensors_nvfp4,
)

from .args import Glm5NextArgs


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _block_fp8(config: Any) -> tuple[str, tuple[int, int] | None]:
    quant = _get(config, "quantization_config") or {}
    method = str(_get(quant, "quant_method", "")).lower()
    fmt = str(_get(quant, "fmt", "")).lower()
    block = tuple(int(x) for x in (_get(quant, "weight_block_size", ()) or ()))
    if method == "fp8" and fmt == "e4m3" and block == (128, 128):
        if str(_get(quant, "activation_scheme", "")).lower() != "dynamic":
            raise ValueError("GLM-5.3 block-FP8 requires dynamic activation quantization")
        return "fp8_block", block
    raise ValueError(
        "GLM-5.3-Flash requires the pinned e4m3 dynamic 128x128 block-FP8 checkpoint"
    )


def parse_config(hf_config: Any) -> ModelConfig:
    text = _get(hf_config, "text_config", hf_config)
    linear = _get(text, "linear_attn_config") or {}
    source_layers = int(_get(text, "num_hidden_layers"))
    num_layers = source_layers
    cap = os.getenv("FREETOKEN_GLM5_NEXT_MAX_LAYERS")
    if cap:
        num_layers = min(source_layers, int(cap))

    layer_types = tuple(str(x) for x in _get(text, "layer_types", ()))[:num_layers]
    if len(layer_types) != num_layers:
        raise ValueError(
            f"GLM-5.3 layer_types has {len(layer_types)} entries for {num_layers} layers"
        )
    allowed = {"linear_attention", "deepseek_sparse_attention"}
    unknown = sorted(set(layer_types) - allowed)
    if unknown:
        raise ValueError(f"unsupported GLM-5.3 layer types: {unknown}")
    kda_ids = tuple(i for i, kind in enumerate(layer_types) if kind == "linear_attention")
    dsa_ids = tuple(i for i, kind in enumerate(layer_types) if kind == "deepseek_sparse_attention")

    # The released lists are zero-based.  Validate them against layer_types rather
    # than trusting two independent pieces of checkpoint metadata.
    listed_kda = tuple(int(i) for i in _get(linear, "kda_layers", ()) if int(i) < num_layers)
    listed_dsa = tuple(int(i) for i in _get(linear, "full_attn_layers", ()) if int(i) < num_layers)
    if listed_kda and listed_kda != kda_ids:
        raise ValueError(f"GLM-5.3 KDA layer list disagrees with layer_types: {listed_kda} != {kda_ids}")
    if listed_dsa and listed_dsa != dsa_ids:
        raise ValueError(f"GLM-5.3 MLA layer list disagrees with layer_types: {listed_dsa} != {dsa_ids}")

    qk_nope = int(_get(text, "qk_nope_head_dim"))
    qk_rope = int(_get(text, "qk_rope_head_dim", 0))
    if bool(_get(text, "mla_use_nope", False)) and qk_rope != 0:
        raise ValueError("GLM-5.3 NoPE MLA checkpoint unexpectedly has a RoPE dimension")
    kv_rank = int(_get(text, "kv_lora_rank"))
    rotary = RotaryConfig(
        head_dim=qk_nope + qk_rope,
        rotary_dim=qk_rope,
        max_position=int(_get(text, "max_position_embeddings")),
        base=float(_get(text, "rope_theta", 10000.0)),
        scaling=None,
    )

    quant_source = text if _get(text, "quantization_config") is not None else hf_config
    if detect_compressed_tensors_nvfp4(quant_source):
        # RedHatAI/GLM-5.3-Flash-NVFP4 targets only routed expert projections.
        # Every resident text-tower Linear remains BF16 per the checkpoint ignore list.
        expert_quant = "nvfp4"
        resident_quant = "none"
        block = None
    else:
        expert_quant, block = _block_fp8(quant_source)
        resident_quant = expert_quant
    index_dim = int(_get(text, "index_head_dim"))
    kpool = int(_get(text, "index_kpool", 1))
    kda_quant = str(
        os.environ.get("_FREETOKEN_CONVERT_GLM5_KDA_QUANT")
        or _get(text, "freetoken_kda_quant", _get(hf_config, "freetoken_kda_quant", "none"))
    ).lower()
    if kda_quant not in {"none", "fp8_pertensor"}:
        raise ValueError(f"unsupported GLM-5.3 KDA quantization: {kda_quant!r}")
    dsa_on = bool(dsa_ids) and os.getenv("FREETOKEN_GLM5_NEXT_DSA", "1") != "0"
    args = Glm5NextArgs(
        hidden_size=int(_get(text, "hidden_size")),
        num_heads=int(_get(text, "num_attention_heads")),
        q_lora_rank=int(_get(text, "q_lora_rank")),
        kv_lora_rank=kv_rank,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=int(_get(text, "v_head_dim")),
        norm_eps=float(_get(text, "rms_norm_eps", 1e-5)),
        max_position=int(_get(text, "max_position_embeddings")),
        layer_types=layer_types,
        kda_layer_ids=kda_ids,
        dsa_layer_ids=dsa_ids,
        kda_num_heads=int(_get(linear, "num_heads")),
        kda_head_dim=int(_get(linear, "head_dim")),
        kda_conv_kernel=int(_get(linear, "short_conv_kernel_size")),
        kda_gate_lower_bound=_get(linear, "gate_lower_bound"),
        kda_quant=kda_quant,
        index_n_heads=int(_get(text, "index_n_heads")),
        index_head_dim=index_dim,
        index_topk=int(_get(text, "index_topk")),
        index_kpool=kpool,
        index_kpool_always_select_tail=bool(
            _get(text, "index_kpool_always_select_tail", True)
        ),
        hc_mult=int(_get(text, "hc_mult")),
        hc_eps=float(_get(text, "hc_eps")),
        hc_sinkhorn_iters=int(_get(text, "hc_sinkhorn_iters")),
    )
    if args.hc_mult != 4:
        raise ValueError(f"GLM-5.3 serving currently requires hc_mult=4, got {args.hc_mult}")
    if kpool < 1 or args.index_topk % kpool:
        raise ValueError(f"invalid GLM-5.3 KPool geometry: topk={args.index_topk}, pool={kpool}")

    groups = []
    if kda_ids:
        groups.append(
            LinearGatedDeltaGroupConfig(
                name="linear",
                layer_ids=kda_ids,
                num_key_heads=args.kda_num_heads,
                num_value_heads=args.kda_num_heads,
                key_head_dim=args.kda_head_dim,
                value_head_dim=args.kda_head_dim,
                conv_kernel_dim=args.kda_conv_kernel,
                output_gate=True,
            )
        )
    if dsa_ids:
        groups.append(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=dsa_ids,
                num_kv_heads=1,
                head_dim=kv_rank + qk_rope,
                rotary_config=rotary,
                mla=True,
                # KPool reconstructs a learned pooled key from cached normalized K
                # and per-channel gate logits, hence two index_head_dim fields.
                index_head_dim=args.packed_index_dim if dsa_on else 0,
                num_index_layers=len(dsa_ids) if dsa_on else 0,
            )
        )

    num_experts = int(_get(text, "n_routed_experts", 0) or 0)
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=1,
        head_dim=kv_rank + qk_rope,
        hidden_size=args.hidden_size,
        vocab_size=int(_get(text, "vocab_size")),
        intermediate_size=int(_get(text, "intermediate_size")),
        rms_norm_eps=args.norm_eps,
        rotary_config=rotary,
        hidden_act=str(_get(text, "hidden_act", "silu")),
        tie_word_embeddings=bool(_get(text, "tie_word_embeddings", False)),
        num_experts=num_experts,
        num_experts_per_tok=int(_get(text, "num_experts_per_tok")),
        moe_intermediate_size=int(_get(text, "moe_intermediate_size")),
        norm_topk_prob=bool(_get(text, "norm_topk_prob", True)),
        model_type=str(_get(hf_config, "model_type", "glm5_next")),
        architectures=list(
            _get(hf_config, "architectures", ["Glm5NextForConditionalGeneration"])
        ),
        moe_enabled=num_experts > 0,
        linear_state_snapshots=False,
        expert_quant=expert_quant,
        weight_block_size=block,
        fp8_block_scale_dtype="float32",
        attn_quant=resident_quant,
        dense_quant=resident_quant,
        moe_weight_format=expert_quant,
        first_k_dense_replace=int(_get(text, "first_k_dense_replace", 0)),
        n_shared_experts=int(_get(text, "n_shared_experts", 0)),
        shared_expert_intermediate_size=(
            int(_get(text, "moe_intermediate_size")) * int(_get(text, "n_shared_experts", 0))
        ),
        routed_scaling_factor=float(_get(text, "routed_scaling_factor", 1.0)),
        n_group=int(_get(text, "n_group", 1)),
        topk_group=int(_get(text, "topk_group", 1)),
        attn_sm_scale=(qk_nope + qk_rope) ** -0.5,
        swiglu_limit=float(_get(text, "swiglu_limit", 10.0)),
        attention_groups=tuple(sorted(groups, key=lambda group: group.layer_ids[0])),
        vision_config=None,
        image_token_id=_get(hf_config, "image_token_id"),
        glm5_next_args=args,
        single_stream_only=True,
    )


__all__ = ["parse_config"]
