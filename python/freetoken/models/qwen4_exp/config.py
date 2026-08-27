"""Engine configuration for the Qwen3.8-Flash-Next Qwen4-Exp text tower."""

from __future__ import annotations

import os
from typing import Any

from freetoken.attention import AttnType
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .args import load_args


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", None) or hf_config
    real_layers = int(text.num_hidden_layers)
    cap = os.getenv("FREETOKEN_QWEN4_MAX_LAYERS")
    num_layers = min(real_layers, int(cap)) if cap else real_layers
    if num_layers <= 0:
        raise ValueError(f"FREETOKEN_QWEN4_MAX_LAYERS must be positive, got {num_layers}")
    if num_layers < real_layers:
        from freetoken.utils import init_logger

        init_logger(__name__).warning(
            "FREETOKEN_QWEN4_MAX_LAYERS: serving a truncated Qwen4 model "
            f"({num_layers}/{real_layers} layers); outputs are not meaningful"
        )
    raw_types = getattr(text, "layer_types", None)
    if raw_types is None:
        interval = int(getattr(text, "full_attention_interval", 4))
        raw_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]
    layer_types = tuple(str(x) for x in raw_types[:num_layers])
    allowed = {"linear_attention", "full_attention", "qwen_sparse_attention"}
    if len(layer_types) != num_layers or set(layer_types) - allowed:
        raise ValueError(f"invalid Qwen4 layer partition: {layer_types}")

    args = load_args(text, layer_types)
    linear_ids = tuple(i for i, kind in enumerate(layer_types) if kind == "linear_attention")
    if set(linear_ids) | set(args.qsa_layer_ids) != set(range(num_layers)):
        raise ValueError("Qwen4 layer partition has holes")

    head_dim = int(getattr(text, "head_dim", 0) or text.hidden_size // text.num_attention_heads)
    rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None) or {}
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type not in {None, "default"}:
        raise ValueError(f"Qwen4 runtime implements default partial RoPE, got {rope!r}")
    rotary_dim = round(head_dim * float(rope.get(
        "partial_rotary_factor", getattr(text, "partial_rotary_factor", 1.0)
    )))
    if rotary_dim > args.index_head_dim:
        raise ValueError("Qwen4 rotary dimensions do not fit the QSA index head")
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(text.max_position_embeddings),
        base=float(rope.get("rope_theta", getattr(text, "rope_theta", 10_000.0))),
        scaling=None,
    )

    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=int(text.linear_num_key_heads),
            num_value_heads=int(text.linear_num_value_heads),
            key_head_dim=int(text.linear_key_head_dim),
            value_head_dim=int(text.linear_value_head_dim),
            conv_kernel_dim=int(text.linear_conv_kernel_dim),
            output_gate=True,
        ),
        FullAttentionGroupConfig(
            name="qsa",
            layer_ids=args.qsa_layer_ids,
            num_kv_heads=int(text.num_key_value_heads),
            head_dim=head_dim,
            rotary_config=rotary,
            index_head_dim=args.index_head_dim,
            num_index_layers=len(args.qsa_layer_ids),
            indexed_attn_type=AttnType.QSA,
        ),
    )

    expert_quant = str(getattr(text, "freetoken_expert_quant", "none"))
    if expert_quant not in {"none", "nvfp4"}:
        raise ValueError(f"unsupported Qwen4 expert quantization: {expert_quant!r}")

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=int(text.num_attention_heads),
        num_kv_heads=int(text.num_key_value_heads),
        head_dim=head_dim,
        hidden_size=int(text.hidden_size),
        vocab_size=int(text.vocab_size),
        intermediate_size=int(getattr(text, "intermediate_size", 0) or 0),
        rms_norm_eps=float(text.rms_norm_eps),
        rotary_config=rotary,
        hidden_act=str(text.hidden_act),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        num_experts=int(text.num_experts),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        model_type=str(getattr(hf_config, "model_type", "qwen4_exp")),
        architectures=list(getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"])),
        moe_enabled=True,
        shared_expert_intermediate_size=int(text.shared_expert_intermediate_size),
        use_qk_norm=True,
        attention_groups=groups,
        vision_config=None,
        image_token_id=getattr(hf_config, "image_token_id", None),
        expert_quant=expert_quant,
        linear_state_snapshots=False,
        qwen4_exp_args=args,
    )


__all__ = ["parse_config"]
