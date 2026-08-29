"""Engine-facing configuration for the text tower of Moonshot Kimi K3."""

from __future__ import annotations

import os
from typing import Any

from sparklab.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .args import KimiK3Args


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_mxfp4(text: Any) -> bool:
    quant = _get(text, "quantization_config")
    if quant is None:
        return False
    method = str(_get(quant, "quant_method", "")).lower()
    fmt = str(_get(quant, "format", "")).lower()
    groups = _get(quant, "config_groups", {}) or {}
    for group in groups.values() if isinstance(groups, dict) else ():
        weight = _get(group, "weights", {}) or {}
        if (
            int(_get(weight, "num_bits", 0) or 0) == 4
            and str(_get(weight, "type", "")).lower() == "float"
            and int(_get(weight, "group_size", 0) or 0) == 32
        ):
            return method == "compressed-tensors" or "mxfp4" in fmt
    return False


def _modelopt_quant_kinds(hf_config: Any) -> set[str]:
    """Return the quant algorithms declared by NVIDIA's mixed K3 export.

    Moonshot's checkpoint keeps its compressed-tensors configuration on the text
    config.  NVIDIA's ``Kimi-K3-NVFP4`` instead puts a ModelOpt
    ``quantized_layers`` inventory on the multimodal wrapper, so looking only at
    ``text_config`` silently classifies the packed expert weights as BF16.
    """
    quant = _get(hf_config, "quantization_config") or {}
    method = str(_get(quant, "quant_method", "")).lower()
    producer = _get(quant, "producer", {}) or {}
    if method != "modelopt_mixed" and str(_get(producer, "name", "")).lower() != "modelopt":
        return set()
    layers = _get(quant, "quantized_layers", {}) or {}
    if not isinstance(layers, dict):
        return set()
    return {
        str(_get(spec or {}, "quant_algo", "")).upper()
        for spec in layers.values()
        if _get(spec or {}, "quant_algo")
    }


def resident_mlp_quant() -> str:
    """Resolved representation for K3's dense and shared-expert MLP weights."""
    return "fp8_pertensor" if os.getenv("SPARKLAB_KIMI_MLP_FP8", "0") == "1" else "none"


def parse_config(hf_config: Any) -> ModelConfig:
    text = _get(hf_config, "text_config", hf_config)
    linear = _get(text, "linear_attn_config") or {}
    num_layers = int(_get(text, "num_hidden_layers"))

    # Kimi's layer lists are deliberately 1-based.  Intersect with the decoder
    # range because released configs may also list a trailing MTP layer.
    kda_ids = tuple(
        sorted({int(i) - 1 for i in _get(linear, "kda_layers", ()) if 1 <= int(i) <= num_layers})
    )
    full_ids = tuple(
        sorted(
            {
                int(i) - 1
                for i in _get(linear, "full_attn_layers", ())
                if 1 <= int(i) <= num_layers
            }
        )
    )
    covered = set(kda_ids) | set(full_ids)
    if covered != set(range(num_layers)) or set(kda_ids) & set(full_ids):
        missing = sorted(set(range(num_layers)) - covered)
        overlap = sorted(set(kda_ids) & set(full_ids))
        raise ValueError(
            f"invalid Kimi K3 attention layer partition: "
            f"missing={missing}, overlap={overlap}"
        )

    qk_nope = int(_get(text, "qk_nope_head_dim"))
    qk_rope = int(_get(text, "qk_rope_head_dim"))
    kv_rank = int(_get(text, "kv_lora_rank"))
    rotary = RotaryConfig(
        head_dim=qk_nope + qk_rope,
        rotary_dim=qk_rope,
        max_position=int(_get(text, "max_position_embeddings")),
        base=float(_get(text, "rope_theta", 10000.0)),
        scaling=_get(text, "rope_scaling"),
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=1,
        head_dim=kv_rank + qk_rope,
        rotary_config=rotary,
        mla=True,
    )
    kda_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=kda_ids,
        num_key_heads=int(_get(linear, "num_heads")),
        num_value_heads=int(_get(linear, "num_heads")),
        key_head_dim=int(_get(linear, "head_dim")),
        value_head_dim=int(_get(linear, "head_dim")),
        conv_kernel_dim=int(_get(linear, "short_conv_kernel_size")),
        output_gate=True,
    )
    groups = tuple(sorted((full_group, kda_group), key=lambda g: g.layer_ids[0]))

    args = KimiK3Args(
        q_lora_rank=int(_get(text, "q_lora_rank")),
        kv_lora_rank=kv_rank,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=int(_get(text, "v_head_dim")),
        kda_num_heads=int(_get(linear, "num_heads")),
        kda_head_dim=int(_get(linear, "head_dim")),
        kda_conv_kernel=int(_get(linear, "short_conv_kernel_size")),
        kda_full_rank_gate=bool(_get(linear, "use_full_rank_gate", False)),
        kda_gate_lower_bound=_get(linear, "gate_lower_bound"),
        attn_res_block_size=int(_get(text, "attn_res_block_size")),
        routed_expert_hidden_size=int(_get(text, "routed_expert_hidden_size")),
        latent_moe_use_norm=bool(_get(text, "latent_moe_use_norm", False)),
        situ_beta=float(_get(text, "activation_situ_beta", 4.0)),
        situ_linear_beta=_get(text, "activation_situ_linear_beta"),
        mla_output_gate=bool(_get(text, "mla_use_output_gate", False)),
    )
    num_experts = int(_get(text, "num_experts", 0) or 0)
    modelopt_kinds = _modelopt_quant_kinds(hf_config)
    mxfp4 = _is_mxfp4(text)
    expert_quant = "nvfp4" if "NVFP4" in modelopt_kinds else ("mxfp4" if mxfp4 else "none")
    resident_quant = "fp8_block" if "FP8_PB_WO" in modelopt_kinds else "none"
    # K3's released NVFP4 checkpoint leaves the dense layer and every shared expert in
    # BF16. On a 128-GiB GB10 those always-resident matrices prevent the mandatory
    # one-slot-per-expert GPU cache from fitting. This opt-in W8A16 mode is resolved into
    # ModelConfig so model construction and both source/FTW loaders cannot disagree.
    mlp_quant = resident_mlp_quant()
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=int(_get(text, "num_attention_heads")),
        num_kv_heads=1,
        head_dim=kv_rank + qk_rope,
        hidden_size=int(_get(text, "hidden_size")),
        vocab_size=int(_get(text, "vocab_size")),
        intermediate_size=int(_get(text, "intermediate_size")),
        rms_norm_eps=float(_get(text, "rms_norm_eps", 1e-5)),
        rotary_config=rotary,
        hidden_act=str(_get(text, "hidden_act", "situ")),
        tie_word_embeddings=bool(_get(text, "tie_word_embeddings", False)),
        num_experts=num_experts,
        num_experts_per_tok=int(_get(text, "num_experts_per_token", 0)),
        moe_intermediate_size=int(_get(text, "moe_intermediate_size", 0)),
        norm_topk_prob=bool(_get(text, "moe_renormalize", True)),
        model_type=str(_get(hf_config, "model_type", "kimi_k3")),
        architectures=list(_get(hf_config, "architectures", ["KimiK3ForConditionalGeneration"])),
        moe_enabled=num_experts > 0,
        expert_hidden_size=args.routed_expert_hidden_size,
        linear_state_snapshots=False,
        expert_quant=expert_quant,
        weight_block_size=(128, 128) if resident_quant == "fp8_block" else None,
        fp8_block_scale_dtype="float32",
        attn_quant=resident_quant,
        dense_quant=mlp_quant,
        # The GB10 resident profile also keeps the untied embedding/head in per-row
        # FP8. K3 does not currently expose a separate embedding-quant field, so the
        # model deliberately keys both vocabulary matrices off this same resolved mode.
        lm_head_quant=mlp_quant,
        moe_weight_format=expert_quant if expert_quant != "none" else None,
        first_k_dense_replace=int(_get(text, "first_k_dense_replace", 0)),
        n_shared_experts=int(_get(text, "num_shared_experts", 0) or 0),
        routed_scaling_factor=float(_get(text, "routed_scaling_factor", 1.0)),
        n_group=int(_get(text, "num_expert_group", 1)),
        topk_group=int(_get(text, "topk_group", 1)),
        attention_groups=groups,
        attn_sm_scale=(qk_nope + qk_rope) ** -0.5,
        vision_config=None,
        image_token_id=_get(hf_config, "media_placeholder_token_id"),
        kimi_k3_args=args,
    )


__all__ = ["parse_config"]
