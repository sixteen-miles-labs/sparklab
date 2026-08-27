"""Engine-facing configuration for the text tower of Moonshot Kimi K3."""

from __future__ import annotations

from typing import Any

from freetoken.models.config import (
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
        linear_state_snapshots=False,
        expert_quant="mxfp4" if _is_mxfp4(text) else "none",
        moe_weight_format="mxfp4" if _is_mxfp4(text) else None,
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
