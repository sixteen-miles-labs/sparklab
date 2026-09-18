"""Prism GGUF metadata and lossless packed-weight mapping."""

from types import SimpleNamespace

import torch

from sparklab.models.gguf.dequant import dequantize
from sparklab.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata


def validate_rotation(m):
    expected = {
        "version": 1,
        "block_size": 1024,
        "transform": "normalized-sylvester-walsh-hadamard",
        "axis": "input-last-dimension",
        "sign_mode": "explicit",
        "gdn_v_grouped": True,
    }
    for key, value in expected.items():
        if m.get("prism.hadamard." + key) != value:
            raise ValueError(f"Unsupported Bonsai rotation metadata: {key}")
    if m.get("prism.hadamard.inverse_weight_names") != ["token_embd.weight"]:
        raise ValueError("Unsupported Bonsai embedding rotation")
    folded = m.get("prism.hadamard.weight_names", [])
    if not folded or len(folded) != len(set(folded)):
        raise ValueError("Missing or duplicate Bonsai folded weight names")
    widths = m["prism.hadamard.sign_widths"]
    values = m["prism.hadamard.sign_values"]
    if len(widths) != len(set(widths)) or any(w <= 0 or w % 1024 for w in widths):
        raise ValueError("Invalid Bonsai sign widths")
    if sum(widths) != len(values) or any(v not in (-1, 1) for v in values):
        raise ValueError("Invalid Bonsai sign values")


def parse_config(shim):
    from sparklab.models.qwen3_5_moe.config import parse_config as qwen_config

    m = shim.metadata
    validate_rotation(m)

    def g(k):
        return m["qwen35." + k]

    text = SimpleNamespace(
        num_hidden_layers=g("block_count"),
        hidden_size=g("embedding_length"),
        intermediate_size=g("feed_forward_length"),
        vocab_size=shim.vocab_size,
        num_attention_heads=g("attention.head_count"),
        num_key_value_heads=g("attention.head_count_kv"),
        head_dim=g("attention.key_length"),
        max_position_embeddings=g("context_length"),
        rms_norm_eps=g("attention.layer_norm_rms_epsilon"),
        hidden_act="silu",
        tie_word_embeddings=False,
        full_attention_interval=g("full_attention_interval"),
        linear_num_key_heads=g("ssm.group_count"),
        linear_num_value_heads=g("ssm.time_step_rank"),
        linear_key_head_dim=g("ssm.state_size"),
        linear_value_head_dim=g("ssm.inner_size") // g("ssm.time_step_rank"),
        linear_conv_kernel_dim=g("ssm.conv_kernel"),
        rope_parameters={
            "rope_theta": g("rope.freq_base"),
            "partial_rotary_factor": g("rope.dimension_count")
            / g("attention.key_length"),
        },
        model_type="qwen35",
        architectures=shim.architectures,
    )
    return qwen_config(text)


def signs_for(m, width):
    start = 0
    for w in m["prism.hadamard.sign_widths"]:
        if w == width:
            return torch.tensor(
                m["prism.hadamard.sign_values"][start : start + w],
                dtype=torch.float32,
                device="cpu",
            )
        start += w
    raise ValueError(f"No Hadamard signs for width {width}")


def mapping(m):
    """Destination -> GGUF tensor list; concatenated along output rows."""
    result = {"model.embed_tokens": ["token_embd.weight"], "lm_head": ["output.weight"]}
    for i in range(m["qwen35.block_count"]):
        src, dst = f"blk.{i}.", f"model.layers.{i}."
        if (i + 1) % m["qwen35.full_attention_interval"]:
            pairs = {
                "linear_attn.in_proj.qkv": ["attn_qkv.weight"],
                "linear_attn.in_proj.z": ["attn_gate.weight"],
                "linear_attn.in_proj.b": ["ssm_beta.weight"],
                "linear_attn.in_proj.a": ["ssm_alpha.weight"],
                "linear_attn.out_proj": ["ssm_out.weight"],
            }
        else:
            pairs = {
                "self_attn.qkv_proj": [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                ],
                "self_attn.o_proj": ["attn_output.weight"],
            }
        pairs.update(
            {
                "mlp.gate_up_proj": ["ffn_gate.weight", "ffn_up.weight"],
                "mlp.down_proj": ["ffn_down.weight"],
            }
        )
        result.update({dst + k: [src + s for s in v] for k, v in pairs.items()})
    return result


def ungroup(t, m, head_dim):
    """GGML tiled V heads -> native grouped V heads, preserving packed rows."""
    nk, nv = m["qwen35.ssm.group_count"], m["qwen35.ssm.time_step_rank"]
    return (
        t.reshape(nv // nk, nk, head_dim, *t.shape[1:])
        .transpose(0, 1)
        .reshape(t.shape)
        .contiguous()
    )


def restore_rows(t, name, m):
    hd = m["qwen35.ssm.state_size"]
    if name.endswith(("attn_qkv.weight", "ssm_conv1d.weight")):
        split = 2 * m["qwen35.ssm.group_count"] * hd
        return torch.cat((t[:split], ungroup(t[split:], m, hd)))
    if name.endswith("attn_gate.weight"):
        return ungroup(t, m, hd)
    if name.endswith(("ssm_beta.weight", "ssm_alpha.weight", "ssm_a", "ssm_dt.bias")):
        return ungroup(t, m, 1)
    return t


def iter_weights(
    model_path, device=None, *, include_moe_experts=True, include_non_moe=True
):
    from sparklab.runtime.distributed import get_tp_info

    if get_tp_info().size != 1:
        raise ValueError("Bonsai GGUF requires TP=1")
    if not include_non_moe:
        return
    m = load_gguf_metadata(model_path)
    validate_rotation(m)
    ts = {t.name: t for t in iter_gguf_tensors(model_path)}
    consumed = set()
    rotated = set(m["prism.hadamard.weight_names"])
    inverse = set(m["prism.hadamard.inverse_weight_names"])
    for dest, sources in mapping(m).items():
        items = [ts[s] for s in sources]
        types = {t.ggml_type for t in items}
        if len(types) != 1 or len({t.shape[-1] for t in items}) != 1:
            raise ValueError(f"Incompatible fused Bonsai tensors: {sources}")
        kind = items[0].ggml_type
        if kind in (142, 143):
            if not all(s in rotated or s in inverse for s in sources):
                raise ValueError(f"Missing ternary rotation: {sources}")
            value = torch.cat([restore_rows(t.packed(), t.name, m) for t in items])
            yield dest + ".qweight", value
            yield dest + ".signs", signs_for(m, items[0].shape[-1])
        elif kind in (0, 1, 30):
            if any(s in rotated or s in inverse for s in sources):
                raise ValueError("Rotated dense Bonsai weights are not supported")
            yield (
                dest + ".weight",
                torch.cat(
                    [
                        restore_rows(
                            dequantize(t.packed(), kind, torch.bfloat16).reshape(
                                t.shape
                            ),
                            t.name,
                            m,
                        )
                        for t in items
                    ]
                ),
            )
        else:
            raise ValueError(f"Unsupported Bonsai weight type {kind}")
        consumed.update(sources)
    norm_map = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ssm_norm.weight": "linear_attn.norm.weight",
        "ssm_conv1d.weight": "linear_attn.conv1d.weight",
        "ssm_dt.bias": "linear_attn.dt_bias",
        "ssm_a": "linear_attn.A_log",
    }
    for name, t in ts.items():
        if name in consumed:
            continue
        if name == "output_norm.weight":
            dest = "model.norm.weight"
        elif name.startswith("blk.") and name.split(".", 2)[2] in norm_map:
            _, layer, suffix = name.split(".", 2)
            dest = f"model.layers.{layer}." + norm_map[suffix]
        else:
            raise ValueError(f"Unexpected Bonsai tensor {name}")
        dtype = (
            torch.float32 if name.endswith(("ssm_a", "ssm_dt.bias")) else torch.bfloat16
        )
        v = restore_rows(
            dequantize(t.packed(), t.ggml_type, torch.float32).reshape(t.shape), name, m
        )
        if name.endswith("ssm_a"):
            v = (-v).log()
        if name.endswith("ssm_conv1d.weight"):
            v = v.unsqueeze(1)
        # GGUF converter already folded +1 into Qwen RMSNorm weights.
        yield dest, v.to(dtype)
    if rotated - consumed or inverse - consumed:
        raise ValueError("Unused Bonsai Hadamard weights")
