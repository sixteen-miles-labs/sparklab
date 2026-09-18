"""Load Prism's Q8_0/BF16 GGUF image projector into the native Qwen vision tower."""

import torch

from sparklab.models.gguf.dequant import dequantize
from sparklab.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata


def vision_config(source):
    m = load_gguf_metadata(str(source))
    if (
        m.get("general.architecture") != "clip"
        or m.get("clip.projector_type") != "qwen3vl_merger"
    ):
        raise ValueError("Expected a Qwen3-VL GGUF image projector")
    if any(m.get("clip.vision.is_deepstack_layers", [])):
        raise ValueError("Deepstack GGUF projectors are not supported")

    def g(k):
        return m["clip.vision." + k]

    return {
        "hidden_size": g("embedding_length"),
        "intermediate_size": g("feed_forward_length"),
        "num_heads": g("attention.head_count"),
        "depth": g("block_count"),
        "out_hidden_size": g("projection_dim"),
        "patch_size": g("patch_size"),
        "spatial_merge_size": g("spatial_merge_size"),
        "temporal_patch_size": 2,
        "num_position_embeddings": (g("image_size") // g("patch_size")) ** 2,
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [],
    }


def image_processor(source, tokenizer):
    from transformers import Qwen2VLImageProcessor

    vc = vision_config(source)
    m = load_gguf_metadata(str(source))
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_id is None or image_id == tokenizer.unk_token_id:
        raise ValueError("Bonsai tokenizer lacks image token")
    processor = Qwen2VLImageProcessor(
        patch_size=vc["patch_size"],
        temporal_patch_size=2,
        merge_size=vc["spatial_merge_size"],
        image_mean=m["clip.vision.image_mean"],
        image_std=m["clip.vision.image_std"],
    )
    return {"vision_config": vc, "image_token_id": image_id}, processor


def load_vision(model, source, device):
    if not hasattr(model, "_bonsai_metadata"):
        raise ValueError(
            "GGUF image projectors currently require the native Bonsai decoder"
        )
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5VisionModel,
        Qwen3_5VisionRotaryEmbedding,
    )

    vc = Qwen3_5VisionConfig(**vision_config(source))
    if vc.out_hidden_size != model.model.embed_tokens.embedding_dim:
        raise ValueError("Bonsai projector width differs from text model")
    vc._attn_implementation = "sdpa"
    with torch.device("meta"):
        tower = Qwen3_5VisionModel(vc)
    expected = tower.state_dict()
    weights, patches = {}, {}
    root = {
        "mm.0": "merger.linear_fc1",
        "mm.2": "merger.linear_fc2",
        "v.post_ln": "merger.norm",
        "v.patch_embd": "patch_embed.proj",
        "v.position_embd": "pos_embed",
    }
    block = {
        "attn_out": "attn.proj",
        "attn_qkv": "attn.qkv",
        "ffn_up": "mlp.linear_fc1",
        "ffn_down": "mlp.linear_fc2",
        "ln1": "norm1",
        "ln2": "norm2",
    }
    for t in iter_gguf_tensors(str(source)):
        raw = t.packed()
        if t.ggml_type == 8:
            blocks = raw.reshape(-1, 34)
            value = (
                blocks[:, :2].contiguous().view(torch.float16).float()
                * blocks[:, 2:].contiguous().view(torch.int8).float()
            ).reshape(t.shape)
        else:
            value = dequantize(raw, t.ggml_type, torch.float32).reshape(t.shape)
        name = t.name
        if name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
            patches[name] = value
            continue
        if name.startswith("v.blk."):
            _, _, layer, part, suffix = name.split(".")
            dest = f"blocks.{layer}.{block[part]}.{suffix}"
        else:
            prefix, suffix = name.rsplit(".", 1)
            if prefix not in root:
                raise ValueError(f"Unexpected projector tensor {name}")
            dest = root[prefix] + "." + suffix
        if dest not in expected or tuple(value.shape) != tuple(expected[dest].shape):
            raise ValueError(f"Projector tensor shape mismatch: {name}")
        weights[dest] = value.to(device=device, dtype=torch.bfloat16)
    weights["patch_embed.proj.weight"] = torch.stack(
        [patches["v.patch_embd.weight"], patches["v.patch_embd.weight.1"]], dim=2
    ).to(device=device, dtype=torch.bfloat16)
    tower.load_state_dict(weights, strict=True, assign=True)
    tower.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(
        vc.hidden_size // vc.num_heads // 2
    ).to(device)
    tower.eval()
    model._vision = tower
    m = model._bonsai_metadata
    model.model._image_token_id = m["tokenizer.ggml.tokens"].index("<|image_pad|>")
    sections = tuple(m["qwen35.rope.dimension_sections"][:3])
    for layer in model.model.layers.op_list:
        if hasattr(layer, "self_attn"):
            layer.self_attn._mrope_section = sections
