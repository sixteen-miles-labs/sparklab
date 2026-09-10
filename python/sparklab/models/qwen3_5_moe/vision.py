"""Native Qwen image tower and three-axis rotary positions.

The image tower uses the matching installed Transformers reference module;
text generation remains in SparkLab's native scheduler and decoder.
"""
from pathlib import Path
import json

import torch
from safetensors import safe_open


def load_vision(model, source, device):
    source = Path(source)
    config = json.loads((source / 'config.json').read_text())
    flash_next = config.get('model_type') == 'qwen4_exp'
    if flash_next:
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig as VisionConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpVisionModel as VisionModel,
            Qwen4ExpVisionRotaryEmbedding as VisionRotary,
        )
    else:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig as VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5VisionModel as VisionModel,
            Qwen3_5VisionRotaryEmbedding as VisionRotary,
        )
    vc = VisionConfig(**config['vision_config'])
    if vc.deepstack_visual_indexes:
        raise ValueError('Qwen vision deepstack is not supported')
    if vc.out_hidden_size != model.model.embed_tokens.weight.shape[1]:
        raise ValueError('Vision output width does not match the text checkpoint')
    vc._attn_implementation = 'sdpa'
    with torch.device('meta'):
        tower = VisionModel(vc)
    expected = tower.state_dict()
    weights = {}
    for path in sorted(source.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as f:
            keys = set(f.keys())
            for raw in keys:
                prefix = next((p for p in ('model.visual.', 'visual.') if raw.startswith(p)), None)
                if prefix is None:
                    continue
                name = raw[len(prefix):]
                if name not in expected:
                    if name.endswith(('weight_scale', 'weight_scale_2', 'input_scale')):
                        continue
                    raise ValueError(f'Unexpected vision tensor: {raw}')
                from .weight import _load_maybe_quantized
                tensor = _load_maybe_quantized(f, raw, keys)
                if tensor.shape != expected[name].shape:
                    raise ValueError(f'Vision tensor shape mismatch: {raw}')
                weights[name] = tensor.to(device=device, dtype=torch.bfloat16)
    tower.load_state_dict(weights, strict=True, assign=True)
    tower.eval()
    # Nonpersistent rotary buffers are not checkpoint tensors.
    tower.rotary_pos_emb = VisionRotary(vc.hidden_size // vc.num_heads // 2).to(device)
    model._vision = tower
    model.model._image_token_id = config['image_token_id']
    text = config.get('text_config', config)
    rope = text.get('rope_parameters', text.get('rope_scaling', {}))
    if rope.get('rope_type', 'default') != 'default' or not rope.get('mrope_interleaved', False):
        raise ValueError('Qwen vision requires default interleaved MRoPE')
    for layer in model.model.layers.op_list:
        attn = getattr(layer, 'self_attn', None)
        if attn is not None:
            attn._mrope_section = tuple(rope['mrope_section'])
            if flash_next:
                rotary_dim = int(text['head_dim'] * rope['partial_rotary_factor'])
                attn.rotary = MultimodalRotary(
                    attn.rotary, attn.head_dim, rotary_dim, rope['rope_theta'], attn._mrope_section,
                )
                attn.index_rotary = MultimodalRotary(
                    attn.index_rotary, attn.index_dim, rotary_dim, rope['rope_theta'], attn._mrope_section,
                )


def image_positions(input_ids, grid_thw, image_token_id, merge):
    """Image-only MRoPE: spatial grid at each placeholder run; text advances all axes."""
    ids = input_ids.tolist()
    grids = iter(grid_thw.tolist())
    positions = torch.empty(3, len(ids), dtype=torch.int64)
    offset = cursor = 0
    while cursor < len(ids):
        if ids[cursor] != image_token_id:
            positions[:, cursor] = offset
            cursor += 1
            offset += 1
            continue
        try:
            t, h, w = next(grids)
        except StopIteration as exc:
            raise ValueError('Image placeholders exceed image grids') from exc
        if t != 1 or h % merge or w % merge:
            raise ValueError('Only still images with aligned spatial grids are supported')
        h, w = h // merge, w // merge
        length = h * w
        if ids[cursor:cursor + length] != [image_token_id] * length:
            raise ValueError('Image grid and placeholder count differ')
        positions[0, cursor:cursor + length] = offset
        positions[1, cursor:cursor + length] = torch.arange(h).repeat_interleave(w) + offset
        positions[2, cursor:cursor + length] = torch.arange(w).repeat(h) + offset
        offset += max(h, w)
        cursor += length
    if next(grids, None) is not None:
        raise ValueError('Unused image grid')
    return positions, offset - len(ids)


def apply_mrope(q, k, positions, head_dim, rotary_dim, base, sections):
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=q.device).float() / rotary_dim))
    freqs = positions.float().unsqueeze(-1) * inv
    mixed = freqs[0].clone()
    for axis in (1, 2):
        mixed[..., axis:sections[axis] * 3:3] = freqs[axis, ..., axis:sections[axis] * 3:3]
    angles = torch.cat((mixed, mixed), -1)
    cos, sin = angles.cos().to(q.dtype).unsqueeze(1), angles.sin().to(q.dtype).unsqueeze(1)
    def rotate(x):
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1] // head_dim, head_dim)
        part = x[..., :rotary_dim]
        half = rotary_dim // 2
        rotated = part * cos + torch.cat((-part[..., half:], part[..., :half]), -1) * sin
        return torch.cat((rotated, x[..., rotary_dim:]), -1).reshape(original_shape)
    return rotate(q), rotate(k)


class MultimodalRotary:
    """Preserve ordinary text RoPE while accepting three-axis image positions."""
    def __init__(self, text_rotary, head_dim, rotary_dim, base, sections):
        self.text_rotary = text_rotary
        self.head_dim, self.rotary_dim = head_dim, rotary_dim
        self.base, self.sections = base, sections

    def forward(self, positions, q, k):
        if positions.ndim == 1:
            return self.text_rotary.forward(positions, q, k)
        return apply_mrope(q, k, positions, self.head_dim, self.rotary_dim, self.base, self.sections)


def request_positions(req, logical):
    """Positions for arbitrary QSA group starts, including prompt/decode boundaries."""
    prompt = req.mm_positions
    if prompt is None:
        return logical
    within_prompt = logical < prompt.shape[1]
    prompt_rows = prompt.index_select(1, logical.clamp(max=prompt.shape[1] - 1).long())
    continuation = (logical + req.mm_delta).expand(3, -1)
    return torch.where(within_prompt.unsqueeze(0), prompt_rows, continuation)
