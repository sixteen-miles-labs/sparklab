"""Bounded OpenAI image parts -> Qwen patches, placeholders and MRoPE."""
import base64
from io import BytesIO
import json
from pathlib import Path

import torch


def has_images(messages):
    return isinstance(messages, list) and any(
        isinstance(m.get('content'), list) and any(p.get('type') == 'image_url' for p in m['content'])
        for m in messages
    )


class QwenImageProcessor:
    def __init__(self, source, tokenizer):
        from transformers import AutoImageProcessor
        self.config = json.loads((Path(source) / 'config.json').read_text())
        self.tokenizer = tokenizer
        self.processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
        self.processor.size = {'shortest_edge': 65536, 'longest_edge': 1048576}

    def prepare(self, messages, tools, kwargs):
        from PIL import Image
        images, rendered = [], []
        for message in messages:
            m = dict(message)
            if isinstance(m.get('content'), list):
                parts = []
                for part in m['content']:
                    if part.get('type') == 'text':
                        parts.append(part.get('text', ''))
                    elif part.get('type') == 'image_url':
                        if len(images) >= 4:
                            raise ValueError('At most four images are supported per request')
                        value = part.get('image_url')
                        url = value.get('url') if isinstance(value, dict) else value
                        if not isinstance(url, str) or not url.startswith('data:image/') or ';base64,' not in url:
                            raise ValueError('Images must be base64 data:image URLs')
                        if len(url) > 16 * 1024 * 1024:
                            raise ValueError('Encoded image exceeds 16 MiB')
                        raw = base64.b64decode(url.split(';base64,', 1)[1], validate=True)
                        with Image.open(BytesIO(raw)) as image:
                            if image.width * image.height > 16_000_000:
                                raise ValueError('Image exceeds 16 megapixels')
                            images.append(image.convert('RGB'))
                        parts.append('<|vision_start|><|image_pad|><|vision_end|>')
                    else:
                        raise ValueError('Only text and image_url content parts are supported')
                m['content'] = ''.join(parts)
            rendered.append(m)
        processed = self.processor(images=images, return_tensors='pt')
        grids = processed['image_grid_thw'].to(torch.int64)
        prompt = self.tokenizer.apply_chat_template(rendered, tools=tools, tokenize=False,
                                                   add_generation_prompt=True, **kwargs)
        marker = '<|image_pad|>'
        if prompt.count(marker) != len(images):
            raise ValueError('Image placeholders do not match uploaded images')
        chunks = prompt.split(marker)
        merge = self.config['vision_config']['spatial_merge_size']
        prompt = chunks[0]
        for grid, suffix in zip(grids, chunks[1:], strict=True):
            prompt += marker * (int(grid.prod()) // merge**2) + suffix
        ids = torch.tensor(self.tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.int32)
        from sparklab.models.qwen3_5_moe.vision import image_positions
        positions, delta = image_positions(ids, grids, self.config['image_token_id'], merge)
        pixels = processed['pixel_values'].float()
        payload = {'pixels': pixels.flatten(), 'pixel_shape': list(pixels.shape),
                   'grids': grids.flatten(), 'positions': positions.flatten(), 'delta': delta}
        return prompt, ids, payload
