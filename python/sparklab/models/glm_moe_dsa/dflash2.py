"""Original BF16 GLM DFlash2 loading; target weights are never modified.

The separately acquired draft retains its upstream license (CC BY-NC-ND 4.0).
This adapter loads it for local evaluation and does not publish converted weights.
"""

from pathlib import Path

import safetensors
import torch

from sparklab.models.loader import drop_page_cache


def load_bf16_draft(draft, device, *, dummy=False):
    expected = draft.state_dict()
    if dummy:
        draft.load_state_dict({name: torch.zeros(tuple(t.shape), dtype=t.dtype, device=device)
                               for name, t in expected.items()})
        return
    path = Path(draft.args.draft_model_path) / "model.safetensors"
    fusions = {
        "self_attn.qkv_proj.weight": ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                      "self_attn.v_proj.weight"),
        "mlp.gate_up_proj.weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
    }
    loaded = {}
    try:
        with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
            consumed = set()

            def get(name):
                consumed.add(name)
                tensor = handle.get_tensor(name)
                if tensor.dtype not in (torch.bfloat16, torch.float32):
                    raise ValueError(f"Expected original BF16/FP32 draft tensor: {name}")
                return tensor

            for name, param in expected.items():
                parts = next((tuple(name[:-len(suffix)] + part for part in source)
                              for suffix, source in fusions.items() if name.endswith(suffix)), None)
                if parts is None:
                    tensor = get(name).to(device=device, dtype=param.dtype)
                else:
                    tensor = torch.empty(tuple(param.shape), dtype=param.dtype, device=device)
                    offset = 0
                    for part in parts:
                        value = get(part)
                        if value.ndim != 2 or value.shape[1] != param.shape[1]:
                            raise ValueError(f"Bad fused draft input shape: {part}")
                        tensor[offset:offset + value.shape[0]].copy_(value)
                        offset += value.shape[0]
                    if offset != param.shape[0]:
                        raise ValueError(f"Bad fused draft shape: {name}")
                loaded[name] = tensor
            if consumed != set(handle.keys()):
                raise ValueError(f"Unexpected DFlash2 tensors: {sorted(set(handle.keys()) - consumed)}")
            draft.load_state_dict(loaded)
    finally:
        drop_page_cache(str(path))
