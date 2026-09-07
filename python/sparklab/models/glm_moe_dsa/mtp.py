"""Opt-in checkpoint-native NextN for full GLM-5.3 (not GLM-5.3 Flash).

The original BF16 draft is loaded separately from the immutable NVFP4 target.
Experts remain resident; only the target participates in disk-cache admission.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import safetensors
import torch

from sparklab.core import Batch, Req, get_global_ctx
from sparklab.layers import BaseOP, LinearReplicated, RMSNorm, RMSNormFused
from sparklab.models.loader import drop_page_cache


class GlmDsaMultiTokenPredictor(BaseOP):
    def __init__(self, config):
        from .model import GlmMoeDsaDecoderLayer

        self._layer_id = config.num_layers
        draft = replace(config, moe_backend="fused", attn_quant="none", dense_quant="none")
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = LinearReplicated(2 * config.hidden_size, config.hidden_size, has_bias=False)
        self.layer = GlmMoeDsaDecoderLayer(draft, self._layer_id)
        self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, embeddings, previous_hidden):
        positions = get_global_ctx().batch.positions
        embeddings = torch.where(positions[:, None] == 0, 0, embeddings)
        hidden = self.eh_proj.forward(torch.cat((
            self.enorm.forward(embeddings), self.hnorm.forward(previous_hidden),
        ), dim=-1))
        hidden, residual = self.layer.forward(hidden)
        # One final norm for both the shared head and feedback to the next step.
        return self.norm.forward(hidden, residual)[0]

    def load_shards(self, directory: str, device: torch.device, *, dummy=False):
        expected = self.state_dict()
        if dummy:
            self.load_state_dict({name: torch.zeros(tuple(t.shape), dtype=t.dtype, device=device)
                                  for name, t in expected.items()})
            return
        folder = Path(directory).resolve()
        index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = f"model.layers.{self._layer_id}."
        selected = {name: filename for name, filename in index.items() if name.startswith(prefix)}
        if not selected:
            raise ValueError(f"No native MTP layer {self._layer_id} in {folder}")
        paths = {}
        for filename in set(selected.values()):
            path = (folder / filename).resolve()
            if not path.is_relative_to(folder) or not path.is_file():
                raise ValueError(f"Missing or unsafe MTP shard: {filename}")
            paths[filename] = path

        loaded = {}
        for name in ("layer.mlp.experts.gate_up_proj", "layer.mlp.experts.down_proj"):
            param = expected[name]
            loaded[name] = torch.empty(tuple(param.shape), dtype=param.dtype, device=device)
        try:
            with ExitStack() as stack:
                handles = {name: stack.enter_context(safetensors.safe_open(
                    str(path), framework="pt", device="cpu")) for name, path in paths.items()}
                consumed = set()

                def get(name):
                    consumed.add(name)
                    return handles[selected[name]].get_tensor(name)

                def copy_expert(destination, name):
                    source = get(name)
                    if source.shape != destination.shape or source.dtype != torch.bfloat16:
                        raise ValueError(f"Invalid native BF16 MTP expert tensor: {name}")
                    destination.copy_(source)

                gate_up = loaded["layer.mlp.experts.gate_up_proj"]
                down = loaded["layer.mlp.experts.down_proj"]
                width = gate_up.shape[1] // 2
                for expert in range(gate_up.shape[0]):
                    base = f"{prefix}mlp.experts.{expert}."
                    copy_expert(gate_up[expert, :width], base + "gate_proj.weight")
                    copy_expert(gate_up[expert, width:], base + "up_proj.weight")
                    copy_expert(down[expert], base + "down_proj.weight")
                for raw in selected:
                    if raw in consumed:
                        continue
                    suffix = raw.removeprefix(prefix)
                    if suffix == "shared_head.norm.weight":
                        name = "norm.weight"
                    elif suffix.startswith(("enorm.", "hnorm.", "eh_proj.")):
                        name = suffix
                    else:
                        name = "layer." + suffix.replace(
                            "mlp.gate.e_score_correction_bias", "mlp.e_score_correction_bias")
                    if name not in expected:
                        raise ValueError(f"Unexpected MTP tensor {raw}")
                    loaded[name] = get(raw).to(device=device, dtype=expected[name].dtype)
                if set(loaded) != set(expected):
                    raise ValueError(f"MTP weights missing: {sorted(set(expected) - set(loaded))}")
                self.load_state_dict(loaded)
        finally:
            for path in paths.values():
                drop_page_cache(str(path))


def propose_nextn(model, batch, next_token, *, prefix_tokens=None):
    """Rebuild only the retained draft prefix, then recurse from its last row.

    Target KV is append-only. A rejected suffix is hidden by fresh prefix-length
    metadata; draft KV for the retained prefix is refreshed with the correction.
    No target forward or target-cache mutation is needed for this operation.
    """
    if model._mtp is None or model._mtp_target_hidden is None or batch.size != 1:
        return None
    req = batch.reqs[0]
    if not req.sampling_params.is_greedy:
        return None
    n = batch.input_ids.numel() if prefix_tokens is None else prefix_tokens
    if not 0 < n <= batch.input_ids.numel():
        raise ValueError("MTP prefix must be a nonempty subset of the verified query")
    start = batch.verify_cached_lens[0] if batch.is_verify else req.cached_len
    end = start + n
    steps = min(model._mtp_steps, max(0, req.max_device_len - end - 1))
    if not steps:
        return None
    ctx = get_global_ctx()
    query = batch.input_ids[:n]
    shifted = torch.cat((query[1:], next_token.reshape(1)))

    def draft_batch(input_ids, position, cached, length, phase):
        draft_req = Req(
            input_ids=torch.zeros(length, dtype=req.input_ids.dtype),
            table_idx=req.table_idx, cached_len=cached, output_len=1,
            uid=req.uid, sampling_params=req.sampling_params, cache_handle=req.cache_handle,
        )
        out = Batch(reqs=[draft_req], phase=phase)
        out.padded_reqs = out.reqs
        out.input_ids = input_ids
        out.positions = position
        out.out_loc = ctx.page_table[req.table_idx, cached:length]
        out.active_table_idx = torch.tensor([req.table_idx], dtype=torch.int64, device=input_ids.device)
        ctx.attn_backend.prepare_metadata(out)
        return out

    first = draft_batch(shifted, batch.positions[:n], start, end, "prefill")
    with ctx.forward_batch(first):
        feedback = model._mtp.forward(model.model.embed_tokens.forward(shifted),
                                      model._mtp_target_hidden[:n])[-1:]
        draft = model.project_draft_logits(feedback).argmax(-1)
    drafts = [draft.reshape(()).int()]
    for step in range(1, steps):
        position = end + step - 1
        current = draft_batch(draft.reshape(1), torch.tensor(
            [position], dtype=torch.int32, device=draft.device), position, position + 1, "decode")
        with ctx.forward_batch(current):
            feedback = model._mtp.forward(model.model.embed_tokens.forward(current.input_ids), feedback)
            draft = model.project_draft_logits(feedback).argmax(-1)
        drafts.append(draft.reshape(()).int())
    return torch.stack(drafts)
