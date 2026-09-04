"""Checkpoint-native NextN predictor for GLM-5.3 Flash."""

from __future__ import annotations

import os
from dataclasses import replace

import safetensors
import torch

from sparklab.core import get_global_ctx
from sparklab.layers import BaseOP, LinearReplicated, RMSNorm, RMSNormFused

from .attention import Glm5NextMLAAttention
from .moe import Glm5NextSparseMoe


class _MTPDecoderLayer(BaseOP):
    """The released layer-45 block: MLA + sparse MoE, without target mHC."""

    def __init__(self, config):
        draft = replace(
            config,
            moe_backend="fused",
            expert_quant="fp8_block",
            attn_quant="none",
            dense_quant="none",
            shared_expert_quant="none",
        )
        self.self_attn = Glm5NextMLAAttention(draft, config.num_layers)
        self.mlp = Glm5NextSparseMoe(draft, config.num_layers)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden
        hidden = self.input_layernorm.forward(hidden)
        hidden = self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward(hidden, residual)
        return self.mlp.forward(hidden), residual


class Glm5NextMultiTokenPredictor(BaseOP):
    def __init__(self, config):
        hidden = config.hidden_size
        self._layer_id = config.num_layers
        self.enorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, config.rms_norm_eps)
        self.eh_proj = LinearReplicated(2 * hidden, hidden, has_bias=False)
        self.layer = _MTPDecoderLayer(config)
        self.norm = RMSNormFused(hidden, config.rms_norm_eps)

    def forward(
        self, token_embeddings: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        batch = get_global_ctx().batch
        token_embeddings = torch.where(
            batch.positions.view(-1, 1) == 0,
            torch.zeros((), dtype=token_embeddings.dtype, device=token_embeddings.device),
            token_embeddings,
        )
        embedded = self.enorm.forward(token_embeddings)
        previous = self.hnorm.forward(target_hidden)
        hidden = self.eh_proj.forward(torch.cat((embedded, previous), dim=-1))
        hidden, residual = self.layer.forward(hidden)
        return self.norm.forward(hidden, residual)[0]

    def load_dummy(self, device: torch.device) -> None:
        loaded = {
            name: torch.zeros(tuple(param.shape), dtype=param.dtype, device=device)
            for name, param in self.state_dict().items()
        }
        self.load_state_dict(loaded)

    def load_sidecar(self, path: str, device: torch.device) -> None:
        """Load the mixed BF16/block-FP8 layer without a multi-GB host fusion."""
        expected = self.state_dict()
        loaded: dict[str, torch.Tensor] = {}

        def allocate(name: str) -> torch.Tensor:
            target = expected[name]
            return torch.empty(tuple(target.shape), dtype=target.dtype, device=device)

        gate_up = allocate("layer.mlp.experts.gate_up_proj")
        gate_up_scale = allocate("layer.mlp.experts.gate_up_scale_inv")
        down = allocate("layer.mlp.experts.down_proj")
        down_scale = allocate("layer.mlp.experts.down_scale_inv")
        loaded.update(
            {
                "layer.mlp.experts.gate_up_proj": gate_up,
                "layer.mlp.experts.gate_up_scale_inv": gate_up_scale,
                "layer.mlp.experts.down_proj": down,
                "layer.mlp.experts.down_scale_inv": down_scale,
            }
        )

        prefix = f"model.language_model.layers.{self._layer_id}."
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            for expert in range(gate_up.shape[0]):
                source = f"{prefix}mlp.experts.{expert}"
                width = gate_up.shape[1] // 2
                scale_width = gate_up_scale.shape[1] // 2
                for role, rows, scale_rows in (
                    ("gate_proj", slice(0, width), slice(0, scale_width)),
                    (
                        "up_proj",
                        slice(width, 2 * width),
                        slice(scale_width, 2 * scale_width),
                    ),
                ):
                    base = f"{source}.{role}"
                    gate_up[expert, rows].copy_(handle.get_tensor(base + ".weight"))
                    gate_up_scale[expert, scale_rows].copy_(
                        handle.get_tensor(base + ".weight_scale")
                    )
                base = f"{source}.down_proj"
                down[expert].copy_(handle.get_tensor(base + ".weight"))
                down_scale[expert].copy_(handle.get_tensor(base + ".weight_scale"))

            for raw_name in handle.keys():
                if not raw_name.startswith(prefix) or ".mlp.experts." in raw_name:
                    continue
                suffix = raw_name.removeprefix(prefix)
                if suffix == "shared_head.norm.weight":
                    name = "norm.weight"
                elif suffix.startswith(("enorm.", "hnorm.", "eh_proj.")):
                    name = suffix
                else:
                    name = "layer." + suffix
                target = expected.get(name)
                if target is None:
                    raise RuntimeError(f"unexpected GLM-5.3 MTP tensor {raw_name!r}")
                loaded[name] = handle.get_tensor(raw_name).to(
                    device=device, dtype=target.dtype
                )

        missing = set(expected) - set(loaded)
        if missing:
            raise RuntimeError(f"GLM-5.3 MTP sidecar is missing tensors: {sorted(missing)}")
        self.load_state_dict(loaded)
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


__all__ = ["Glm5NextMultiTokenPredictor"]
