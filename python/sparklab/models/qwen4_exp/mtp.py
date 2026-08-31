"""Native Qwen4-Exp multi-token predictor.

The publisher stores the one-layer draft head in a standalone safetensors file.
Its routed experts are small enough to remain resident as native ModelOpt NVFP4
on GB10, independently of the target model's immutable 48-layer expert cache.
"""

from __future__ import annotations

import os

import safetensors
import torch

from sparklab.core import get_global_ctx
from sparklab.layers import (
    BaseOP,
    GemmaPlusOneRMSNorm,
    LinearReplicated,
    OPList,
)
from sparklab.moe.fused import fused_topk
from sparklab.models.qwen3_5_moe.moe import _SharedExpert

from .attention import Qwen4ExpAttention
from .hyper import Qwen4GatedResidual


class _ResidentNvfp4MTPMoE(BaseOP):
    def __init__(self, config):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        # Hidden from BaseOP.state_dict: these are materialized directly from the
        # standalone sidecar after the target state dict has loaded.
        self._expert_banks: tuple[torch.Tensor, ...] | None = None

    def load_experts(self, handle, device: torch.device) -> None:
        e, h, i = self.num_experts, self.hidden_size, self.intermediate_size
        fp8 = torch.float8_e4m3fn
        gate_up_packed = torch.empty(e, 2 * i, h // 2, dtype=torch.uint8, device=device)
        gate_up_scale = torch.empty(e, 2 * i, h // 16, dtype=fp8, device=device)
        gate_up_global = torch.empty(e, 2 * i, dtype=torch.float16, device=device)
        down_packed = torch.empty(e, h, i // 2, dtype=torch.uint8, device=device)
        down_scale = torch.empty(e, h, i // 16, dtype=fp8, device=device)
        down_global = torch.empty(e, h, dtype=torch.float16, device=device)

        for expert in range(e):
            prefix = f"mtp.layers.0.mlp.experts.{expert}"
            for role, row in (("gate_proj", slice(0, i)), ("up_proj", slice(i, 2 * i))):
                base = f"{prefix}.{role}"
                gate_up_packed[expert, row].copy_(handle.get_tensor(base + ".weight"))
                gate_up_scale[expert, row].copy_(handle.get_tensor(base + ".weight_scale"))
                gate_up_global[expert, row].fill_(
                    float(handle.get_tensor(base + ".weight_scale_2").item())
                )
            base = f"{prefix}.down_proj"
            down_packed[expert].copy_(handle.get_tensor(base + ".weight"))
            down_scale[expert].copy_(handle.get_tensor(base + ".weight_scale"))
            down_global[expert].fill_(
                float(handle.get_tensor(base + ".weight_scale_2").item())
            )
        self._expert_banks = (
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            down_packed,
            down_scale,
            down_global,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._expert_banks is None:
            raise RuntimeError("Qwen4 MTP expert banks were not loaded")
        router_logits = self.gate.forward(hidden_states)
        # Evaluate the resident shared branch before the routed kernel, whose
        # fused implementation may reuse its input storage.
        shared_input = hidden_states.clone()
        shared = self.shared_expert.forward(shared_input)
        shared.mul_(torch.sigmoid(self.shared_expert_gate.forward(shared_input)))
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=True,
        )
        if get_global_ctx().batch.uses_prefill_kernels:
            from sparklab.moe.fused_nvfp4 import fused_experts_nvfp4

            routed = fused_experts_nvfp4(
                hidden_states,
                *self._expert_banks,
                topk_weights,
                topk_ids,
                self.num_experts,
            )
        else:
            from sparklab.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

            routed = fused_experts_decode_nvfp4_marlin(
                hidden_states, *self._expert_banks, topk_weights, topk_ids
            )
        return routed + shared


class _MTPDecoderLayer(BaseOP):
    def __init__(self, config):
        args = config.qwen4_exp_args
        self.self_attn = Qwen4ExpAttention(config, config.num_layers)
        self.mlp = _ResidentNvfp4MTPMoE(config)
        self.attn_hyper_connection = Qwen4GatedResidual(
            config.hidden_size, args.hc_count, args.hc_lowrank, config.rms_norm_eps
        )
        self.mlp_hyper_connection = Qwen4GatedResidual(
            config.hidden_size, args.hc_count, args.hc_lowrank, config.rms_norm_eps
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        branch, residual, inject = self.attn_hyper_connection.forward(hidden)
        branch = self.self_attn.forward(branch)
        hidden = self.attn_hyper_connection.inject(branch, residual, inject)
        branch, residual, inject = self.mlp_hyper_connection.forward(hidden)
        branch = self.mlp.forward(branch)
        return self.mlp_hyper_connection.inject(branch, residual, inject)


class Qwen4ExpMultiTokenPredictor(BaseOP):
    def __init__(self, config):
        args = config.qwen4_exp_args
        h, hc = config.hidden_size, args.hc_count
        self.fc_embedding = LinearReplicated(h, h, has_bias=False)
        self.fc_hidden = LinearReplicated(h, h, has_bias=False)
        self.layers = OPList([_MTPDecoderLayer(config)])
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(h, config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(h * hc, config.rms_norm_eps)
        self.hyper_connection_mixer = Qwen4GatedResidual(
            h, hc, args.hc_lowrank, config.rms_norm_eps, use_combine=False
        )
        self.hidden_size = h
        self.hc_count = hc

    def forward(
        self, token_embeddings: torch.Tensor, target_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, h, hc = target_hidden.shape[0], self.hidden_size, self.hc_count
        embed = self.fc_embedding.forward(
            self.pre_fc_norm_embedding.forward(token_embeddings)
        )
        hidden = self.pre_fc_norm_hidden.forward(target_hidden)
        hidden = self.fc_hidden.forward(hidden.view(n, hc, h))
        hidden = (hidden + embed.unsqueeze(1)).flatten(-2)
        multi_hidden = self.layers.op_list[0].forward(hidden)
        return self.hyper_connection_mixer.forward(multi_hidden), multi_hidden

    def load_sidecar(self, path: str, device: torch.device) -> None:
        expected = self.state_dict()
        pending: dict[str, dict[str, torch.Tensor]] = {}
        loaded: dict[str, torch.Tensor] = {}

        def emit(name: str, tensor: torch.Tensor) -> None:
            target = expected.get(name)
            if target is None:
                raise RuntimeError(f"unexpected Qwen4 MTP tensor {name!r}")
            loaded[name] = tensor.to(device=device, dtype=target.dtype)

        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            for raw in handle.keys():
                if ".experts." in raw or raw.endswith(".input_scale"):
                    continue
                name = raw.removeprefix("mtp.")
                name = name.replace(".self_attn.indexer.index_qk_proj.", ".self_attn.index_qk_proj.")
                name = name.replace(".self_attn.indexer.q_layernorm.", ".self_attn.index_q_norm.")
                name = name.replace(".self_attn.indexer.k_layernorm.", ".self_attn.index_k_norm.")
                fusion = None
                for fused, parts in {
                    ".self_attn.qkv_proj.weight": (
                        ".self_attn.q_proj.weight",
                        ".self_attn.k_proj.weight",
                        ".self_attn.v_proj.weight",
                    ),
                    ".mlp.shared_expert.gate_up_proj.weight": (
                        ".mlp.shared_expert.gate_proj.weight",
                        ".mlp.shared_expert.up_proj.weight",
                    ),
                }.items():
                    for index, suffix in enumerate(parts):
                        if name.endswith(suffix):
                            out = name[: -len(suffix)] + fused
                            slots = pending.setdefault(out, {})
                            slots[str(index)] = handle.get_tensor(raw)
                            if len(slots) == len(parts):
                                emit(out, torch.cat([slots[str(i)] for i in range(len(parts))]))
                                del pending[out]
                            fusion = True
                            break
                    if fusion:
                        break
                if fusion:
                    continue
                emit(name, handle.get_tensor(raw))
            if pending:
                raise RuntimeError(f"incomplete Qwen4 MTP projection groups: {list(pending)}")
            missing = set(expected) - set(loaded)
            if missing:
                raise RuntimeError(f"Qwen4 MTP sidecar is missing tensors: {sorted(missing)}")
            self.load_state_dict(loaded)
            self.layers.op_list[0].mlp.load_experts(handle, device)
        try:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


__all__ = ["Qwen4ExpMultiTokenPredictor"]
