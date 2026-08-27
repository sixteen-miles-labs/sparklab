from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hyper import Qwen4GatedResidual
from .ple import Qwen4PLE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.qwen4_exp_args
        self._layer_id = layer_id
        self._linear = config.is_linear_layer(layer_id)
        if self._linear:
            group = config.linear_attention_group()
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=group.num_key_heads,
                num_v_heads=group.num_value_heads,
                head_k_dim=group.key_head_dim,
                head_v_dim=group.value_head_dim,
                conv_kernel_size=group.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                output_gate_activation=args.output_gate_activation,
            )
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen3_5MoE(config, layer_id)
        self.ple = (
            Qwen4PLE(config, layer_id, args.ple_layer_ids.index(layer_id))
            if layer_id in args.ple_layer_ids else None
        )
        self.attn_hyper_connection = Qwen4GatedResidual(
            config.hidden_size, args.hc_count, args.hc_lowrank, config.rms_norm_eps
        )
        self.mlp_hyper_connection = Qwen4GatedResidual(
            config.hidden_size, args.hc_count, args.hc_lowrank, config.rms_norm_eps
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden)
        branch, residual, inject = self.attn_hyper_connection.forward(hidden)
        branch = (
            self.linear_attn.forward(branch) if self._linear
            else self.self_attn.forward(branch)
        )
        hidden = Qwen4GatedResidual.inject(branch, residual, inject)
        branch, residual, inject = self.mlp_hyper_connection.forward(hidden)
        branch = self.mlp.forward(branch)
        return Qwen4GatedResidual.inject(branch, residual, inject)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig):
        args = config.qwen4_exp_args
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([
            Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)
        ])
        self.hyper_connection_mixer = Qwen4GatedResidual(
            config.hidden_size, args.hc_count, args.hc_lowrank,
            config.rms_norm_eps, use_combine=False,
        )
        self._hc_count = args.hc_count

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self._hc_count)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden)
        return self.hyper_connection_mixer.forward(hidden)


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=(self.model.embed_tokens if config.tie_word_embeddings else None),
        )
        super().__init__()

    def prepare_for_weight_load(self, model_path: str, *, dummy: bool = False) -> None:
        for layer in self.model.layers.op_list:
            if layer.ple is not None:
                layer.ple.bind(model_path, dummy=dummy)

    def forward(self) -> torch.Tensor:
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(hidden)


__all__ = ["Qwen4ExpForCausalLM"]
