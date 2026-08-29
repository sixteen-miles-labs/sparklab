from __future__ import annotations

import torch
from sparklab.core import get_global_ctx
from sparklab.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from sparklab.models.blocks import BaseLLMModel
from sparklab.utils import nvtx_annotate

from .attention import KimiMLAAttention
from .kda import KimiDeltaAttention
from .moe import KimiSparseMoeBlock
from .ops import KimiSituMLP, apply_attention_residual


class KimiK3DecoderLayer(BaseOP):
    def __init__(self, config, layer_id: int):
        args = config.kimi_k3_args
        assert args is not None
        self._layer_id = layer_id
        self.self_attn = (
            KimiDeltaAttention(config, layer_id)
            if config.is_linear_layer(layer_id)
            else KimiMLAAttention(config, layer_id)
        )
        if layer_id >= config.first_k_dense_replace:
            self.block_sparse_moe = KimiSparseMoeBlock(config, layer_id)
            self.mlp = None
        else:
            self.block_sparse_moe = None
            self.mlp = KimiSituMLP(
                config.hidden_size,
                config.intermediate_size,
                args.situ_beta,
                args.situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self.mlp_res_proj = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self.block_size = args.attn_res_block_size
        self.eps = config.rms_norm_eps

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, hidden_states: torch.Tensor, block_residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = apply_attention_residual(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj.weight,
                self.self_attention_res_norm.weight,
                self.eps,
            )
        if self._layer_id % self.block_size == 0:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
            prefix_sum = None

        hidden_states = self.self_attn.forward(self.input_layernorm.forward(hidden_states))
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        hidden_states = apply_attention_residual(
            prefix_sum,
            block_residual,
            self.mlp_res_proj.weight,
            self.mlp_res_norm.weight,
            self.eps,
        )
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        if self.block_sparse_moe is not None:
            hidden_states = self.block_sparse_moe.forward(hidden_states)
        else:
            hidden_states = self.mlp.forward(hidden_states)
        return prefix_sum + hidden_states, block_residual


class KimiK3Model(BaseOP):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [KimiK3DecoderLayer(config, layer) for layer in range(config.num_layers)]
        )
        self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_proj = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eps = config.rms_norm_eps

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens.forward(input_ids)
        blocks = hidden.new_zeros(hidden.shape[0], 0, hidden.shape[-1])
        for layer in self.layers.op_list:
            hidden, blocks = layer.forward(hidden, blocks)
        hidden = apply_attention_residual(
            hidden,
            blocks,
            self.output_attn_res_proj.weight,
            self.output_attn_res_norm.weight,
            self.eps,
        )
        return self.norm.forward(hidden)


class KimiK3ForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self.model = KimiK3Model(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        for layer in self.model.layers.op_list:
            if isinstance(layer.self_attn, KimiMLAAttention):
                layer.self_attn.prepare_for_runtime()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["KimiK3ForCausalLM"]
