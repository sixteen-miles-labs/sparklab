from __future__ import annotations

import torch
from sparklab.core import get_global_ctx
from sparklab.kernels.triton.fp8_pertensor_linear import fp8_pertensor_linear
from sparklab.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from sparklab.models.blocks import BaseLLMModel
from sparklab.runtime.distributed import get_tp_info
from sparklab.utils import nvtx_annotate

from .attention import KimiMLAAttention
from .kda import KimiDeltaAttention
from .moe import KimiSparseMoeBlock
from .ops import KimiSituMLP, apply_attention_residual


class KimiFp8Embedding(VocabParallelEmbedding):
    """Per-row FP8 token embedding for the opt-in GB10 resident profile."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        if get_tp_info().size != 1:
            raise NotImplementedError("Kimi K3 FP8 embedding currently requires TP=1")
        super().__init__(num_embeddings, embedding_dim)
        self.weight = torch.empty(
            self.num_embeddings_tp, embedding_dim, dtype=torch.float8_e4m3fn
        )
        self.weight_scale = torch.empty(self.num_embeddings_tp, dtype=torch.float32)

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from sparklab.kernels import indexing

        values = indexing(weights=self.weight, indices=x).to(torch.bfloat16)
        scales = self.weight_scale[x.long()].to(values.dtype).unsqueeze(-1)
        return values * scales


class KimiFp8LMHead(ParallelLMHead):
    """Per-row FP8 W8A16 head for K3's untied vocabulary projection."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        if get_tp_info().size != 1:
            raise NotImplementedError("Kimi K3 FP8 lm_head currently requires TP=1")
        super().__init__(num_embeddings, embedding_dim, tie_word_embeddings=False)
        self.weight = torch.empty(num_embeddings, embedding_dim, dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty(num_embeddings, dtype=torch.float32)

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fp8_pertensor_linear(x, self.weight, self.weight_scale)


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
                quantization=config.dense_quant,
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
        self.embed_tokens = (
            KimiFp8Embedding(config.vocab_size, config.hidden_size)
            if config.lm_head_quant == "fp8_pertensor"
            else VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        )
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
        if config.lm_head_quant == "fp8_pertensor" and config.tie_word_embeddings:
            raise NotImplementedError(
                "Kimi K3's FP8 resident profile requires untied embeddings"
            )
        self.model = KimiK3Model(config)
        self.lm_head = (
            KimiFp8LMHead(config.vocab_size, config.hidden_size)
            if config.lm_head_quant == "fp8_pertensor"
            else ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
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


__all__ = ["KimiFp8Embedding", "KimiFp8LMHead", "KimiK3ForCausalLM"]
