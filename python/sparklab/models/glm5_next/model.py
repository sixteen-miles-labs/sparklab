"""Text-only GLM-5.3-Flash decoder with four-stream mHC residuals."""

from __future__ import annotations

import torch
from sparklab.core import get_global_ctx
from sparklab.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from sparklab.models.blocks import BaseLLMModel
from sparklab.utils import nvtx_annotate

from .attention import Glm5NextMLAAttention
from .hyper import Glm5NextHyperConnection
from .kda import Glm5NextDeltaAttention
from .mlp import Glm5NextMLP
from .moe import Glm5NextSparseMoe


class Glm5NextDecoderLayer(BaseOP):
    def __init__(self, config, layer_id: int):
        args = config.glm5_next_args
        assert args is not None
        self._layer_id = layer_id
        self.self_attn = (
            Glm5NextDeltaAttention(config, layer_id)
            if config.is_linear_layer(layer_id)
            else Glm5NextMLAAttention(config, layer_id)
        )
        self.mlp = (
            Glm5NextSparseMoe(config, layer_id)
            if layer_id >= config.first_k_dense_replace
            else Glm5NextMLP(
                config.hidden_size,
                config.intermediate_size,
                config.swiglu_limit,
                quantization=config.dense_quant,
            )
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        hc_args = (
            config.hidden_size,
            args.hc_mult,
            args.hc_eps,
            args.hc_sinkhorn_iters,
            config.rms_norm_eps,
        )
        self.attn_hc = Glm5NextHyperConnection(*hc_args)
        self.ffn_hc = Glm5NextHyperConnection(*hc_args)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        residual = streams
        post, comb, hidden = self.attn_hc.forward(streams)
        hidden = self.self_attn.forward(self.input_layernorm.forward(hidden))
        streams = self.attn_hc.expand(hidden, residual, post, comb)

        residual = streams
        post, comb, hidden = self.ffn_hc.forward(streams)
        hidden = self.mlp.forward(self.post_attention_layernorm.forward(hidden))
        return self.ffn_hc.expand(hidden, residual, post, comb)


class Glm5NextModel(BaseOP):
    def __init__(self, config):
        args = config.glm5_next_args
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Glm5NextDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_mult = args.hc_mult

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens.forward(input_ids)
        streams = hidden.unsqueeze(1).expand(-1, self.hc_mult, -1)
        for layer in self.layers.op_list:
            streams = layer.forward(streams)
        return self.norm.forward(streams.mean(dim=1))


class Glm5NextForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self.model = Glm5NextModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        for layer in self.model.layers.op_list:
            layer.attn_hc.prepare_for_runtime()
            layer.ffn_hc.prepare_for_runtime()
            if isinstance(layer.self_attn, Glm5NextMLAAttention):
                layer.self_attn.prepare_for_runtime()
            elif isinstance(layer.self_attn, Glm5NextDeltaAttention):
                layer.self_attn.prepare_for_runtime()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Glm5NextForCausalLM"]
