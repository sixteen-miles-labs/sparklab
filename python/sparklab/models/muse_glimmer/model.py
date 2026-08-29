from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sparklab.core import get_global_ctx
from sparklab.layers import (
    BaseOP,
    GemmaPlusOneRMSNorm,
    GemmaPlusOneRMSNormFused,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
    silu_and_mul,
)
from sparklab.models.blocks import BaseLLMModel
from sparklab.utils import nvtx_annotate

from .attention import MuseGlimmerAttention
from sparklab.models.quant_linear import make_col_merged, make_replicated

if TYPE_CHECKING:
    from sparklab.models.config import ModelConfig


class MuseGlimmerMLP(BaseOP):
    """SwiGLU MLP (gate|up fused), NVFP4 (W4A16) on the quantized checkpoint else bf16."""

    def __init__(self, config: ModelConfig):
        self.gate_up_proj = make_col_merged(
            config,
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )
        self.down_proj = make_replicated(
            config, config.intermediate_size, config.hidden_size, has_bias=False
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class MuseGlimmerDecoderLayer(BaseOP):
    """Sandwich block with centered (1+w) norms:

        h = post_attention_layernorm(attn(input_layernorm(x)));  x = x + h
        h = post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)));  x = x + h

    The pre-norms use ``rms_norm_eps``, the post-norms the tighter ``post_norm_eps``
    (1e-8). The (1+w) scaling stays a runtime fp32 add (GemmaPlusOneRMSNorm) -- baking
    +1 into the bf16 weight would round away exactly the precision the centered form
    exists to keep.
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self.self_attn = MuseGlimmerAttention(config, layer_id)
        self.mlp = MuseGlimmerMLP(config)
        H = config.hidden_size
        post_eps = config.post_norm_eps if config.post_norm_eps is not None else config.rms_norm_eps
        self.input_layernorm = GemmaPlusOneRMSNorm(H, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaPlusOneRMSNorm(H, eps=post_eps)
        self.pre_feedforward_layernorm = GemmaPlusOneRMSNormFused(H, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaPlusOneRMSNorm(H, eps=post_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h)
        h = self.post_attention_layernorm.forward(h)
        # Fused add + norm: residual becomes x + h, pre_ff its norm.
        pre_ff, residual = self.pre_feedforward_layernorm.forward(h, residual)
        h = self.mlp.forward(pre_ff)
        h = self.post_feedforward_layernorm.forward(h)
        return residual + h


class MuseGlimmerModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # NormedEmbedding: a weightless RMSNorm on top of the embeddings. Kept separate
        # from the embedding matrix (the reference cannot fold it either -- the DFlash
        # drafter embeds without the norm); no weight, so it never appears in state_dict.
        self.embed_norm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, with_scale=False
        )
        self.layers = OPList(
            [MuseGlimmerDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        # Final norm scales by the raw checkpoint weight (plain RMSNorm, not the
        # centered (1+w) form the decoder norms use).
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_norm.forward(self.embed_tokens.forward(input_ids))
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class MuseGlimmerForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = MuseGlimmerModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self._output_multiplier = config.output_multiplier
        self._final_logit_softcapping = config.final_logit_softcapping
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        # Gemma-style logit post-processing, with a pre-scale: T * tanh(logits * mult / T).
        if self._output_multiplier is not None:
            logits = logits * self._output_multiplier
        if self._final_logit_softcapping is not None:
            cap = self._final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits


__all__ = ["MuseGlimmerForCausalLM"]
