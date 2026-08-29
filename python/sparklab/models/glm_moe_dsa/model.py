from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from sparklab.core import get_global_ctx
from sparklab.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from sparklab.models.blocks import BaseLLMModel
from sparklab.utils import nvtx_annotate

from .attention import GlmMoeDsaAttention
from .mlp import GlmDsaGatedMLP
from .moe import GlmMoeDsaSparseBlock

if TYPE_CHECKING:
    from sparklab.models.config import ModelConfig


class GlmFp8LMHead(ParallelLMHead):
    """W8A16 lm_head (fp8-e4m3 weight + per-row scale, quantized at load).

    The full-vocab logits GEMV reads the whole ~1.9 GiB bf16 head every decode step;
    fp8 halves that. Selected by ``ModelConfig.lm_head_quant == "fp8_pertensor"``
    (weight.py quantizes at load off the same field). GLM-5.2 does not tie embeddings,
    so the fp8 weight is head-only.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim, tie_word_embeddings=False)
        self.weight = torch.empty(num_embeddings, embedding_dim, dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty(num_embeddings, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from sparklab.kernels.triton.fp8_pertensor_linear import fp8_pertensor_linear

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fp8_pertensor_linear(x, self.weight, self.weight_scale)


class GlmMoeDsaDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = GlmMoeDsaAttention(config, layer_id)
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = GlmMoeDsaSparseBlock(config, layer_id)
        else:
            self.mlp = GlmDsaGatedMLP(
                config.hidden_size, config.intermediate_size, quant=config.dense_quant
            )
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class GlmMoeDsaModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [GlmMoeDsaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class GlmMoeDsaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = GlmMoeDsaModel(config)
        if config.lm_head_quant == "fp8_pertensor" and not config.tie_word_embeddings:
            self.lm_head: BaseOP = GlmFp8LMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing hook (engine calls it before the pool family's solve_num_pages):
        materialize every layer's bmm-ready kv_b split and free the checkpoint-layout
        originals, so the ~2.2 GiB repack is measured by the sizing pass instead of
        overcommitting the KV budget on the first forward (gpt_oss precedent)."""
        import torch

        for layer in self.model.layers.op_list:
            layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["GlmMoeDsaForCausalLM"]
