from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sparklab.core import get_global_ctx
from sparklab.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sparklab.models.blocks import BaseLLMModel
from sparklab.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from sparklab.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        dflash_args = getattr(config, "dflash2_args", None)
        self._dflash_capture_ids = (
            frozenset(dflash_args.target_layer_ids) if dflash_args is not None else frozenset()
        )
        self._dflash_captures: list[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        captures: list[torch.Tensor] = []
        for layer_id, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if layer_id in self._dflash_capture_ids:
                captures.append(x + residual)
        self._dflash_captures = captures
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from sparklab.kernels.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        self._mtp = None
        self._dflash = None
        self._mtp_steps = int(getattr(config, "speculative_tokens", 0) or 0)
        if getattr(config, "speculative_method", "none") == "dflash2":
            from .dflash2 import Qwen38DFlash2

            self._dflash = Qwen38DFlash2(
                config.dflash2_args, config.num_layers, self._mtp_steps
            )
        elif self._mtp_steps:
            from .mtp import Qwen3_5MultiTokenPredictor

            self._mtp = Qwen3_5MultiTokenPredictor(config)
        self._mtp_target_hidden: torch.Tensor | None = None
        super().__init__()

    def prepare_for_runtime(self) -> None:
        if self._dflash is not None:
            self._dflash.prepare_for_runtime(self.lm_head)

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        output = self.model.forward(batch.input_ids)
        if self._dflash is not None:
            self._dflash.materialize_target_hidden(
                self.model._dflash_captures, batch.positions, batch.out_loc
            )
        if self._mtp is not None:
            self._mtp_target_hidden = output
        return self.lm_head.forward(output)

    def replay_speculative_state(self) -> None:
        """Rebuild target and DFlash caches without the unused vocabulary projection."""
        if self._dflash is None:
            self.model.forward(get_global_ctx().batch.input_ids)
            return
        batch = get_global_ctx().batch
        self.model.forward(batch.input_ids)
        self._dflash.materialize_target_hidden(
            self.model._dflash_captures, batch.positions, batch.out_loc
        )

    def speculative_state_dict(self):
        if self._dflash is not None:
            return self._dflash.state_dict()
        return {} if self._mtp is None else self._mtp.state_dict()

    def load_speculative_state_dict(self, state_dict) -> None:
        if self._dflash is not None:
            self._dflash.load_state_dict(state_dict)
            return
        if self._mtp is None:
            if state_dict:
                raise RuntimeError("received Qwen MTP weights while MTP is disabled")
            return
        self._mtp.load_state_dict(state_dict)

    def propose_mtp(self, batch, next_token: torch.Tensor) -> torch.Tensor | None:
        if self._dflash is not None:
            return self._dflash.propose(self, batch, next_token)
        if self._mtp is None or self._mtp_target_hidden is None or batch.size != 1:
            return None
        from sparklab.core import Batch, Req

        req = batch.reqs[0]
        if not req.sampling_params.is_greedy:
            return None
        ctx = get_global_ctx()
        project_logits = getattr(self.lm_head, "forward_all", self.lm_head.forward)
        query = batch.input_ids
        shifted = torch.cat((query[1:], next_token.reshape(1)))
        original_input = batch.input_ids
        batch.input_ids = shifted
        try:
            with ctx.forward_batch(batch):
                feedback = self._mtp.forward(
                    self.model.embed_tokens.forward(shifted), self._mtp_target_hidden
                )
                last = batch.attn_metadata.get_last_indices(1).to(torch.long)
                feedback = feedback.index_select(0, last)
                draft = torch.argmax(project_logits(feedback), dim=-1)
        finally:
            batch.input_ids = original_input

        drafts = [draft.squeeze(0).to(torch.int32)]

        steps = min(
            self._mtp_steps, max(1, req.max_device_len - req.device_len)
        )
        for step in range(1, steps):
            position = req.device_len + step - 1
            draft_req = Req(
                input_ids=torch.zeros(position + 1, dtype=req.input_ids.dtype),
                table_idx=req.table_idx,
                cached_len=position,
                output_len=1,
                uid=req.uid,
                sampling_params=req.sampling_params,
                cache_handle=req.cache_handle,
            )
            draft_req.linear_slot_idx = req.linear_slot_idx
            draft_batch = Batch(reqs=[draft_req], phase="decode")
            draft_batch.padded_reqs = draft_batch.reqs
            draft_batch.input_ids = draft.reshape(1)
            draft_batch.positions = torch.tensor(
                [position], dtype=torch.int32, device=draft.device
            )
            draft_batch.out_loc = ctx.page_table[
                draft_req.table_idx, position : position + 1
            ]
            ctx.attn_backend.prepare_metadata(draft_batch)
            with ctx.forward_batch(draft_batch):
                feedback = self._mtp.forward(
                    self.model.embed_tokens.forward(draft_batch.input_ids), feedback
                )
                draft = torch.argmax(project_logits(feedback), dim=-1)
            drafts.append(draft.squeeze(0).to(torch.int32))
        return torch.stack(drafts)


__all__ = ["Qwen3_5MoEForCausalLM"]
