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
        self._mtp_steps = int(getattr(config, "speculative_tokens", 0) or 0)
        self._mtp = None
        if self._mtp_steps:
            from .mtp import Glm5NextMultiTokenPredictor

            self._mtp = Glm5NextMultiTokenPredictor(config)
        self._mtp_path: str | None = None
        self._mtp_target_hidden: torch.Tensor | None = None
        super().__init__()

    def prepare_for_weight_load(self, model_path: str, *, dummy: bool = False) -> None:
        if self._mtp is None or dummy:
            return
        from .weight import find_mtp_sidecar

        self._mtp_path = find_mtp_sidecar(model_path)
        if self._mtp_path is None:
            raise FileNotFoundError(
                "GLM-5.3 Flash MTP requires model_mtp.safetensors beside the "
                "checkpoint or SPARKLAB_GLM5_MTP_PATH"
            )

    def load_speculative_weights(
        self, model_path: str, device: torch.device, *, dummy: bool = False
    ) -> None:
        del model_path
        if self._mtp is None:
            return
        if dummy:
            self._mtp.load_dummy(device)
            return
        if self._mtp_path is None:
            raise RuntimeError("GLM-5.3 MTP sidecar path was not prepared")
        self._mtp.load_sidecar(self._mtp_path, device)

    def prepare_for_runtime(self) -> None:
        for layer in self.model.layers.op_list:
            layer.attn_hc.prepare_for_runtime()
            layer.ffn_hc.prepare_for_runtime()
            if isinstance(layer.self_attn, Glm5NextMLAAttention):
                layer.self_attn.prepare_for_runtime()
            elif isinstance(layer.self_attn, Glm5NextDeltaAttention):
                layer.self_attn.prepare_for_runtime()
        if self._mtp is not None:
            self._mtp.layer.self_attn.prepare_for_runtime()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        if self._mtp is not None:
            self._mtp_target_hidden = output
        return self.lm_head.forward(output)

    def propose_mtp(self, batch, next_token: torch.Tensor) -> torch.Tensor | None:
        if self._mtp is None or self._mtp_target_hidden is None or batch.size != 1:
            return None
        from sparklab.core import Batch, Req
        from sparklab.layers.linear import _linear_forward

        req = batch.reqs[0]
        if not req.sampling_params.is_greedy:
            return None
        ctx = get_global_ctx()
        head = self.lm_head.tied_embedding or self.lm_head

        def project_logits(hidden: torch.Tensor) -> torch.Tensor:
            # ParallelLMHead.forward applies prefill last-row selection. The MTP
            # path has already selected its row, so project directly like Qwen4.
            return _linear_forward(hidden, head.weight, self.lm_head.bias)

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
        steps = min(self._mtp_steps, max(1, req.max_device_len - req.device_len))
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
            draft_batch = Batch(reqs=[draft_req], phase="decode")
            draft_batch.padded_reqs = draft_batch.reqs
            draft_batch.input_ids = draft.reshape(1)
            draft_batch.positions = torch.tensor(
                [position], dtype=torch.int32, device=draft.device
            )
            draft_batch.out_loc = ctx.page_table[
                draft_req.table_idx, position : position + 1
            ]
            draft_batch.active_table_idx = torch.tensor(
                [draft_req.table_idx], dtype=torch.int64, device=draft.device
            )
            ctx.attn_backend.prepare_metadata(draft_batch)
            with ctx.forward_batch(draft_batch):
                feedback = self._mtp.forward(
                    self.model.embed_tokens.forward(draft_batch.input_ids), feedback
                )
                draft = torch.argmax(project_logits(feedback), dim=-1)
            drafts.append(draft.squeeze(0).to(torch.int32))
        return torch.stack(drafts)


__all__ = ["Glm5NextForCausalLM"]
