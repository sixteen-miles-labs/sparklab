from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sparklab.core import get_global_ctx
from sparklab.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from sparklab.models.blocks import BaseLLMModel
from sparklab.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from sparklab.models.qwen3_5_moe.moe import Qwen3_5MoE
from sparklab.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hyper import Qwen4GatedResidual
from .ple import Qwen4PLE
from .mtp import Qwen4ExpMultiTokenPredictor

if TYPE_CHECKING:
    from sparklab.models.config import ModelConfig


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
        hidden = self.attn_hyper_connection.inject(branch, residual, inject)
        branch, residual, inject = self.mlp_hyper_connection.forward(hidden)
        branch = self.mlp.forward(branch)
        return self.mlp_hyper_connection.inject(branch, residual, inject)


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

    def forward(
        self, input_ids: torch.Tensor, *, return_multi: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self._hc_count)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden)
        sample = self.hyper_connection_mixer.forward(hidden)
        return (sample, hidden) if return_multi else sample


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=(self.model.embed_tokens if config.tie_word_embeddings else None),
        )
        # Private so BaseOP's strict target state dict remains unchanged. The
        # standalone sidecar is loaded explicitly after the target + expert cache.
        self._mtp = (
            Qwen4ExpMultiTokenPredictor(config)
            if int(getattr(config, "speculative_tokens", 0) or 0) else None
        )
        self._mtp_steps = int(getattr(config, "speculative_tokens", 0) or 0)
        self._mtp_path: str | None = None
        self._mtp_target_hidden: torch.Tensor | None = None
        super().__init__()

    def prepare_for_weight_load(self, model_path: str, *, dummy: bool = False) -> None:
        for layer in self.model.layers.op_list:
            if layer.ple is not None:
                layer.ple.bind(model_path, dummy=dummy)
        if self._mtp is not None and not dummy:
            from .weight import find_mtp_sidecar

            self._mtp_path = find_mtp_sidecar(model_path)
            if self._mtp_path is None:
                raise FileNotFoundError(
                    "Qwen4 MTP requires nvfp4_experts_mtp.safetensors beside the "
                    "checkpoint (or SPARKLAB_QWEN4_MTP_PATH)"
                )

    def prepare_for_runtime(self) -> None:
        if self._mtp is not None:
            if self._mtp_path is None:
                raise RuntimeError("Qwen4 MTP sidecar path was not prepared")
            self._mtp.load_sidecar(self._mtp_path, self.model.embed_tokens.weight.device)

    def prepare_cuda_graph_inputs(self, batch) -> None:
        for layer in self.model.layers.op_list:
            if layer.ple is not None:
                layer.ple.prepare_cuda_graph_inputs(batch)

    def forward(self) -> torch.Tensor:
        if self._mtp is None:
            hidden = self.model.forward(get_global_ctx().batch.input_ids)
            return self.lm_head.forward(hidden)
        hidden, multi = self.model.forward(
            get_global_ctx().batch.input_ids, return_multi=True
        )
        self._mtp_target_hidden = multi
        return self.lm_head.forward(hidden)

    def propose_mtp(self, batch, next_token: torch.Tensor) -> torch.Tensor | None:
        """Greedily propose up to the configured MTP width for one request.

        Step zero reuses the target batch shape exactly: token ids are shifted
        left by one and the sampled target token fills the last row, matching
        vLLM's Qwen4 MTP alignment. Later one-token steps advance only the draft
        layer's independent QSA cache.
        """
        if self._mtp is None or self._mtp_target_hidden is None or batch.size != 1:
            return None
        import torch.nn.functional as F
        from sparklab.core import Batch, Req

        ctx = get_global_ctx()
        req = batch.reqs[0]
        if not req.sampling_params.is_greedy:
            return None
        query = batch.input_ids
        shifted = torch.cat((query[1:], next_token.reshape(1)))
        original_input = batch.input_ids
        batch.input_ids = shifted
        try:
            with ctx.forward_batch(batch):
                sample_hidden, feedback = self._mtp.forward(
                    self.model.embed_tokens.forward(shifted), self._mtp_target_hidden
                )
        finally:
            batch.input_ids = original_input

        last = batch.attn_metadata.get_last_indices(1).to(torch.long)
        sample_hidden = sample_hidden.index_select(0, last)
        feedback = feedback.index_select(0, last)
        head = self.lm_head.tied_embedding or self.lm_head
        draft = torch.argmax(F.linear(sample_hidden, head.weight, self.lm_head.bias), -1)
        drafts = [draft.squeeze(0).to(torch.int32)]

        # The scheduler reserves the configured lookahead before this forward,
        # including with page_size=1, so every configured draft step has a
        # physical KV row.
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
                sample_hidden, feedback = self._mtp.forward(
                    self.model.embed_tokens.forward(draft_batch.input_ids), feedback
                )
            draft = torch.argmax(
                F.linear(sample_hidden, head.weight, self.lm_head.bias), -1
            )
            drafts.append(draft.squeeze(0).to(torch.int32))
        return torch.stack(drafts)


__all__ = ["Qwen4ExpForCausalLM"]
