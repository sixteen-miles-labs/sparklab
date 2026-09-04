"""Native DFlash2 block drafter for dense Qwen3.8.

The draft has no embedding or language-model head.  It consumes five captured
target-layer features, reuses the target embedding/head, and predicts a masked
block in parallel.  The architecture follows the MIT-licensed DGX Spark
DFlash2 reference; the runtime integration and cache layout are SparkLab-native.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sparklab.core import Batch, Req, get_global_ctx
from sparklab.kernels.triton.nvfp4_linear import (
    Nvfp4DenseColMerged,
    Nvfp4DenseLinear,
)
from sparklab.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    RMSNorm,
    silu_and_mul,
)
from sparklab.layers.rotary import get_rope

try:
    from flashinfer import top_k as _flashinfer_top_k
except ImportError:  # pragma: no cover - exercised only in minimal installations
    _flashinfer_top_k = None


def _selector_topk(scores: torch.Tensor, k: int):
    """Use FlashInfer's radix top-k for the selector's large vocabulary scan."""
    if scores.is_cuda and _flashinfer_top_k is not None:
        return _flashinfer_top_k(scores, k, sorted=True, deterministic=True)
    return torch.topk(scores, k, dim=-1)


@dataclass(frozen=True)
class DFlash2Args:
    draft_model_path: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    sliding_window: int
    checkpoint_block_size: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int


def parse_dflash2_args(hf_config: Any, draft_model_path: str) -> DFlash2Args:
    cfg = getattr(hf_config, "text_config", hf_config)
    dflash = getattr(cfg, "dflash_config", None) or getattr(hf_config, "dflash_config", {})
    if not isinstance(dflash, dict):
        dflash = dflash.to_dict()
    architectures = tuple(getattr(hf_config, "architectures", ()) or ())
    if "DFlash2DraftModel" not in architectures:
        raise ValueError(
            "--speculative-method dflash2 requires a DFlash2DraftModel checkpoint"
        )
    layer_ids = tuple(int(x) for x in dflash.get("target_layer_ids", ()))
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    if not layer_ids or len(layer_ids) != num_layers:
        raise ValueError(
            "DFlash2 target_layer_ids must contain one target feature per draft layer"
        )
    rope = getattr(cfg, "rope_parameters", None) or {}
    return DFlash2Args(
        draft_model_path=draft_model_path,
        num_layers=num_layers,
        hidden_size=int(getattr(cfg, "hidden_size")),
        intermediate_size=int(getattr(cfg, "intermediate_size")),
        num_attention_heads=int(getattr(cfg, "num_attention_heads")),
        num_key_value_heads=int(getattr(cfg, "num_key_value_heads")),
        head_dim=int(getattr(cfg, "head_dim")),
        vocab_size=int(getattr(cfg, "vocab_size")),
        max_position_embeddings=int(getattr(cfg, "max_position_embeddings")),
        rope_theta=float(rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))),
        rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
        sliding_window=int(getattr(cfg, "sliding_window", 0) or 0),
        checkpoint_block_size=int(dflash.get("block_size", 0) or 0),
        conv_kernel_size=int(dflash.get("conv_kernel_size", 0) or 0),
        conv_group_size=int(dflash.get("conv_group_size", 0) or 0),
        selector_rank=int(dflash.get("selector_rank", 0) or 0),
        selector_top_k=int(dflash.get("selector_top_k", 0) or 0),
        target_layer_ids=layer_ids,
        mask_token_id=int(dflash["mask_token_id"]),
    )


class DFlashGroupedConv(BaseOP):
    def __init__(self, args: DFlash2Args, block_size: int):
        if args.hidden_size % args.conv_group_size:
            raise ValueError("DFlash2 conv_group_size must divide hidden_size")
        self.block_size = block_size
        self.taps = args.conv_kernel_size
        self.group_size = args.conv_group_size
        self.num_groups = args.hidden_size // self.group_size
        self.base_kernel = torch.empty(2, self.taps, args.hidden_size)
        self.kernel_projection = LinearReplicated(
            args.hidden_size, 2 * self.taps * self.num_groups, has_bias=False
        )

    def _convolve(
        self, hidden: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        blocks = hidden.unflatten(-1, (self.num_groups, self.group_size))
        base = self.base_kernel[side].view(
            1, self.taps, self.num_groups, self.group_size
        )
        coefficients = base + delta.unsqueeze(-1)
        out = coefficients[:, 0] * blocks
        position = torch.arange(hidden.shape[0], device=hidden.device) % self.block_size
        for tap in range(1, self.taps):
            shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            out = out + coefficients[:, tap] * shifted * (
                position >= tap
            ).view(-1, 1, 1)
        return out.flatten(-2)

    def prepare(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection.forward(hidden).reshape(
            hidden.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(self, hidden: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return self._convolve(hidden, coefficients, 1)


class DFlashAttention(BaseOP):
    def __init__(self, args: DFlash2Args, layer_id: int):
        self.layer_id = layer_id
        self.num_q = args.num_attention_heads
        self.num_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.q_size = self.num_q * self.head_dim
        self.kv_size = self.num_kv * self.head_dim
        self.sliding_window = args.sliding_window
        self.qkv_proj = Nvfp4DenseColMerged(
            args.hidden_size, [self.q_size, self.kv_size, self.kv_size]
        )
        self.o_proj = Nvfp4DenseLinear(self.q_size, args.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=args.max_position_embeddings,
            base=args.rope_theta,
            rope_scaling=None,
        )

    def _project(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj.forward(hidden)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], -1)
        q = self.q_norm.forward(q.contiguous().view(-1, self.head_dim)).view_as(q)
        k = self.k_norm.forward(k.contiguous().view(-1, self.head_dim)).view_as(k)
        q, k = self.rotary.forward(positions, q, k)
        return q, k, v.contiguous()

    def materialize_context(
        self, hidden: torch.Tensor, positions: torch.Tensor, out_loc: torch.Tensor
    ) -> None:
        qkv = self.qkv_proj.forward(hidden)
        _, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], -1)
        k = self.k_norm.forward(k.contiguous().view(-1, self.head_dim)).view_as(k)
        dummy_q = torch.empty_like(k)
        _, k = self.rotary.forward(positions, dummy_q, k)
        get_global_ctx().kv_cache.store_kv(
            k.view(-1, self.kv_size),
            v.contiguous().view(-1, self.kv_size),
            out_loc,
            self.layer_id,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        if batch.size != 1:
            raise RuntimeError("native DFlash2 currently supports batch size 1")
        q, k, v = self._project(hidden, batch.positions)
        req = batch.reqs[0]
        block = hidden.shape[0]
        prefix_keep = req.cached_len
        if self.sliding_window:
            prefix_keep = min(prefix_keep, max(0, self.sliding_window - block))
        start = req.cached_len - prefix_keep
        slots = ctx.page_table[req.table_idx, start : req.cached_len].to(torch.long)
        k_cache = ctx.kv_cache.k_cache(self.layer_id).view(
            -1, self.num_kv, self.head_dim
        )
        v_cache = ctx.kv_cache.v_cache(self.layer_id).view(
            -1, self.num_kv, self.head_dim
        )
        prefix_k = k_cache.index_select(0, slots)
        prefix_v = v_cache.index_select(0, slots)
        all_k = torch.cat((prefix_k, k.view(-1, self.num_kv, self.head_dim)), 0)
        all_v = torch.cat((prefix_v, v.view(-1, self.num_kv, self.head_dim)), 0)
        q4 = q.view(block, self.num_q, self.head_dim).transpose(0, 1).unsqueeze(0)
        k4 = all_k.transpose(0, 1).unsqueeze(0)
        v4 = all_v.transpose(0, 1).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            q4, k4, v4, is_causal=False, enable_gqa=True
        )
        output = output.squeeze(0).transpose(0, 1).reshape(block, self.q_size)
        return self.o_proj.forward(output)


class DFlashMLP(BaseOP):
    def __init__(self, args: DFlash2Args):
        self.gate_up_proj = Nvfp4DenseColMerged(
            args.hidden_size, [args.intermediate_size, args.intermediate_size]
        )
        self.down_proj = Nvfp4DenseLinear(args.intermediate_size, args.hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(hidden)))


class DFlashDecoderLayer(BaseOP):
    def __init__(self, args: DFlash2Args, layer_id: int, block_size: int):
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.self_attn = DFlashAttention(args, layer_id)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = DFlashMLP(args)
        self.attention_conv = DFlashGroupedConv(args, block_size)
        self.mlp_conv = DFlashGroupedConv(args, block_size)

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            residual = residual + hidden
            hidden = self.input_layernorm.forward(residual)
        hidden, kernel = self.attention_conv.prepare(hidden)
        hidden = self.attention_conv.finish(self.self_attn.forward(hidden), kernel)
        residual = residual + hidden
        hidden = self.post_attention_layernorm.forward(residual)
        hidden, kernel = self.mlp_conv.prepare(hidden)
        hidden = self.mlp_conv.finish(self.mlp.forward(hidden), kernel)
        return hidden, residual


class CandidateSelector(BaseOP):
    def __init__(self, args: DFlash2Args):
        self.predecessor_codebook = torch.empty(
            args.vocab_size, args.selector_rank
        )
        self.successor_codebook = torch.empty(args.vocab_size, args.selector_rank)
        self.hidden_projection = LinearReplicated(
            args.hidden_size, args.selector_rank, has_bias=False
        )
        self.top_k = args.selector_top_k

    def greedy_path(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.hidden_projection.forward(hidden)
        predecessor_ids = torch.cat(
            (anchor.reshape(1, 1).expand(1, self.top_k), candidate_ids[:-1]), 0
        )
        predecessors = self.predecessor_codebook[predecessor_ids]
        successors = self.successor_codebook[candidate_ids]
        scores = unary_logits[:, None, :] + torch.einsum(
            "lpr,lcr,lr->lpc", predecessors, successors, projected
        )
        if scores.is_cuda:
            from sparklab.kernels.triton.dflash2 import greedy_selector_walk

            return greedy_selector_walk(candidate_ids, scores).to(torch.int32)
        index = scores[0, 0].argmax()
        path = [candidate_ids[0, index]]
        for edge in range(1, candidate_ids.shape[0]):
            index = scores[edge, index].argmax()
            path.append(candidate_ids[edge, index])
        return torch.stack(path).to(torch.int32)


class Qwen38DFlash2(BaseOP):
    def __init__(self, args: DFlash2Args, target_num_layers: int, block_size: int):
        if args.hidden_size <= 0 or args.selector_rank <= 0:
            raise ValueError("invalid DFlash2 checkpoint geometry")
        if any(layer < 0 or layer >= target_num_layers for layer in args.target_layer_ids):
            raise ValueError("DFlash2 target layer id is outside the target tower")
        self.args = args
        self.block_size = block_size
        first_layer_id = target_num_layers
        self.layers = OPList(
            [
                DFlashDecoderLayer(args, first_layer_id + i, block_size)
                for i in range(args.num_layers)
            ]
        )
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.fc = LinearReplicated(
            len(args.target_layer_ids) * args.hidden_size,
            args.hidden_size,
            has_bias=False,
        )
        self.hidden_norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.candidate_selector = CandidateSelector(args)
        self._selector_lm_head = None

    def prepare_for_runtime(self, target_lm_head) -> None:
        """Use a compact proposal-only head while target verification stays exact."""
        from sparklab.kernels.triton.nvfp4_linear import (
            Nvfp4LMHead,
            quantize_nvfp4_lm_head,
        )

        if isinstance(target_lm_head, Nvfp4LMHead):
            self._selector_lm_head = target_lm_head
            return
        weight = getattr(target_lm_head, "weight", None)
        if weight is None:
            raise RuntimeError("DFlash2 selector requires a target language-model head")
        try:
            self._selector_lm_head = quantize_nvfp4_lm_head(weight)
        except ImportError:
            # FlashInfer also accelerates top-k, but keep the correctness-first
            # BF16 proposal path usable in minimal installations.
            self._selector_lm_head = target_lm_head

    def materialize_target_hidden(
        self,
        captures: list[torch.Tensor],
        positions: torch.Tensor,
        out_loc: torch.Tensor,
    ) -> None:
        if len(captures) != len(self.args.target_layer_ids):
            raise RuntimeError("DFlash2 target hidden capture count mismatch")
        hidden = self.hidden_norm.forward(self.fc.forward(torch.cat(captures, -1)))
        for layer in self.layers.op_list:
            layer.self_attn.materialize_context(hidden, positions, out_loc)

    def propose(self, target, batch: Batch, next_token: torch.Tensor) -> torch.Tensor | None:
        if batch.size != 1 or not batch.reqs[0].sampling_params.is_greedy:
            return None
        req = batch.reqs[0]
        block_size = min(self.block_size, max(1, req.max_device_len - req.device_len))
        if block_size <= 1:
            return None
        ids = torch.full(
            (block_size,),
            self.args.mask_token_id,
            dtype=torch.int32,
            device=next_token.device,
        )
        ids[0].copy_(next_token.reshape(()).to(torch.int32))
        prefix = req.device_len
        draft_req = Req(
            input_ids=torch.zeros(prefix + block_size, dtype=req.input_ids.dtype),
            table_idx=req.table_idx,
            cached_len=prefix,
            output_len=1,
            uid=req.uid,
            sampling_params=req.sampling_params,
            cache_handle=req.cache_handle,
        )
        draft_req.linear_slot_idx = req.linear_slot_idx
        draft_batch = Batch(reqs=[draft_req], phase="decode")
        draft_batch.padded_reqs = draft_batch.reqs
        draft_batch.input_ids = ids
        draft_batch.positions = torch.arange(
            prefix, prefix + block_size, dtype=torch.int32, device=ids.device
        )
        draft_batch.out_loc = get_global_ctx().page_table[
            req.table_idx, prefix : prefix + block_size
        ]
        draft_batch.return_all_logits = True
        with get_global_ctx().forward_batch(draft_batch):
            hidden = target.model.embed_tokens.forward(ids)
            residual: torch.Tensor | None = None
            for layer in self.layers.op_list:
                hidden, residual = layer.forward(hidden, residual)
            hidden = self.norm.forward(hidden + residual)
            predictive = hidden[1:]
            selector_head = self._selector_lm_head or target.lm_head
            project = getattr(selector_head, "forward_all", selector_head.forward)
            logits = project(predictive)
            unary, candidates = _selector_topk(
                logits.float(), self.candidate_selector.top_k
            )
            return self.candidate_selector.greedy_path(
                candidates.to(torch.long), unary, predictive, ids[:1].to(torch.long)
            )


__all__ = ["DFlash2Args", "Qwen38DFlash2", "parse_dflash2_args"]
