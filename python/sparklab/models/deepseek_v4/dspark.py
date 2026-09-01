"""Fused DeepSeek-V4 DSpark draft model.

The three checkpoint ``mtp.*`` blocks share the target embedding and LM head.
Target auxiliary states seed their sliding-window context KV; a parallel block
of anchor/noise queries is then refined by a sequential low-rank Markov bias.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sparklab.core import get_global_ctx
from sparklab.kernels.triton.dsv4.hc import hc_pre_combine

from .args import DeepseekV4Args
from .layers import Linear, RMSNorm
from .model import Block


def draft_query_geometry(
    device_len: int, page_size: int, max_steps: int
) -> tuple[int, int]:
    """Return the sampled anchor position and page-local DSpark block width."""
    if device_len <= 0 or page_size <= 0 or max_steps <= 0:
        return max(0, device_len - 1), 0
    anchor_pos = device_len - 1
    page_end = (anchor_pos // page_size + 1) * page_size
    return anchor_pos, min(max_steps, page_end - anchor_pos)


class DSparkDraft(nn.Module):
    def __init__(self, args: DeepseekV4Args, steps: int, sample_method: str):
        super().__init__()
        self.args = args
        self.steps = int(steps)
        self.sample_method = sample_method
        self.hc_mult = args.hc_mult
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        n_aux = len(args.dspark_target_layer_ids)
        self.main_proj = Linear(n_aux * args.dim, args.dim, kind="fp8")
        self.main_norm = RMSNorm(args.dim, args.norm_eps)
        self.layers = nn.ModuleList(
            [Block(args.n_layers + i, args) for i in range(args.n_mtp_layers)]
        )
        self.norm = RMSNorm(args.dim, args.norm_eps)
        hc_dim = args.hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(
            torch.empty(args.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(args.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        self.markov_w1 = nn.Parameter(
            torch.empty(args.vocab_size, args.dspark_markov_rank, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.markov_w2 = nn.Parameter(
            torch.empty(args.vocab_size, args.dspark_markov_rank, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.confidence_weight = nn.Parameter(
            torch.empty(1, args.dim + args.dspark_markov_rank, dtype=torch.float32),
            requires_grad=False,
        )
        self._bound = False
        self.last_draft_probs: torch.Tensor | None = None

    def bind(self, pool, device: torch.device) -> None:
        if self._bound:
            return
        for layer in self.layers:
            layer.attn.bind(pool, device)
        self._bound = True

    def mark_for_rebind(self) -> None:
        self._bound = False

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(
            xf.square().mean(-1, keepdim=True) + self.norm_eps
        )
        mixes = F.linear(xf, self.hc_head_fn) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base)
            + self.hc_eps
        )
        m = shape[0] * shape[1]
        return hc_pre_combine(
            xf.view(m, self.hc_mult, self.args.dim),
            pre.view(m, self.hc_mult),
            dtype,
        ).view(*shape[:2], self.args.dim)

    @torch.inference_mode()
    def store_target_context(
        self,
        aux_states: list[torch.Tensor],
        *,
        segments,
        positions: torch.Tensor,
    ) -> None:
        """Project selected post-target-layer states and persist draft context KV."""
        if not aux_states:
            return
        pool = get_global_ctx().kv_cache
        self.bind(pool, pool.device)
        main = self.main_norm(self.main_proj(torch.cat(aux_states, dim=-1)))
        for off, n, table_idx, start_pos in segments:
            pos = positions[off : off + n].long()
            slots = get_global_ctx().attn_backend.window_slots_of(
                table_idx, start_pos, start_pos + n
            )
            part = main[:, off : off + n]
            for layer in self.layers:
                layer.attn.store_dspark_context(part, pos, slots)

    @torch.inference_mode()
    def propose(
        self,
        batch,
        next_token: torch.Tensor,
        *,
        embedding_weight: torch.Tensor,
        head_weight: torch.Tensor,
    ) -> torch.Tensor | None:
        if batch.size != 1:
            return None
        ctx = get_global_ctx()
        pool = ctx.kv_cache
        self.bind(pool, pool.device)
        req = batch.reqs[0]
        self.last_draft_probs = None
        if self.sample_method == "greedy" and not req.sampling_params.is_greedy:
            return None
        # ``next_token`` is the just-sampled anchor at device_len - 1.  Draft
        # outputs begin at device_len; positioning the anchor one row later
        # shifts every RoPE phase and can cross into an unallocated page.
        query_start, n_query = draft_query_geometry(
            req.device_len, pool.P, self.steps
        )
        if n_query <= 0:
            return None
        anchor_slot = ctx.attn_backend.window_slots_of(
            req.table_idx, query_start, query_start + 1
        )
        if bool((anchor_slot < 0).any()):
            # The sampled anchor opened a new page.  Its target forward on the
            # next scheduler step allocates that page, after which drafting
            # resumes; cache allocation remains scheduler-owned.
            return None

        anchor = next_token.reshape(1).long()
        noise = torch.full(
            (n_query - 1,),
            self.args.dspark_noise_token_id,
            dtype=torch.long,
            device=anchor.device,
        )
        input_ids = torch.cat((anchor, noise)).view(1, -1)
        positions = torch.arange(
            query_start,
            query_start + n_query,
            dtype=torch.long,
            device=anchor.device,
        )
        context_width = max(0, self.args.window_size - n_query)
        context_start = max(0, query_start - context_width)
        hidden = F.embedding(input_ids, embedding_weight)
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            hidden = layer.dspark_forward(
                hidden,
                input_ids,
                positions=positions,
                table_idx=req.table_idx,
                context_start=context_start,
            )
        head_hidden = self.hc_head(hidden)
        base_logits = F.linear(self.norm(head_hidden)[0], head_weight)

        previous = anchor
        drafts = []
        draft_probs = []
        for row in range(n_query):
            markov = F.embedding(previous, self.markov_w1)
            logits = base_logits[row : row + 1] + F.linear(markov, self.markov_w2)
            if req.sampling_params.is_greedy:
                token = torch.argmax(logits, dim=-1)
            else:
                from sparklab.runtime.engine.sample import sampling_distribution

                probs = sampling_distribution(logits, req.sampling_params)
                token = torch.multinomial(probs, 1).squeeze(-1)
                draft_probs.append(probs.squeeze(0))
            drafts.append(token.squeeze(0).to(torch.int32))
            previous = token
        if draft_probs:
            self.last_draft_probs = torch.stack(draft_probs)
        return torch.stack(drafts)


__all__ = ["DSparkDraft"]
