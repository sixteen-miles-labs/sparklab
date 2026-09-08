"""Fixed-shape draft graphs with independent FlashInfer planning buffers."""
from copy import copy

import torch

from sparklab.core import get_global_ctx


class MTPDraftGraph:
    def __init__(self, model, backend, batch, shifted, hidden):
        self.model = model
        self.backend = backend
        self.input_ids = shifted.clone()
        self.hidden = hidden.clone()
        self.positions = batch.positions.clone()
        self.out_loc = batch.out_loc.clone()
        self.batch = copy(batch)
        # Both initial feedback and recursive steps are causal continuations.
        # This selects the skinny routed kernel, including for a one-row prefix.
        self.batch.phase = "verify"
        self.batch.input_ids = self.input_ids
        self.batch.positions = self.positions
        self.batch.out_loc = self.out_loc
        backend.prepare_for_capture(self.batch)
        self.graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=hidden.device)
        with get_global_ctx().forward_batch(self.batch):
            self._forward()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(self.graph, stream=capture_stream):
                self.draft, self.feedback = self._forward()
            torch.cuda.current_stream().wait_stream(capture_stream)

    def _forward(self):
        hidden = self.model._mtp.forward(
            self.model.model.embed_tokens.forward(self.input_ids), self.hidden
        )
        # One request; its last query is fixed by this graph's row count.
        feedback = hidden[-1:]
        project = getattr(self.model.lm_head, "forward_all", self.model.lm_head.forward)
        draft = torch.argmax(project(feedback), dim=-1)
        return draft, feedback

    def replay(self, batch, shifted, hidden):
        self.input_ids.copy_(shifted)
        self.hidden.copy_(hidden)
        self.positions.copy_(batch.positions)
        self.out_loc.copy_(batch.out_loc)
        active = copy(batch)
        active.phase = "verify"
        active.input_ids = shifted
        self.backend.prepare_metadata(active)
        self.backend.prepare_for_replay(active)
        self.graph.replay()
        return self.draft, self.feedback


def run_mtp_draft_graph(model, batch, shifted, hidden):
    """Return None outside the small, single-request FlashInfer draft domain."""
    from sparklab.attention.fi import FlashInferBackend

    backend = get_global_ctx().attn_backend
    if (not isinstance(backend, FlashInferBackend)
            or not 1 <= shifted.numel() <= model._mtp_steps + 1):
        return None
    graphs = getattr(model, "_mtp_draft_graphs", None)
    if graphs is None:
        # Plans modify GPU scratch and pinned host staging. Target verification
        # and eager prefill must not replan the draft graphs' captured buffers.
        draft_backend = FlashInferBackend(backend.config)
        draft_backend.init_capture_graph(
            max_seq_len=get_global_ctx().page_table.shape[1], bs_list=[1]
        )
        model._mtp_draft_graph_backend = draft_backend
        model._mtp_draft_graphs = graphs = {}
    rows = shifted.numel()
    if rows not in graphs:
        graphs[rows] = MTPDraftGraph(
            model, model._mtp_draft_graph_backend, batch, shifted, hidden
        )
    return graphs[rows].replay(batch, shifted, hidden)
