from types import SimpleNamespace

import torch

from sparklab.attention.base import HybridBackend
from sparklab.core import Batch


class _RecordingBackend:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append(("forward", args, kwargs))
        return self.name

    def prepare_metadata(self, batch):
        self.calls.append(("prepare", batch))
        return self.name


def _batch(phase):
    batch = Batch(reqs=[SimpleNamespace()], phase=phase)
    batch.padded_reqs = batch.reqs
    return batch


def test_hybrid_attention_routes_verification_through_multitoken_backend():
    prefill = _RecordingBackend("prefill")
    decode = _RecordingBackend("decode")
    backend = HybridBackend(prefill, decode)
    batch = _batch("verify")

    assert backend.prepare_metadata(batch) == "prefill"
    assert backend.forward(
        torch.empty(0), torch.empty(0), torch.empty(0), 0, batch
    ) == "prefill"
    assert [call[0] for call in prefill.calls] == ["prepare", "forward"]
    assert decode.calls == []


def test_hybrid_attention_keeps_speculative_replay_on_sequential_backend():
    prefill = _RecordingBackend("prefill")
    decode = _RecordingBackend("decode")
    backend = HybridBackend(prefill, decode)
    batch = _batch("prefill")
    batch.is_speculative_replay = True

    assert backend.prepare_metadata(batch) == "prefill"
    assert backend.forward(
        torch.empty(0), torch.empty(0), torch.empty(0), 0, batch
    ) == "prefill"
    assert [call[0] for call in prefill.calls] == ["prepare", "forward"]
    assert decode.calls == []


def test_hybrid_attention_keeps_single_token_decode_on_decode_backend():
    prefill = _RecordingBackend("prefill")
    decode = _RecordingBackend("decode")
    backend = HybridBackend(prefill, decode)
    batch = _batch("decode")

    assert backend.prepare_metadata(batch) == "decode"
    assert backend.forward(
        torch.empty(0), torch.empty(0), torch.empty(0), 0, batch
    ) == "decode"
    assert prefill.calls == []
    assert [call[0] for call in decode.calls] == ["prepare", "forward"]
