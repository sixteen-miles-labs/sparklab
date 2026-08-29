"""Req.append_host writes into the preallocated buffer: value-equivalent to the
old per-step torch.cat, no reallocation, and existing views stay stable."""

import pytest
import torch

from sparklab.core import Req, SamplingParams
from sparklab.runtime.scheduler.prefill import ChunkedReq


def _mk(cls, input_ids, output_len=4):
    return cls(
        input_ids=input_ids,
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )


def test_append_host_matches_cat_without_reallocating():
    req = _mk(Req, torch.arange(6, dtype=torch.int32))
    ref = torch.arange(6, dtype=torch.int32)
    base_ptr = req.input_ids.data_ptr()
    view = req.input_ids[:3]
    for t in (101, 102, 103, 104):
        tok = torch.tensor([t], dtype=torch.int32)
        ref = torch.cat([ref, tok])
        req.append_host(tok)
        assert torch.equal(req.input_ids, ref)
        assert req.input_ids.data_ptr() == base_ptr
        assert req.input_ids.dtype == torch.int32
    assert len(req.input_ids) == req.max_device_len
    assert torch.equal(view, torch.arange(3, dtype=torch.int32))


def test_chunked_req_keeps_prompt_view_and_rejects_append():
    ids = torch.arange(6, dtype=torch.int32)
    req = _mk(ChunkedReq, ids)
    assert req.input_ids is ids
    with pytest.raises(NotImplementedError):
        req.append_host(torch.tensor([1], dtype=torch.int32))
