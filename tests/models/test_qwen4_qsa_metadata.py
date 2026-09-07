"""Optimized addressing must exactly match the original QSA metadata builder."""

from types import SimpleNamespace

import pytest
import torch

from sparklab.attention.qsa import QSAAttnBackend, QSACaptureData


def _backend(monkeypatch, fast):
    backend = QSAAttnBackend.__new__(QSAAttnBackend)
    backend.device = torch.device("cpu")
    backend.args = SimpleNamespace(index_compress_ratio=4, index_block_topk=8)
    backend._fast_metadata = fast
    backend._single_query_id = None
    page_table = torch.arange(192, dtype=torch.int32).view(3, 64).flip(1)
    monkeypatch.setattr(
        "sparklab.attention.qsa.get_global_ctx",
        lambda: SimpleNamespace(page_table=page_table),
    )
    return backend


def _batch(cached, count, table=1):
    req = SimpleNamespace(
        cached_len=cached, extend_len=count, device_len=cached + count, table_idx=table
    )
    return SimpleNamespace(reqs=[req])


def _equal(actual, expected):
    assert actual.qo_indptr == expected.qo_indptr
    assert actual.capture_decode == expected.capture_decode
    for name in [
        "last_indices",
        "q_to_req",
        "dense_indptr",
        "dense_indices",
        "dense_q_positions",
    ]:
        a, b = getattr(actual, name), getattr(expected, name)
        if b is None:
            assert a is None
        else:
            assert a.dtype == b.dtype
            assert a.shape == b.shape
            assert torch.equal(a, b)


@pytest.mark.parametrize("cached", [0, 1, 3, 8, 31, 34, 35])
def test_single_query_metadata_matches_original_builder(monkeypatch, cached):
    backend = _backend(monkeypatch, False)
    reference = _batch(cached, 1)
    backend.prepare_metadata(reference)
    backend._fast_metadata = True
    candidate = _batch(cached, 1)
    backend.prepare_metadata(candidate)
    _equal(candidate.attn_metadata, reference.attn_metadata)

    # Preparing another request must not overwrite a previous request's scalars.
    before = candidate.attn_metadata.dense_q_positions
    if before is not None:
        saved = before.clone()
        backend.prepare_metadata(_batch(7, 1, table=2))
        assert torch.equal(before, saved)


@pytest.mark.parametrize("cached", [0, 1, 3, 8, 30])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("captured", [False, True])
def test_prefix_metadata_matches_fresh_eager_builder(
    monkeypatch, cached, accepted, captured
):
    backend = _backend(monkeypatch, False)
    source, reference = _batch(cached, 5), _batch(cached, accepted)
    backend.prepare_metadata(source)
    backend.prepare_metadata(reference)
    if captured:
        backend.capture = QSACaptureData.create(
            64, torch.device("cpu"), max_queries_per_req=5
        )
        backend._point_to_capture(source)
    source_metadata = source.attn_metadata
    source_indices = source_metadata.dense_indices.clone()
    backend._fast_metadata = True
    candidate = _batch(cached, accepted)

    backend.prepare_prefix_metadata(source, candidate)

    _equal(candidate.attn_metadata, reference.attn_metadata)
    assert source.attn_metadata is source_metadata
    assert torch.equal(source_metadata.dense_indices, source_indices)
    assert (
        candidate.attn_metadata.dense_indices.data_ptr()
        == source_metadata.dense_indices.data_ptr()
    )


@pytest.mark.parametrize("cached", [32, 36])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_sparse_source_rebuilds_metadata_including_dense_prefix(
    monkeypatch, cached, accepted
):
    backend = _backend(monkeypatch, False)
    source, reference = _batch(cached, 4), _batch(cached, accepted)
    backend.prepare_metadata(source)
    assert source.attn_metadata.dense_indices is None
    backend.prepare_metadata(reference)
    backend._fast_metadata = True
    candidate = _batch(cached, accepted)
    backend.prepare_prefix_metadata(source, candidate)
    _equal(candidate.attn_metadata, reference.attn_metadata)


def test_prefix_reuse_rejects_changed_request_origin(monkeypatch):
    backend = _backend(monkeypatch, True)
    source = _batch(7, 4)
    backend.prepare_metadata(source)
    with pytest.raises(ValueError, match="unchanged accepted-prefix origin"):
        backend.prepare_prefix_metadata(source, _batch(8, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA arithmetic needs CUDA")
@pytest.mark.parametrize("cached", [0, 17, 127])
@pytest.mark.parametrize(
    "mode,count",
    [("single", 1)]
    + [
        (mode, count)
        for mode in ["eager-prefix", "capture-prefix"]
        for count in range(1, 6)
    ],
)
def test_cuda_attention_is_bitwise_equal_with_reused_metadata(
    monkeypatch, cached, mode, count
):
    from sparklab.kernels.triton.attention import paged_attention

    torch.manual_seed(412)
    device = torch.device("cuda")
    backend = QSAAttnBackend.__new__(QSAAttnBackend)
    backend.device = device
    backend.args = SimpleNamespace(index_compress_ratio=4, index_block_topk=512)
    backend._fast_metadata = False
    backend._single_query_id = None
    page_table = torch.randperm(512, dtype=torch.int32, device=device).view(2, 256)
    monkeypatch.setattr(
        "sparklab.attention.qsa.get_global_ctx",
        lambda: SimpleNamespace(page_table=page_table),
    )
    source, reference = _batch(cached, 5), _batch(cached, count)
    backend.prepare_metadata(source)
    backend.prepare_metadata(reference)
    if mode == "capture-prefix":
        backend.capture = QSACaptureData.create(256, device, max_queries_per_req=5)
        backend._point_to_capture(source)
    backend._fast_metadata = True
    candidate = _batch(cached, count)
    if mode == "single":
        backend.prepare_metadata(candidate)
    else:
        backend.prepare_prefix_metadata(source, candidate)
    _equal(candidate.attn_metadata, reference.attn_metadata)
    q = torch.randn(count, 8, 256, dtype=torch.bfloat16, device=device)
    k = torch.randn(512, 2, 256, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)

    def forward(md):
        return paged_attention(
            q=q,
            k_cache=k,
            v_cache=v,
            indptr=md.dense_indptr,
            indices=md.dense_indices,
            q_to_req=md.q_to_req,
            q_positions=md.dense_q_positions,
            sm_scale=256**-0.5,
        )

    assert torch.equal(
        forward(candidate.attn_metadata), forward(reference.attn_metadata)
    )


@pytest.mark.parametrize("width", [0, 1, 3, 4])
@pytest.mark.parametrize("max_bs", [1, 3])
def test_capture_capacity_includes_target_and_every_draft_row(
    monkeypatch, width, max_bs
):
    backend = _backend(monkeypatch, False)
    backend.config = SimpleNamespace(speculative_tokens=width)
    backend.capture = None
    backend.init_capture_graph(64, [1, max_bs])
    capacity = max_bs * max(4, width + 1)
    assert backend.capture.dense_indptr.numel() == capacity + 1
    assert backend.capture.dense_q_positions.numel() == capacity
    assert backend.capture.q_to_req.numel() == capacity
    assert backend.capture.last_indices.numel() == capacity
    assert backend.capture.dense_indices.numel() == capacity * 35

    batch = _batch(2, width + 1)
    backend.prepare_metadata(batch)
    backend._point_to_capture(batch)
    assert batch.attn_metadata.capture_decode
    assert batch.attn_metadata.dense_q_positions.tolist() == list(range(2, width + 3))
