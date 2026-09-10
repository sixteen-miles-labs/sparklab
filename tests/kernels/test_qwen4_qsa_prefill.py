import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("cached,tokens,topk,ratio,heads,dim", [
    (0, 40, 4, 4, 3, 16),       # dense prefix, causal boundary, all tail lengths
    (15, 37, 4, 4, 3, 16),      # cached prefix straddling the sparse boundary
    (63, 35, 8, 2, 4, 32),      # entirely sparse, partial chunks
    (8001, 33, 512, 4, 16, 128), # production budget and long cached context
    (0, 0, 4, 4, 3, 16),
])
@pytest.mark.parametrize("tied", [False, True])
def test_prefill_selection_matches_per_query_reference(cached, tokens, topk, ratio, heads, dim, tied):
    from sparklab.kernels.triton.qwen4 import qsa_index_scores
    from sparklab.kernels.triton.qwen4_qsa_prefill import qsa_prefill_indices

    torch.manual_seed(810)
    query = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn((cached + tokens) // ratio, dim, device="cuda", dtype=torch.bfloat16)
    if tied:
        query.zero_()
    physical = torch.randperm(cached + tokens, device="cuda", dtype=torch.int32)
    actual, counts = qsa_prefill_indices(
        query, keys, physical, cached_len=cached, ratio=ratio, topk=topk, chunk_size=17,
    )
    expected = []
    for local in range(tokens):
        visible = cached + local + 1
        complete = visible // ratio
        if complete <= topk:
            rows = physical[:visible]
        else:
            score = qsa_index_scores(query[local], keys[:complete])
            selected = torch.topk(score, topk, sorted=False).indices
            logical = (selected[:, None] * ratio + torch.arange(ratio, device="cuda")).flatten()
            logical = torch.cat((logical, torch.arange(complete * ratio, visible, device="cuda"))).sort().values
            rows = physical[logical]
        expected.append(rows)
        assert counts[local] == rows.numel()
    expected = torch.cat(expected) if expected else physical.new_empty(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_backend_packs_mixed_prefill_and_decode_requests(monkeypatch):
    from types import SimpleNamespace

    from sparklab.attention.qsa import QSAAttnBackend
    from sparklab.kernels.triton.attention import paged_attention

    torch.manual_seed(918)
    device = torch.device("cuda")
    page_table = torch.randperm(384, dtype=torch.int32, device=device).view(3, 128)
    index_cache = torch.randn(384, 16, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(384, 1, 128, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn_like(k_cache)
    backend = QSAAttnBackend.__new__(QSAAttnBackend)
    backend.device = device
    backend.args = SimpleNamespace(index_compress_ratio=4, index_block_topk=4, index_head_dim=16)
    backend.config = SimpleNamespace(num_kv_heads=1, head_dim=128)
    backend.sm_scale = 128 ** -0.5
    backend._idx_slot = {0: 0}
    backend._fused_selection = True
    backend._fast_metadata = False
    backend.kvcache = SimpleNamespace(
        store_kv=lambda *args: None, store_index_k=lambda *args: None,
        index_k_cache=lambda slot: index_cache,
        k_cache=lambda layer: k_cache, v_cache=lambda layer: v_cache,
    )
    # Pooling is independent of selection; provide an already populated cache.
    backend._pool_completed_keys = lambda *args: None
    monkeypatch.setattr("sparklab.attention.qsa.get_global_ctx", lambda: SimpleNamespace(page_table=page_table))
    reqs = [SimpleNamespace(table_idx=i, cached_len=c, extend_len=n, device_len=c+n)
            for i, (c, n) in enumerate([(0, 40), (0, 5), (63, 1)])]
    batch = SimpleNamespace(reqs=reqs, out_loc=torch.arange(46, device=device))
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.dense_indices is None
    query = torch.randn(46, 2, 128, dtype=torch.bfloat16, device=device)
    index_query = torch.randn(46, 3, 16, dtype=torch.bfloat16, device=device)
    actual = backend.qsa_forward(query, None, None, index_query, None, None, None, 0, batch)

    rows = []
    for i, req in enumerate(reqs):
        q0, q1 = batch.attn_metadata.qo_indptr[i:i+2]
        physical = page_table[i, :req.device_len]
        pooled = index_cache[physical[3:req.device_len//4*4:4].long()]
        rows.extend(backend._selected_rows(iq, pooled, physical, req.cached_len+j+1)
                    for j, iq in enumerate(index_query[q0:q1]))
    counts = torch.tensor([r.numel() for r in rows], dtype=torch.int32, device=device)
    expected = paged_attention(
        q=query, k_cache=k_cache, v_cache=v_cache,
        indptr=torch.cat((counts.new_zeros(1), counts.cumsum(0))), indices=torch.cat(rows),
        q_to_req=batch.attn_metadata.q_to_req, q_positions=counts.long()-1,
        sm_scale=backend.sm_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
