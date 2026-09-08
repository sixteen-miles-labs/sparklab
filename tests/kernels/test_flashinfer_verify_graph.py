"""Verify graph replay retains causal, multi-query attention as KV length changes."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 4, 5])
def test_verify_graph_matches_eager_with_changing_prefix(monkeypatch, rows):
    pytest.importorskip("flashinfer")
    import sparklab.core as core
    from sparklab.core import Batch, Context, Req, SamplingParams
    from sparklab.attention.fi import FlashInferBackend
    from sparklab.models.config import FullAttentionGroupConfig
    from sparklab.runtime.distributed import set_tp_info, try_get_tp_info
    from sparklab.runtime.kvcache import create_kvcache_pool
    from tests.models.test_qwen3_5_mtp import _config
    from sparklab.models.qwen3_5_moe.config import parse_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    hf = _config()
    hf.text_config.head_dim = 256
    config = parse_config(hf)
    device = torch.device("cuda")
    page_size = 1
    ctx = Context(page_size=page_size)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    ctx.kv_cache = create_kvcache_pool(config, 128 // page_size, page_size,
                                      torch.bfloat16, device)
    pages = torch.randperm(128 // page_size, device=device)
    ctx.page_table = (pages[:, None] * page_size + torch.arange(page_size, device=device)).flatten().to(torch.int32)[None]
    backend = FlashInferBackend(config)
    ctx.attn_backend = backend
    layer = next(g.layer_ids[0] for g in config.attention_groups
                 if isinstance(g, FullAttentionGroupConfig))

    def batch_for(prefix):
        req = Req(torch.zeros(prefix + rows, dtype=torch.int32), 0, prefix,
                  8, 1, SamplingParams(), None)
        batch = Batch(reqs=[req], phase="verify")
        batch.padded_reqs = batch.reqs
        batch.input_ids = torch.zeros(rows, dtype=torch.int32, device=device)
        batch.out_loc = ctx.page_table[0, prefix:prefix + rows].clone()
        return batch

    # The scheduler plans metadata before attaching input tensors.
    unprepared = batch_for(7)
    del unprepared.input_ids
    backend.prepare_metadata(unprepared)

    torch.manual_seed(41)
    for cache in (ctx.kv_cache.k_cache(layer), ctx.kv_cache.v_cache(layer)):
        cache.copy_(torch.randn(cache.shape, dtype=torch.bfloat16, device=device))
    q = torch.randn(rows, 4, 256, dtype=torch.bfloat16, device=device)
    k = torch.randn(rows, 256, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    captured = batch_for(7)
    backend.init_capture_graph(max_seq_len=128, bs_list=[1])
    backend.prepare_for_capture(captured)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backend.forward(q, k, v, layer, captured)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = backend.forward(q, k, v, layer, captured)
    for prefix in (7, 43, 91, 2):
        q.normal_(); k.normal_(); v.normal_()
        batch = batch_for(prefix)
        backend.prepare_metadata(batch)
        expected = backend.forward(q, k, v, layer, batch).clone()
        # A new eager plan must not corrupt the captured wrapper's fixed buffers.
        replay = batch_for(prefix)
        backend.prepare_metadata(replay)
        captured.out_loc.copy_(replay.out_loc)
        backend.prepare_for_replay(replay)
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
