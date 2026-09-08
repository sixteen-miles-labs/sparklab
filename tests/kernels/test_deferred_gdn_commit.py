"""Deferred GDN state must match canonical commits across graph replays and owners."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deferred_state_matches_materialized_verification_with_graph():
    from sparklab.models.config import LinearGatedDeltaGroupConfig
    from sparklab.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla
    from sparklab.runtime.kvcache.linear_state_pool import LinearStatePool

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1), num_key_heads=2, num_value_heads=4,
        key_head_dim=128, value_head_dim=128, conv_kernel_dim=4, output_gate=True,
    )
    pools = [LinearStatePool(group, 5, torch.bfloat16, torch.device("cuda"), 1) for _ in range(2)]
    normal, deferred = pools
    torch.manual_seed(39)
    normal.recurrent_states.normal_(std=0.1)
    normal.conv_states.normal_(std=0.1)
    deferred.recurrent_states.copy_(normal.recurrent_states)
    deferred.conv_states.copy_(normal.conv_states)
    for pool in pools:
        pool.enable_verify_transactions(5)
    deferred.enable_deferred_verify_commits()
    q = torch.randn(1, 5, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 5, 4, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(5, 4, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    alog = torch.zeros(4, device="cuda")
    bias = torch.zeros_like(alog)

    def run(pool):
        return [gdn_decode_fla(
            q, k, v, a, b, A_log=alog, dt_bias=bias,
            state_source=pool.recurrent_states[li], indices=indices,
            cu_seqlens=cu, scale=128 ** -0.5, disable_state_update=True,
            intermediate_states_buffer=pool.verify_recurrent_states[li],
            intermediate_state_indices=pool.verify_state_indices,
            cached_initial_state_step=pool.cached_initial_state_step,
        ) for li in range(2)]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(deferred)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = run(deferred)
    for iteration, (live, length) in enumerate([(1, 1), (1, 5), (1, 2), (1, 4), (2, 3), (2, 1), (1, 5)]):
        indices.fill_(live)
        for tensor in (q, k, v, a, b):
            tensor.normal_()
        normal.verify_conv_inputs.normal_()
        deferred.verify_conv_inputs.copy_(normal.verify_conv_inputs)
        for pool in pools:
            pool.prepare_verify_state(live)
            pool.snapshot_verify_inputs(live, 4)
        expected = run(normal)
        graph.replay()
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        torch.testing.assert_close(deferred.verify_recurrent_states, normal.verify_recurrent_states, atol=0, rtol=0)
        for pool in pools:
            pool.commit_verify_prefix(4, live, length)
        torch.testing.assert_close(deferred.conv_states, normal.conv_states, atol=0, rtol=0)
        if iteration == 2:
            # Snapshot consumers force materialization before copying the state.
            for pool in pools:
                pool.copy_from(live, 3)
            torch.testing.assert_close(deferred.recurrent_states, normal.recurrent_states, atol=0, rtol=0)
    deferred.materialize_verify_state()
    torch.testing.assert_close(deferred.recurrent_states, normal.recurrent_states, atol=0, rtol=0)
