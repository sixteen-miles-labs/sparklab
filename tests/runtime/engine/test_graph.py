from types import SimpleNamespace

import torch

from sparklab.runtime.engine.graph import (
    GraphRunner,
    MTPVerificationCaptureBuffer,
    MTPVerificationGraphRunner,
)


def test_cuda_graph_eligibility_delegates_to_attention_backend():
    runner = GraphRunner.__new__(GraphRunner)
    runner.max_graph_bs = 1
    batch = SimpleNamespace(is_decode=True, size=1)

    runner.attn_backend = SimpleNamespace(supports_cuda_graph=lambda _batch: True)
    assert runner.can_use_cuda_graph(batch)

    runner.attn_backend = SimpleNamespace(supports_cuda_graph=lambda _batch: False)
    assert not runner.can_use_cuda_graph(batch)


def test_cuda_graph_eligibility_keeps_phase_and_batch_guards():
    runner = GraphRunner.__new__(GraphRunner)
    runner.max_graph_bs = 1
    runner.attn_backend = SimpleNamespace(supports_cuda_graph=lambda _batch: True)

    assert not runner.can_use_cuda_graph(SimpleNamespace(is_decode=False, size=1))
    assert not runner.can_use_cuda_graph(SimpleNamespace(is_decode=True, size=2))


def test_mtp_verification_capture_buffer_separates_rows_from_requests():
    buffer = MTPVerificationCaptureBuffer.init(3, 16, torch.device("cpu"))
    batch = SimpleNamespace()

    buffer.set_batch(batch)

    assert batch.input_ids.shape == (3,)
    assert batch.linear_table_idx.shape == (1,)
    assert batch.fla_metadata.cu_seqlens.tolist() == [0, 3]
    assert batch.fla_metadata.has_initial_state.tolist() == [True]


def test_mtp_verification_graph_domain_is_fixed_width_and_dense():
    runner = MTPVerificationGraphRunner.__new__(MTPVerificationGraphRunner)
    runner.rows = 3
    runner.dflash = False
    runner.attn_backend = SimpleNamespace(supports_cuda_graph=lambda _batch: True)

    eligible = SimpleNamespace(
        is_verify=True, size=1, input_ids=torch.zeros(3, dtype=torch.int32)
    )
    assert runner.can_use_cuda_graph(eligible)
    eligible.input_ids = torch.zeros(2, dtype=torch.int32)
    assert not runner.can_use_cuda_graph(eligible)
    eligible.input_ids = torch.zeros(3, dtype=torch.int32)
    eligible.is_verify = False
    assert not runner.can_use_cuda_graph(eligible)


def test_dflash_capture_preserves_multi_query_attention_and_dynamic_positions():
    buffer = MTPVerificationCaptureBuffer.init(8, 16, torch.device("cpu"))
    metadata = buffer.dflash_metadata(128)
    assert not metadata.is_decode
    assert metadata.max_q_len == 8
    assert metadata.cu_seqlens_q_gpu.tolist() == [0, 8]
    assert metadata.q_to_req.tolist() == [0] * 8
    assert metadata.indices.shape == (128,)
    assert metadata.q_positions.data_ptr() == buffer.positions.data_ptr()
    buffer.positions.copy_(torch.arange(31, 39, dtype=torch.int32))
    assert metadata.q_positions.tolist() == list(range(31, 39))
