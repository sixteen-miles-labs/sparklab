from types import SimpleNamespace

from sparklab.runtime.engine.graph import GraphRunner


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
