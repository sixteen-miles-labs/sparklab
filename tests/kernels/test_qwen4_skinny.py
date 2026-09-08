import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows", [1, 3])
def test_bf16_skinny_linear_matches_torch(rows: int) -> None:
    from sparklab.kernels.triton.qwen4_skinny import bf16_skinny_linear

    torch.manual_seed(7)
    x = torch.randn(rows, 96, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(65, 96, device="cuda", dtype=torch.bfloat16) * 0.1

    actual = bf16_skinny_linear(x, weight)
    expected = F.linear(x, weight)

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


def test_qwen4_skinny_linear_falls_back_on_cpu() -> None:
    from sparklab.kernels.triton.qwen4_skinny import qwen4_skinny_linear

    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)

    torch.testing.assert_close(
        qwen4_skinny_linear(x, weight, bias), F.linear(x, weight, bias)
    )


def test_glm53_lm_head_shape_uses_measured_sm121_plan() -> None:
    from sparklab.kernels.triton.qwen4_skinny import _SM121_PLANS

    assert (154_880, 4_096) in _SM121_PLANS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape,rows", [((9216, 2048), 1), ((9216, 2048), 3), ((2048, 4096), 2)])
def test_qwen3_draft_skinny_graph_matches_torch(monkeypatch, shape, rows):
    import sparklab.kernels.triton.qwen4_skinny as module
    monkeypatch.setenv("SPARKLAB_QWEN3_MTP_SKINNY", "1")
    monkeypatch.setattr(module, "_is_sm121", lambda _: True)
    torch.manual_seed(39)
    weight = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.01
    x = torch.randn(rows, shape[1], device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module.qwen4_skinny_linear(x, weight)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = module.qwen4_skinny_linear(x, weight)
    for _ in range(2):
        x.normal_()
        graph.replay()
        torch.testing.assert_close(actual, F.linear(x, weight), atol=2e-3, rtol=2e-2)
