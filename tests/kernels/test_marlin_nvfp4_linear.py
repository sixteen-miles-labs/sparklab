import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 4])
def test_marlin_nvfp4_linear_matches_native_and_replays(rows):
    pytest.importorskip("vllm")
    from sparklab.layers.marlin_linear import MarlinNVFP4Linear
    from sparklab.kernels.triton.nvfp4_linear import nvfp4_dense_linear

    torch.manual_seed(29)
    n, k = 512, 256
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randint(1, 64, (n, k // 16), device="cuda").to(torch.float8_e4m3fn)
    global_scale = torch.full((n,), .001, device="cuda", dtype=torch.float16)
    original = [t.view(torch.uint8).clone() for t in (weight, scales, global_scale)]
    layer = MarlinNVFP4Linear(weight, scales, global_scale)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer.forward(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = layer.forward(x)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        x.normal_()
        graph.replay()
        reference = nvfp4_dense_linear(x, weight, scales, global_scale)
        rel = (output.float() - reference.float()).norm() / reference.float().norm()
        assert rel < .01
    for actual, expected in zip((weight, scales, global_scale), original):
        torch.testing.assert_close(actual.view(torch.uint8), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="one global scale"):
        MarlinNVFP4Linear(weight, scales, torch.arange(n, device="cuda").float())
