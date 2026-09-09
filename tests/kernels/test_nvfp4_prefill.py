import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _dequant(packed, scales, global_scale):
    lut = torch.tensor(
        [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    values = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1)
    values = values.reshape(packed.shape[0], -1)
    return values * scales.float().repeat_interleave(16, dim=-1) * global_scale[:, None]


@pytest.mark.parametrize("rows", [128, 257])
def test_prefill_matches_independently_dequantized_fp4_operands(rows):
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("GB10 required")
    flashinfer = pytest.importorskip("flashinfer")
    from flashinfer.quantization import SfLayout
    from sparklab.kernels.triton.nvfp4_linear import Nvfp4DenseColMerged

    torch.manual_seed(83)
    n, k = 256, 256
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randint(1, 12, (n, k // 16), device="cuda").to(torch.float8_e4m3fn)
    global_scale = torch.full((n,), .01, device="cuda", dtype=torch.float16)
    global_scale[128:] *= 4
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    original = weight.clone()
    layer = Nvfp4DenseColMerged(k, [128, 128], has_bias=True, prefill_backend="flashinfer")
    layer.load_state_dict({"weight": weight, "weight_scale": scales,
                           "weight_global": global_scale, "bias": bias})
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    inverse = 2688.0 / x.float().abs().amax().reshape(1)
    packed_x, scales_x = flashinfer.nvfp4_quantize(x, inverse, sfLayout=SfLayout.layout_linear)
    scales_x = scales_x.view(torch.float8_e4m3fn).reshape(rows, k // 16)
    x_ref = _dequant(packed_x, scales_x, inverse.reciprocal().expand(rows))
    w_ref = _dequant(weight, scales, global_scale.float())
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = (x_ref @ w_ref.T).to(torch.bfloat16) + bias
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    actual = layer.forward(x)
    relative_error = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative_error < .008
    torch.testing.assert_close(weight, original, atol=0, rtol=0)
    torch.testing.assert_close(layer.forward(torch.zeros_like(x)), bias.expand(rows, -1), atol=0, rtol=0)


def test_small_rows_keep_native_numerics_and_graph_replay():
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("GB10 required")
    pytest.importorskip("flashinfer")
    from sparklab.kernels.triton.nvfp4_linear import Nvfp4DenseLinear

    torch.manual_seed(84)
    n, k = 256, 256
    state = {
        "weight": torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8),
        "weight_scale": torch.ones(n, k // 16, device="cuda").to(torch.float8_e4m3fn),
        "weight_global": torch.full((n,), .01, device="cuda", dtype=torch.float16),
    }
    native = Nvfp4DenseLinear(k, n)
    fast = Nvfp4DenseLinear(k, n, prefill_backend="flashinfer")
    native.load_state_dict(dict(state))
    fast.load_state_dict(dict(state))
    for rows in (1, 12, 127):
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(fast.forward(x), native.forward(x), atol=0, rtol=0)
    x = torch.randn(12, k, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fast.forward(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = fast.forward(x)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        x.normal_()
        graph.replay()
        torch.testing.assert_close(actual, native.forward(x), atol=0, rtol=0)
