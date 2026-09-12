import pytest
import torch

from sparklab.models.deepseek_v41.ops import dequant, fp8_roundtrip


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("rows,out_features,in_features", [(1, 96, 256), (3, 65, 256)])
def test_mxfp4_linear_matches_reference(rows, out_features, in_features):
    from sparklab.kernels.triton.dsv41_linear import mxfp4_linear

    torch.manual_seed(411)
    x = torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda")
    packed = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8, device="cuda")
    codes = torch.randint(120, 132, (out_features, in_features // 32), dtype=torch.uint8, device="cuda")
    rounded = fp8_roundtrip(x)
    expected = torch.nn.functional.linear(
        rounded.float(), dequant(packed, codes.view(torch.float8_e8m0fnu), fp4=True)
    ).to(torch.bfloat16)
    actual = mxfp4_linear(rounded, packed, codes)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1.5e-2)


def test_v41_head_skinny_linear_matches_fp32_reference():
    from sparklab.kernels.triton.dsv4.skinny import bf16_skinny_linear

    torch.manual_seed(412)
    x = torch.randn(1, 5120, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(129280, 5120, dtype=torch.bfloat16, device="cuda")
    actual = bf16_skinny_linear(x, weight, out_dtype=torch.float32)
    expected = torch.nn.functional.linear(x.float(), weight.float())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-3)


def test_dense_tile_scales_expand_once_per_output_row():
    from types import SimpleNamespace

    from sparklab.models.deepseek_v41.weight import DiskWeights

    store = DiskWeights.__new__(DiskWeights)
    store.derived = {}
    store.fds = {}
    store.cache = {}
    store.cache_bytes = 0
    compact = torch.tensor([[120, 121], [122, 123]], dtype=torch.uint8, device="cuda")
    store.get = lambda name: compact.view(torch.float8_e8m0fnu)
    actual = store.linear_scale_codes("scale", 33)
    assert actual.shape == (33, 2)
    torch.testing.assert_close(actual[:32], compact[:1].expand(32, 2))
    torch.testing.assert_close(actual[32:], compact[1:2])
    assert store.linear_scale_codes("scale", 33).data_ptr() == actual.data_ptr()
