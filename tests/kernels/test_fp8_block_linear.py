"""Numerical coverage for the generic block-FP8 dense linear kernels."""

from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))


@pytest.mark.parametrize("tokens", [1, 8])
def test_block_fp8_linear_accepts_fp32_checkpoint_scales(tokens: int):
    """Kimi K3 stores resident block scales as FP32, for decode and prefill."""
    from sparklab.kernels.triton.fp8_block_linear import block_fp8_linear

    torch.manual_seed(20260828 + tokens)
    n = k = 256
    weight = (torch.randn(n, k, device="cuda") * 1.5).to(torch.float8_e4m3fn)
    scale = torch.empty(n // 128, k // 128, dtype=torch.float32, device="cuda").uniform_(0.002, 0.02)
    x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")

    actual = block_fp8_linear(x, weight, scale).float()
    dequant = (
        weight.float().reshape(n // 128, 128, k // 128, 128)
        * scale[:, None, :, None]
    ).reshape(n, k)
    expected = x.float() @ dequant.T

    assert torch.isfinite(actual).all()
    rel_l2 = _relative_l2(actual, expected)
    cosine = float(torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0,
    ))
    if tokens == 1:  # W8A16 split-K decode
        assert rel_l2 < 0.01
        assert cosine > 0.9999
    else:  # dynamically quantized W8A8 prefill
        assert rel_l2 < 0.08
        assert cosine > 0.995
