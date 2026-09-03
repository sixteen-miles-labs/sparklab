import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows", [1, 3, 6])
def test_dsv4_bf16_skinny_linear_matches_torch(rows: int) -> None:
    from sparklab.kernels.triton.dsv4.skinny import bf16_skinny_linear

    torch.manual_seed(17)
    x = torch.randn(rows, 96, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(65, 96, device="cuda", dtype=torch.bfloat16) * 0.1

    actual = bf16_skinny_linear(x, weight)
    expected = F.linear(x, weight)

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dsv4_markov_argmax_matches_torch() -> None:
    from sparklab.kernels.triton.dsv4.skinny import dsv4_markov_argmax

    torch.manual_seed(23)
    base = torch.randn(1, 1025, device="cuda", dtype=torch.bfloat16)
    markov = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1025, 256, device="cuda", dtype=torch.bfloat16)

    actual = dsv4_markov_argmax(base, markov, weight)
    expected = torch.argmax(base + F.linear(markov, weight), dim=-1)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 1),
    reason="needs SM121 CUDA",
)
def test_dsv4_small_m_fp8_plan_matches_general_plan(monkeypatch) -> None:
    from sparklab.kernels.triton.dsv4.fp8_linear import block_fp8_linear

    torch.manual_seed(29)
    x = torch.randn(5, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, 4096, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.full(
        (1024 // 128, 4096 // 128), 127, device="cuda", dtype=torch.uint8
    )

    actual = block_fp8_linear(x, weight, scale)
    monkeypatch.setenv("SPARKLAB_DISABLE_DSV4_SMALL_M", "1")
    expected = block_fp8_linear(x, weight, scale)

    torch.testing.assert_close(actual, expected)


def test_dsv4_skinny_helpers_fall_back_on_cpu() -> None:
    from sparklab.kernels.triton.dsv4.skinny import (
        dsv4_head_linear,
        dsv4_markov_argmax,
    )

    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    torch.testing.assert_close(dsv4_head_linear(x, weight), F.linear(x, weight))

    base = torch.randn(1, 3)
    markov = torch.randn(1, 4)
    torch.testing.assert_close(
        dsv4_markov_argmax(base, markov, weight),
        torch.argmax(base + F.linear(markov, weight), dim=-1),
    )
