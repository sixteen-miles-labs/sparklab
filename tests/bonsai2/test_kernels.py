import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(raw, kind):
    blocks = raw.cpu().reshape(-1, 34 if kind == 142 else 28)
    scale_bytes = blocks[:, :2] if kind == 142 else blocks[:, 26:28]
    scale = scale_bytes.contiguous().view(torch.float16).float()
    if kind == 142:
        q = blocks[:, 2:].int()
        vals = torch.stack(
            [((q >> (2 * i)) & 3) - 1 for i in range(4)], dim=-1
        ).reshape(-1, 128)
    else:
        q = blocks[:, :26].int()
        chunks = []
        # Independent scalar-layout transcription of Prism dequantize_row_ptq1_0.
        for start, length, digits in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
            for n in range(digits):
                chunks.append(
                    (((q[:, start : start + length] * 3**n) % 256) * 3 // 256) - 1
                )
        vals = torch.cat(chunks, dim=1)
    return (vals * scale).reshape(raw.shape[0], -1)


@pytest.mark.parametrize("kind", [142, 143])
@pytest.mark.parametrize("m", [1, 3, 17, 65])
def test_packed_linear_and_embedding(kind, m):
    from sparklab.kernels.triton.bonsai import embedding, linear

    gen = torch.Generator().manual_seed(42)
    size = 34 if kind == 142 else 28
    raw = torch.randint(0, 256, (64, 16, size), dtype=torch.uint8, generator=gen)
    scales = (torch.rand(64, 16, generator=gen) * 0.03 + 0.01).half()
    scale_offset = 0 if kind == 142 else 26
    raw[:, :, scale_offset : scale_offset + 2] = scales.view(torch.uint8).reshape(
        64, 16, 2
    )
    raw = raw.reshape(64, -1)
    w = reference(raw, kind).cuda()
    x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
    got = linear(x, raw.cuda(), kind)
    expected = x.float() @ w.T if m <= 4 else x.float() @ w.bfloat16().float().T
    torch.testing.assert_close(got.float(), expected, atol=0.035, rtol=0.015)
    ids = torch.tensor([0, 17, 63], device="cuda")
    torch.testing.assert_close(
        embedding(ids, raw.cuda(), kind, 2048), w[ids].bfloat16(), rtol=0, atol=0
    )


def test_rotation_matches_sylvester_and_inverse():
    from sparklab.kernels.triton.bonsai import rotate

    h = torch.ones(1, 1)
    for _ in range(10):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h = (h / 32).cuda()
    x = torch.randn(3, 2048, device="cuda")
    signs = torch.where(torch.arange(2048, device="cuda") % 3 == 0, -1.0, 1.0)
    expected = ((x * signs).reshape(-1, 1024) @ h).reshape_as(x)
    torch.testing.assert_close(rotate(x, signs), expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(
        rotate(rotate(x, signs), signs, inverse=True), x, rtol=2e-4, atol=2e-5
    )


@pytest.mark.parametrize("width", [384, 5120, 17408])
def test_ptq_block_reduction_with_fallback_and_model_widths(width):
    from sparklab.kernels.triton.bonsai import linear

    gen = torch.Generator().manual_seed(73)
    raw = torch.randint(
        0, 256, (32, width // 128, 28), dtype=torch.uint8, generator=gen
    )
    scales = (torch.rand(32, width // 128, generator=gen) * 0.02 + 0.01).half()
    raw[:, :, 26:28] = scales.view(torch.uint8).reshape(32, width // 128, 2)
    raw = raw.reshape(32, -1)
    weight = reference(raw, 143).cuda()
    x = torch.randn(1, width, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(
        linear(x, raw.cuda(), 143).float(),
        x.float() @ weight.T,
        atol=0.035,
        rtol=0.015,
    )
