from __future__ import annotations

import pytest
import torch

from freetoken.checkpoint.convert import _quantize_nvfp4_bank
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 converter requires CUDA")
def test_nvfp4_conversion_round_trip_preserves_expert_scale():
    torch.manual_seed(41)
    source = (torch.randn(2, 128, 256) * 0.1).to(torch.bfloat16)
    source[1] *= 4
    packed, scales, globals_ = _quantize_nvfp4_bank(source, torch.device("cuda"))

    assert packed.shape == (2, 128, 128) and packed.dtype == torch.uint8
    assert scales.shape == (2, 128, 16) and scales.dtype == torch.float8_e4m3fn
    assert globals_.shape == (2, 128) and globals_.dtype == torch.float16
    restored = dequant_nvfp4(
        packed.cuda(),
        scales.cuda(),
        globals_.cuda(),
        torch.arange(2, dtype=torch.int32, device="cuda"),
        dtype=torch.bfloat16,
    ).cpu()
    error = (source.float() - restored.float()).abs()
    relative_mae = error.mean() / source.float().abs().mean()
    assert relative_mae < 0.12
    torch.testing.assert_close(
        restored.float().abs().amax(dim=(1, 2)),
        source.float().abs().amax(dim=(1, 2)),
        rtol=0.08,
        atol=0.01,
    )
