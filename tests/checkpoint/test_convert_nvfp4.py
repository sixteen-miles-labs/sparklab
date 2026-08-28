from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from freetoken.checkpoint.convert import (
    _quantize_nvfp4_bank,
    _source_fingerprint,
    _write_config_override,
)
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4


def test_conversion_kda_override_is_scoped_and_written_to_nested_config(
    tmp_path, monkeypatch
):
    module = importlib.import_module("freetoken.checkpoint.convert")
    monkeypatch.setenv(module._GLM_KDA_QUANT_ENV, "previous")
    seen = []

    def fake_convert(*args, **kwargs):
        seen.append(module.os.environ[module._GLM_KDA_QUANT_ENV])
        return {"fingerprint": "test"}

    monkeypatch.setattr(module, "_convert_checkpoint", fake_convert)
    result = module.convert_checkpoint(
        "source", "output", kda_quantization="fp8_pertensor"
    )

    assert result == {"fingerprint": "test"}
    assert seen == ["fp8_pertensor"]
    assert module.os.environ[module._GLM_KDA_QUANT_ENV] == "previous"

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": {}}))
    _write_config_override(str(tmp_path), "freetoken_kda_quant", "fp8_pertensor")
    value = json.loads(config.read_text())
    assert value["freetoken_kda_quant"] == "fp8_pertensor"
    assert value["text_config"]["freetoken_kda_quant"] == "fp8_pertensor"


def test_source_fingerprint_distinguishes_glm_kda_artifact_quantization(
    tmp_path, monkeypatch
):
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"checkpoint")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 1))

    def config(kda_quant: str):
        return SimpleNamespace(
            expert_quant="nvfp4",
            architectures=["Glm5NextForConditionalGeneration"],
            glm5_next_args=SimpleNamespace(kda_quant=kda_quant),
        )

    original = _source_fingerprint(str(tmp_path), config("none"), device="cuda")
    optimized = _source_fingerprint(
        str(tmp_path), config("fp8_pertensor"), device="cuda"
    )

    assert original != optimized


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
