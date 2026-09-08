"""Per-tensor FP8 linear: the W8A16 kernels against the dequant reference, and the W8A8
(torch._scaled_mm) path against a W8A8 reference.

Which of the two runs is fixed by the deployment -- an ``input_scale`` in the checkpoint plus
sm_89+ -- and never by M, so each is checked against the reference matching its own contract:
W8A16 keeps the activation in bf16 and matches the exact dequant reference, W8A8 quantizes it
to fp8 and is held to a reference that applies the same quantization.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from sparklab.kernels.triton.e4m3_compat import e4m3_native

DEV = "cuda"
FP8 = torch.float8_e4m3fn


def _quant_parts(part_rows: list[int], K: int, seed: int = 0):
    """A fused per-tensor-FP8 weight: each part quantized under its own scalar, exactly how
    modelopt stores q/k/v (and how the loader concatenates them into a per-row vector)."""
    torch.manual_seed(seed)
    N = sum(part_rows)
    wf = torch.randn(N, K, device=DEV) * 0.05
    rows, qs, scales = 0, [], []
    for p in part_rows:
        block = wf[rows : rows + p]
        s = block.abs().max() / 448.0
        qs.append((block / s).clamp(-448, 448).to(FP8))
        scales.append(s.expand(p))
        rows += p
    return torch.cat(qs, 0), torch.cat(scales).contiguous().float()


def _dequant(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w8.to(torch.float32) * scale[:, None]


# 1 = split-K GEMV; 2..16 = decode batch (the CUDA-graph ladder); 64/300 = prefill.
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 64, 300])
# (5120, 14336) and (5120, 16384) are Qwen3.5-27B's fused qkv_proj / in_proj_qkvz;
# (1024, 6144) is a standalone o_proj shape; 6112 leaves a k-mask tail.
@pytest.mark.parametrize("K,part_rows", [
    (5120, [12288, 1024, 1024]),
    (1024, [6144]),
    (6112, [512, 128]),
])
def test_w8a16_matches_dequant_reference(M: int, K: int, part_rows: list[int]):
    """Without an ``input_scale`` every M stays on the W8A16 kernels, activation exact."""
    from sparklab.kernels.triton.fp8_pertensor_linear import fp8_pertensor_linear

    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    y = fp8_pertensor_linear(x, w8, scale)
    y_ref = (x.float() @ _dequant(w8, scale).t()).to(torch.bfloat16)
    rel = (y.float() - y_ref.float()).abs().max() / y_ref.float().abs().max().clamp(min=1e-6)
    assert rel.item() < 2e-2, rel.item()


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
@pytest.mark.parametrize("M", [1, 2, 4, 16, 64])
@pytest.mark.parametrize("part_rows,uniform", [
    ([12288, 1024, 1024], False),  # fused -> piecewise-constant scale -> row-wise
    ([6144], True),                # standalone -> one scalar -> tensor-wise
])
def test_w8a8_matches_w8a8_reference(M: int, part_rows: list[int], uniform: bool):
    from sparklab.kernels.triton.fp8_pertensor_linear import fp8_pertensor_linear

    K = 2048
    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    y = fp8_pertensor_linear(x, w8, scale, None, input_scale, uniform)

    xq = (x.float() / input_scale).clamp(-448, 448).to(FP8)
    y_ref = (xq.to(torch.float32) * input_scale) @ _dequant(w8, scale).t()
    rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
    assert rel < 1e-2, rel


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
def test_batch_size_does_not_change_the_numeric_scheme():
    """A deployment that can run W8A8 must run it at every M, so that a reply reproduces at
    bs=1 regardless of how many other requests shared its forward. Feeding the same row alone
    and as part of a batch must therefore agree bit-for-bit."""
    from sparklab.kernels.triton.fp8_pertensor_linear import fp8_pertensor_linear

    K, part_rows = 2048, [1024, 256]
    w8, scale = _quant_parts(part_rows, K)
    x = torch.randn(8, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    batched = fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    alone = fp8_pertensor_linear(x[:1], w8, scale, None, input_scale, False)
    assert torch.equal(alone, batched[:1])


def test_layer_load_marks_uniform_scale_and_optional_input_scale():
    from sparklab.kernels.triton.fp8_pertensor_linear import (
        Fp8PerTensorColMerged,
        Fp8PerTensorLinear,
    )

    K = 512
    w8, scale = _quant_parts([256, 64, 64], K)
    merged = Fp8PerTensorColMerged(K, [256, 64, 64])
    merged.load_state_dict({
        "weight": w8, "weight_scale": scale, "input_scale": torch.tensor(0.01, device=DEV),
    })
    assert merged._uniform_scale is False
    assert merged.input_scale is not None

    single = Fp8PerTensorLinear(K, 384)
    flat = scale[:1].expand(384).contiguous()
    single.load_state_dict({"weight": w8, "weight_scale": flat})  # no input_scale
    assert single._uniform_scale is True
    assert single.input_scale is None

    # a reload must not trip over the input_scale it kept from the first load
    single.load_state_dict({"weight": w8, "weight_scale": flat})
    assert single.input_scale is None


@pytest.mark.parametrize("rows", [1, 4, 5, 8])
@pytest.mark.parametrize("uniform", [False, True])
def test_vllm_skinny_fp8_preserves_scales_and_graph_replay(monkeypatch, rows, uniform):
    pytest.importorskip("vllm")
    from sparklab.kernels.triton.fp8_pertensor_linear import _scaled_mm

    torch.manual_seed(63)
    n, k = 512, 256
    w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((n,), .02, device="cuda", dtype=torch.float32)
    if not uniform:
        scales[n // 2:] = .07
    input_scale = torch.tensor(.013, device="cuda", dtype=torch.float32)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setenv("SPARKLAB_FP8_MM_BACKEND", "vllm")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _scaled_mm(x, w, scales, input_scale, uniform, x.dtype)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = _scaled_mm(x, w, scales, input_scale, uniform, x.dtype)
    torch.cuda.current_stream().wait_stream(stream)
    monkeypatch.setenv("SPARKLAB_FP8_MM_BACKEND", "torch")
    for _ in range(2):
        x.normal_()
        graph.replay()
        expected = _scaled_mm(x, w, scales, input_scale, uniform, x.dtype)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
