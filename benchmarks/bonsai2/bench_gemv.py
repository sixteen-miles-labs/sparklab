"""Compare scalar unpacking and block-reuse GEMV on real Bonsai projections."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton
from sparklab.kernels.triton.bonsai import _gemv, _gemv_blocks
from sparklab.models.gguf.reader import iter_gguf_tensors


def baseline(x, w, kind):
    n, k = w.shape[0], x.shape[-1]
    bk = 1024 if k % 1024 == 0 else 128
    splits = k // bk
    partial = torch.empty((1, n, splits), device=x.device, dtype=torch.float32)
    _gemv[(n // 4, splits, 1)](x, w, partial, k, n, kind, bk, 4, splits)
    return partial.sum(-1).to(x.dtype)


def candidate(x, w, kind, groups, bn, warps, buffer):
    n, k = w.shape[0], x.shape[-1]
    splits = k // (groups * 128)
    _gemv_blocks[(n // bn, splits, 1)](
        x, w, buffer, k, n, kind, groups, bn, splits, num_warps=warps
    )
    return buffer.sum(-1).to(x.dtype)


def run(a):
    wanted = {
        "output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_qkv.weight",
    }
    results = []
    for t in iter_gguf_tensors(str(a.model)):
        if t.name not in wanted:
            continue
        w = t.packed().cuda()
        n, k = t.shape
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        expected = baseline(x, w, t.ggml_type)
        old_ms = triton.testing.do_bench_cudagraph(
            partial(baseline, x, w, t.ggml_type), rep=100
        )
        variants = []
        for groups in (4, 8, 16):
            if k % (groups * 128):
                continue
            splits = k // (groups * 128)
            p = torch.empty((1, n, splits), device="cuda", dtype=torch.float32)
            for bn in a.rows:
                for warps in (4, 8):
                    f = partial(candidate, x, w, t.ggml_type, groups, bn, warps, p)

                    got = f()
                    torch.testing.assert_close(got, expected, rtol=0.016, atol=0.002)
                    ms = triton.testing.do_bench_cudagraph(f, rep=100)
                    variants.append(
                        {
                            "groups": groups,
                            "rows": bn,
                            "warps": warps,
                            "ms": ms,
                            "max_error": float(
                                (got.float() - expected.float()).abs().max()
                            ),
                        }
                    )
        result = {
            "tensor": t.name,
            "shape": list(t.shape),
            "old_ms": old_ms,
            "variants": sorted(variants, key=lambda v: v["ms"]),
        }
        results.append(result)
        a.output.write_text(json.dumps(results, indent=2) + "\n")
        print(t.name, old_ms, result["variants"][:3], flush=True)
        del w, x, expected
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[4, 8, 16])
    run(p.parse_args())
