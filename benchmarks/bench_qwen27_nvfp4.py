#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sweep GB10 W4A16 small-row split-K and K tiles for Qwen's dense projections."""
import json
import torch
import triton
from sparklab.kernels.triton import nvfp4_linear as k

def main():
    torch.manual_seed(17)
    original_pick = k._pick_split_k_bm16
    original_kw = k._GEMM_BLOCK_KW
    for n, inner in [(34816, 5120), (5120, 17408), (6144, 5120), (5120, 4096)]:
        w = torch.randint(0, 256, (n, inner // 2), dtype=torch.uint8, device="cuda")
        s = torch.ones((n, inner // 16), device="cuda").to(torch.float8_e4m3fn)
        g = torch.full((n,), 0.02, device="cuda", dtype=torch.float16)
        w, s = k.nvfp4_transpose_resident(w, s)
        a = torch.randn((8, inner), device="cuda", dtype=torch.bfloat16)
        ref = k.nvfp4_dense_linear_t(a, w, s, g)
        fn = lambda: k.nvfp4_dense_linear_t(a, w, s, g)
        baseline_ms = triton.testing.do_bench_cudagraph(fn, rep=120)
        rows = []
        try:
            for kw in [16, 32]:
                for split in [1, 2, 4, 8, 16]:
                    k._GEMM_BLOCK_KW = kw
                    k._pick_split_k_bm16 = lambda *_, sk=split: sk
                    out = fn()
                    ms = triton.testing.do_bench_cudagraph(fn, rep=120)
                    rows.append({"kw": kw, "split": split, "ms": ms,
                                 "max_abs_error": (out.float() - ref.float()).abs().max().item()})
        finally:
            k._GEMM_BLOCK_KW = original_kw
            k._pick_split_k_bm16 = original_pick
        rows.sort(key=lambda row: row["ms"])
        print(json.dumps({"n": n, "k": inner, "baseline_ms": baseline_ms, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
