"""CLI: convert an HF safetensors checkpoint to a FreeToken Weight (FTW) checkpoint.

    ft checkpoint --model <hf_dir> --out <ftw_dir> \
        [--dtype bfloat16] [--moe-backend offload] [--nvfp4-backend triton] \
        [--shard-gib 8]

The output dir is self-contained: point the server's ``--model`` at it to load via the FTW
fast path (auto-detected).
"""

from __future__ import annotations

import argparse
import time

import torch

from .convert import convert_checkpoint

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main(argv: list[str] | None = None, prog: str = "freetoken.checkpoint") -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("--model", required=True, help="source HF safetensors checkpoint dir")
    p.add_argument("--out", required=True, help="output FTW checkpoint dir")
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    p.add_argument("--moe-backend", default="offload",
                   help="offload (experts -> banks) or e.g. triton (experts stay dense)")
    p.add_argument(
        "--nvfp4-backend",
        choices=("auto", "marlin", "flashinfer", "triton"),
        default="triton",
        help="NVFP4 expert layout/kernel to bake into FTW (default: triton)",
    )
    p.add_argument("--shard-gib", type=float, default=8.0, help="max shard size in GiB")
    p.add_argument("--device", default=None, help="CUDA device for repack (default cuda:0)")
    ns = p.parse_args(argv)

    shard_limit = int(ns.shard_gib * (1 << 30))
    shard_limit -= shard_limit % 4096  # keep aligned
    t = time.perf_counter()
    index = convert_checkpoint(
        ns.model, ns.out, dtype=_DTYPES[ns.dtype],
        moe_backend=ns.moe_backend, nvfp4_backend=ns.nvfp4_backend,
        shard_limit=shard_limit, device=ns.device,
    )
    dt = time.perf_counter() - t
    c = index["counts"]
    gib = index["total_bytes"] / (1 << 30)
    print(f"\nwrote FTW checkpoint -> {ns.out}")
    print(f"  tensors: {c['weight']} weight + {c['experts_bank']} experts_bank")
    print(f"  FTW: {gib:.2f} GiB across {len(index['shards'])} shard(s) "
          f"(<= {ns.shard_gib} GiB each)")
    print(f"  quant_format: {index['quant_format']}  fingerprint={index['fingerprint']}")
    print(f"  converted in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
