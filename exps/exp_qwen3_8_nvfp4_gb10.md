# Qwen3.8-Flash-Next publisher NVFP4 on NVIDIA GB10

Last updated: 2026-08-28

## Result

The complete pinned `Inferact/Qwen3.8-Flash-Next-NVFP4` checkpoint was prepared into
SparkLab's FTW layout and ran through the OpenAI-compatible streaming API at **12.58
decode tok/s** with **0.786 s warm TTFT** on one NVIDIA GB10. Both numbers pass the
Frontier performance thresholds. The recipe remains Experimental because the exact 64K
context, capability, quality, and endurance gates were not run.

Compact evidence:
[`GB10-QWEN38-NVFP4-001`](../benchmarks/gb10/results/GB10-QWEN38-NVFP4-001.json).

## Artifact and system

| Component | Value |
|---|---|
| Source | `Inferact/Qwen3.8-Flash-Next-NVFP4` |
| Revision | `103a7608316173ca6edd49929544244de7ffda70` |
| Prepared artifact | FTW NVFP4, 180,433,108,992 bytes total, fingerprint `47e11ddb878adf4c` |
| FTW tensor store | 78,032,617,472 physical bytes, 10 shards, 1,175 tensors |
| External n-gram bank | 102,400,491,520 bytes, BF16, 320,001,536 x 160 |
| Validation | 21 source shards and all prepared tensors validated |
| Engine revision | `d979d9952cabd8b70cd3a65a1991f2543e5c0358`, clean tracked worktree |
| Hardware | NVIDIA GB10, SM121, 128 GB unified memory, NVMe |
| Software | Linux 6.17.0-1014-nvidia; CUDA 13.0; PyTorch 2.11.0+cu130; Triton 3.6.0 |

Preparation preserved the publisher's ModelOpt NVFP4 routed experts without
conversion-time requantization. Other served tensors retain their published precision.

## Method

The selected run used AIME-25 problem 0, batch size one, greedy sampling, 64 requested
completion tokens, one warm request, and one measured request. Timing spans the streaming
HTTP API. The runtime used QSA attention, NVMe-backed MoE, 3 GiB of host expert cache,
16 disk readers, automatic expert-cache sizing, sparse prefill through 512 tokens, and no
CUDA graph.

```bash
CUDA_VISIBLE_DEVICES=0 SPARKLAB_DISK_READ_WORKERS=16 PYTHONPATH=python:. \
  .venv/bin/python benchmarks/bench_decode_moe.py \
  --model /path/to/qwen3.8-flash-next/prepared/0.5.0 \
  --recipe qwen3.8-flash-next \
  --backend offload --storage disk --attention-backend qsa \
  --nvfp4-backend triton --host-cache-gb 3 \
  --prefill-hit-d2d --prefill-sparse-max-tokens 512 \
  --page-size 16 --cache-type naive --no-graph \
  --collect-moe-stats --include-output --greedy --decode 64
```

## Measurements

| Metric | Value |
|---|---:|
| Decode throughput | 12.584 tok/s |
| Warm TTFT | 0.786 s |
| Inter-token p50 / p99 | 78.21 / 97.31 ms |
| Device allocation | 87.46 GiB |
| Minimum host memory available | 18.26 GiB |
| Expert cache miss rate | 0.241% |
| Expert disk reads | 1,392 operations / 0.600 GiB |
| Output hash | `d8a4f705f746` |

The API returned 63 completion tokens and produced 62 timed decode steps. The request
swapped five pages in and no pages out. It completed without an OOM or service restart.
The 64-token cap stopped before a final mathematical answer, so this run is performance
and serving evidence rather than task-accuracy evidence.

## Comparison

The superseded official FP8 recipe measured 4.99 tok/s and 0.580 s warm TTFT. This
publisher-NVFP4 FTW artifact is 2.52x faster during decode, with a lower expert-cache miss
rate (0.24% versus 5.39%) and less physical expert I/O (0.60 GiB versus 7.68 GiB). Warm
TTFT is 0.206 s slower. The checkpoint and recipe tuples differ, so the comparison is
informative rather than a controlled quantization-only experiment.
