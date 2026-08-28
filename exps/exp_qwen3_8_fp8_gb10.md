# Qwen3.8-Flash-Next official FP8 on NVIDIA GB10

Last updated: 2026-08-27

## Result

The complete pinned `Qwen/Qwen3.8-Flash-Next-FP8` checkpoint ran through the
OpenAI-compatible streaming API at **4.99 decode tok/s** with **0.580 s warm TTFT** on one
NVIDIA GB10. The recipe remains Experimental: throughput was 0.01 tok/s below the Frontier
floor, and the exact 64K context, capability, quality, and endurance gates were not run.

Compact evidence: [`GB10-QWEN38-FP8-001`](../benchmarks/gb10/results/GB10-QWEN38-FP8-001.json).

## Artifact and system

| Component | Value |
|---|---|
| Source | `Qwen/Qwen3.8-Flash-Next-FP8` |
| Revision | `970c569adaca6b35532111fd6b27351b2baefe50` |
| Prepared artifact | FTW FP8, 181,907,095,552 bytes, fingerprint `cb278ff3c47c29b2` |
| External n-gram bank | 51,200,245,760 bytes, FP8 E4M3, 320,001,536 x 160 |
| Validation | 144 source files verified; FTW structure and all n-gram payload bytes checked |
| Engine revision | `092d8e2e6b4a29803e0d99679e0ad217b3d97cc0`, clean tracked worktree |
| Hardware | NVIDIA GB10, SM121, 128 GB unified memory, NVMe |
| Software | Linux 6.17.0-1014-nvidia; CUDA 13.0; PyTorch 2.11.0+cu130; Triton 3.6.0 |

The conversion preserved the source checkpoint's native 128x128 block-FP8 expert weights
and BF16 inverse scales. It did not dequantize and requantize routed experts.

## Method

The selected run used AIME-25 problem 0, batch size one, greedy sampling, 64 requested
completion tokens, one warm request, and one measured request. Timing spans the streaming
HTTP API. The runtime used QSA attention, NVMe-backed MoE, 3 GiB of host expert cache,
16 disk readers, automatic expert-cache sizing, sparse prefill through 512 tokens, and no
CUDA graph.

```bash
FREETOKEN_DISK_READ_WORKERS=16 PYTHONPATH=python:. \
  .venv/bin/python benchmarks/bench_decode_moe.py \
  --model /path/to/Qwen3.8-Flash-Next-FP8-970c569a \
  --recipe qwen3.8-flash-next \
  --backend offload --storage disk --attention-backend qsa \
  --host-cache-gb 3 --prefill-hit-d2d --prefill-sparse-max-tokens 512 \
  --page-size 16 --cache-type naive --no-graph \
  --collect-moe-stats --include-output --greedy --decode 64
```

## Measurements

| Metric | Value |
|---|---:|
| Decode throughput | 4.987 tok/s |
| Warm TTFT | 0.580 s |
| Inter-token p50 / p99 | 205.98 / 346.22 ms |
| Device allocation | 77.99 GiB |
| Minimum host memory available | 27.67 GiB |
| Expert cache miss rate | 5.39% |
| Expert disk reads | 6,696 operations / 7.68 GiB |
| Output hash | `11b2b1eb274f` |

The API returned 63 completion tokens and produced 62 timed decode steps. The request read
516 pre-existing swap pages back into memory and swapped no pages out. Spark Lab therefore
records the run as measured evidence but does not grant admission on this non-zero-swap host.

## Fixes required to complete the run

The first attempts exposed three concrete runtime issues, all fixed before the selected
measurement: the recipe's 2 GiB host-cache budget was below the 2.34 GiB disk-prefill
minimum; expert scale rows of 200 and 400 bytes needed tail-safe aligned copies; and sparse
FP8 prefill needed to sort logical expert IDs and translate them to cache slots in-kernel.
The selected run completed without an OOM or service restart.
