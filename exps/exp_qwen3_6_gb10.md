# Qwen3.6-35B-A3B NVFP4 on NVIDIA GB10

Last updated: 2026-08-27

## Result

The complete pinned `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint ran through the
OpenAI-compatible streaming API at **67.46 decode tok/s** with **0.320 s warm TTFT** on one
NVIDIA GB10. This passes the Fast tier's performance thresholds, but the recipe remains
Experimental because the 32K context and 60-minute stability gates were not run and the
host had unrelated swap pages resident before launch.

The compact evidence is
[`GB10-QWEN36-FAST-001`](../benchmarks/gb10/results/GB10-QWEN36-FAST-001.json).

## Artifact and system

| Component | Value |
|---|---|
| Model | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Revision | `491c2f1ea524c639598bf8fa787a93fed5a6fbce` |
| Repository / indexed tensor bytes | 23,450,872,529 / 23,407,580,856 |
| Checksum validation | All 17 repository files passed `hf cache verify` |
| Engine revision | `7a785ce8dd1e3d2f3431ec4248567f0b637d4589`, clean tracked worktree |
| GPU | NVIDIA GB10, SM121, 128 GB unified memory |
| Software | CUDA 13.0; PyTorch 2.11.0+cu130; Triton 3.6.0 |

FreeToken retained the checkpoint's native mixed layout: NVFP4 routed/shared experts and
LM head, per-tensor FP8 attention projections, and the publisher's remaining source dtypes.

## Method

The selected run used AIME-25 problem 0, batch size one, greedy sampling, 64 requested
completion tokens, one warm request, and one measured request. Timing spans the real
streaming HTTP API rather than an isolated kernel. The runtime used all 10,240 routed-expert
cache slots in RAM, the Triton NVFP4 backend, an 8,192-token KV allocation, FlashInfer
attention selected by `auto`, hybrid radix caching, and a batch-size-one CUDA graph.

```bash
PYTHONPATH=python .venv/bin/python benchmarks/bench_decode_moe.py \
  --model /path/to/491c2f1ea524c639598bf8fa787a93fed5a6fbce \
  --recipe qwen3.6-35b-a3b \
  --backend offload --storage ram \
  --nvfp4-backend triton --cache-rate 1.0 \
  --num-tokens 8192 --prefill-hit-d2d \
  --collect-moe-stats --include-output --greedy --decode 64
```

The API returned 63 completion tokens, yielding 62 timed inter-token steps. This is the
same fixed-cap off-by-one behavior recorded in earlier serving experiments and is stated
explicitly rather than normalizing the count.

## Measurements

| Trial | Provenance / system isolation | tok/s | TTFT(s) | Output hash | Swap in / out |
|---|---|---:|---:|---|---:|
| Initial | Shared worktree became dirty during run | 65.70 | 0.326 | `2205dfe14c21` | 0 / 0 pages |
| Clean concurrent control | Clean worktree; Qwen3.8 download active | 67.19 | 1.116 | `2205dfe14c21` | 0 / 0 pages |
| **Selected isolated run** | **Clean worktree; competing transfer paused** | **67.46** | **0.320** | `2205dfe14c21` | **0 / 0 pages** |

The isolated run measured 14.824 ms/token, 14.356 ms p50 and 28.410 ms p99 inter-token
latency, 20.50 GiB device allocation, 61.69 GiB minimum `MemAvailable`, 39.89 W average
board power, and 50 C peak GPU temperature. All three runs reproduced the same greedy text
hash. The concurrent control demonstrates why its download-inflated TTFT was excluded from
the catalog while retaining it as an explicit control.

## Interpretation

The measured latency clears the Fast performance floor of 20 tok/s and 5 s warm TTFT by a
wide margin. It does not promote the recipe: the short probe did not reach its mathematical
final answer, usable 32K context was not tested, and no 60-minute agent trace was run.
Additionally, Spark Lab's doctor correctly reported `supported_not_ready` because about
1.94 GiB of swap from unrelated host activity existed before the experiment. The measured
request itself caused no swap-in or swap-out.
