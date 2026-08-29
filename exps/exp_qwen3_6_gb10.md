# Qwen3.6-35B-A3B NVFP4 on NVIDIA GB10

Last updated: 2026-08-29

## Result

The pinned `oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW` artifact is **Fast-certified** on one
NVIDIA GB10. The certification performance probe measured **67.79 decode tok/s** and
**0.329 s warm TTFT**. The same artifact passed exact 32,768-token recall, reasoning/tool/
coding capability probes, bounded resident memory, and an uninterrupted 60.28-minute
endurance run with 122 requests, zero swap growth, zero OOMs, and zero parser failures.

The certification evidence is
[`GB10-QWEN36-FAST-002`](../benchmarks/gb10/results/GB10-QWEN36-FAST-002.json). The earlier
[`GB10-QWEN36-FAST-001`](../benchmarks/gb10/results/GB10-QWEN36-FAST-001.json) remains the
historical source-checkpoint performance probe.

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

The FTW certification clears every operational Fast gate by a wide margin. Its fixed
five-problem AIME quality sample scored 0/5 because all five runs exhausted the 2,048-token
reasoning budget before emitting a final answer. That result is retained explicitly:
certification establishes complete-checkpoint serving correctness, parser behavior, exact
32K context, bounded memory, Fast latency, and endurance—not a model-quality promise.
