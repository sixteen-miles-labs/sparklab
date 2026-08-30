# Qwen3.8-Flash-Next publisher NVFP4 on NVIDIA GB10

Last updated: 2026-08-30

## Result

The complete pinned `Inferact/Qwen3.8-Flash-Next-NVFP4` checkpoint was prepared into
SparkLab's FTW layout and ran through the OpenAI-compatible streaming API at **16.06
decode tok/s** with **0.434 s warm TTFT** on one NVIDIA GB10. Recipe 0.6.0 preloads all
24,576 routed experts into deterministic immutable slots, removing steady-state expert
disk staging, the pageable host LRU, and decode-time cache ownership updates. A 128K-token
KV cap retains operating-system headroom. Relative to the previous 12.58 tok/s result,
throughput improved by 27.6% and warm TTFT fell by 44.7%.

Both numbers pass the Frontier performance thresholds. Exact 64K recall, reasoning and
tool parsing, a coding-agent task, and the fixed five-problem quality sample also passed.
The recipe remains Experimental because the optimization result came from a dirty
worktree and the clean-revision 60-minute endurance gate was not run.

Compact evidence:
[`GB10-QWEN38-NVFP4-OPT-001`](../benchmarks/gb10/results/GB10-QWEN38-NVFP4-OPT-001.json).

## Artifact and system

| Component | Value |
|---|---|
| Source | `Inferact/Qwen3.8-Flash-Next-NVFP4` |
| Revision | `103a7608316173ca6edd49929544244de7ffda70` |
| Prepared artifact | FTW NVFP4, 180,433,108,992 bytes total, fingerprint `47e11ddb878adf4c` |
| FTW tensor store | 78,032,617,472 physical bytes, 10 shards, 1,175 tensors |
| External n-gram bank | 102,400,491,520 bytes, BF16, 320,001,536 x 160 |
| Validation | 21 source shards and all prepared tensors validated |
| Engine revision | `8544aa517b0f9adc424985d3c399cdc0f4e16444`, dirty optimization worktree |
| Hardware | NVIDIA GB10, SM121, 128 GB unified memory, NVMe |
| Software | Linux 6.17.0-1014-nvidia; CUDA 13.0; PyTorch 2.11.0+cu130; Triton 3.6.0 |

Preparation preserved the publisher's ModelOpt NVFP4 routed experts without
conversion-time requantization. Other served tensors retain their published precision.

## Method

The selected run used AIME-25 problem 0, batch size one, greedy sampling, 64 requested
completion tokens, one warm request, and one measured request. Timing spans the streaming
HTTP API. The runtime used QSA attention, a complete immutable 24,576-slot expert cache,
no pageable host expert cache, a 131,072-token KV capacity, and no CUDA graph. The
one-time expert preload took 35.78 seconds.

```bash
CUDA_VISIBLE_DEVICES=0 SPARKLAB_DISK_READ_WORKERS=16 PYTHONPATH=python:. \
  .venv/bin/python benchmarks/bench_decode_moe.py \
  --model /path/to/qwen3.8-flash-next/prepared/0.5.0 \
  --recipe qwen3.8-flash-next \
  --backend offload --storage disk --attention-backend qsa \
  --nvfp4-backend triton --host-cache-gb 0 --preload-all \
  --num-tokens 131072 \
  --page-size 16 --cache-type naive --no-graph \
  --include-output --greedy --decode 64
```

The optimized path combines several changes:

- an arbitrary-top-k fused router handles Qwen3.8's top-10 selection without the eager
  softmax/top-k/divide/cast chain (6.19 microseconds versus 38.79 microseconds in the
  isolated GB10 router probe);
- full expert preload gives every `(layer, expert)` a fixed physical slot, so decode and
  prefill bypass cache lookup, eviction, ownership updates, host-LRU work, and expert I/O;
- dense QSA metadata is built once per batch, completed four-token index-key pools are
  materialized once in the existing cache, and dense attention skips the unused query
  projection;
- sparse QSA scoring and chronological row expansion use fused Triton kernels, while
  completed key groups gather only their new physical rows instead of converting the
  complete page-table prefix on every prefill chunk;
- one-token PLE hashing is span-local, while its dilated depthwise convolution, SiLU, and
  state update run in one kernel;
- grouped plus-one RMSNorm and the Hyper-Connection scale/SiLU, four-stream gate/reduce,
  injection-gate, and residual-update chains are fused with reference-exact BF16 rounding;
- the exact NVFP4 decode tile uses a wider K step that improved Qwen3.8's routed-expert
  microbenchmark by 4.4%;
- on integrated GB10 memory, startup advises Linux to discard clean FTW/n-gram page cache
  before solving the unified-memory cache budget; and
- the recipe caps KV capacity at 128K tokens instead of filling all otherwise-free memory.

## Measurements

| Metric | Value |
|---|---:|
| Decode throughput | 16.061 tok/s |
| Warm TTFT | 0.434 s |
| Inter-token p50 / p99 | 61.82 / 71.43 ms |
| Device allocation | 76.43 GiB |
| Minimum host memory available | 32.46 GiB |
| Expert cache miss rate | 0% by immutable mapping |
| Routed-expert disk reads | 0 during steady-state decode |
| Output hash | `c1c0854490c6` |

The API returned 64 completion tokens and produced 63 timed decode steps. The measured
request grew neither swap nor expert disk I/O and completed without an OOM or service
restart. Its hash exactly matched the immediate pre-optimization control. The 64-token cap
stopped before a final mathematical answer, so this run is performance and serving
evidence rather than task-accuracy evidence.

Longer diagnostics exercised correctness, quality, and the sparse path:

- A controlled 4,193-token prompt plus 512-token decode measured 14.303 tok/s with fused
  QSA selection versus 13.888 tok/s with the eager selector, a 3.0% gain. The greedy
  hashes diverged after accumulated selector floating-point differences, so exact recall
  and task quality—not hash equality—are the acceptance checks for this approximate top-k
  path.
- The fixed AIME-25 five-problem sample scored 3/5, with both misses exhausting the
  2,048-token output budget. It sustained 14.365 token-weighted tok/s, averaged 0.791 s
  TTFT, and completed 9.60 uninterrupted minutes with zero routed-expert reads, swap
  growth, OOMs, restarts, or parser failures.
- An exact 65,536-token sparse-QSA recall run returned `SPARK-7319` in 208.58 seconds,
  retained 33.59 GiB of available memory, and recorded zero routed-expert disk reads,
  zero swap growth, and zero OOM kills. Including server startup and the one-time expert
  preload, the benchmark completed in 263.74 seconds.
- Separate capability probes passed reasoning-channel parsing, structured tool calls, and
  an executable coding-agent task.
- A separate 512-token Hyper-Connection A/B measured 15.79 tok/s fused versus 15.24
  tok/s eager, a 3.6% gain, with the identical `da138b79cff5` greedy output hash.

## Comparison

The immediate pre-optimization control at engine revision `8544aa5` measured 12.92 tok/s
and 0.771 s warm TTFT with the same artifact, prompt, sampling, and host. The optimized
dynamic-cache path measured 13.89 tok/s with the same output hash; immutable preload then
raised the fixed probe to 15.54 tok/s, while reference-exact Hyper-Connection fusion
raised the final result to 16.06 tok/s and cut TTFT to 0.434 s. Capping KV at 128K
reduced device allocation from 97.62 to 76.43 GiB while retaining the same output hash;
the 0.7% throughput difference from the unconstrained sample is within short-run noise.

Against the earlier recipe's exact-64K recall run, request time fell from 689.37 to 208.58
seconds (69.7% lower, or 3.31x faster), while routed-expert physical reads fell from
550.83 GB to zero. Both runs observed exactly 65,536 prompt tokens and returned the same
planted key; because their recipe versions and prefill chunk sizes differ, this is a
version-to-version result rather than a single-variable comparison.

The superseded official FP8 recipe measured 4.99 tok/s and 0.580 s warm TTFT. The
checkpoint and recipe tuples differ, so that comparison remains informative rather than a
controlled quantization-only experiment.
