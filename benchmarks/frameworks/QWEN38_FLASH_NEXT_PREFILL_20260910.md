# Qwen3.8-Flash-Next prefill optimization — 2026-09-10

Batched PLE reads and sparse-attention selection reduce long-prompt processing
cost in the native Qwen4 runtime. On one DGX Spark, the paired 30-request ~8K/256
workload reduced mean TTFT by **15.8%** and full-request
time by **11.3%**. Weights, activation/storage
precision, sampling, KV capacity, and the selected MTP profile are unchanged.

| Workload, input / output tokens | TTFT, baseline → optimized | Decode tok/s, baseline → optimized | Full request, baseline → optimized |
|---|---:|---:|---:|
| Short, 96 / 128 | 0.243 → 0.243 s | 42.20 → 42.14 | 3.252 → 3.256 s |
| Random, 1024 / 256 | 2.213 → 1.981 s | 30.65 → 30.57 | 10.748 → 10.536 s |
| SPEED-Bench, ~8K / 256 | 23.010 → 19.374 s | 26.10 → 26.28 | 32.880 → 29.171 s |

Short is the median of three repeats after one warmup. The other rows are means
over 30 distinct requests after three separate warmups. Decode timing is
`(completion_tokens - 1) / (request_s - ttft_s)`, through stream completion.
Both native runs reuse prefix cache for the short repeated prompt. The same
frozen chat messages, thinking settings, and token budgets are used in both runs.

## Changes

Large PLE cache misses are submitted in batches of 512 rows to the existing
16-worker I/O pool. This removes most per-row Future/queue overhead while keeping
parallel NVMe reads. Raw bytes, FP8 scaling, returned row order, cache insertion,
and the small decode lookup path are preserved. Serial reads were rejected after
a cold-file test showed a substantial regression. The prototype with 2,048-row
jobs reduced a cold 8K lookup from 4.1–4.5 seconds to 1.9 seconds; the table above
measures the final 512-row implementation in the full model.

Sparse QSA prompt selection processes up to 128 queries at once. It retains the
existing FP32 score reduction and causal visibility, packs selected physical rows
in chronological order, and uses the original one-dimensional top-k whenever a
selection boundary is tied. Dense prefixes and partial block tails are retained.
Small decode and MTP verification batches keep their existing selection path.
The 8K-shaped component test selected exactly the same 15,081,760 physical rows;
warmed selection took 44.6 ms versus 198.5 ms for the old query loop.

The changes apply automatically to native Qwen4 prefill. No recipe flag,
quantization choice, or portfolio number was changed.

## Validation

- **223 focused tests passed:** exact attention-row selection across causal
  boundaries, cached prefixes, tails and tied scores; scaled FP8/BF16 PLE reads
  across source shards, async lookups, cache eviction and short-read errors; plus
  mixed prefill/decode request packing and existing Qwen4 model and metadata tests.
- All 140 speed requests including warmups completed with the requested token
  counts. Paired measured outputs were identical for **63/63** requests.
- The same 52-case arithmetic/recall/JSON smoke suite, run after the speed workloads, scored
  **42/52 → 42/52**, with
  **0 new failures** and **52/52 identical answers**. Existing baseline failures are retained in the report.

## Reproduction and limits

Fresh sequential servers used a clean baseline at
`3f259fb04d7d28a4d78b49f96502083e8586012f` and an isolated candidate checkout of that
revision plus the recorded source-file hashes. Both used NVIDIA source
`fab0aecb760cec45227f6656abcaafa11abca87a`, FTW fingerprint `94e1ee0daa442357`, full
24,576-expert preload, BF16 resident projections/KV, FP32 recurrent state, Triton
NVFP4, a 12,288-token sequence cap and 131,072 KV tokens. Each server uses the same
existing model-page-cache release policy at preload. No other benchmarks or
downloads ran alongside speed measurements. Host swap and memory observations
are retained in the result. Neither run had an OOM or memory-guard event.
Minimum available memory was 22.7 GiB before and 27.1 GiB after. The optimized
run recorded no swap-out pages; the baseline recorded 7,737, with pre-existing
host swap present in both runs.

The measured fast MTP3 profile remains opt-in: separate FP8 proposal head,
immediate redraft and light verification snapshots. Its historical MGSM tuning
subset was 92/100 versus 94/100 for the older profile. This optimization and its
smoke tests do not establish general quality parity, reverse that historical
limitation, or certify concurrency, long context, or production endurance.
MGSM was not rerun. The previous external-engine measurements remain historical;
vLLM and SGLang were not rerun for this change.

[Versioned evidence](../gb10/results/GB10-QWENNVIDIA-006.json) includes commands,
source hashes, per-request metrics and output hashes, quality results and memory
telemetry. Raw logs, scripts and requests are in
`results/qwen38-flash-next-opt-20260910` at the repository root. The corpus matches
the [earlier framework comparison](QWEN38_FLASH_NEXT_20260909.md), with suite SHA256
`38316b0c391b4dc37c57923315389d69663a98aa90b20a5c8b1e04dc6f15f7fb`.
