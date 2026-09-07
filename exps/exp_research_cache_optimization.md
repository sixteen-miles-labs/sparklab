# Research-tier cache optimization — GLM selected, Kimi pending

Order: full GLM-5.3 first, then Kimi K3. This is not GLM-5.3 Flash.
The GLM 2,850-slot profile is selected for the Experimental recipe and portfolio:
1.11 tok/s, 1.958 s warm TTFT. This is not Research certification. Kimi K3
acquisition and end-to-end optimization have not started.

## Selected result and answer checks

The comparable 256-token result improves decode throughput by 36.4% over
the historical selected 0.812615 tok/s run. The README keeps only the scalar
portfolio values; methodology and limitations remain here and in the model page.

| Normal-EOS check | Cache slots | Tokens | Decode tok/s | Result |
|---|---:|---:|---:|---|
| AIME problem 0 | 2850 | 841 | 1.090915 | Correct final answer 70, EOS |
| AIME problem 1 | 2850 | 1024 | 1.086261 | Token cap, no final answer |
| AIME problem 0 control | 675 | 844 | 0.815443 | Correct final answer 70, EOS |

The selected profile retained at least 29.496 GiB available RAM through both
answer checks, with zero scoped swap-out and OOM-kill deltas. Text differs from
the control; only the completed answer for problem 0 agrees. Two smoke checks
do not establish general quality or certification. No Kimi metrics are changed.

Summary: [GB10-GLM53-RESEARCH-OPT-002](../benchmarks/gb10/results/GB10-GLM53-RESEARCH-OPT-002.json).
Raw answer checks:
[candidate](../benchmarks/gb10/results/GB10-GLM53-RESEARCH-OPT-002-aime2850.json),
[control](../benchmarks/gb10/results/GB10-GLM53-RESEARCH-OPT-002-aime675.json).
Machine-specific home prefixes in checked-in raw evidence are replaced by
`$HOME`; numerical measurements are unchanged.

Recipe artifact version 0.1.0 and the pinned FTW are retained: this changes the
runtime profile, not the checkpoint representation. Runtime admission uses a
conservative 96-GiB estimate plus the planner's separate safety reserve. The
20-reader setting used for measurements is explicit in the model's run command.

Pre-PR verification after integrating the current main branch:

- CPU workflow suite: 1,122 passed, 75 skipped.
- CUDA offload/NVFP4 suites: 55 passed, four skipped.
- CPU interpretation of the actual layer-LRU Triton kernel: three passed.
- CLI, README, and model-table regression tests agree with recipe metrics.

These regression runs do not replace the hardware performance evidence above.

## GLM-5.3 fresh control, 2026-09-06

Raw evidence: [64-token control](../benchmarks/gb10/results/GB10-GLM53-RESEARCH-OPT-002-baseline64.json).
The original process exited normally before the CPU-only changes below.
Engine revision `c6d4956c06f64a1ef9a9e52efd01fa693992fd58`, tracked tree clean.
FTW fingerprint `a0e799b03bceb4bf`.

Configuration: 675 layer-LRU slots, FlashInfer b12x experts, disk storage,
20 readers, zero host expert LRU, 2,048 KV tokens, memory ratio 0.90,
sparse prefill threshold 256, shared-expert overlap, eager execution.
AIME-25 problem 0, greedy, 64 output tokens, one warmup and one measured request.

| Measurement | Control |
|---|---:|
| Decode | 0.843469 tok/s |
| Warm TTFT | 2.215869 s |
| First-request TTFT | 50.8515 s |
| Expert miss rate | 74.5265% |
| Measured-request physical expert reads | 565.50 GiB |
| Minimum available RAM over server lifecycle | 73.076 GiB |
| Swap-out pages / scoped OOM-kill delta | 0 / 0 |

Output SHA1 `e19ef21a678b` matches the earlier 64-token experiment. The response
is length-capped, so this is output reproducibility, not a completed quality pass.
The disk counter includes the measured request's prefill and decode, not startup
or the first request. Pre-existing swap-in activity remains a caveat.

## Candidate changes (validation recorded below)

1. Remove the redundant `torch.unique` assertion in layer-LRU decode admission.
   `unique_count <= query_count <= cache_capacity` already follows from the
   shape check. The old assertion introduced a dynamic-output GPU operation
   and host synchronization on every offloaded layer.
2. Preserve all current query routes when their count exceeds the layer quota.
   The previous CPU reference raises `no evictable slot` when the cache is full;
   the GPU kernel can encounter an all-MAX victim score and select a live slot.
   Prefer the existing quota-based victims, then borrow the globally oldest
   non-current slot only when no preferred victim exists.
3. Let opt-in sparse prefill use the total cache capacity, not equal per-layer
   quota, as its route-set capacity limit. This relies on change 2. Kimi's
   top-16 routing exceeds the default 896/92 = 9–10 protected slots/layer;
   the old quota gate forces even one-token prefills to load all 896 experts.
   Larger GLM prompts also encounter this gate. The optimal token threshold
   and cache-pollution tradeoff still need measurements.

CPU verification:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/moe/test_offload.py -q
# 36 passed, 4 CUDA tests skipped
CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/moe/test_layer_lru_interpreter.py -q
# 3 passed: actual Triton kernel interpreted on CPU, versus CPU reference
```

The regression was observed failing before the fix. Interpreter cases cover
unequal quotas, full cache, full-layer queries, repeated routes, ownership,
copy plans, LRU timestamps, and subsequent quota reclamation. CPU tests do not
prove compiled CUDA execution, numerical output parity, or end-to-end speed.

## Experiment chronology and benchmark coordination

### Checkpoint cleanup plan (2026-09-06)

Kimi's pinned repository is accessible at
`793f1f8436cd7de11e7912c41b3d49d4c9e4d11c`: 214 files, 194 FTW shards,
1,610,940,993,828 bytes including metadata. Before cleanup, disk free space was
1,001,042,706,432 bytes, insufficient for acquisition plus its safety margin.

Validated deletion targets, with physical allocated bytes (hard links counted
once across the two Inferact directories):

| Exact directory | Allocated bytes |
|---|---:|
| `$HOME/models/models/glm-5.3/source/ce67b36f3669` | 475478843392 |
| `$HOME/models/qwen38-nvidia/source/fab0aecb760c` | 132734746624 |
| `$HOME/.sparklab/models/qwen3.8-flash-next/prepared/0.5.0` | 180456607744 |
| `$HOME/.sparklab/models/qwen3.8-flash-next/prepared/0.5.0-fp8-ple` | 51200557056 |

Their canonical paths equal the explicit paths above; no active model server,
benchmark, or open target files were found. Current GLM FTW
`a0e799b03bceb4bf` and NVIDIA Qwen FTW `94e1ee0daa442357` passed the native artifact
validator, including all shard sizes and NVIDIA's two local sidecars. Neither
prepared tree contains symlinks into the source downloads. Both prepared trees
are retained. The Inferact variants (`47e11ddb878adf4c`) are superseded by NVIDIA.
Original sources can be downloaded again at the recipe's pinned revisions;
the experimental Inferact FP8-PLE variant requires regeneration, not trash restore.

DeepSeek's source is deliberately retained: its validator rejects a duplicated
name shared by `weight` and `speculative_weight` entries. This is a separate
validation issue, not evidence that the source is expendable.

Cleanup completed with non-forced recursive removal of exactly the four listed
directories. Free space afterward: **1,840,911,863,808 bytes** (about 1.67 TiB).
The standard Inferact FTW is recoverable from
`oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW@5ab790b83f149a96594237a35905d84be24599a3`;
its original source revision is `103a7608316173ca6edd49929544244de7ffda70`.
The experimental FP8-PLE variant must be regenerated. These were deletions,
not moves to trash; no current prepared artifact was removed.

### GPU validation and first isolation runs

The initial offload suite passed **40/40 on CUDA**. The first route-first
64-token GLM run measured 0.839516 tok/s with output hash `e04fe1589d9a`, different
from the control; the sweep stopped automatically. Restricting sparse prefill to
one token restored hash `e19ef21a678b` at 0.846710 tok/s and 2.498334 s warm TTFT.
This isolates the text difference to multi-token prefill, not the decode fix.
The restriction makes synthetic startup warmup scan complete expert layers and
is an isolation setting, not a selected recipe.

The next, 1,350-slot run failed in the donor W4A16 descriptor construction:
`OverflowError: Value overflow: 4246732800 exceeds range of l`.
The flattened GLM gate/up bank exceeds int32 extent at **683 slots**. This is
not memory exhaustion. Its backend exited before a usable measurement. The
queue was stopped; the failed benchmark and shutting-down API were terminated.

Two further candidate fixes now have targeted GPU tests:

- `materialize_routed_layer` loads only selected experts at logical positions
  in an E-row view, preserving the old b12x prefill geometry. Inverse maps and
  copy plans cover overlapping source/destination slots; the numerical test
  exactly matches full-layer prefill after overwriting the cache with another
  layer to expose missing copies.
- Above the descriptor limit, b12x gathers routed rows from the large GPU cache
  into a bounded compute bank. Batch-one decode avoids `torch.unique`; copying
  stays on-device. The targeted test forces this path with a small threshold
  and exactly matches ordinary decode. The extra memory traffic must be
  included in end-to-end measurements, not assumed free.

The stable-layout/large-cache candidate ran sequentially with the original
256-token sparse threshold and 12-GiB memory guard. Raw local paths:
`/tmp/glm53-opt-stable2-cache{675,1350,2700,2850}-64.jsonl`.
These initial short probes were not sufficient for recipe selection.

The stable-layout 675-slot run reproduced the control hash `e19ef21a678b` at
0.845760 tok/s, 2.538650 s warm TTFT, and **25.500066 s first-request TTFT**
(control: 50.8515 s). At 1,350 slots, the compact path measured 0.900019 tok/s,
62.2937% misses, 475.282 GiB measured-request reads, and 60.107 GiB minimum
available RAM; its output hash `e04fe1589d9a` differed, so the queue stopped.

Synthetic probes at GLM's actual 6,144-hidden / 2,048-intermediate dimensions,
top-8 routing, and 675 physical slots reproduced small compact-path numerical
differences. Three seeds had relative RMS deltas of 0.000133–0.000638;
FP32-reference relative RMS errors were 0.00404–0.00445 for both paths, with
no material worsening in these probes. This is not end-to-end quality proof.
Larger-cache exploratory runs compare against the 1,350-slot compact-path hash,
not the original hash. Completed-answer checks remain required before selection.

### Larger-cache short probes

| Cache slots | Decode tok/s | Warm TTFT s | Lifecycle minimum available GiB | Output SHA1 |
|---|---:|---:|---:|---|
| 675, control | 0.843469 | 2.215869 | 73.076 | `e19ef21a678b` |
| 675, candidate | 0.845760 | 2.538650 | — | `e19ef21a678b` |
| 1350 | 0.900019 | 2.518284 | 60.107 | `e04fe1589d9a` |
| 2700 | 1.110749 | 2.113318 | 32.861 | `e04fe1589d9a` |
| 2850 | 1.113522 | 2.089217 | 29.899 | `e19ef21a678b` |

All are 64-token probes, not completed-answer checks. The 2,700/2,850 runs
recorded zero swap-out, swap growth, and OOM-kill deltas. The latter reproduced
the original control prefix but differed from the 1,350-slot comparison hash,
so the exploratory sweep stopped as designed. Cache size changes route
placement as well as bank geometry; no global bit-exactness is claimed.
The 2,700 run's miss rate was 45.2831%, down from 74.5265% in the control.

A GPU-copy microbenchmark used actual GLM packed row sizes and eight routes.
Allocating `index_select` took 1.548 ms; the fastest fused-copy setting took
1.523 ms (20 timed iterations, medians). This small isolated difference does
not justify extra buffer-lifetime complexity; no fused-copy change was made.
Local evidence: `/tmp/probe-glm53-gather.log`.

### Longer validation

Cleanup waited for the user's explicit “finished” confirmation and is now
complete as recorded above. The 2,850-slot 256-token trial completed:

- Decode **1.108134 tok/s**, warm TTFT **1.957731 s**, first TTFT 25.857734 s.
- Minimum lifecycle available memory 29.690 GiB; zero swap-out and OOM-kill
  deltas, no memory-guard stop.
- Miss rate 44.61%; measured-request physical reads 1,356.89 GiB.
- Output SHA1 `5b69d85cc226`, different from the published 256-token control
  `bdbf1a17f134`; still length-capped, with no final answer established.
- About 36.4% higher throughput and 22.6% lower warm TTFT than the historical
  selected 256-token run. This is one candidate run, not a repeated result.

Raw evidence is retained in
[`GB10-GLM53-RESEARCH-OPT-002-candidate256.json`](../benchmarks/gb10/results/GB10-GLM53-RESEARCH-OPT-002-candidate256.json).
The four short candidate runs are retained in
[`GB10-GLM53-RESEARCH-OPT-002-short-sweep.json`](../benchmarks/gb10/results/GB10-GLM53-RESEARCH-OPT-002-short-sweep.json).
The latest CPU regression run passed 43 tests, with five CUDA tests skipped.

A bounded sequential completed-answer queue finished after the performance
process exited and the GPU was confirmed idle. It ran normal-EOS AIME
problems 0 and 1 at 2,850 slots, then problem 0 at 675 slots, each with a
1,024-token output cap and the 12-GiB lifecycle memory guard. Local queue:
`/tmp/glm53-completed-answer-queue.py`; progress:
`/tmp/glm53-completed-answer-queue.log`; result paths:
`/tmp/glm53-opt-stable2-cache{2850,675}-aime1024.jsonl`.
The queue performed no recipe promotion or checkpoint acquisition. Incorrect
or capped responses are retained for comparison with the control, while
process/memory failures stop the queue. Kimi acquisition has not started.

- Keep a 12-GiB available-memory guard over startup and both requests. Stop
  on memory pressure, swap-out growth, process failure, or output differences.
- The 256-token performance and normal-EOS smoke checks are complete above.
  Broader quality, context, parser, and endurance validation remain outstanding.
- Further tuning can measure disk-reader count and cache size beyond this
  selected profile. Whole-model CUDA graphs remain unsupported for disk offload.
- Acquire Kimi's pinned FTW and repeat the correctness/performance process.
  Post-cleanup free disk is about 1.67 TiB, sufficient for its 1.465-TiB
  download with about 214 GiB remaining. Do not overlap download disk traffic
  with GPU model performance measurements.
