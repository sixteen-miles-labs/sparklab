# GLM-5.2 NVFP4 on NVIDIA GB10

Last updated: 2026-08-27

## Experiment status

The pinned checkpoint, both FTW layouts, the G1--G7 ladder, the selected
64-token run, and two 256-token stability trials are complete. The serving path
is usable at about 0.8 tok/s, but the experiment does **not** pass the strict
Research / bring-up gate because neither 256-token trial had zero swap growth.
The primary trial swapped out 170 pages (680 KiB); the lower-cache retry swapped
out 138 pages (552 KiB), despite more than 72 GiB minimum `MemAvailable`.

This experiment has two distinct outcomes:

1. **Research bring-up:** prove that the complete official checkpoint converts,
   loads, and produces validated output with bounded unified memory and no swap
   growth.
2. **Promotion beyond Research:** additionally pass the product latency, context,
   endurance, API, and agent gates below.

A successful bring-up does not by itself make GLM-5.2 a Frontier model.

## Spark Lab tier gate

GLM-5.2 is retained as an Experimental Research fallback rather than a Frontier
fallback. Assign the result according to the exact checkpoint and recipe tested:

| Status | Required result on one GB10 |
|---|---|
| Research / bring-up | Full official checkpoint; at least 64 validated greedy tokens; bounded memory; no swap growth from the recorded pre-run baseline; all limitations and physical I/O reported |
| Experiment usability | At least 0.5 tok/s sustained over the 256-token confirmation, or a measured SSD/cache-locality bound explaining a lower result |
| Frontier candidate | At least 5 tok/s and warm TTFT at most 20 s on `GB10-INTERACTIVE-001` |
| Frontier certified | Frontier candidate performance, at least 64K usable context, a stable 60-minute agent trace, reasoning/tool-parser correctness, fixed coding-agent task completion, and the complete benchmark evidence contract |

Interpretation:

- Below 0.5 tok/s: correct feasibility result, but not yet usable.
- From 0.5 to below 5 tok/s: Spark Lab Research tier.
- At least 5 tok/s: eligible for Frontier only after every other certification
  gate passes.

The 64-token AIME probe in this report is the tuning and correctness workload.
It is not a substitute for the separate 4K-input/512-output
`GB10-INTERACTIVE-001` tier benchmark.

## Target and system

| Component | Configuration |
|---|---|
| System | NVIDIA GB10, 128 GB unified memory |
| CPU | 20 physical Arm cores (10 Cortex-X925 + 10 Cortex-A725) |
| OS | Ubuntu NVIDIA kernel 6.17.0-1014-nvidia, aarch64 |
| NVIDIA driver / CUDA | 580.126.09 / CUDA 13.0 |
| Python | 3.11.15 in `.venv` |
| PyTorch / Triton | 2.11.0+cu130 / 3.6.0 |
| FlashInfer / SGLang kernel | 0.6.15.post1 / 0.4.5 |
| Model | `nvidia/GLM-5.2-NVFP4` |
| Pinned model revision | `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Source checkpoint | `/home/lidaiqing/models/GLM-5.2-NVFP4` |
| FTW checkpoint | `/home/lidaiqing/ftw/GLM-5.2-NVFP4-b12x` |

The pinned repository advertises 57 files totaling 464,874,323,992 bytes. Its
weight index contains 47 safetensors shards, 232,385 tensor keys, and
464,795,267,072 indexed tensor bytes.

The parsed model has 78 transformer layers, of which the final 75 are MoE layers,
256 routed experts per MoE layer, and top-8 routing. It uses a 6,144 hidden size,
2,048 expert intermediate size, MLA latent KV, and DSA with 21 full IndexShare
indexer layers. FreeToken preserves routed experts in the checkpoint's NVFP4
format and, by default, converts the large resident BF16 projections to per-row
W8A16 FP8.

## Preflight validation

- 55 GLM/DSA model, KV-pool, and attention-backend tests passed.
- The expanded GLM, DSA, NVFP4 backend, offload-cache, and FTW-row gate passed
  51 tests with 4 hardware-specific skips.
- A five-layer dummy-weight server selected `dsa` attention and reached API-ready
  state on GB10.
- A forced `--nvfp4-backend flashinfer` preflight selected the SM12x `b12x`
  expert backend and reported a 21,233,664-byte packed expert row.

## GB10 bandwidth profile

The model-specific profile uses the exact GLM-5.2 expert geometry rather than a
smaller GLM-4.7 proxy.

| Measurement | Result |
|---|---:|
| CPU STREAM read | 103.8 GB/s |
| Unified-memory linear H2D / D2H | 58.6 / 59.2 GB/s |
| GLM-5.2 NVFP4 CPU MoE | 41.9 GB/s |
| GLM-5.2 NVFP4 GPU expert gather | 86.0 GB/s |
| CPU / GPU-gather ratio | 0.49 |
| Recommended backend | `offload` |

The profile is saved at
`/home/lidaiqing/results/glm5.2-gb10/benchbw-nvfp4.json`.

The SM12x cache-copy microbenchmark measured a 20.25 MiB b12x expert row. At
batch size 1, two misses copied at 100.4--100.8 GB/s and eight misses copied at
103.8--104.1 GB/s. Results were unchanged between 256 and 1,024 cache slots, so
copy bandwidth is not sensitive to the tested cache allocation size.

## Checkpoint conversion

The conversion must explicitly select the same backend-owned layout used at
serve time. The experiment added `--nvfp4-backend` to `ft checkpoint` and both
serving benchmark harnesses; result JSON records the choice.

```bash
CUDA_HOME=/usr/local/cuda \
PATH=/usr/local/cuda/bin:$PATH \
FREETOKEN_CONVERT_PROGRESS=1 \
.venv/bin/ft checkpoint \
  --model /home/lidaiqing/models/GLM-5.2-NVFP4 \
  --out /home/lidaiqing/ftw/GLM-5.2-NVFP4-b12x \
  --dtype bfloat16 \
  --moe-backend offload \
  --nvfp4-backend flashinfer \
  --shard-gib 8 \
  --device cuda:0
```

Before conversion, all 47 source shards were read and checked against their LFS
SHA-256 objects. Every safetensors key was checked against the index: 232,385
keys, no missing or extra keys, and 464,795,267,072 indexed tensor bytes. The
full validation took 647.1 s. After it passed, eight unrecoverable transfer
fragments (`*.incomplete`, 6.316 GB total) were deleted.

| Layout | Time | Exact bytes | Shards | Tensors | Expert row | Fingerprint |
|---|---:|---:|---:|---:|---:|---|
| Native Triton (`nvfp4`) | 722.2 s | 429,106,159,616 | 77 | 2,191 | 21,254,144 B | `d3affa2cd69360b1` |
| FlashInfer (`nvfp4_b12x`) | 1,012.9 s | 428,713,099,264 | 77 | 2,043 | 21,233,664 B | `d3affa2cd69360b1` |

Both layouts passed exact format/version/alignment, shard extent, dtype, shape,
byte-count, non-overlap, bank-count, descriptor-count, and sampled physical-read
validation. The b12x resident metadata is identical to native. The first b12x
validator invocation used an incorrect hard-coded expected byte total and
stopped before content checks; correcting that expectation produced the clean
result above. This was a validator correction, not a checkpoint failure.

Conversion was bounded to roughly one active layer of shared mapping, with more
than 100 GiB `MemAvailable`, but the host did page because it already had swap in
use: the conversion interval included both swap-in and swap-out activity. This
is recorded as a conversion limitation rather than silently treated as a
zero-swap result.

Validation artifacts:

- `/home/lidaiqing/results/glm5.2-gb10/source-validation.json`
- `/home/lidaiqing/results/glm5.2-gb10/native-validation.json`
- `/home/lidaiqing/results/glm5.2-gb10/b12x-validation.json`
- `/home/lidaiqing/results/glm5.2-gb10/conversion-native.log`
- `/home/lidaiqing/results/glm5.2-gb10/conversion-b12x.log`

## Serving benchmark

All comparison runs use the first AIME-25 problem, batch size 1, greedy decode,
eager execution, a fixed KV budget, one warm request, and one measured request.
The source and FTW shard pages are evicted with targeted
`POSIX_FADV_DONTNEED` before cold initialization so Linux file cache does not
consume GB10 unified-memory headroom.

### Optimization ladder

Physical I/O is the measured-request delta. The API returned one fewer
completion token than the requested cap in fixed-length runs, so the 16, 64,
and 256 rows below contain 15, 63, and 255 returned completion tokens.

| Gate / configuration | Requested decode | tok/s | TTFT | Miss rate | Physical I/O | Output hash | Result |
|---|---:|---:|---:|---:|---:|---|---|
| G1 native synchronous | 1 | control | 46.04 s | n/a | 380.05 GiB | `93ef0dd82710` | one-token correctness (`The`) |
| G1 native synchronous | 16 | 0.615 | 45.94 s | 100% | 558.20 GiB | `5fb5f7a03094` | control |
| G1 native synchronous | 64 | 0.619 | 45.76 s | 100% | 1,128.28 GiB | `9571e64d566e` | control |
| G2 b12x synchronous | 16 | 0.628 | 46.51 s | 100% | 557.67 GiB | `5fb5f7a03094` | exact native hash |
| G3 b12x sparse prefill | 16 | 0.621 | 3.24 s | 100% | 189.84 GiB | `e0eaa4f7fb20` | 14.4x lower TTFT |
| G4 + shared overlap | 16 | 0.625 | 3.19 s | 100% | 189.84 GiB | `e0eaa4f7fb20` | exact G3 hash; 1,200 overlap calls |
| G5 layer-LRU, 512 slots | 16 | 0.799 | 45.34 s | 76.9% | 516.61 GiB | `e0eaa4f7fb20` | all sparse layers fell back |
| G5 layer-LRU, 600 slots | 16 | 0.821 | 2.78 s | 70.9% | 137.16 GiB | `e0eaa4f7fb20` | all 75 layers sparse |
| G6 integrated auto budget | 16 | 0.796 | 45.89 s | 76.1% | 515.13 GiB | `e0eaa4f7fb20` | safe 522 slots, below top-8 threshold |
| G7 native hybrid | 16 | 0.672 | 3.03 s | 78.9% | 151.70 GiB | `5fb5f7a03094` | exact native hash; one GPU fetch/layer |

G3--G5's 16-token sparse text differs from the synchronous control only in
spacing (`b>9` versus `b > 9`), but therefore is not bitwise identical. The
selected 64-token run below does match the native 64-token hash exactly.

The initial G7 run exposed a native sparse-prefill bug: cache-slot ids reached
the Triton grouped kernel with a `None` row-domain size. Commit `b5db743` fixes
the domain to the persistent-cache row count, adds a regression test, and passed
38 focused MoE tests with 4 expected hardware skips. The rerun above is clean.

### Selected configuration and stability

The selected b12x cache is 675 slots (9 slots per MoE layer). A 750-slot tuning
probe failed before producing output because its packed leading stride was
2,359,296,000, beyond FlashInfer/CUTLASS's signed 32-bit descriptor range. The
derived hard ceiling for this model shape is 682 slots; 675 is the largest
balanced layer quota below it.

| Configuration | Requested decode | tok/s | ms/token | TTFT | Miss rate | Physical I/O | Peak GPU / NVMe temp | Swap in / out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native synchronous control, cache 512 | 64 | 0.619 | 1,614.5 | 45.76 s | 100% | 1,128.28 GiB | 46 / 51.9 C | 11,094 / 0 pages |
| Selected b12x, cache 675 | 64 | 0.808 | 1,237.3 | 2.61 s | 72.4% | 552.58 GiB | 46 / 50.9 C | 2,569 / 0 pages |
| Selected b12x, cache 675 | 256 | 0.802 | 1,247.3 | 2.57 s | 73.5% | 2,236.02 GiB | 47 / 53.9 C | 6,267 / 170 pages |
| Stability retry, cache 600 | 256 | 0.790 | 1,265.7 | 2.61 s | 74.7% | 2,270.39 GiB | 48 / 53.9 C | 2,861 / 138 pages |

The selected 64-token output hash, `9571e64d566e`, exactly matches the native
control. It improves TTFT by 17.6x, decode throughput by 30.5%, and measured
physical I/O by 51.0%. The primary 256-token measurement sustained 0.802 tok/s
for 254 timed decode steps; p50/p99 inter-token latency was 1,243/1,490 ms,
minimum `MemAvailable` was 72.48 GiB, board power averaged 16.57 W (17.78 W
peak), and estimated request energy was 5.29 kJ. There was no throughput or
temperature drift, but its nonzero swap delta fails the strict gate.

```bash
CUDA_HOME=/usr/local/cuda \
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
FREETOKEN_DISK_READ_WORKERS=20 \
.venv/bin/python benchmarks/bench_decode_moe.py \
  --model /home/lidaiqing/ftw/GLM-5.2-NVFP4-b12x \
  --backend offload --storage disk --nvfp4-backend flashinfer \
  --host-cache-gb 0 --cache 675 --cache-policy layer_lru \
  --decode 256 --num-tokens 2048 --mem-ratio 0.90 \
  --disable-prefill-overlap --prefill-sparse-max-tokens 256 \
  --shared-expert-overlap --collect-moe-stats \
  --greedy --no-graph --include-output --server-timeout 1200 \
  --json /home/lidaiqing/results/glm5.2-gb10/selected-b12x-cache675-decode256.jsonl
```

### Cold and distinct-prompt evidence

A clean detached `b5db743` server ran AIME-25 problems 0--2 at a 64-token cap.
The first cold request and two warm-but-distinct requests averaged 0.817 tok/s.
TTFT was 55.55 s cold and 48.09 s mean for the two distinct warm requests. All
three hit the token cap, so their `0/3` accuracy is not an AIME quality result.
All three requests recorded zero swap-out.

This suite also falsified the intended G5 held-out benefit at this cache size:
each prompt had more than the 9-row per-layer quota, so all 75 layers correctly
fell back to full-layer prefill. Layer-LRU improves repeated-prompt decode and
the one-token cached-prefix prefill, but it does not lower physical reads for
these distinct prompts. The suite read 2,755.78 GiB at an effective 9.71 GiB/s.

### Verdict and remaining certification work

- The exact 753B checkpoint is validated, both FTW layouts load, G1--G7 execute,
  64-token correctness is exact, and the 256-token path is thermally and
  throughput stable.
- The 0.802 tok/s confirmation clears the experiment-usability performance
  threshold, but no Spark Lab tier is awarded because the prerequisite
  zero-swap-growth gate failed twice.
- It is far below the 5 tok/s Frontier-candidate threshold and has not run the
  separate `GB10-INTERACTIVE-001` 4K/512 test, 64K context, 60-minute endurance,
  API/tool-parser, or coding-agent certification gates.
- Automatic cache sizing is safe but needs a top-8-aware minimum or an explicit
  sparse fallback warning. The b12x planner also needs a 682-slot kernel-layout
  cap so invalid larger allocations fail at configuration time.
- Server shutdown consistently reports four leaked multiprocessing semaphores;
  this did not affect request results but remains cleanup work.

The compact catalog evidence is checked in as
[`GB10-GLM52-RESEARCH-001`](../benchmarks/gb10/results/GB10-GLM52-RESEARCH-001.json).
It records the measured performance and failed admission separately.

Primary serving results are in
`/home/lidaiqing/results/glm5.2-gb10/selected-b12x-cache675-decode64.jsonl`,
`/home/lidaiqing/results/glm5.2-gb10/selected-b12x-cache675-decode256.jsonl`,
and
`/home/lidaiqing/results/glm5.2-gb10/heldout-aime0-2-b12x-cache675-decode64.jsonl`.
G1--G6 used clean revision `83b0758547ee396d6e7ccdbd9706d644a5ad0612`;
G7 and the primary selected/stability runs used clean pushed revision
`b5db743d4d5103c452835fc25af9e178a8c1c523`. The cache-600 stability retry ran
while unrelated Kimi-K3 tracked edits existed in the shared checkout and is
therefore supplementary (`git_tracked_dirty=true`), not primary evidence.
