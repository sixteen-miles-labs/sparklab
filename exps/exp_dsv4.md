# DeepSeek-V4 Inference Experiments

Last updated: 2026-08-23

## Summary

We converted `deepseek-ai/DeepSeek-V4-Flash-0731` to the FreeToken Weight (FTW)
format and ran a disk-backed inference smoke test on the local RTX 3090 system.

Key findings:

- The first conversion attempt exhausted 62 GiB of RAM and 8 GiB of swap because
  completed shared-anonymous expert-bank mappings remained resident.
- Releasing completed `HostBank` mappings with `MADV_REMOVE`, and syncing plus
  evicting completed FTW output shards, kept memory bounded throughout conversion.
- The fixed conversion completed in 154.3 seconds without meaningful swap growth.
- Disk-backed hybrid inference now works. Staging CPU-overflow rows before CPU
  submission fixed the previously unsupported disk/hybrid combination.
- Parallel, coalesced direct reads raise effective disk throughput from about
  1.35 to 2.83 GiB/s. The optimized 16-token run reaches 2.26 tok/s and cuts warm
  TTFT from 104.5 to 60.3 seconds, with the same greedy output hash.
- For sustained generation, a safe 40 GiB pageable host cache reaches 1.87 tok/s
  over 62 measured decode intervals. A 44 GiB trial caused swap pressure and was
  rejected.
- Combined hybrid staging plus a borrowable per-layer host LRU raises the matched
  64-token result to 1.982 tok/s, 6.1% above the previous hybrid result and 10.4%
  above the matched optimized offload control, without changing output.
- Memory-bounded double-buffered disk prefill cuts warm TTFT by 35.7%, but its
  smaller host LRU reduces sustained decode to 1.860 tok/s. It is useful for short
  requests but is not selected for the long-CoT AIME evaluation.
- The selected single-buffer configuration completed all 30 AIME-25 problems in
  one long-lived server at 1.910 mean decode tok/s (1.902 token-weighted), only
  3.7% below the matched 64-token probe. It scored 9/30 pass@1 with a 1,024-token
  output limit; 21 answers reached that limit, so this is a bounded systems run
  rather than a full-length quality ceiling.

## Test system and model

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GiB VRAM |
| Host memory | 62 GiB RAM, 8 GiB swap |
| Storage | WD_BLACK SN850X 4 TB NVMe, ext4, mounted at `/mnt/ssd` |
| Software | FreeToken development checkout; PyTorch 2.11.0+cu130; CUDA 13.0 |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Source checkpoint | `/mnt/ssd/freetoken/models/DeepSeek-V4-Flash-0731` |
| FTW checkpoint | `/mnt/ssd/freetoken/ftw/DeepSeek-V4-Flash-0731` |

The source checkpoint is approximately 156 GiB across 48 indexed safetensors
shards. All indexed shards were present before conversion.

## Experiment 1: FTW conversion and memory failure

The original serial conversion reached 19 of 43 expert layers before the machine
became unresponsive. Kernel and sysstat records showed global OOM reclaim rather
than an NVMe, filesystem, GPU, or thermal fault.

Observed evidence:

- The affected user service peaked at 59.6 GiB RAM and 7.7 GiB swap.
- OOM kills began at 16:23 and cascaded across desktop and user services.
- About 53 GiB was attributed to cache/shared memory near the failure.
- The conversion process remained alive during heavy reclaim, making the machine
  appear frozen.
- No relevant NVMe, ext4, NVIDIA Xid, PCIe, or thermal errors were logged.

Two retention paths were corrected:

1. `FTWWriter` now flushes and `fsync`s each completed shard, then advises Linux
   to evict it with `POSIX_FADV_DONTNEED`.
2. `HostBank.release()` now uses `MADV_REMOVE` for shared-anonymous mappings,
   with `MADV_DONTNEED` as a portability fallback. In the real DeepSeek run,
   `MADV_DONTNEED` alone allowed roughly 3.2 GiB per completed expert layer to
   remain resident under sustained writes.

## Experiment 2: Fixed FTW conversion

Command:

```bash
FREETOKEN_CONVERT_PROGRESS=1 .venv/bin/ft checkpoint \
  --model /mnt/ssd/freetoken/models/DeepSeek-V4-Flash-0731 \
  --out /mnt/ssd/freetoken/ftw/DeepSeek-V4-Flash-0731 \
  --dtype bfloat16 \
  --moe-backend offload \
  --shard-gib 8 \
  --device cuda:0
```

| Metric | Result |
|---|---:|
| Expert layers converted | 43 / 43 |
| Dense weight entries | 1,521 |
| Expert-bank entries | 172 |
| FTW size | 146.65 GiB |
| FTW shards | 23 |
| Quantization format | `ds_fp4` |
| Fingerprint | `dbe5ee6fba52f0a3` |
| Conversion time | 154.3 s |
| Expert-row descriptors validated | 44,032 |

At expert layer 26, the system still reported approximately 57 GiB available RAM,
about 1 GiB process RSS, and no new swap growth. Memory remained healthy through
finalization.

Conversion log:
`/mnt/ssd/freetoken/results/dsv4/conversion-retry.log`

## Experiment 3: Synchronous disk-backed smoke inference

The smoke test used greedy decoding, the first AIME-25 prompt, a 1 GiB host expert
LRU, automatic GPU expert-cache sizing, eager decode, and a 2,048-token KV allocation.
Disk mode requires prefill overlap to be disabled.

Command:

```bash
CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13 \
PATH="$CUDA_HOME/bin:$PATH" \
.venv/bin/python benchmarks/bench_decode_moe.py \
  --model /mnt/ssd/freetoken/ftw/DeepSeek-V4-Flash-0731 \
  --backend offload \
  --storage disk \
  --host-cache-gb 1 \
  --decode 4 \
  --num-tokens 2048 \
  --disable-prefill-overlap \
  --collect-moe-stats \
  --greedy \
  --no-graph \
  --json /mnt/ssd/freetoken/results/dsv4/disk-smoke.json
```

| Metric | Result |
|---|---:|
| Prompt tokens | 48 |
| Requested / actual completion tokens | 4 / 3 |
| Measured decode steps | 2 |
| Decode throughput | 0.57 tok/s |
| Time per token | 1,755.19 ms |
| Warm TTFT | 119.69 s |
| VRAM | 18.96 GiB |
| GPU expert-cache slots | 726 |
| Expert-cache miss rate | 66.28% |
| Missing / active experts per layer | 3.98 / 6.00 |
| Physical disk reads | 142.45 GiB |
| Physical disk throughput | 1.43 GiB/s |
| Host LRU capacity | 80 entries / 1 GiB |
| Host LRU hit rate | 0.69% |
| Host LRU evictions | 513 |
| Prefill admission bypasses | 10,928 |
| Output hash | `a3d0eccca7d6` |
| Output sample | `We need answer` |

Result: `/mnt/ssd/freetoken/results/dsv4/disk-smoke.json`

This is a smoke result, not a stable throughput measurement: only two decode
intervals were measured, and the request returned three completion tokens despite
requesting four. A longer run is required for representative decode percentiles.

## Experiment 4: 16-token disk-backed decode

The primary measurement repeated the same configuration with `--decode 16` and
wrote its result to `disk-greedy-16.json`. As in the earlier Qwen experiments, the
reported actual completion was one token below the requested maximum.

| Metric | Result |
|---|---:|
| Prompt tokens | 48 |
| Requested / actual completion tokens | 16 / 15 |
| Measured decode steps | 14 |
| Decode throughput | 0.65 tok/s |
| Time per token | 1,547.47 ms |
| Decode interval | 21.665 s |
| Event latency p50 / p99 | 1,559.30 / 2,111.84 ms |
| Warm TTFT | 120.51 s |
| VRAM | 18.96 GiB |
| GPU expert-cache slots | 726 |
| Expert-cache miss rate | 49.48% |
| Missing / active experts per layer | 2.97 / 6.00 |
| Physical disk reads | 159.91 GiB |
| Physical disk throughput | 1.38 GiB/s |
| Host LRU capacity | 80 entries / 1 GiB |
| Host LRU hit rate | 0.62% |
| Host LRU evictions | 1,915 |
| Prefill admission bypasses | 10,928 |
| Output hash | `03c286207d6b` |
| Output sample | `We need answer math. Problem: Find sum of all integer bases b>` |

Result: `/mnt/ssd/freetoken/results/dsv4/disk-greedy-16.json`

The longer run confirms that synchronous NVMe misses dominate. Decode throughput
improved relative to the tiny smoke window because the GPU expert-cache miss rate
fell from 66.28% to 49.48%, but prefill still scans enough expert data to produce
an approximately two-minute TTFT.

## Experiment 5: Enable disk-backed hybrid inference

The original engine rejected `--moe-storage disk --moe-backend hybrid`. That
guard was necessary because the CPU executor reads raw expert IDs from persistent
host-bank pointers, while disk mode exposes a single shared layer-shaped staging
bank. Without staging CPU-assigned rows first, the executor could read stale rows
from another layer.

The hybrid decode path now extracts the unique CPU-overflow IDs after routing and
synchronously stages those rows before `decode_submit`. GPU-assigned misses retain
the existing `copy_missing` path. Disk mode remains eager-only. At this experiment
stage it still required prefill overlap to be disabled; Experiment 8 adds a bounded
overlap implementation.

Validation:

- 31 focused MoE/checkpoint tests passed.
- A real DeepSeek hybrid smoke inference completed at fetch cap 3.
- The smoke output hash, `a3d0eccca7d6`, matches the disk-offload smoke run.
- Every longer greedy run produced `03c286207d6b`, matching the original 16-token
  disk-offload result.

Smoke result:
`/mnt/ssd/freetoken/results/dsv4/disk-hybrid-smoke-f3.json`

## Experiment 6: Disk hybrid host-cache and fetch-cap sweep

All longer measurements used the same 48-token AIME-25 prompt, requested 16
completion tokens (15 returned, 14 measured decode intervals), greedy sampling,
automatic 726-slot GPU expert cache, eager execution, and a 2,048-token KV budget.

| Backend / fetch cap | Host cache | Decode tok/s | ms/token | Warm TTFT | Host hit rate | Physical reads | Output hash |
|---|---:|---:|---:|---:|---:|---:|---|
| offload | 1 GiB | 0.646 | 1,547.47 | 120.51 s | 0.62% | 159.91 GiB | `03c286207d6b` |
| hybrid / 3 | 16 GiB | 0.705 | 1,419.15 | 111.33 s | 11.59% | 142.62 GiB | `03c286207d6b` |
| hybrid / 3 | 32 GiB | **2.146** | **465.98** | **104.49 s** | 28.48% | 115.36 GiB | `03c286207d6b` |
| hybrid / 4 | 32 GiB | 2.083 | 480.03 | 105.97 s | 28.47% | 115.30 GiB | `03c286207d6b` |
| hybrid / 6 | 32 GiB | 2.021 | 494.77 | 104.92 s | 28.76% | 114.63 GiB | `03c286207d6b` |
| offload | 32 GiB | 2.358 | 424.01 | 104.57 s | 28.76% | 114.63 GiB | `03c286207d6b` |

The 32 GiB cache never filled: the best hybrid run occupied 1,861 of 2,570
entries and had zero decode-admission evictions. Increasing it further is not
expected to improve this short repeated-prompt workload. Fetch cap 3 is the best
measured hybrid balance on this 8-core AVX2 host and RTX 3090. Fetching all six
routes remains valid hybrid mode but loses performance to its bookkeeping without
getting useful CPU/GPU co-compute. Plain offload is 9.9% faster at the same cache
budget, but does not exercise CPU/GPU hybrid execution.

The original synchronous-reader recommendation below is superseded by Experiment 7.

Recommended maximum-throughput disk-hybrid command on this machine at this point:

```bash
CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13 \
PATH="$CUDA_HOME/bin:$PATH" \
.venv/bin/python benchmarks/bench_decode_moe.py \
  --model /mnt/ssd/freetoken/ftw/DeepSeek-V4-Flash-0731 \
  --backend hybrid \
  --storage disk \
  --host-cache-gb 40 \
  --hybrid-fetch 3 \
  --decode 16 \
  --num-tokens 2048 \
  --disable-prefill-overlap \
  --collect-moe-stats \
  --greedy \
  --no-graph
```

Result files:

- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-f3-cache16-decode16.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-f3-cache32-decode16.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-f4-cache32-decode16.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-f6-cache32-decode16.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-offload-cache32-decode16.json`

## Experiment 7: Parallel/coalesced FTW reads and sustained decode

Profiling showed that the synchronous staging loop read the four independent
DeepSeek expert banks serially and delivered only about 1.35 GiB/s from the SN850X.
An isolated full-layer FTW test measured 1.65 GiB/s with one reader and a plateau
near 3.07 GiB/s with 12-16 readers.

The disk source now:

- owns a persistent, bounded 16-worker read pool;
- coalesces consecutive aligned expert rows into direct reads targeting the pinned
  staging banks, avoiding transient allocation and an extra CPU copy;
- falls back to the general aligned-row path for formats whose rows cannot be read
  directly;
- explicitly enters PyTorch inference mode in worker threads;
- leaves the large CPU-only host LRU pageable while keeping only the GPU-facing
  one-layer staging banks pinned.

The pageable LRU matters operationally. A pinned 40 GiB cache forced approximately
5.5 GiB of unrelated pages into swap. Repeating the workload with the pageable LRU
held about 12 GiB available RAM and did not increase `pswpout`. A 44 GiB pageable
trial still caused about 2.5 GiB of additional swap-out during initialization, so
40 GiB is the maximum safe tested budget on this 62 GiB machine.

### End-to-end results

| Reader/cache configuration | Requested / actual | Decode tok/s | ms/token | Warm TTFT | Host hit rate | Physical reads | Disk rate | Output hash |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| serial rows, 32 GiB | 16 / 15 | 2.146 | 465.98 | 104.49 s | 28.48% | 115.36 GiB | 1.35 GiB/s | `03c286207d6b` |
| parallel rows, 32 GiB | 16 / 15 | 1.999 | 500.22 | 66.39 s | 28.48% | 115.36 GiB | 2.45 GiB/s | `03c286207d6b` |
| parallel + coalesced, 32 GiB | 16 / 15 | **2.260** | **442.39** | **60.29 s** | 28.48% | 115.36 GiB | **2.83 GiB/s** | `03c286207d6b` |
| parallel + coalesced, 32 GiB | 64 / 63 | 1.702 | 587.49 | 59.40 s | 38.20% | 140.25 GiB | 2.39 GiB/s | `fbf178b2bde5` |
| parallel + coalesced, pageable 40 GiB | 64 / 63 | **1.868** | **535.46** | **58.02 s** | **46.11%** | **122.30 GiB** | 2.26 GiB/s | `fbf178b2bde5` |

The short-run result improves hybrid decode by 5.3% and TTFT by 42.3% relative to
the previous best. The 64-token run is the more representative sustained result:
40 GiB improves throughput by 9.7% over 32 GiB and removes 17.95 GiB of physical
reads. All comparable runs retain identical greedy hashes.

Final recommended command:

```bash
CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13 \
PATH="$CUDA_HOME/bin:$PATH" \
.venv/bin/python benchmarks/bench_decode_moe.py \
  --model /mnt/ssd/freetoken/ftw/DeepSeek-V4-Flash-0731 \
  --backend hybrid \
  --storage disk \
  --host-cache-gb 40 \
  --hybrid-fetch 3 \
  --mem-ratio 0.90 \
  --decode 64 \
  --num-tokens 2048 \
  --disable-prefill-overlap \
  --collect-moe-stats \
  --greedy \
  --no-graph \
  --json /mnt/ssd/freetoken/results/dsv4/final-disk-hybrid.json
```

Key result files:

- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-coalesced16-f3-cache32-decode16.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-coalesced16-f3-cache32-decode64.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-coalesced16-pageable-cache40-decode64.json`

## Experiment 8: Combined staging, layer-aware LRU, and bounded prefill overlap

Hybrid decode previously staged CPU-overflow experts, submitted the CPU work, and
then issued a separate disk staging call for GPU cache misses. The new path merges
both ID sets into one deduplicated disk request before CPU submission. This gives
the disk source one larger coalescing/read-pool batch and prevents `copy_missing`
from restaging the GPU subset.

The optional `FREETOKEN_DISK_CACHE_POLICY=layer_lru` policy protects a fair cache
floor for each of the 43 MoE layers. Capacity that a layer is not using remains
borrowable, and eviction first targets the oldest entry from a layer above its
floor. This avoids both the cross-layer churn of a single unconstrained LRU and the
125 unused entries observed with an initial hard-partition prototype.

All rows below use the same AIME-25 problem 0 prompt, greedy sampling, 64 requested
tokens (63 returned; 62 measured intervals), a 40 GiB host-cache setting,
automatic 726-slot GPU expert cache, eager decode, memory ratio 0.90, 16 disk-read
workers, and fetch cap 3 unless noted.

| Configuration | Decode tok/s | ms/token | Warm TTFT | Host hit | Physical reads | Host evictions | Output hash |
|---|---:|---:|---:|---:|---:|---:|---|
| Matched optimized offload control | 1.795 | 557.15 | 57.70 s | 46.31% | 122.82 GiB | 2,068 | `fbf178b2bde5` |
| Previous hybrid baseline | 1.868 | 535.46 | 58.02 s | 46.11% | 122.30 GiB | 2,026 | `fbf178b2bde5` |
| Combined hybrid staging | 1.881 | 531.74 | 57.73 s | 46.11% | 122.30 GiB | 2,026 | `fbf178b2bde5` |
| Hard per-layer LRU prototype | 1.923 | 520.07 | 58.08 s | 46.54% | 121.32 GiB | 1,797 | `fbf178b2bde5` |
| Borrowable per-layer LRU, 7 CPU workers | 1.965 | 508.90 | 57.78 s | 48.57% | 116.73 GiB | 1,579 | `fbf178b2bde5` |
| Borrowable per-layer LRU, 8 CPU workers | **1.982** | **504.43** | 57.90 s | **48.57%** | **116.73 GiB** | **1,579** | `fbf178b2bde5` |
| Same, fetch cap 2 | 1.963 | 509.44 | 57.74 s | 48.11% | 118.14 GiB | 1,692 | `fbf178b2bde5` |
| Two-buffer prefill, 8 CPU workers | 1.860 | 537.59 | **37.25 s** | 44.08% | 126.93 GiB | 2,142 | `fbf178b2bde5` |

The selected sustained-decode configuration improves on the previous hybrid
baseline by 6.1% and on the matched offload control by 10.4%. Fetch cap 2 loses
1.0%, so cap 3 remains selected. Eight CPU workers improve 0.9% over seven without
changing memory usage or output.

For disk prefill overlap, the source owns two parity-selected pinned layer buffers.
Layer N+1 begins its O_DIRECT read while layer N's GPU work executes. The second
approximately 3.2 GiB pinned layer is deducted from the pageable host LRU budget,
reducing capacity from 3,212 to 2,956 entries and keeping total host allocation at
the previously validated level. If the configured LRU budget is smaller than the
second staging layer, startup now rejects overlap before allocating or pinning that
buffer. The 64-token trial caused no meaningful swap-out growth and cut TTFT by
35.7%, but reduced sustained throughput by 6.2% relative to the selected
single-buffer configuration. Therefore:

- use single-buffer mode (`--disable-prefill-overlap`) for Figure-3-style long-CoT
  AIME requests;
- use bounded two-buffer prefill only when TTFT dominates short requests;
- do not exceed the tested 40 GiB setting on this 62 GiB host. The rejected 44 GiB
  configuration remains unsafe.

Selected long-CoT command:

```bash
CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13 \
PATH="$CUDA_HOME/bin:$PATH" \
FREETOKEN_DISK_READ_WORKERS=16 \
FREETOKEN_DISK_CACHE_POLICY=layer_lru \
.venv/bin/python benchmarks/bench_decode_moe.py \
  --model /mnt/ssd/freetoken/ftw/DeepSeek-V4-Flash-0731 \
  --backend hybrid --storage disk --host-cache-gb 40 \
  --hybrid-fetch 3 --cpu-threads 8 --mem-ratio 0.90 \
  --decode 64 --num-tokens 2048 --disable-prefill-overlap \
  --collect-moe-stats --greedy --no-graph
```

Focused validation: 38 checkpoint/MoE tests passed. The 16-token overlap pilot
also retained the established `03c286207d6b` greedy output hash and completed with
zero additional swap-out.

Key result files:

- `/mnt/ssd/freetoken/results/dsv4/disk-offload-coalesced16-pageable-cache40-decode64.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-combined-stage-cache40-decode64.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-combined-stage-layer-lru-soft-cache40-cpu8-decode64.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-double-prefill-layer-lru-cache40-cpu8-decode64.json`
- `/mnt/ssd/freetoken/results/dsv4/disk-hybrid-layer-lru-cache40-cpu8-fetch2-decode64.json`

## Experiment 9: Full Figure-3-style AIME-25 evaluation

[Figure 3 of the FreeToken paper](https://arxiv.org/pdf/2608.16157) defines W1 as
single-turn AIME math reasoning with long chain-of-thought decoding and no tools.
It reports per-request mean decode throughput and mean TTFT. This local run uses
that evaluation shape, including the paper's eight CPU threads for DSV4, but it is
not a reproduction of the paper's hardware result. The paper uses an RTX 5090 with
the model resident in 180 GiB of host DRAM; this machine has an RTX 3090, 62 GiB of
RAM, and serves most routed expert traffic from NVMe.

Protocol:

- all 30 problems from `math-ai/aime25`, in dataset order, exactly once (pass@1);
- dataset SHA-256
  `b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331`;
- one long-lived server, one request at a time, thinking enabled, normal EOS, no
  tools, and a 1,024-token maximum output;
- checkpoint temperature 1.0 and top-p 1.0, plus the runner's top-k 64 fallback;
- selected Experiment 8 configuration: hybrid execution, 40 GiB host-cache
  budget, borrowable layer-aware LRU, fetch cap 3, eight CPU workers, 16 disk-read
  workers, single-buffer prefill, eager decode, and memory ratio 0.90;
- each request row was appended, flushed, and `fsync`ed immediately. The final
  JSONL contains problem indices 0 through 29 once each, followed by one summary.

Exact command:

```bash
CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13 \
PATH="$CUDA_HOME/bin:$PATH" \
FREETOKEN_DISK_READ_WORKERS=16 \
FREETOKEN_DISK_CACHE_POLICY=layer_lru \
.venv/bin/python benchmarks/bench_aime_suite.py \
  --model /mnt/ssd/freetoken/ftw/DeepSeek-V4-Flash-0731 \
  --aime /home/lidaiqing/.cache/huggingface/hub/datasets--math-ai--aime25/snapshots/563bb8404243c5f09de6ec262f2db674fe5bce9b/test.jsonl \
  --problems all --decode 1024 \
  --backend hybrid --storage disk --host-cache-gb 40 \
  --hybrid-fetch 3 --cpu-threads 8 --mem-ratio 0.90 \
  --num-tokens 2048 --disable-prefill-overlap --no-graph \
  --include-output \
  --json /mnt/ssd/freetoken/results/dsv4/aime25-figure3-full-max1024.jsonl
```

Results:

| Metric | Result |
|---|---:|
| AIME-25 pass@1 | **9 / 30 (30.0%)** |
| Natural EOS / output-cap finishes | 9 / 21 |
| Completion tokens | 27,311 total; 910.4 mean; 1,023 median |
| Mean decode throughput | **1.910 tok/s** (523.54 ms/token) |
| Token-weighted decode throughput | **1.902 tok/s** over 27,281 intervals |
| First request / subsequent-request mean | 1.925 / 1.910 tok/s |
| Mean / p50 / p95 TTFT | 67.36 / 58.46 / 116.48 s |
| Mean per-request inter-token p50 / p95 | 515.09 / 691.40 ms |
| Host-cache hit rate | 67.47% |
| Physical expert reads | 13.43 TiB across 4,417,848 read operations |
| Effective physical disk-read rate | 1.519 GiB/s |
| Host-cache evictions / bypasses | 832,974 / 268,276 |
| Maximum reported VRAM | 19.03 GiB |

The long-run mean is 3.65% below the selected 1.982 tok/s 64-token result. That is
a small long-context penalty, and the almost identical first-request and
subsequent-request means show no sustained throughput decay across the suite. The
TTFT median remains close to the 64-token probe's 57.9 seconds; the 67.4-second
mean and 116.5-second p95 are driven by longer prompts (48 to 804 prompt tokens).

All nine correct responses ended naturally. Every incorrect capped response used
the server's full local length budget: a requested limit of 1,024 is reported by
the current API as 1,023 completion tokens, giving 1,022 measured inter-token
intervals. Because 21/30 requests were capped, 30.0% is a lower-bound quality
measurement for this protocol, not evidence that the model's full-length AIME-25
accuracy is 30.0%. A quality-oriented follow-up should use the normal 16k-scale
reasoning budget and multiple samples; that is much more expensive and is not
needed to validate this machine's sustained serving performance.

The paper reports 22--25 tok/s for DSV4 across its RTX 5090 workloads. The local
1.910 tok/s result must not be read as a direct regression against that number:
the paper keeps the expert pool in a much larger host-memory system, whereas this
run moved 13.43 TiB from NVMe. The full-suite evidence instead confirms that the
Experiment 8 setting is the best safe configuration tested on this 62 GiB host:
it preserves almost all of its short-probe throughput, held 13.6 GiB of memory
available, caused no additional swap-out, and stayed below 20 GiB reported VRAM.
Further large gains require reducing NVMe expert traffic (more host memory or a
smaller/more aggressively quantized expert pool), not raising the already-rejected
44 GiB host-cache allocation.

Raw result:

- `/mnt/ssd/freetoken/results/dsv4/aime25-figure3-full-max1024.jsonl`

## Next experiments

1. Reconcile the observed 1,024-requested / 1,023-reported completion-token
   convention with the OpenAI-compatible API contract.
2. Run a quality-oriented AIME evaluation with a 16k-scale output budget and
   multiple samples if accuracy, rather than systems performance, is the target.
3. Compare disk-backed output against a trusted external DeepSeek-V4 reference.
