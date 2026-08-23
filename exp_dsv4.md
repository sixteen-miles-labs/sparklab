# DeepSeek-V4 Inference Experiments

Last updated: 2026-08-22

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
- A 32 GiB host expert cache plus a three-expert hybrid fetch cap is the fastest
  measured hybrid configuration: 2.15 tok/s, 3.33x the original 1 GiB disk
  baseline, with the same greedy output hash.
- Prefill remains synchronous and I/O-bound. Even the best decode setting has a
  warm TTFT of about 104.5 seconds for the 48-token test prompt.

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
the existing `copy_missing` path. Disk mode remains eager-only and still requires
prefill overlap to be disabled.

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

Recommended maximum-throughput disk-hybrid command on this machine:

```bash
CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13 \
PATH="$CUDA_HOME/bin:$PATH" \
.venv/bin/python benchmarks/bench_decode_moe.py \
  --model /mnt/ssd/freetoken/ftw/DeepSeek-V4-Flash-0731 \
  --backend hybrid \
  --storage disk \
  --host-cache-gb 32 \
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

## Next experiments

1. Add asynchronous, queue-depth-aware expert reads; synchronous prefill still
   dominates TTFT.
2. Run 64+ decode tokens and multiple prompts to characterize steady-state cache
   reuse and tail latency.
3. Confirm the requested/actual completion-token accounting convention.
4. Compare disk-backed output against a trusted external DeepSeek-V4 reference.
