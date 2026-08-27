# DeepSeek-V4 Stage A on NVIDIA GB10

Last updated: 2026-08-26

## Result

The best safe Stage A configuration measured on this GB10 is disk-backed MoE
offload with prefill overlap, a 4 GiB pinned host budget, a 5,900-entry GPU
expert cache, and 20 disk-read workers. On the 64-token AIME-25 probe it reached
**9.217 decode tok/s** (108.489 ms/token) with a 14.045 s warm TTFT. A final
independent run reached 8.967 tok/s with the same routing, I/O, and output hash.

This is 1.93x the 4.764 tok/s best no-overlap control. Prefill overlap reduced
the measured expert-cache miss rate from 21.42% to 3.15% and physical expert I/O
from 180.41 to 142.63 GiB. Greedy output stayed identical across every matched
configuration.

The next larger 6,000-entry cache was rejected: it used 1.25 GiB more device
memory and reached only 8.890 tok/s in the sustained probe, with exactly the same
miss count and physical reads as 5,900. The selected setting also completed with
2.79--3.52 GiB free after initialization across its sustained runs and no
observed swap growth.

## System and environment

| Component | Configuration |
|---|---|
| System | NVIDIA GB10, 128 GB unified memory |
| CPU | 20 physical Arm cores (10 Cortex-X925 + 10 Cortex-A725), performance governor |
| OS | Ubuntu NVIDIA kernel 6.17.0-1014-nvidia, aarch64 |
| NVIDIA driver / CUDA | 580.126.09 / CUDA 13.0 |
| Python | 3.11.15 in `.venv` |
| PyTorch / Triton | 2.11.0+cu130 / 3.6.0 |
| FlashInfer / SGLang kernel | 0.6.15.post1 / 0.4.5 |
| FreeToken revision | `ddc3b34` plus the benchmark-metadata change in this experiment |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Source checkpoint | `/home/lidaiqing/models/DeepSeek-V4-Flash-0731` |
| FTW checkpoint | `/home/lidaiqing/ftw/DeepSeek-V4-Flash-0731` |

The environment was created with `uv` and the checkout installed editable with
the development dependencies. CUDA allocation and a real GB10 tensor operation
were verified before model work. The focused DSV4 validation sets passed 64/64
and 44/44 tests. The broader suite passed 1,366 tests with 7 skipped and 13
deselected; its remaining failures were unrelated fixture/import issues. The
three initially missing-FlashInfer failures passed after installing the optional
FlashInfer dependency.

## Hardware profile and backend choice

`benchbw` measured the following ceilings and DSV4 kernels:

| Measurement | Result |
|---|---:|
| CPU STREAM read | 104.57 GB/s |
| Unified-memory linear H2D / D2H | 58.94 / 59.11 GB/s |
| DSV4 DS-FP4 CPU MoE | 27.70 GB/s |
| DSV4 DS-FP4 GPU expert gather | 85.18 GB/s |
| CPU / GPU-gather ratio | 0.325 |
| Recommended backend | `offload` |

The profile is saved at
`/home/lidaiqing/results/dsv4-gb10/benchbw-dsfp4.json`. The large gap between
CPU MoE and GPU expert gather is why Stage A uses `offload`, not `hybrid`.

## Checkpoint preparation

The downloaded checkpoint contains all 48 indexed safetensors shards. All
72,317 indexed tensor keys were present and no incomplete download files
remained.

Conversion command:

```bash
CUDA_HOME=/usr/local/cuda \
PATH=/usr/local/cuda/bin:$PATH \
FREETOKEN_CONVERT_PROGRESS=1 \
.venv/bin/ft checkpoint \
  --model /home/lidaiqing/models/DeepSeek-V4-Flash-0731 \
  --out /home/lidaiqing/ftw/DeepSeek-V4-Flash-0731 \
  --dtype bfloat16 \
  --moe-backend offload \
  --shard-gib 8 \
  --device cuda:0
```

| FTW property | Result |
|---|---:|
| Conversion time | 265.1 s |
| Expert layers | 43 / 43 |
| Dense weights / expert banks | 1,521 / 172 |
| Data size | 146.65 GiB (157,460,918,272 bytes) |
| Shards | 23 |
| Quantization | `ds_fp4` |
| Fingerprint | `72acedb1d9578b22` |

All 23 on-disk shard sizes and their sum match the FTW index. All 1,693 tensor
ranges are in bounds, the reader sees the indexed 1,521 weights and 172 expert
banks, and the fingerprint matches. The conversion log is
`/home/lidaiqing/results/dsv4-gb10/conversion.log`.

## Stage A tuning

Every serving probe used the first AIME-25 problem, batch size 1, greedy decode,
a 2,048-token KV allocation, eager execution, and one warm request followed by
one measured request. The 16-token probes returned 15 completion tokens and 14
measured intervals; the 64-token probes returned 63 completion tokens and 62
measured intervals.

| Configuration | Workers | Decode | tok/s | ms/token | TTFT | Miss rate | Physical I/O |
|---|---:|---:|---:|---:|---:|---:|---:|
| No overlap, auto cache (6,217) | 16 | 16 | 2.923 | 342.068 | 16.341 s | 45.61% | 159.04 GiB |
| No overlap, cache 6,217 | 16 | 64 | 4.709 | 212.354 | 16.304 s | 21.42% | 180.41 GiB |
| No overlap, cache 6,400 | 16 | 64 | 4.679 | 213.743 | 16.389 s | 21.42% | 180.41 GiB |
| No overlap, cache 6,217 | 20 | 64 | 4.764 | 209.921 | 16.219 s | 21.42% | 180.41 GiB |
| Overlap, host 4 GiB, cache 5,900 | 20 | 16 | 6.191 | 161.531 | 14.214 s | 13.23% | 142.63 GiB |
| **Overlap, host 4 GiB, cache 5,900** | **20** | **64** | **9.217** | **108.489** | **14.045 s** | **3.15%** | **142.63 GiB** |
| Overlap, host 4 GiB, cache 5,900 (confirmation) | 20 | 64 | 8.967 | 111.516 | 14.043 s | 3.15% | 142.63 GiB |
| Overlap, host 4 GiB, cache 6,000 | 20 | 16 | 6.266 | 159.593 | 14.313 s | 13.23% | 142.63 GiB |
| Overlap, host 4 GiB, cache 6,000 | 20 | 64 | 8.890 | 112.486 | 14.118 s | 3.15% | 142.63 GiB |

The 20-worker setting narrowly beat both 16 and 32 workers and matches the 20
physical CPU cores. A 4 GiB host budget is enough for two pinned layer staging
buffers plus a 65-entry host LRU. Enabling overlap is the material optimization:
it hides the disk-prefill path and leaves a much hotter GPU cache for measured
decode. Increasing the no-overlap cache to 6,400 did not change the working set,
and reduced throughput while consuming more unified memory.

All 16-token runs produced output SHA-1 `03c286207d6b`; all 64-token runs produced
`fbf178b2bde5`, matching the existing RTX controls.

## Selected command

Before a cold run, evict only the FTW shard pages from the Linux page cache. On
this unified-memory system, cached model files reduce the memory that CUDA
reports as free even when Linux still reports ample `MemAvailable`; targeted
`POSIX_FADV_DONTNEED` avoids requiring a global cache drop.

```bash
CUDA_HOME=/usr/local/cuda \
PATH=/usr/local/cuda/bin:$PATH \
FREETOKEN_DISK_READ_WORKERS=20 \
FREETOKEN_AIME25_JSONL=/home/lidaiqing/datasets/aime25/test.jsonl \
.venv/bin/python benchmarks/bench_decode_moe.py \
  --model /home/lidaiqing/ftw/DeepSeek-V4-Flash-0731 \
  --backend offload \
  --storage disk \
  --host-cache-gb 4 \
  --cache 5900 \
  --decode 64 \
  --num-tokens 2048 \
  --mem-ratio 0.90 \
  --collect-moe-stats \
  --greedy \
  --no-graph \
  --include-output \
  --json /home/lidaiqing/results/dsv4-gb10/stage-a-final-overlap-host4-cache5900-decode64.jsonl
```

The confirmation JSON records the requested configuration, 20 resolved reader
workers, two staging buffers, and the cache geometry: 5,900 expert slots, 16 DSV4
KV pages of 128 tokens, 43 MoE layers, 256 experts per layer, and 13,369,344 bytes
per cached expert. Both sustained 5,900-slot runs used 45,820 disk reads and
142.63 GiB physical I/O, with 0.56% host-LRU hits. The best run's p50/p99 event
latencies were 94.058/569.993 ms and server-reported device allocation was 83.40
GiB.

Result artifacts are under `/home/lidaiqing/results/dsv4-gb10/`. The best measured
row is `stage-a-overlap-host4-cache5900-decode64.jsonl`; the fully self-describing
confirmation is `stage-a-final-overlap-host4-cache5900-decode64.jsonl`.
