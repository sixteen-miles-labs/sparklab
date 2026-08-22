# Inference Experiments

Last updated: 2026-08-22

## Summary

We validated `nvidia/Qwen3.6-35B-A3B-NVFP4` inference on an RTX 3090 using both the existing
RAM-backed expert path and the new synchronous NVMe-backed path.

Key findings:

- Disk-backed inference works without loading the complete 16.9 GiB routed-expert pool into host RAM.
- RAM and disk produced the same greedy output hash and expert-cache behavior.
- The correctness-first disk path decoded at 2.14 tok/s versus 18.16 tok/s from RAM.
- A 1 GiB bounded host expert LRU works correctly, but its 1.65% hit rate is too low to
  improve throughput on this prefill-heavy smoke test.

## Test system and model

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GiB VRAM, SM86 |
| CPU | Intel Core i9-9900K, 8 cores / 16 threads |
| Host memory | 64 GiB DDR4 |
| GPU link | PCIe 3.0 ×16 |
| Storage | Local ext4 NVMe mounted at `/mnt/ssd` |
| Software | FreeToken 0.1.2 development checkout; PyTorch 2.11; CUDA 13.0 |
| Model | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| FTW checkpoint | `/mnt/ssd/freetoken/ftw/Qwen3.6-35B-A3B-NVFP4` |

Checkpoint sizes:

- Hugging Face checkpoint: approximately 22 GiB
- Converted FTW checkpoint: approximately 20 GiB
- FTW disk backend: `O_DIRECT`

## Hardware comparison

### Systems from the FreeToken paper

The following values reproduce Table 1 from the
[FreeToken paper](https://arxiv.org/pdf/2608.16157):

| System | GPU | VRAM | PCIe | B_P | CPU (threads) | DRAM | B_H |
|---|---|---:|---:|---:|---|---:|---:|
| 5090 | RTX 5090 | 32 GB | 5.0 ×16 | 52.7 GB/s | 2× Xeon Gold 6459C (32) | DDR5 180 GiB | 77.3 GB/s |
| 4090 | RTX 4090 | 24 GB | 4.0 ×16 | 25.1 GB/s | 2× Xeon Platinum 8358P (32) | DDR4 240 GiB | 63.2 GB/s |
| 3090 | RTX 3090 | 24 GB | 4.0 ×16 | 25.3 GB/s | 2× Xeon Gold 6330 (28) | DDR4 180 GiB | 56.7 GB/s |
| 5090 desktop | RTX 5090 | 32 GB | 5.0 ×16 | 49.0 GB/s | Ryzen 9 9950X3D (32) | DDR5 192 GiB | 53.8 GB/s |
| 4060 laptop | RTX 4060 Laptop | 8 GB | 4.0 ×8 | 11.8 GB/s | Core i9-13900H (20) | LPDDR5 32 GiB | 47.5 GB/s |
| PRO 6000 | RTX PRO 6000 | 96 GB | 5.0 ×16 | 51.5 GB/s | Xeon Platinum 8559C (48) | DDR5 512 GiB | 178 GB/s |

`B_P` is measured host-to-device expert-transfer bandwidth. `B_H` is measured effective
bandwidth of the CPU-side MoE expert kernel. The paper measured both on deployed tensor
shapes rather than using peak hardware specifications.

### Our systems

| System | GPU / memory | Interconnect | B_P | CPU | Memory | B_H |
|---|---|---|---:|---|---|---:|
| Current 3090 | RTX 3090, 24 GiB VRAM | PCIe 3.0 ×16 | 12.08 GB/s | Core i9-9900K, 16 threads | 64 GiB DDR4 | 23.28 GB/s |
| GB10 | GB10 Blackwell, 128 GB unified | Coherent unified memory | N/A | 20-core Arm | 128 GB LPDDR5x | TBD |

The current-system values were measured with `ft bench bw --dtype nvfp4` using Qwen NVFP4
expert shapes, so they should not be treated as directly equivalent to every paper row.

GB10 does not copy experts over a discrete PCIe host-to-device link, so `B_P` is not
applicable. Its [official specification](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
lists 273 GB/s unified-memory bandwidth. We leave `B_H` as `TBD` until the FreeToken expert
kernel is measured on that machine.

## RTX 3090 method comparison

The paper rows below transcribe the RTX 3090 bars from
[Figure 5](https://arxiv.org/pdf/2608.16157). Methods are rows and experiment attributes and
results are columns.

| Result source | Method | Model format | Workload | Expert storage / execution | Decode |
|---|---|---|---|---|---:|
| Figure 5 | FreeToken | Qwen3.6-35B-A3B BF16 | W2 OpenCode + SWE | RAM hybrid GPU/CPU | 36.2 tok/s |
| Figure 5 | KTransformers | Qwen3.6-35B-A3B BF16 | W2 OpenCode + SWE | RAM, prefill-updated placement | 27.4 tok/s |
| Figure 5 | llama.cpp | Qwen3.6-35B-A3B BF16 | W2 OpenCode + SWE | RAM, static placement | 22.1 tok/s |
| Figure 5 | Ollama | Qwen3.6-35B-A3B BF16 | W2 OpenCode + SWE | RAM, static placement | 18.2 tok/s |
| Local controlled baseline | FreeToken RAM | Qwen3.6-35B-A3B NVFP4 | AIME reasoning | Complete FTW expert banks in RAM | 25.33 tok/s |
| Local matched control | FreeToken RAM | Qwen3.6-35B-A3B NVFP4 | AIME reasoning | Complete FTW expert banks in RAM | 18.16 tok/s |
| Local disk prototype | FreeToken disk | Qwen3.6-35B-A3B NVFP4 | AIME reasoning | Synchronous FTW rows from NVMe | 2.14 tok/s |
| Local bounded-cache prototype | FreeToken disk + 1 GiB LRU | Qwen3.6-35B-A3B NVFP4 | AIME reasoning | NVMe + 604-entry pinned RAM LRU | 2.11 tok/s |

These groups are not directly comparable. Figure 5 measures a coding-agent trajectory with
BF16 weights on the paper's rented RTX 3090 server (25.3 GB/s host-to-GPU and 56.7 GB/s CPU
MoE bandwidth). Our rows measure a short greedy AIME request with NVFP4 weights on the local
i9-9900K system (12.08 GB/s host-to-GPU and 23.28 GB/s CPU MoE bandwidth). The two local RAM
rows also use different KV allocations: 16,384 tokens for the controlled baseline and 4,096
tokens for the matched RAM/disk comparison.

## Experiment 1: Bandwidth baseline

Command:

```bash
ft bench bw --dtype nvfp4
```

| Method | CPU streaming | Host to GPU | Qwen NVFP4 CPU MoE | Expert gather |
|---|---:|---:|---:|---:|
| `ft bench bw` | ~28 GB/s | 12.08 GB/s | 23.28 GB/s | 7.35 GB/s |

## Experiment 2: Controlled RAM baseline

| Setting | Value |
|---|---|
| Expert source | Complete FTW expert banks in host RAM |
| GPU expert cache | 512 slots |
| KV allocation | 16,384 tokens |
| Decode mode | Batch size 1, greedy |

| Method | Decode | Time/token | VRAM | Cache miss | Missing / active per layer | Output hash |
|---|---:|---:|---:|---:|---:|---|
| RAM-backed FTW | 25.33 tok/s | 39.48 ms | 4.59 GiB | 63.31% | 5.07 / 8.00 | `c85ab4fe14ee` |

Result: `/mnt/ssd/freetoken/results/baseline/controlled-cache512.json`

## Experiment 3: RAM versus synchronous disk

### Goal

Verify that experts can be read directly from NVMe with bounded host staging memory while
preserving inference output.

The disk prototype keeps one expert layer in pinned host memory. It reads required FTW rows
from NVMe, stages them in that buffer, and reuses the existing host-to-GPU cache-copy path.

### Shared configuration

| Setting | Value |
|---|---|
| Batch size | 1 |
| Sampling | Greedy |
| GPU expert cache | 512 slots |
| KV allocation | 4,096 tokens |
| Requested / actual output | 16 / 15 tokens |
| Prompt | Identical in both runs |
| CUDA graphs | Disabled |
| MoE prefill overlap | Disabled |

### Results

| Method | Decode | Time/token | Warm TTFT | Prefill | VRAM | Cache miss | Missing / active per layer | Output hash |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| RAM-backed FTW | 18.16 tok/s | 55.06 ms | 2.54 s | 16.17 tok/s | 4.35 GiB | 63.81% | 5.105 / 8.0 | `8400f78e0fc8` |
| Synchronous disk | 2.14 tok/s | 466.90 ms | 23.68 s | 1.78 tok/s | 4.35 GiB | 63.81% | 5.105 / 8.0 | `8400f78e0fc8` |
| Disk / RAM | 0.12× | 8.48× | 9.32× | 0.11× | 1.00× | Same | Same | Match |

Disk I/O during the measured request:

| Method | Bank-row reads | Logical bytes | Physical bytes | Read-and-stage time | Physical throughput |
|---|---:|---:|---:|---:|---:|
| Synchronous disk | 79,818 | 23,621,019,648 | 23,648,264,192 (22.02 GiB) | 26.63 s | 0.83 GiB/s |

Result files:

- RAM: `/mnt/ssd/freetoken/results/disk/ram-control.json`
- Disk: `/mnt/ssd/freetoken/results/disk/sync-smoke.json`

### Conclusion

The identical output hash and cache statistics validate the disk path against the RAM
control. The approximately 8.5× decode slowdown is expected because reads are serialized;
this implementation is a correctness reference, not the performance target.

## Experiment 4: Bounded host expert LRU

### Goal

Add a configurable RAM tier between NVMe and the GPU expert cache without allowing the
complete expert pool to become resident in host memory.

The cache is keyed by `(layer, expert)`, stores all six NVFP4 banks for each entry, and uses
LRU eviction. `--moe-host-cache-gb 1` produced a capacity of 604 complete experts and
1,072,472,064 logical cache bytes. Page-aligned pinned allocation is checked against the
configured byte budget. The existing one-layer pinned staging buffer remains
fixed overhead outside this byte budget.

### Results

| Method | Host-cache budget | Entries | Host hit | Evictions | Disk read | Decode | Warm TTFT | Output hash |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Disk without host LRU | 0 GiB | 0 | — | — | 22.02 GiB | 2.14 tok/s | 23.68 s | `8400f78e0fc8` |
| Disk with host LRU | 1 GiB | 604 / 604 | 1.65% (220 / 13,303) | 13,083 | 21.66 GiB | 2.11 tok/s | 25.24 s | `8400f78e0fc8` |

Result: `/mnt/ssd/freetoken/results/disk/host-lru-1g.json`

### Conclusion

The byte limit, hit path, and repeated eviction path all worked while preserving the exact
RAM-control output. The cache avoided 220 expert reads and reduced physical disk traffic by
0.36 GiB (1.6%), but throughput did not improve. The repeated prefill touches most experts
in every layer and replaces the previous request's useful decode entries before they can be
reused. The next policy should isolate or bypass prefill admissions, then add asynchronous
reads so cache misses no longer serialize the inference stream.

## Next experiments

1. Prevent prefill scans from polluting the host expert LRU.
2. Coalesce duplicate expert reads.
3. Add asynchronous, queue-depth-aware disk reads.
4. Prefetch experts for upcoming layers.
5. Sweep 0.5, 1, 2, 4, 8, and 16 GiB host-cache budgets.
6. Measure peak process RSS and verify the host-memory budget.
7. Compare cold, warm, repeated, and distinct prompts.
8. Run `ft bench bw --dtype nvfp4` on GB10 and fill in its `B_H` result.

## Overall inference results

Methods are rows and measured metrics are columns. The controlled baseline used a larger KV
allocation than the matched comparison, so it should be treated as a separate configuration.

| Method | Expert source | KV tokens | Decode | Time/token | Prefill | Warm TTFT | VRAM | Cache miss | Output hash | Disk read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Controlled RAM baseline | Complete FTW banks in RAM | 16,384 | 25.33 tok/s | 39.48 ms | — | 1.53 s | 4.59 GiB | 63.31% | `c85ab4fe14ee` | — |
| Matched RAM control | Complete FTW banks in RAM | 4,096 | 18.16 tok/s | 55.06 ms | 16.17 tok/s | 2.54 s | 4.35 GiB | 63.81% | `8400f78e0fc8` | — |
| Synchronous disk | FTW rows from NVMe | 4,096 | 2.14 tok/s | 466.90 ms | 1.78 tok/s | 23.68 s | 4.35 GiB | 63.81% | `8400f78e0fc8` | 22.02 GiB |
| Disk + 1 GiB host LRU | NVMe + 604-entry RAM LRU | 4,096 | 2.11 tok/s | 473.91 ms | 1.69 tok/s | 25.24 s | 4.35 GiB | 63.81% | `8400f78e0fc8` | 21.66 GiB |
