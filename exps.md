# Inference Experiments

Last updated: 2026-08-22

## Summary

We validated `nvidia/Qwen3.6-35B-A3B-NVFP4` inference on an RTX 3090 using both the existing
RAM-backed expert path and the new synchronous NVMe-backed path.

Key findings:

- Disk-backed inference works without loading the complete 16.9 GiB routed-expert pool into host RAM.
- RAM and disk produced the same greedy output hash and expert-cache behavior.
- The correctness-first disk path decoded at 2.14 tok/s versus 18.16 tok/s from RAM.
- The next optimization is a bounded host expert cache with asynchronous reads and prefetch.

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

## Next experiments

1. Add a byte-budgeted pinned-host expert LRU.
2. Coalesce duplicate expert reads.
3. Add asynchronous, queue-depth-aware disk reads.
4. Prefetch experts for upcoming layers.
5. Measure peak process RSS and verify the host-memory budget.
6. Compare cold, warm, repeated, and distinct prompts.
7. Run `ft bench bw --dtype nvfp4` on GB10 and fill in its `B_H` result.

## Overall inference results

Methods are rows and measured metrics are columns. The controlled baseline used a larger KV
allocation than the matched comparison, so it should be treated as a separate configuration.

| Method | Expert source | KV tokens | Decode | Time/token | Prefill | Warm TTFT | VRAM | Cache miss | Output hash | Disk read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Controlled RAM baseline | Complete FTW banks in RAM | 16,384 | 25.33 tok/s | 39.48 ms | — | 1.53 s | 4.59 GiB | 63.31% | `c85ab4fe14ee` | — |
| Matched RAM control | Complete FTW banks in RAM | 4,096 | 18.16 tok/s | 55.06 ms | 16.17 tok/s | 2.54 s | 4.35 GiB | 63.81% | `8400f78e0fc8` | — |
| Synchronous disk | FTW rows from NVMe | 4,096 | 2.14 tok/s | 466.90 ms | 1.78 tok/s | 23.68 s | 4.35 GiB | 63.81% | `8400f78e0fc8` | 22.02 GiB |
