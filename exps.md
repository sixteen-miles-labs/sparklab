# Inference Experiments

## Qwen3.6-35B-A3B-NVFP4 on RTX 3090

Date: 2026-08-22

### System

- GPU: NVIDIA GeForce RTX 3090, 24 GiB VRAM (SM86)
- Host memory: 64 GiB
- Model: `nvidia/Qwen3.6-35B-A3B-NVFP4`
- FTW checkpoint: `/mnt/ssd/freetoken/ftw/Qwen3.6-35B-A3B-NVFP4`
- Storage: local ext4 NVMe mounted at `/mnt/ssd`
- PyTorch: 2.11 with CUDA 13.0
- FreeToken: 0.1.2 development checkout

The checkpoint occupies approximately 22 GiB in its downloaded Hugging Face form and 20
GiB after FTW conversion. FTW expert-row reads used the `O_DIRECT` path.

### Paper Table 1 with local systems

The first six rows reproduce Table 1 from the
[FreeToken paper](https://arxiv.org/pdf/2608.16157). `B_P` is host-to-device expert-transfer
bandwidth and `B_H` is effective CPU-side MoE expert-kernel bandwidth; the paper reports
measurements on deployed tensor shapes, not hardware peak specifications. The final two
rows add our systems and therefore are not paper results.

| System | GPU (VRAM) | PCIe | B_P (GB/s) | CPU (threads) | DRAM (GiB) | B_H (GB/s) |
|---|---|---:|---:|---|---:|---:|
| 5090 | RTX 5090 (32 GB) | 5.0 ×16 | 52.7 | 2× Xeon Gold 6459C (32) | DDR5 180 | 77.3 |
| 4090 | RTX 4090 (24 GB) | 4.0 ×16 | 25.1 | 2× Xeon Platinum 8358P (32) | DDR4 240 | 63.2 |
| 3090 | RTX 3090 (24 GB) | 4.0 ×16 | 25.3 | 2× Xeon Gold 6330 (28) | DDR4 180 | 56.7 |
| 5090 desktop | RTX 5090 (32 GB) | 5.0 ×16 | 49.0 | Ryzen 9 9950X3D (32) | DDR5 192 | 53.8 |
| 4060 laptop | RTX 4060 Laptop (8 GB) | 4.0 ×8 | 11.8 | Core i9-13900H (20) | LPDDR5 32 | 47.5 |
| PRO 6000 | RTX PRO 6000 (96 GB) | 5.0 ×16 | 51.5 | Xeon Platinum 8559C (48) | DDR5 512 | 178 |
| **Current 3090** | **RTX 3090 (24 GiB)** | **3.0 ×16** | **12.08** | **Core i9-9900K (16)** | **DDR4 64** | **23.28** |
| **GB10** | **GB10 Blackwell (128 GB unified)** | **Unified memory** | **N/A** | **20-core Arm (20)** | **LPDDR5x unified 128** | **TBD** |

The current-system `B_P` and `B_H` values come from `ft bench bw --dtype nvfp4`; they are
useful local measurements but are not directly interchangeable with every paper row because
our benchmark used Qwen NVFP4 expert shapes. The GB10 has coherent CPU/GPU unified memory,
so there is no discrete PCIe host-to-device expert-transfer stage to report as `B_P`. Its
official [hardware specification](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) is 128
GB LPDDR5x at 273 GB/s, but that peak is not substituted for `B_H`; FreeToken still needs to
measure the effective expert kernel on that machine.

### Initial bandwidth and RAM-backed baseline

`ft bench bw --dtype nvfp4` measured approximately 28 GB/s CPU streaming bandwidth, 12.08
GB/s host-to-device bandwidth, 23.28 GB/s Qwen NVFP4 CPU MoE bandwidth, and 7.35 GB/s gather
bandwidth.

With the regular RAM-backed FTW offload path, a 512-slot GPU expert cache, and a controlled
16,384-token KV allocation, decode reached 25.33 tok/s at 39.48 ms/token. VRAM usage was
4.59 GiB and the GPU expert-cache miss rate was 63.31% (5.07 missing experts out of 8 active
experts per layer). The deterministic output hash was `c85ab4fe14ee`.

Result: `/mnt/ssd/freetoken/results/baseline/controlled-cache512.json`

### Matched RAM versus synchronous disk experiment

The first disk implementation is a correctness-oriented synchronous path. It keeps one
full expert layer in pinned host staging memory, reads requested expert rows from FTW, and
then reuses the existing host-to-device cache-copy path. It does not load the complete 16.9
GiB routed-expert banks into host RAM. Disk mode currently requires the offload backend,
FTW, CUDA graphs disabled, and MoE prefill overlap disabled.

Both runs used batch size one, greedy decoding, a 512-slot GPU expert cache, a 4,096-token KV
allocation, identical prompts, and 16 requested output tokens. The server emitted 15 output
tokens in both cases.

| Metric | RAM-backed | Synchronous disk |
|---|---:|---:|
| Decode throughput | 18.16 tok/s | 2.14 tok/s |
| Time per token | 55.06 ms | 466.90 ms |
| Warm TTFT | 2.54 s | 23.68 s |
| Prefill throughput | 16.17 tok/s | 1.78 tok/s |
| Server VRAM | 4.35 GiB | 4.35 GiB |
| GPU expert-cache miss rate | 63.81% | 63.81% |
| Missing/active experts per layer | 5.105 / 8.0 | 5.105 / 8.0 |
| Greedy output SHA-1 prefix | `8400f78e0fc8` | `8400f78e0fc8` |

The matching output hash and cache activity validate the synchronous disk path against the
RAM control for this smoke test. During the measured disk request, FreeToken issued 79,818
bank-row reads, moved 23,648,264,192 physical bytes (22.02 GiB), and spent 26.63 seconds in
the synchronous read-and-stage operation, corresponding to approximately 0.83 GiB/s.

Results:

- RAM: `/mnt/ssd/freetoken/results/disk/ram-control.json`
- Disk: `/mnt/ssd/freetoken/results/disk/sync-smoke.json`

### Interpretation and next experiment

The experiment establishes that routed experts can be served explicitly from NVMe without
placing the complete expert bank in host RAM. The approximately 8.5x decode slowdown is
expected from the deliberately serialized implementation and is not a performance target.

The next step is a byte-budgeted pinned-host expert LRU, followed by coalesced asynchronous
reads and layer-aware prefetch. Subsequent experiments should report peak process RSS,
host-cache hit rate, disk bytes per generated token, queue depth, and cold versus warm
prompt behavior.
