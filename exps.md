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
