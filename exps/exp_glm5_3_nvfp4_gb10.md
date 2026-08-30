# GLM-5.3 Flash NVFP4 on NVIDIA GB10

Last updated: 2026-08-30

## Result

The complete pinned `oakmindai/GLM-5.3-Flash-NVFP4-FTW` artifact was served through
SparkLab's OpenAI-compatible streaming API on one NVIDIA GB10. Fusing the model's
multi-stream hyper-connections and preparing KDA convolution weights once at load time
raised the fixed-geometry probe from **4.994 to 6.271 tok/s**, a **25.58%** improvement,
while reducing decode latency from **200.24 to 159.45 ms/token**. Warm TTFT remained
effectively flat at **5.681 s**.

The optimized probe now passes the Frontier throughput and TTFT thresholds. It is not a
Frontier admission result: the 256-token response stopped before the final answer, the
host had pre-existing swap use, and the context, capability, quality, and endurance gates
remain outstanding. A second optimized run reproduced the same greedy output hash at
6.284 tok/s. Compact evidence:
[`GB10-GLM53-MHC-003`](../benchmarks/gb10/results/GB10-GLM53-MHC-003.json).

## Artifact and system

| Component | Value |
|---|---|
| Source | `RedHatAI/GLM-5.3-Flash-NVFP4` at `9eaeadaf026871a90640e32c0604f6ab0b2d641d` |
| Runtime artifact | `oakmindai/GLM-5.3-Flash-NVFP4-FTW` at `f296cec0baceb2276121efe76f14d61b62c1e47d` |
| Prepared artifact | FTW NVFP4 + KDA FP8, 184,716,947,456 physical bytes, fingerprint `4c021651a1e61802` |
| FTW index | 23 shards, 1,634 entries, 184,715,638,396 logical bytes |
| Validation | Every indexed shard, tensor range, and physical boundary validated |
| Engine revision | `8c35d0da66ccc318ba28d4fa53e30a82f49181e9`, clean tracked worktree |
| Hardware | NVIDIA GB10, SM121, 128 GB unified memory, NVMe |
| Software | Linux 6.17.0-1014-nvidia; CUDA 13.0; PyTorch 2.11.0+cu130; Triton 3.6.0 |

The runtime artifact preserves the publisher's NVFP4 routed experts and stores the
bandwidth-dominant KDA q/k/v/o projections as per-output-row FP8 W8A16. Recurrent gates,
DSA projections, shared experts, embeddings, and the output head retain source precision.

## Method

The paired runs used AIME-25 problem 0, batch size one, greedy sampling, 256 requested
completion tokens, one warm request, and one measured request. Timing spans the streaming
HTTP API. Both processes ran the same clean revision and pinned artifact with 6,711 expert
cache slots and 8,691 KV tokens. The control set `SPARKLAB_DEBUG_GLM53_EAGER_MHC=1`; the
optimized run changed only the mHC execution path. Disabling the installed beta kernel
cache ensured both paths compiled against the current stable runtime.

The runtime used DSA attention, NVMe-backed MoE, no pageable host expert LRU, one pinned
staging layer, 16 disk readers, a 0.96 unified-memory ratio, sparse prefill through 512
tokens, and no CUDA graph.

```bash
CUDA_VISIBLE_DEVICES=0 SPARKLAB_DISABLE_KERNEL_CACHE=1 \
SPARKLAB_DISK_READ_WORKERS=16 PYTHONPATH=python:. \
  .venv/bin/python benchmarks/bench_decode_moe.py \
  --model /path/to/glm-5.3-flash/prepared/0.3.2 \
  --recipe glm-5.3-flash \
  --backend offload --storage disk --attention-backend dsa \
  --nvfp4-backend triton --host-cache-gb 0 \
  --cache 6711 --mem-ratio 0.96 --num-tokens 8691 \
  --disable-prefill-overlap \
  --prefill-hit-d2d --prefill-sparse-max-tokens 512 \
  --page-size 1 --cache-type naive --no-graph \
  --collect-moe-stats --include-output --greedy --decode 256
```

## Measurements

| Metric | Value |
|---|---:|
| Decode throughput | 6.271 tok/s |
| Decode latency | 159.454 ms/token |
| Cold / warm TTFT | 10.687 / 5.681 s |
| Inter-token p50 / p99 | 154.48 / 240.40 ms |
| Device allocation | 101.58 GiB |
| Minimum host memory available | 4.63 GiB |
| Expert cache miss rate | 8.50% |
| Expert disk reads | 59,604 operations / 131.12 GiB |
| Estimated request energy | 1,339 J |
| Output hash | `d3523265958c`, reproduced twice |

| Paired comparison | Eager mHC | Fused mHC | Change |
|---|---:|---:|---:|
| Decode throughput | 4.994 tok/s | 6.271 tok/s | +25.58% |
| Decode latency | 200.242 ms/token | 159.454 ms/token | -20.37% |
| Inter-token p50 | 195.161 ms | 154.483 ms | -20.84% |
| Warm TTFT | 5.678 s | 5.681 s | -0.04% |
| Estimated request energy | 1,560 J | 1,339 J | -14.15% |

The API returned 256 completion tokens and produced 255 timed decode steps. The scoped
run had no OOM, restart, swap growth, or swap-out pages, but recorded 191 swap-in pages
from pre-existing host swap use. It therefore does not pass the strict memory gate.

## Findings and limitations

- GLM-5.3 has no RoPE attention sub-dimension. Its NoPE-only DSA path required the
  optional zero-width dot product to compile away; that path now has a GPU regression test.
- FlashInfer 0.6.15's SM12x route packer fixes a 4,096-route tile. With 288 experts rounded
  to 512 lanes, it exceeds Triton's 1,048,576-element tensor limit, so this recipe uses
  FreeToken's native Triton NVFP4 backend.
- Two KDA bugs caused the earlier repeated output: the fused kernel used an older
  clamp/softplus forget gate instead of GLM-5.3's bounded sigmoid rule, and it treated
  channel-major convolution views as feature-contiguous, scrambling Q/K/V addresses.
- The model applies hyper-connections twice in each of 45 blocks: 90 small eager GPU
  sequences per generated token. A one-token microbenchmark fell from 599.88 to 58.10
  microseconds per forward-plus-expand call after fusing RMS normalization, Sinkhorn and
  pre-collapse, and post-expansion. That is a 10.3x kernel-path speedup.
- KDA q/k/v depthwise-convolution weights are now packed once in `prepare_for_runtime`
  instead of concatenated on every step. This removes another approximately 0.23 ms/token.
- Fused FP32 mHC changes reduction order relative to eager PyTorch. CUDA regression tests
  establish numerical equivalence, and the optimized output hash and expert routes
  reproduced across two full runs. The eager and optimized greedy hashes differ, so the
  evidence does not claim bitwise equivalence to the old path.
- The optimized route performed 2.80 GiB more physical expert I/O than the eager route;
  the 25.58% throughput gain therefore includes that additional storage work.
- A real layer-0 weight probe from the earlier KDA correction matches the reference
  recurrence at cosine 0.99999994, and an earlier complete-checkpoint probe produced the
  correct AIME solution. The current fixed response stopped at its token cap, so the
  exact 64K context, capability, full quality, and endurance gates remain outstanding.
