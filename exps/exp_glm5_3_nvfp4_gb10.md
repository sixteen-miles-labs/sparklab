# GLM-5.3 Flash NVFP4 on NVIDIA GB10

Last updated: 2026-08-28

## Result

The complete pinned `RedHatAI/GLM-5.3-Flash-NVFP4` checkpoint was converted into
Spark Lab's FTW NVFP4 layout and served through the OpenAI-compatible streaming API on
one NVIDIA GB10. After correcting the GLM-5.3 KDA recurrence and tensor layout, the
measured request decoded at **3.27 tok/s** with **7.121 s warm
TTFT**.

The greedy AIME-25 probe produced the correct solution set, `b = 21, 49`, whose requested
sum is the reference answer `70`. This is successful full-checkpoint loading and a
single output-correctness probe, but it is not a full quality suite or Frontier admission
result; the performance thresholds were not met. Compact evidence:
[`GB10-GLM53-NVFP4-001`](../benchmarks/gb10/results/GB10-GLM53-NVFP4-001.json).

## Artifact and system

| Component | Value |
|---|---|
| Source | `RedHatAI/GLM-5.3-Flash-NVFP4` |
| Revision | `9eaeadaf026871a90640e32c0604f6ab0b2d641d` |
| Prepared artifact | FTW NVFP4, 189,276,450,816 physical bytes, fingerprint `93b1de335dd523e5` |
| FTW index | 24 shards, 1,498 entries, 189,275,141,756 logical bytes |
| Validation | 146,745 source tensors in 10 shards and every prepared entry validated |
| Engine revision | `a1c3c3d59e7beb72bda90458f646155b8de7ff7a`, dirty tracked worktree |
| Hardware | NVIDIA GB10, SM121, 128 GB unified memory, NVMe |
| Software | Linux 6.17.0-1014-nvidia; CUDA 13.0; PyTorch 2.11.0+cu130; Triton 3.6.0 |

The source index publishes a stale logical total: 190,197,820,540 bytes versus
190,204,505,212 bytes computed from its tensor headers. Every name, declared shard,
tensor range, and physical shard boundary validates; Spark Lab records the 6,684,672-byte
metadata discrepancy without treating the complete snapshot as corrupt.

## Method

The run used AIME-25 problem 0, batch size one, greedy sampling, 256 requested completion
tokens, one warm request, and one measured request. Timing spans the streaming HTTP API.
The runtime used DSA attention, NVMe-backed MoE, a 4 GiB host expert-cache budget, 16 disk
readers, automatic GPU expert-cache sizing, sparse prefill through 512 tokens, and no CUDA
graph.

```bash
CUDA_VISIBLE_DEVICES=0 FREETOKEN_DISK_READ_WORKERS=16 PYTHONPATH=python:. \
  .venv/bin/python benchmarks/bench_decode_moe.py \
  --model /path/to/glm-5.3-flash/prepared/0.3.1 \
  --recipe glm-5.3-flash \
  --backend offload --storage disk --attention-backend dsa \
  --nvfp4-backend triton --host-cache-gb 4 \
  --prefill-hit-d2d --prefill-sparse-max-tokens 512 \
  --page-size 1 --cache-type naive --no-graph \
  --collect-moe-stats --include-output --greedy --decode 256
```

## Measurements

| Metric | Value |
|---|---:|
| Decode throughput | 3.273 tok/s |
| Warm TTFT | 7.121 s |
| Inter-token p50 / p99 | 283.72 / 527.10 ms |
| Device allocation | 92.04 GiB |
| Minimum host memory available | 10.95 GiB |
| Expert cache miss rate | 12.25% |
| Expert disk reads | 83,856 operations / 184.47 GiB |
| Output hash | `97e9756e0869` |

The API returned 255 completion tokens and produced 254 timed decode steps. No pages were
swapped out during the measured request, but 193 pages were swapped in and the host had
pre-existing swap usage. The service completed without an OOM or restart.

## Findings and limitations

- GLM-5.3 has no RoPE attention sub-dimension. Its NoPE-only DSA path required the
  optional zero-width dot product to compile away; that path now has a GPU regression test.
- FlashInfer 0.6.15's SM12x route packer fixes a 4,096-route tile. With 288 experts rounded
  to 512 lanes, it exceeds Triton's 1,048,576-element tensor limit, so this recipe uses
  FreeToken's native Triton NVFP4 backend.
- Two KDA bugs caused the earlier repeated output: the fused kernel used an older
  clamp/softplus forget gate instead of GLM-5.3's bounded sigmoid rule, and it treated
  channel-major convolution views as feature-contiguous, scrambling Q/K/V addresses.
- With both fixes, a real layer-0 weight probe matches the reference recurrence at
  cosine 0.99999994 and the complete checkpoint produces the correct AIME solution.
  The exact 64K context, capability, full quality, and endurance gates remain outstanding.
