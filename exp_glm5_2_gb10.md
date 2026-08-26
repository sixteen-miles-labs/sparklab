# GLM-5.2 NVFP4 on NVIDIA GB10

Last updated: 2026-08-26

## Experiment status

The full checkpoint transfer, FTW conversion, and serving measurements are in
progress. This report is complete only when the selected 64-token result and a
stable 256-token confirmation have been recorded and audited.

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

Conversion and integrity results: **TBD**.

## Serving benchmark

All comparison runs use the first AIME-25 problem, batch size 1, greedy decode,
eager execution, a fixed KV budget, one warm request, and one measured request.
The source and FTW shard pages are evicted with targeted
`POSIX_FADV_DONTNEED` before cold initialization so Linux file cache does not
consume GB10 unified-memory headroom.

| Configuration | Decode | tok/s | ms/token | TTFT | Miss rate | Physical I/O | Output hash |
|---|---:|---:|---:|---:|---:|---:|---|
| Synchronous control | 64 | TBD | TBD | TBD | TBD | TBD | TBD |
| Selected GB10 run | 64 | TBD | TBD | TBD | TBD | TBD | TBD |
| Selected confirmation | 256 | TBD | TBD | TBD | TBD | TBD | TBD |

Selected command and conclusion: **TBD**.
