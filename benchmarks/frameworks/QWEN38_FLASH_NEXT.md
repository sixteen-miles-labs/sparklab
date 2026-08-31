# Qwen3.8-Flash-Next cross-framework benchmark

This report measures batch-one text serving of Qwen3.8-Flash-Next on one NVIDIA
DGX Spark (GB10, 128 GB coherent memory, ARM64 Linux). It is separate from the
[Qwen3.6 comparison](README.md) because Qwen3.8-Flash-Next has a substantially
different QSA/Gated-DeltaNet architecture, 24,576 routed experts, and a 102.4 GB
external n-gram table.

## Results

| Framework | Version | Artifact | Status | Decode tok/s | Warm TTFT | Detail |
|---|---:|---|---|---:|---:|---|
| SparkLab | 0.1.0 | FTW NVFP4, pinned revision `cbbcf69f…` | Measured with startup warning | 16.42 | 0.386 s | QSA, CUDA graph, full immutable expert preload |
| FreeToken | 0.1.2 (`4b94bdc`) | Same FTW NVFP4 | Unsupported | — | — | Artifact ABI mismatch: expected `model.layers.0.linear_attn.out_proj.weight_scale` |
| vLLM | 0.28.0 | Publisher ModelOpt NVFP4 | Not run safely | — | — | Published checkpoint is 135 GB before runtime/KV overhead, exceeding GB10 coherent capacity |
| SGLang | 0.5.18 | Publisher ModelOpt NVFP4 | Not run safely | — | — | Same resident-checkpoint capacity constraint |
| llama.cpp | `9723942ad` | ggml-org Q8_0 GGUF | Not run safely | — | — | Official GGUF is 163 GB, exceeding coherent capacity |
| Ollama | 0.33.2 | Official `125b-mlx` | Unavailable | — | — | Published Ollama variants are MLX artifacts, not runnable by the NVIDIA Linux backend |
| KTransformers | 0.7.0.post1 | — | Unavailable | — | — | `kt-kernel` has no ARM64 wheel; this host cannot install the runtime |

“Not run safely” is not a performance score. The benchmark watchdog deliberately
does not start a framework when its immutable weight allocation already exceeds
the machine's 121.7 GiB usable coherent-memory capacity before KV cache, compute
buffers, and the 12 GiB operating-system reserve.

## Method

- One fixed reasoning prompt, concurrency one, temperature zero.
- One 32-token warmup followed by three 256-token measured streams.
- Warm TTFT is request start to the first non-empty content/reasoning SSE delta.
- Decode throughput is `(completion_tokens - 1) / (last delta - first delta)`.
- Context/KV capacity is 32,832 tokens with a 32,768-token sequence cap.
- SparkLab used QSA, page size 16, naive cache, Triton NVFP4 expert kernels,
  batch-one CUDA-graph replay, zero host expert LRU, and full expert preload.
- Every server runs alone under a watchdog that terminates it below 12 GiB
  `MemAvailable` or 2 GiB free swap.

The measured trial decode rates were 16.07, 17.08, and 16.42 tok/s. The median
TTFT was 0.386 seconds. All three requests returned exactly 256 completion tokens.
Full preload loaded 24,576 routed experts in 34.58 seconds and disabled decode-time
disk staging and expert-LRU bookkeeping.

## Operational caveat

The kernel log recorded five recoverable NVIDIA driver allocation warnings
(`NV_ERR_NO_MEMORY`) at 01:53:36 while resident weights were loading. There was no
Linux OOM kill (`oom_kill` remained zero), the server did not exit, swap did not grow,
and it subsequently completed full preload, CUDA-graph capture, and all four requests.
After graph capture the runtime reported 1.17 GiB of free device allocation headroom.
The throughput is therefore valid, but this exact full-preload/32K configuration has
a narrow driver-memory margin and should not be described as OOM-free.

This closely reproduces the earlier certified 64-token probe
`GB10-QWEN38-NVFP4-OPT-002` (16.61 tok/s, 0.403 s TTFT), while using the common
cross-framework prompt and three longer measured requests.

## FreeToken baseline limitation

FreeToken's current CLI recognizes the model through its `qsa_sparse` backend, but
FreeToken 0.1.2 cannot load SparkLab's pinned FTW artifact. The backend exits during
weight loading because its expected linear-attention scale key is absent from the
newer artifact ABI. Downloading FreeToken's separately documented 135 GB checkpoint
would change both the checkpoint tuple and execution policy, so it is not presented
as a controlled baseline in this report.

## Reproduce

```bash
python benchmarks/frameworks/run_framework.py sparklab \
  --model-family qwen38 \
  --ftw "$HOME/.sparklab/models/qwen3.8-flash-next/prepared/0.5.0" \
  --output-dir benchmarks/frameworks/results/qwen3.8-flash-next \
  --startup-timeout 1800
```

Raw evidence:

- [`results/qwen3.8-flash-next/sparklab.json`](results/qwen3.8-flash-next/sparklab.json)
- [`results/qwen3.8-flash-next/sparklab.log`](results/qwen3.8-flash-next/sparklab.log)
- [`results/qwen3.8-flash-next/freetoken.json`](results/qwen3.8-flash-next/freetoken.json)
- [`results/qwen3.8-flash-next/freetoken.log`](results/qwen3.8-flash-next/freetoken.log)
