---
pipeline_tag: text-generation
base_model:
- nvidia/Kimi-K3-NVFP4
- moonshotai/Kimi-K3
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
library_name: sparklab
tags:
- kimi-k3
- nvidia
- modelopt
- nvfp4
- quantized
- sparklab
- ftw
- dgx-spark
---

# Kimi K3 NVFP4 — SparkLab FTW

This is a ready-to-run **SparkLab FTW** checkpoint for
[`nvidia/Kimi-K3-NVFP4`](https://huggingface.co/nvidia/Kimi-K3-NVFP4), optimized
for text inference on one **NVIDIA DGX Spark** with the Grace Blackwell GB10
Superchip.

> **SparkLab source:** https://github.com/sixteen-miles-labs/sparklab

The artifact is large: its FTW payload occupies **1,610,936,311,808 bytes**
(approximately 1.465 TiB) across 194 shards. Store it on fast local NVMe.

## What this repository contains

This repository does not introduce a new model, fine-tune, or quantization. It
repackages NVIDIA's published checkpoint into SparkLab's self-contained, aligned FTW
format so individual routed-expert rows can be read from NVMe:

1. [Moonshot AI](https://huggingface.co/moonshotai) developed
   [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3).
2. [NVIDIA](https://huggingface.co/nvidia) produced
   [Kimi-K3-NVFP4](https://huggingface.co/nvidia/Kimi-K3-NVFP4) with
   [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).
3. [SparkLab](https://github.com/sixteen-miles-labs/sparklab) provides the native
   inference runtime, FTW conversion, GB10 kernels, NVMe-backed MoE execution, model
   recipes, readiness checks, capacity planning, and serving workflow.
4. [OakMind AI](https://huggingface.co/oakmindai) performed, validated, documented,
   and published this FTW conversion.

The exact source revision is
`f8c5234a0a880bcc6cbf779a315e7ee2f405b812`. The FTW fingerprint is
`534cbc4565d4279d`. Routed experts retain NVIDIA's ModelOpt NVFP4 representation,
supported attention/KDA projections retain block FP8, and the other text tensors retain
their source precision. The current artifact is text-only; SparkLab's vision path is not
included or validated.

## Why FTW is required here

Kimi K3 has 2.8T total parameters and activates 16 of 896 routed experts per token. The
whole model cannot reside in the GB10's 128 GB coherent memory. FTW makes expert rows
independently addressable, allowing SparkLab to retain a bounded GPU expert cache while
streaming misses from NVMe.

FTW improves artifact loading and enables this bounded execution policy; it does not make
the model interactive. The measured configuration remains storage-bound.

## Run with SparkLab on NVIDIA DGX Spark

Install SparkLab:

```bash
git clone https://github.com/sixteen-miles-labs/sparklab.git
cd sparklab
./install.sh
```

SparkLab can download this pinned runtime artifact automatically:

```bash
sparklab doctor --storage-path ~/models
sparklab plan kimi-k3 --root ~/models --prepare
sparklab pull kimi-k3 --root ~/models --prepare
```

Or download the repository directly:

```bash
hf download oakmindai/Kimi-K3-NVFP4-FTW \
  --local-dir ~/models/Kimi-K3-NVFP4-FTW
```

The validated one-GB10 server configuration is:

```bash
sparklab serve \
  --model ~/models/Kimi-K3-NVFP4-FTW \
  --kimi-mlp-fp8 \
  --disable-startup-prefill-warmup \
  --moe-backend offload \
  --moe-storage disk \
  --moe-host-cache-gb 0 \
  --moe-cache-size 896 \
  --moe-cache-policy layer_lru \
  --memory-ratio 0.85 \
  --num-tokens 1024 \
  --disable-moe-prefill-overlap \
  --moe-prefill-sparse-max-tokens 256 \
  --nvfp4-backend triton \
  --cuda-graph-max-bs 0 \
  --max-running-requests 1 \
  --host 127.0.0.1 \
  --port 8000
```

The resident-profile flag quantizes 188 BF16 dense/shared/embedding/head
matrices to per-output-row FP8 while loading. This saves 14.176 GiB of resident memory;
it does not alter the routed NVFP4 expert banks stored in this repository. Disabling the
flag does not fit the validated 128 GB configuration.

Send a request after `/health` reports that the server is ready:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Kimi-K3-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 16,
    "stream": false
  }'
```

## GB10 validation

SparkLab promoted the complete artifact through exact 1-, 16-, 64-, and 256-token
single-request probes under no-swap service cgroups.

| Measurement | Result |
|---|---:|
| Decode throughput | 0.1613 tok/s |
| Warm TTFT | 395.405 s |
| Completion | 256 / 256 tokens |
| Minimum `MemAvailable` | 17.68 GiB |
| Device allocation | 74.86 GiB |
| Runtime OOM kills | 0 |
| Runtime swap-out | 0 |

This is a bounded-capacity result, not a quality or interactivity claim. The 256-token
response stopped while reasoning before emitting the expected AIME answer, and its greedy
text diverged from the shorter ladder after a shared prefix. The experiment report and
machine-readable evidence are in
[`exps/exp_kimik3_gb10.md`](https://github.com/sixteen-miles-labs/sparklab/blob/main/exps/exp_kimik3_gb10.md)
and
[`GB10-KIMI-001.json`](https://github.com/sixteen-miles-labs/sparklab/blob/main/benchmarks/gb10/results/GB10-KIMI-001.json).

## Limitations

- This FTW artifact and the published SparkLab path are text-only.
- The validated configuration is batch one on a single 128 GB GB10.
- TTFT is approximately 6.5 minutes because the current 129-token prefill fallback scans
  all 896 experts across 92 MoE layers.
- Decode is NVMe-bound at an 80.23% expert-cache miss rate in the 256-token probe.
- Cross-rung greedy determinism, answer correctness, long-context behavior, concurrency,
  agent capability, and endurance are not established.
- The checkpoint is Experimental and is not a SparkLab certification claim.

## Credits and license

- Architecture and base weights: [Moonshot AI / Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)
- NVFP4/FP8 checkpoint: [NVIDIA](https://huggingface.co/nvidia/Kimi-K3-NVFP4), using
  [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- Native runtime, FTW format, conversion, kernels, workflow, and deployment:
  [SparkLab](https://github.com/sixteen-miles-labs/sparklab)
- Research ancestry: [FreeToken](https://github.com/FlashML-org/FreeToken)
- FTW conversion and publishing: [OakMind AI](https://huggingface.co/oakmindai)

Use is governed by the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)
and the additional upstream Kimi K3 terms included in `LICENSE`. Review the
[base model card](https://huggingface.co/moonshotai/Kimi-K3) and
[quantized source card](https://huggingface.co/nvidia/Kimi-K3-NVFP4/tree/f8c5234a0a880bcc6cbf779a315e7ee2f405b812)
for the complete intended-use, evaluation, safety, and license information.

## Citation

```bibtex
@software{sparklab2026,
  title  = {SparkLab: Frontier Inference on NVIDIA DGX Spark},
  author = {Sixteen Miles Labs},
  year   = {2026},
  url    = {https://github.com/sixteen-miles-labs/sparklab}
}

@software{freetoken2025,
  title = {FreeToken},
  author = {FlashML.org},
  year = {2025},
  url = {https://github.com/FlashML-org/FreeToken}
}
```
