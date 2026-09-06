---
pipeline_tag: text-generation
base_model: Inferact/Qwen3.8-27B-NVFP4
license: apache-2.0
library_name: sparklab
tags: [sparklab, ftw, nvfp4, dgx-spark, text-generation, modelopt]
---

# Qwen3.8-27B NVFP4 — SparkLab FTW

This is an **experimental, text-only SparkLab FTW checkpoint** for Qwen3.8-27B
on one NVIDIA DGX Spark with 128 GB coherent unified memory.

[SparkLab](https://github.com/sixteen-miles-labs/sparklab) provides a GB10-native
inference engine, model recipes, hardware checks, memory planning, artifact
preparation, and OpenAI-/Anthropic-compatible serving APIs.

## What this repository contains

This repository does not introduce a newly trained model or a new quantization.
It repackages [Inferact/Qwen3.8-27B-NVFP4](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4)
into the native FreeToken Weight (FTW) layout used by SparkLab.

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a dense 27B model.
- Quantized source revision: `6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`.
- FTW fingerprint: `af78f7411817bb73`.
- Three FTW weight shards, totaling **24,617,562,112 bytes** (24.62 GB).
- `freetoken_weight.json`, configuration, tokenizer, chat template, generation settings, and upstream license.

The conversion prepares the published ModelOpt NVFP4 weights for the native
loader; non-quantized tensors remain in their runtime conversion layout. No
additional model training is performed. This is a text-model artifact, not a
standard Transformers/vLLM safetensors checkpoint. Inherited processor metadata
does not enable images or video in SparkLab.

The optional DFlash2 draft is **not included**. Download it separately as described
below; its weights and license belong to its own repository.

## Why use FTW?

FTW performs tensor-layout preparation ahead of time for reproducible native
loading. This dense model runs resident in unified memory; it does not require
NVMe expert offloading. FTW by itself is not a claim of increased model quality
or faster steady-state generation.

## Run with SparkLab on NVIDIA DGX Spark

Use ARM64 Linux / DGX OS, GB10/SM121, CUDA 13, and local NVMe. A current source
installation is recommended, especially for the optimized DFlash2 path.

```bash
git clone https://github.com/sixteen-miles-labs/sparklab.git
cd sparklab
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

hf download oakmindai/Qwen3.8-27B-NVFP4-FTW \
  --local-dir /path/to/models/Qwen3.8-27B-NVFP4-FTW

sparklab doctor --storage-path /path/to/models
sparklab serve \
  --model /path/to/models/Qwen3.8-27B-NVFP4-FTW \
  --nvfp4-backend triton \
  --cache-type radix --page-size 16 \
  --cuda-graph-max-bs 1 --max-running-requests 1 \
  --num-tokens 65536 --max-seq-len-override 65536 \
  --host 127.0.0.1 --port 1919
```

Replace `/path/to/models` with your local model directory. These instructions
download the prebuilt target directly; the current catalog recipe may still
prepare it from Inferact rather than download this Hub repository.

Once ready, check the model ID and send a request:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-NVFP4-FTW","messages":[{"role":"user","content":"Hello!"}],"temperature":0,"max_tokens":128,"stream":true}'
```

### Optional DFlash2 single-stream profile

Download the pinned draft:

```bash
hf download maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal \
  --revision bd7a934213c47a9e7ef69eef36bb3325f47fd1f1 \
  --local-dir /path/to/models/qwen3.8-27b-dflash2
```

Instead of the target-only server, launch the measured 16K-capacity configuration:

```bash
sparklab serve \
  --model /path/to/models/Qwen3.8-27B-NVFP4-FTW \
  --nvfp4-backend triton --attention-backend triton \
  --cache-type radix --page-size 16 \
  --cuda-graph-max-bs 1 --max-running-requests 1 \
  --num-tokens 16384 --max-seq-len-override 16384 \
  --speculative-method dflash2 --speculative-tokens 12 \
  --speculative-draft-model /path/to/models/qwen3.8-27b-dflash2 \
  --host 127.0.0.1 --port 1919
```

## Performance and validation limits

All numbers below are **single-stream**, not aggregate concurrent throughput.

| Profile / workload | Decode | Warm TTFT |
|---|---:|---:|
| Recorded target-only baseline | 8.83 tok/s | 0.144 s |
| Optimized DFlash2-12, 128-token probe | 45.88 tok/s | 0.152 s |

The DFlash2 result is a three-trial median and reproduced the original target-only
output on that short probe. Against a fresh matched DFlash2-8 control (37.69 tok/s),
the improvement was 21.7%. A separate 512-token, thinking-off sweep measured
59.58 tok/s for math, 37.51 for coding, and 19.26 for prose. These are individual
workloads, not general task-suite averages. Full longer traces can differ due to
floating-point rounding from verification grouping.

See [DFlash2 evidence](https://github.com/sixteen-miles-labs/sparklab/blob/main/benchmarks/gb10/results/GB10-QWEN38-DFLASH-004.json)
and [target-only evidence](https://github.com/sixteen-miles-labs/sparklab/blob/main/benchmarks/gb10/results/GB10-QWEN38-27B-001.json).

- Target-only exact 65,536-token recall and reasoning/tool/coding probes passed.
- The declared upstream 262K context has not been validated on this GB10 path.
- DFlash2 is opt-in and batch-one greedy; 64K speculative context, sampling,
  concurrency, and endurance certification remain outstanding.
- Short output parity does not establish broad quality equivalence. Full Fast-tier
  certification, including a clean-revision 60-minute endurance run, remains pending.
- This upload reuses the previously tested artifact; it is not a new GPU benchmark.

See the [SparkLab model guide](https://github.com/sixteen-miles-labs/sparklab/blob/main/docs/models/qwen3.8-27b.md)
for current instructions and limitations.

## Credits and license

- Model architecture and base weights: [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B).
- Published NVFP4 checkpoint: [Inferact](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4), using the ModelOpt format.
- Quantization tooling: [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).
- Native inference, conversion, GB10 optimization, and deployment: [SparkLab](https://github.com/sixteen-miles-labs/sparklab).
- Original FTW format and upstream runtime foundations: [FreeToken](https://github.com/FlashML-org/FreeToken); see SparkLab's [NOTICE](https://github.com/sixteen-miles-labs/sparklab/blob/main/NOTICE).
- FTW publishing: [OakMind AI](https://huggingface.co/oakmindai).
- Optional external draft: [maurienne-ai](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal).

The pinned source declares Apache License 2.0; its `LICENSE` is included unchanged.
Refer to the [source model card](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/blob/6128240ebaf4eaa7bad2b3d1c72c37d677c5f462/README.md)
and [Qwen documentation](https://huggingface.co/Qwen/Qwen3.8-27B) for upstream model
information, limitations, and intended-use guidance. Upstream multimodal results
are not SparkLab deployment certification. Evaluate generated answers for
correctness and suitability for your application.
