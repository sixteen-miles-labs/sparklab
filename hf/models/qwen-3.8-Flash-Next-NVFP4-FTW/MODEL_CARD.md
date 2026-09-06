---
pipeline_tag: text-generation
license: other
license_name: nvidia-open-model-license
license_link: https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4#licenseterms-of-use
base_model: nvidia/Qwen3.8-Flash-Next-NVFP4
library_name: sparklab
tags: [sparklab, ftw, nvfp4, dgx-spark, text-generation, modelopt]
---

# Qwen3.8-Flash-Next NVFP4 — SparkLab FTW

This repository packages Qwen3.8-Flash-Next as an **experimental, text-only
SparkLab FTW artifact** for one NVIDIA DGX Spark with 128 GB coherent unified
memory. It is not a standard Transformers or vLLM safetensors checkpoint.

[SparkLab](https://github.com/sixteen-miles-labs/sparklab) provides GB10-native
inference, model recipes, hardware checks, memory planning, artifact preparation,
and OpenAI-/Anthropic-compatible serving APIs.

## What this repository contains

This is a repackaging, not a newly trained model or an additional quantization:

1. [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) developed the base model.
2. [NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) published the mixed-precision checkpoint using Model Optimizer.
3. SparkLab packages the tensors for its native FTW loader and GB10 execution path.
4. [OakMind AI](https://huggingface.co/oakmindai) published this conversion.

The pinned NVIDIA source revision is `fab0aecb760cec45227f6656abcaafa11abca87a`.
The FTW fingerprint is `94e1ee0daa442357`. Total weight payload is
**131,931,279,080 bytes**, excluding tokenizer and metadata files.

| Component | Published layout and precision |
|---|---|
| Target weights | Ten `freetoken-*.ftw` shards and `freetoken_weight.json`; native NVFP4 routed experts, BF16 resident projections |
| PLE n-gram bank | Approximately 51.2 GB FP8 `qwen4_ngram.bin`, with its published global scale in `qwen4_ngram.json` |
| Native MTP module | `nvfp4_experts_mtp.safetensors`; despite the legacy filename, draft experts are 128-by-128 block-scaled FP8 and remaining draft tensors retain their published precision |
| Runtime metadata | Configuration, tokenizer, chat template, and generation configuration |

Conversion preserves published weight precision and scales without conversion-time
requantization. The complete indexed MTP module is assembled from the source shards.
Inherited image/video processor metadata does not enable multimodal serving here.

## Why use FTW?

FTW prepares tensor layouts ahead of time and stores routed experts in addressable
banks for SparkLab's native loader. This deployment preloads all **24,576 target
routed experts** into immutable unified-memory slots, eliminating steady-state
routed-expert disk reads. The PLE bank remains disk-backed on local NVMe.

The artifact's disk size is not its resident-memory requirement. FTW alone is
not a claim of better model quality or higher steady-state decode throughput.

## Run with SparkLab on NVIDIA DGX Spark

Requires ARM64 Linux / DGX OS, GB10/SM121, CUDA 13, and fast local NVMe.
Use a current SparkLab source build with NVIDIA mixed-precision loading, scaled
FP8 PLE, indexed MTP extraction, and accepted-prefix commits. **The previously
released 0.1.2 wheel is not sufficient.**

```bash
git clone https://github.com/sixteen-miles-labs/sparklab.git
cd sparklab
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

sparklab doctor --storage-path /path/to/models
sparklab plan qwen3.8-flash-next --root /path/to/models --prepare
sparklab pull qwen3.8-flash-next --root /path/to/models --prepare
sparklab run qwen3.8-flash-next --root /path/to/models
```

Replace `/path/to/models` with your local NVMe directory. Recipe 0.9.0 downloads
this prebuilt artifact at immutable revision
`f547c96e86d0e50908c1415f4525c4325555691e` and verifies its identity. A later
documentation-only Hub revision does not require changing that weight pin.
The catalog budgets approximately 402 GB free disk, including preparation and
safety allowance; `plan` is authoritative.

Once the server is ready:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128,"temperature":0,"stream":true}'
```

### Optional single-stream MTP

Instead of the target-only launch above, enable three draft tokens:

```bash
sparklab run qwen3.8-flash-next --root /path/to/models -- --speculative-tokens 3
```

Native MTP is opt-in and limited to batch-one greedy requests. Dense-QSA
verification uses a fixed-shape CUDA graph; sparse-QSA verification runs eagerly.
The target-only recipe's admission limit is not a claim of concurrent MTP support.

## Performance and validation limits

The selected NVIDIA MTP3 accepted-prefix profile measured **31.97 tok/s** and
**0.260 s warm TTFT**, three-trial medians on a short 128-token greedy
single-stream probe. This was 13.9% faster than its NVIDIA MTP3 baseline and
performed no rejection replay. See
[GB10-QWENNVIDIA-002](https://github.com/sixteen-miles-labs/sparklab/blob/main/benchmarks/gb10/results/GB10-QWENNVIDIA-002.json).

The three selected runs reproduced the same output, but differed from target-only
and pre-optimization MTP. This does not establish general quality equivalence.
Quality and endurance certification remain outstanding. The configured
131,072-token KV pool is a capacity setting, not a certified context length.

The NVIDIA artifact does **not** inherit the previous Inferact artifact's quality,
64K recall, concurrency, or endurance evidence. No quality advantage over Inferact
has been established. Experimental reduced-vocabulary drafting is not enabled
by the commands above and is not the selected portfolio profile.

See the [model guide](https://github.com/sixteen-miles-labs/sparklab/blob/main/docs/models/qwen3.8-flash-next.md)
for current runtime limitations and separately labeled experiments.

## Previous artifact compatibility

The NVIDIA artifact replaced the Inferact-derived artifact on `main` on September
5, 2026. The previous artifact remains available at immutable revision
`5ab790b83f149a96594237a35905d84be24599a3`; existing revision-pinned recipes
continue to resolve it. Do not mix its shards or sidecars with the NVIDIA artifact.

## Credits and license

- Base model: [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
- Quantized source: [NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) and [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).
- Native runtime, GB10 optimization, conversion, and deployment workflow: [SparkLab](https://github.com/sixteen-miles-labs/sparklab).
- Original FreeToken Weight format and upstream runtime foundations: [FreeToken](https://github.com/FlashML-org/FreeToken); see SparkLab's [NOTICE](https://github.com/sixteen-miles-labs/sparklab/blob/main/NOTICE).
- FTW conversion publishing: [OakMind AI](https://huggingface.co/oakmindai).

The upstream NVIDIA Open Model License and additional Qwen Community License
terms apply; see the [upstream license/terms](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4#licenseterms-of-use).
The existing `LICENSE` file contains Qwen's additional terms and is not a
substitute for the NVIDIA license. SparkLab's software license does not replace
the model's governing terms.

## Upstream model information

Consult the [pinned NVIDIA model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/fab0aecb760cec45227f6656abcaafa11abca87a/README.md)
and [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for architecture,
training and calibration information, upstream evaluations, and intended-use
limitations. Upstream multimodal and long-context results are not SparkLab
certification. Generated answers can be incorrect or biased; evaluate the
deployed artifact on your intended tasks before relying on it.
