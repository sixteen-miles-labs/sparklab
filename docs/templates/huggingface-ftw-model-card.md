<!--
SparkLab FTW model-card template

Before publishing:
1. Replace every double-brace token. `rg -n '\{\{[A-Z0-9_]+\}\}' README.md` must return nothing.
2. Remove optional sections and flags that do not apply to this model.
3. Pin the exact source revision; never publish against `main`.
4. Keep upstream model, quantization, engine, conversion, and publisher credits distinct.
5. Confirm the YAML metadata, license, modalities, and runtime commands.
-->
---
pipeline_tag: text-generation
base_model:
- {{QUANTIZED_SOURCE_REPO}}
- {{BASE_MODEL_REPO}}
license: {{LICENSE_ID}}
library_name: sparklab
tags:
- {{MODEL_FAMILY_TAG}}
- {{QUANTIZATION_TAG}}
- quantized
- sparklab
- ftw
---

# {{MODEL_NAME}} {{QUANTIZATION_NAME}} — SparkLab FTW

This is a ready-to-run **SparkLab** checkpoint for
[`{{MODEL_NAME}}`](https://huggingface.co/{{BASE_MODEL_REPO}}), optimized for the
**NVIDIA DGX Spark** and its Grace Blackwell GB10 Superchip.

**SparkLab** is a GB10-native frontier inference lab. It packages tested model recipes,
hardware readiness checks, artifact preparation, unified-memory planning, NVMe-backed MoE
execution, and OpenAI-/Anthropic-compatible APIs for one DGX Spark.

> **SparkLab source:** {{SPARKLAB_REPOSITORY_URL}}

The commands and settings below follow SparkLab's supported DGX Spark path: ARM64 Linux
or DGX OS, CUDA 13, SM121 kernels, 128 GB coherent unified memory, and local NVMe.

> **Runtime scope:** this artifact is validated for {{VALIDATED_INPUTS}} input and
> {{VALIDATED_OUTPUTS}} output. {{OUT_OF_SCOPE_MODALITIES}}

## What this repository contains

This repository does **not** introduce a new model or a new quantization. It repackages
the existing [`{{QUANTIZATION_NAME}}` checkpoint](https://huggingface.co/{{QUANTIZED_SOURCE_REPO}})
into the FTW execution format:

1. **[{{BASE_MODEL_AUTHOR}}](https://huggingface.co/{{BASE_MODEL_AUTHOR}})** developed the
   original [`{{MODEL_NAME}}`](https://huggingface.co/{{BASE_MODEL_REPO}}) model.
2. **[{{QUANTIZATION_AUTHOR}}]({{QUANTIZATION_AUTHOR_URL}})** produced the
   [`{{QUANTIZED_SOURCE_REPO}}`](https://huggingface.co/{{QUANTIZED_SOURCE_REPO}})
   quantized checkpoint using {{QUANTIZATION_TOOL}}.
3. **[SparkLab]({{SPARKLAB_REPOSITORY_URL}})** provides the native inference runtime,
   FTW format, conversion tooling, low-precision kernels, model recipes, GB10 readiness
   checks, capacity planning, artifact lifecycle, and serving workflow.
4. **[{{PUBLISHER_NAME}}]({{PUBLISHER_URL}})** performed, validated, documented, and
   published this FTW conversion.

The exact source revision is `{{SOURCE_COMMIT_SHA}}`. Conversion is
precision-preserving: the model remains {{QUANTIZATION_NAME}}, while its tensors are
aligned and sharded for SparkLab's native loader. No training or additional quantization
was performed.

## Why use FTW?

SparkLab may be able to load the original Hugging Face safetensors checkpoint directly.
FTW is an optional deployment format that performs layout work ahead of time. For MoE
models, it stores routed experts in independently addressable expert banks and enables
SparkLab's native loading and expert-caching paths.

{{RESIDENCY_OR_NVME_EXPLANATION}}

FTW does not change the model's expected output quality. Do not assume it improves
steady-state decode speed unless this card cites a controlled measurement.

## Run with SparkLab on NVIDIA DGX Spark

Install SparkLab, then download this checkpoint:

```bash
git clone {{SPARKLAB_REPOSITORY_URL}}
cd {{SPARKLAB_REPOSITORY_DIRECTORY}}
./install.sh

hf download {{FTW_REPOSITORY}} \
  --local-dir ~/models/{{LOCAL_MODEL_DIRECTORY}}
```

Verify the platform before loading the model:

```bash
sparklab doctor --storage-path ~/models/{{LOCAL_MODEL_DIRECTORY}}
```

Start the OpenAI-compatible API server with the validated configuration:

```bash
sparklab serve \
  --model ~/models/{{LOCAL_MODEL_DIRECTORY}} \
  {{VALIDATED_RUNTIME_FLAGS}} \
  --host 127.0.0.1 \
  --port 8000
```

The configuration above was validated for {{VALIDATED_CONFIGURATION_SCOPE}}. It is not a
claim about untested context lengths, concurrency levels, sampling settings, modalities,
or hardware.

Send a chat-completions request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "{{LOCAL_MODEL_DIRECTORY}}",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Credits and license

Please credit each project for its part of this artifact:

- Model architecture and base weights:
  [{{BASE_MODEL_AUTHOR}}](https://huggingface.co/{{BASE_MODEL_REPO}})
- Quantization: [{{QUANTIZATION_AUTHOR}}]({{QUANTIZATION_AUTHOR_URL}}), using
  [{{QUANTIZATION_TOOL}}]({{QUANTIZATION_TOOL_URL}})
- Native inference runtime, FTW format, conversion, kernels, model workflow, and deployment:
  **[SparkLab]({{SPARKLAB_REPOSITORY_URL}})**
- Research ancestry: [FreeToken](https://github.com/FlashML-org/FreeToken)
- FTW conversion and publishing: [{{PUBLISHER_NAME}}]({{PUBLISHER_URL}})

The upstream model is distributed under {{LICENSE_NAME}}. This repository preserves its
license and provenance. Review the
[base model card](https://huggingface.co/{{BASE_MODEL_REPO}}) and
[quantized source card](https://huggingface.co/{{QUANTIZED_SOURCE_REPO}}) for complete
license terms, limitations, training and evaluation details, and intended-use guidance.

## Upstream model information

This FTW card intentionally links to the immutable upstream sources instead of copying a
large model card that may drift over time:

- Base model: https://huggingface.co/{{BASE_MODEL_REPO}}
- Quantized source: https://huggingface.co/{{QUANTIZED_SOURCE_REPO}}/tree/{{SOURCE_COMMIT_SHA}}

{{OPTIONAL_UPSTREAM_EXCERPT}}
