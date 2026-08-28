# Spark Lab

> Turn one NVIDIA GB10 into a private frontier-AI workstation.

Spark Lab is a GB10-native frontier inference lab. It packages tested model
recipes, unified-memory diagnostics, NVMe-backed MoE execution, and
OpenAI-/Anthropic-compatible APIs around the FreeToken research engine.

The supported production target is intentionally narrow:

- NVIDIA GB10 Grace Blackwell Superchip, SM121
- 128 GB coherent unified memory
- ARM64 Linux / DGX OS
- CUDA 13
- Local NVMe for checkpoints and disk-backed experts

This repository is in the staged rebrand period. The `sparklab` product CLI is
available, while the `freetoken` Python package and `ft` CLI remain compatible.
No recipe is called Certified until its GB10 correctness, latency, context,
stability, and agent gates have all passed.

## Start here

Install the current distribution with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install "freetoken[accel]"
```

Or install this checkout:

```bash
git clone https://github.com/FlashML-org/FreeToken.git
cd FreeToken
uv venv
source .venv/bin/activate
uv pip install -e ".[accel]"
```

Inspect the GB10 before loading a checkpoint:

```bash
sparklab doctor
sparklab doctor --storage-path /path/to/models --json
sparklab models
```

For a versioned recipe, plan capacity, acquire its immutable checkpoint, prepare
FTW when required, and launch it through fail-closed GB10 admission:

```bash
sparklab plan qwen3.8-flash-next --prepare
sparklab pull qwen3.8-flash-next --prepare
sparklab run qwen3.8-flash-next
sparklab shell
sparklab launch codex
```

`sparklab serve --model /path/to/checkpoint` remains the expert/compatibility
path for checkpoints outside the recipe catalog.

The server exposes OpenAI Chat Completions and Responses APIs plus the Anthropic
Messages API on `http://127.0.0.1:1919` by default.

## Model portfolio

Spark Lab has three recipe tiers:

| Model | Parameter | Quantization | Status | tok/s | TTFT(s) |
|---|---|---|---|---:|---:|
| **Fast — routine chat, editing, and short agent loops** |  |  |  |  |  |
| [Qwen3.6-35B-A3B](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | 35B total / 3B active | NVFP4 | Experimental | 67.46 | 0.320 |
| **Frontier — quality-first coding, reasoning, and long agent work** |  |  |  |  |  |
| [Qwen3.8-Flash-Next](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4) | 125B LM + 55B auxiliary / 6B active | NVFP4 · FTW upload pending | Experimental | — | — |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 284B total / 13B active | DS-FP4 | Preview | 9.22 | 14.045 |
| [GLM-5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 320B total / 18B active | FP8 | Experimental | — | — |
| **Research — correct, bounded execution beyond the interactive envelope** |  |  |  |  |  |
| [GLM-5.2](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | 753B total / 40B active | NVFP4 | Experimental | 0.80 | 2.570 |
| [Kimi K3](https://huggingface.co/nvidia/Kimi-K3-NVFP4) | 2.8T total / 16 of 896 experts | NVFP4 · FTW upload pending | Experimental | — | — |

Model links point to the selected source checkpoints. When Spark Lab publishes a
converted FTW checkpoint, its Hugging Face link appears in the Quantization column; the
Qwen3.8 and Kimi K3 NVFP4 FTW links will be added after those artifacts are uploaded.

GLM-5.2 remains an Experimental Research fallback outside the primary lineup. Its measured
0.802 tok/s and 2.57 s TTFT come from the selected 256-token GB10 trial, which failed the
strict Research admission gate because swap-out grew by 680 KiB; see the
[`GLM-5.2 GB10 experiment`](exps/exp_glm5_2_gb10.md). A target remains Preview or
Experimental until its complete checkpoint passes the published gate; architecture smoke
tests alone do not change status.

Parameter values use the publishers' architecture counts. Qwen3.8's 55B auxiliary
parameters are its 51B n-gram embedding plus 4B MTP module; Kimi K3's publisher reports
expert activation count rather than an active-parameter total.

Run `sparklab models --json` for exact checkpoint IDs, recipe versions,
implementation state, evidence IDs, and limitations. Tier names describe intended
roles until a recipe's current `status` becomes `certified`.

DeepSeek V4 is the measured GB10 baseline: 9.217 decode tok/s and 14.045 s warm
TTFT on the fixed 64-token probe, with identical output across matched controls.
The compact evidence is checked in as
[`GB10-BASELINE-001`](benchmarks/gb10/results/GB10-BASELINE-001.json); it is a
Preview result, not yet a complete Frontier certification.

Qwen3.6 NVFP4 measured 67.46 decode tok/s and 0.320 s warm TTFT on the same class
of fixed 64-token GB10 probe. The compact
[`GB10-QWEN36-FAST-001`](benchmarks/gb10/results/GB10-QWEN36-FAST-001.json)
evidence establishes Fast-class latency, but the recipe stays Experimental until
its context and endurance gates pass.

The Beta 0.1 Qwen3.8 recipe now targets Inferact's publisher-quantized ModelOpt NVFP4
checkpoint. Preparation preserves those NVFP4 routed experts and the published precision
of the remaining text tower; no BF16-to-NVFP4 requantization is performed. This immutable
0.5.0 checkpoint has not yet been prepared or measured on GB10, so the archived
[`GB10-QWEN38-FP8-001`](benchmarks/gb10/results/GB10-QWEN38-FP8-001.json) and
[`GB10-QWEN38-FRONTIER-001`](benchmarks/gb10/results/GB10-QWEN38-FRONTIER-001.json)
results remain historical and do not certify it.

## Why Spark Lab

- **Made for unified memory:** platform checks and upcoming capacity planning use
  the physical GB10 memory pool rather than pretending it is discrete VRAM.
- **Frontier beyond memory:** FTW and NVMe-backed expert streaming can address MoE
  checkpoints larger than physical memory.
- **Agent ready:** reasoning, tool calls, prefix reuse, and compatible APIs are
  first-class validation targets.
- **Measured, not implied:** product claims must point to versioned GB10 evidence.

See the [quick start](docs/quickstart.md), [model catalog](docs/models.md),
[CLI reference](docs/cli.md), [installation guide](docs/install.md), and
[rebrand plan](docs/spark-lab-rebrand-plan.md).

## Compatibility and research attribution

Spark Lab is the product identity. FreeToken remains the internal engine and the
name of the research paper during this migration. Existing `ft` commands,
`freetoken.*` imports, and API protocols continue to work; see the
[migration guide](docs/migration.md).

If you use the engine for research, cite:

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

FreeToken builds on ideas and code from mini-sglang, SGLang, vLLM,
FlashInfer, flash-linear-attention, LightLLM, and llama.cpp.

## License

[Apache License 2.0](LICENSE).
