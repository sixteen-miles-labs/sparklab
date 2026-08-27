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

| Model | Quantization | Status | tok/s | TTFT |
|---|---|---|---:|---:|
| **Fast — routine chat, editing, and short agent loops** |  |  |  |  |
| [Qwen3.6-35B-A3B](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | NVFP4 | Experimental | — | — |
| **Frontier — quality-first coding, reasoning, and long agent work** |  |  |  |  |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | NVFP4 · FTW upload pending | Certified | 12.51 | 0.870 s |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | DS-FP4 | Preview | 9.22 | 14.045 s |
| [GLM-5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | FP8 | Experimental | — | — |
| **Research — correct, bounded execution beyond the interactive envelope** |  |  |  |  |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | MXFP4 | Experimental | — | — |

Model links point to the original publisher checkpoints. When Spark Lab publishes a
converted FTW checkpoint, its Hugging Face link appears in the Quantization column; the
Qwen3.8 FTW-NVFP4 link will be added after that artifact is uploaded.

GLM-5.2 remains an Experimental fallback outside the primary lineup. A target remains
Preview or Experimental until its complete checkpoint passes the published gate;
architecture smoke tests alone do not change status.

Run `sparklab models --json` for exact checkpoint IDs, recipe versions,
implementation state, evidence IDs, and limitations. Tier names describe intended
roles until a recipe's current `status` becomes `certified`.

DeepSeek V4 is the measured GB10 baseline: 9.217 decode tok/s and 14.045 s warm
TTFT on the fixed 64-token probe, with identical output across matched controls.
The compact evidence is checked in as
[`GB10-BASELINE-001`](benchmarks/gb10/results/GB10-BASELINE-001.json); it is a
Preview result, not yet a complete Frontier certification.

Qwen3.8-Flash-Next is certified for the text-only Frontier tier at 12.51 decode
tok/s, 0.870 s warm TTFT, exact 65,536-token recall, and a 60.50-minute
uninterrupted endurance run. Its compact evidence is
[`GB10-QWEN38-FRONTIER-001`](benchmarks/gb10/results/GB10-QWEN38-FRONTIER-001.json).

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
