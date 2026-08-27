# Spark Lab model portfolio

The catalog is a set of versioned **checkpoint + GB10 recipe** entries. A model
family, parameter count, or successful import does not earn a product tier.

Use the catalog as the authoritative machine-readable view:

```bash
sparklab models
sparklab models --role primary
sparklab models --json
```

## Tiers and gates

| Tier | User promise | Required GB10 gate |
|---|---|---|
| Fast | Routine chat, editing, and short agent loops | ≥20 decode tok/s, ≤5 s warm TTFT, ≥32K usable context, no normal-operation NVMe stalls, and a stable 60-minute agent trace |
| Frontier | Hard coding, reasoning, and long agent work | ≥5 decode tok/s, ≤20 s warm TTFT, ≥64K usable context, bounded NVMe traffic, and the same 60-minute stability gate |
| Research | Complete or novel models outside the interactive envelope | Correct full-model output and bounded memory; no latency promise |

Every tier also requires output correctness, reasoning/tool parsing, a fixed
coding-agent task, and versioned benchmark evidence. Status means:

- **Certified:** every gate for the intended tier passed on the release image.
- **Preview:** end-to-end evidence exists but at least one product gate remains.
- **Experimental:** engineering evaluation only; no usability promise.

## Current recipes

| Model | Quantization | Recipe | Status | GB10 performance |
|---|---|---|---|---|
| **Fast — routine chat, editing, and short agent loops** |  |  |  |  |
| [Qwen3.6-35B-A3B](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | NVFP4 | `qwen3.6-35b-a3b` | Experimental; primary target | Not yet measured |
| **Frontier — hard coding, reasoning, and long agent work** |  |  |  |  |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | NVFP4 | `qwen3.8-flash-next` | Certified | 12.51 decode tok/s · 0.870 s warm TTFT · exact 64K context · 60.5 min endurance |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | DS-FP4 | `deepseek-v4` | Preview | 9.217 decode tok/s · 14.045 s warm TTFT on the fixed baseline probe |
| [GLM-5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | FP8 | `glm-5.3-flash` | Experimental | Not yet measured |
| [GLM-5.2](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | NVFP4 | `glm-5.2` | Experimental fallback | Not yet measured |
| **Research — complete or novel models outside the interactive envelope** |  |  |  |  |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | MXFP4 | `kimi-k3` | Experimental | Not yet measured |

Measured values are copied from the evidence named by the recipe. “Not yet measured”
means no accepted complete-checkpoint GB10 performance evidence is attached.

The primary lineup is Qwen3.6 NVFP4 for Fast; Qwen3.8 Flash Next, GLM-5.3 Flash,
and DeepSeek V4 Flash for Frontier; and Kimi K3 for Research. Qwen3.8's Frontier
evidence records a 3/5 fixed AIME sample, two reasoning-budget cap cases, bounded
NVMe behavior, and zero model-attributed swap growth.

## Engine architecture support

The underlying FreeToken engine can load additional checkpoints, including
GLM-4.7, Qwen3.x, GPT-OSS, Gemma-4, MiniMax, and Muse-Glimmer variants. That
compatibility is not a Spark Lab support or performance claim. Only entries
returned by `sparklab models` participate in the GB10 product portfolio.

## FTW and NVMe execution

FTW is the engine's self-contained fast-load format. Conversion is optional for
resident checkpoints and required by recipes such as full Kimi K3 whose routed
experts must be independently addressable from NVMe:

```bash
sparklab checkpoint --model /path/to/hf-checkpoint --out /path/to/model-ftw
```

Recipe-backed `sparklab pull <recipe> --prepare` performs this conversion at the
pinned revision. Direct conversion and serving flags remain an expert interface
documented in [cli.md](cli.md).
