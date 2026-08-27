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

| Intended tier | Recipe | Checkpoint | Current status |
|---|---|---|---|
| Fast | `qwen3.8-flash-next` | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Preview; primary Fast-layer target. Version 0.3.0 previously earned Frontier certification, but 0.4.0 has not passed the Fast gate |
| Fast fallback | `qwen3.6-35b-a3b` | [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | Experimental fallback outside the primary lineup |
| Frontier | `deepseek-v4` | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | Preview; measured baseline, product gates still open |
| Frontier | `glm-5.3-flash` | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | Experimental; text runtime implemented, complete-checkpoint GB10 validation in progress |
| Frontier fallback | `glm-5.2` | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | Experimental fallback outside the primary lineup |
| Research | `kimi-k3` | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | Experimental; text architecture implemented, full run pending |

The primary lineup is Qwen3.8 for Fast, GLM-5.3 Flash and DeepSeek V4 Flash for
Frontier, and Kimi K3 for Research. Qwen3.8's prior Frontier evidence records a
3/5 fixed AIME sample, two reasoning-budget cap cases, bounded NVMe behavior, and
zero model-attributed swap growth. It cannot be reused as Fast certification.

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
