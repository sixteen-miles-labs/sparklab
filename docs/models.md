# SparkLab model portfolio

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
| Research | Complete or novel models outside the interactive envelope | Correct full-model output, bounded memory, and no swap growth; no latency promise |

Every tier also requires output correctness, reasoning/tool parsing, a fixed
coding-agent task, and versioned benchmark evidence. Status means:

- **Certified:** every gate for the intended tier passed on the release image.
- **Preview:** end-to-end evidence exists but at least one product gate remains.
- **Experimental:** engineering evaluation only; no usability promise.

## Current recipes

| Model | Parameter | Quantization | Recipe | Status | tok/s | TTFT(s) |
|---|---|---|---|---|---:|---:|
| **Fast — routine chat, editing, and short agent loops** |  |  |  |  |  |  |
| [Qwen3.6-35B-A3B](https://huggingface.co/oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW) | 35B total / 3B active | NVFP4 · FTW + optional MTP2 | `qwen3.6-35b-a3b` | Certified | 67.79 | 0.329 |
| **Frontier — hard coding, reasoning, and long agent work** |  |  |  |  |  |  |
| [Qwen3.8-27B](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4) | 27B dense | NVFP4 · FTW | `qwen3.8-27b` | Experimental | 8.83 | 0.144 |
| [Qwen3.8-Flash-Next](https://huggingface.co/oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW) | 125B LM + 55B auxiliary / 6B active | NVFP4 · FTW + MTP3 | `qwen3.8-flash-next` | Experimental | 30.67 | 0.258 |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 284B total / 13B active | DS-FP4 · FTW + optional DSpark5 | `deepseek-v4` | Preview | 13.15 | 0.518 |
| [GLM-5.3 Flash](https://huggingface.co/oakmindai/GLM-5.3-Flash-NVFP4-FTW) | 320B total / 18B active | NVFP4 + KDA FP8 · FTW | `glm-5.3-flash` | Experimental | 6.27 | 5.681 |
| **Research — complete or novel models outside the interactive envelope** |  |  |  |  |  |  |
| [GLM-5.3](https://huggingface.co/oakmindai/GLM-5.3-NVFP4-FTW) | 753B total / 40B active | NVFP4 + resident FP8 · FTW | `glm-5.3` | Experimental fallback | 0.81 | 2.530 |
| [Kimi K3](https://huggingface.co/oakmindai/Kimi-K3-NVFP4-FTW) | 2.8T total / 16 of 896 experts | ModelOpt NVFP4/FP8 · FTW | `kimi-k3` | Experimental | 0.16 | 395.405 |

Model links point to the selected source or published FTW checkpoint. Qwen3.6, GLM-5.3,
and Kimi K3 use pinned prebuilt artifacts with reproducible source-conversion paths.
Parameter counts come from model publishers, and performance values come from the
evidence attached to each recipe. Certification applies only to that exact checkpoint
and recipe version.

Qwen3.6's published FTW artifact includes its native BF16 MTP weights, but the portfolio
row continues to report the certified target-only profile. The optimized two-draft path
measured 74.42 tok/s on a matched 256-token GB10 probe versus 45.38 tok/s target-only;
it remains opt-in pending the full certification suite.

## Evidence and caveats

| Model | Current result | Evidence |
|---|---|---|
| Qwen3.6-35B-A3B | Fast-certified target-only profile, including exact 32K recall and a 60-minute zero-swap run. Optimized native MTP2 reached 74.42 tok/s versus its 45.38 tok/s eager control; it remains opt-in pending full certification. | [GB10-QWEN36-FAST-002](../benchmarks/gb10/results/GB10-QWEN36-FAST-002.json), [original MTP sweep](../benchmarks/gb10/results/GB10-QWEN36-MTP-003.json), [optimized MTP](../benchmarks/gb10/results/GB10-QWEN36-MTP-004.json) |
| Qwen3.8-27B | Passes Frontier speed, exact 64K recall, reasoning/tool parsing, and the coding-agent probe; the clean-revision 60-minute endurance gate remains outstanding. | [GB10-QWEN38-27B-001](../benchmarks/gb10/results/GB10-QWEN38-27B-001.json) |
| Qwen3.8-Flash-Next | The selected MTP3 profile measured a three-trial median 30.67 tok/s at 0.258 s warm TTFT. The clean-revision endurance gate remains outstanding. | [GB10-QWEN38-MTP-007](../benchmarks/gb10/results/GB10-QWEN38-MTP-007.json) |
| DeepSeek V4 Flash | The selected three-trial DSpark5 burst profile reaches 13.15 tok/s. Target-only remains the default and wins the 256-token sustained control, 10.52 versus 9.31 tok/s. | [GB10-DSV4-SMALLM-005](../benchmarks/gb10/results/GB10-DSV4-SMALLM-005.json) |
| GLM-5.3 Flash | Passes Frontier speed; NVMe-sensitive TTFT and remaining certification gates keep it Experimental. | [GB10-GLM53-MHC-003](../benchmarks/gb10/results/GB10-GLM53-MHC-003.json) |
| GLM-5.2 | Below Frontier speed and recorded swap growth, so it remains Experimental. | [Experiment](../exps/exp_glm5_2_gb10.md) |
| GLM-5.3 | Correctness is not established; the measured output reached its length cap before answering. | [GB10-GLM53-RESEARCH-001](../benchmarks/gb10/results/GB10-GLM53-RESEARCH-001.json) |
| Kimi K3 | Complete-checkpoint serving was measured, but correctness and cross-run determinism are not established. | [GB10-KIMI-001](../benchmarks/gb10/results/GB10-KIMI-001.json) |

## Running a recipe

Use the recipe workflow for a validated checkpoint and configuration:

```bash
sparklab plan <recipe> --prepare
sparklab pull <recipe> --prepare
sparklab run <recipe>
```

`--prepare` uses a pinned prebuilt FTW artifact when one is available. Use
`--from-source` to reproduce conversion locally. FTW preserves the source checkpoint's
precision while arranging weights for fast loading; separately quantized artifacts do
not inherit the source recipe's certification.

SparkLab can load additional uncataloged architectures, but only recipes returned by
`sparklab models` are part of the supported GB10 portfolio. Direct conversion and serving
options are documented in [cli.md](cli.md).
