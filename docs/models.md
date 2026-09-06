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
| [Qwen3.6-35B-A3B](https://huggingface.co/oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW) | 35B total / 3B active | NVFP4 · FTW + optional MTP2 | `qwen3.6-35b-a3b` | Experimental | 80.55 | 0.367 |
| [Qwen3.8-27B](https://huggingface.co/oakmindai/Qwen3.8-27B-NVFP4-FTW) | 27B dense | NVFP4 · FTW + optional DFlash2-12 | `qwen3.8-27b` | Experimental | 45.88 | 0.152 |
| **Frontier — hard coding, reasoning, and long agent work** |  |  |  |  |  |  |
| [Qwen3.8-Flash-Next](https://huggingface.co/oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW) | 125B LM + 55B auxiliary / 6B active | NVFP4 · FTW + optional MTP3 | `qwen3.8-flash-next` | Experimental | 31.97 | 0.260 |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 284B total / 13B active | DS-FP4 · FTW + optional DSpark5 | `deepseek-v4` | Preview | 14.02 | 0.515 |
| [GLM-5.3 Flash](https://huggingface.co/oakmindai/GLM-5.3-Flash-NVFP4-FTW) | 320B total / 18B active | NVFP4 + KDA FP8 · FTW + optional MTP3 | `glm-5.3-flash` | Experimental | 7.77 | 6.395 |
| **Research — complete or novel models outside the interactive envelope** |  |  |  |  |  |  |
| [GLM-5.3](https://huggingface.co/oakmindai/GLM-5.3-NVFP4-FTW) | 753B total / 40B active | NVFP4 + resident FP8 · FTW | `glm-5.3` | Experimental fallback | 0.81 | 2.530 |
| [Kimi K3](https://huggingface.co/oakmindai/Kimi-K3-NVFP4-FTW) | 2.8T total / 16 of 896 experts | ModelOpt NVFP4/FP8 · FTW | `kimi-k3` | Experimental | 0.16 | 395.405 |

Model links point to the selected source or published FTW checkpoint. Qwen3.6, GLM-5.3,
and Kimi K3 use pinned prebuilt artifacts with reproducible source-conversion paths.
The current Qwen3.8-Flash-Next recipe requires a
[source installation](install.md#method-2-install-from-source); the released 0.1.2 wheel
lacks its required runtime support.

Parameter counts come from model publishers, and performance values come from the
evidence attached to each recipe. Certification applies only to that exact checkpoint
and recipe version. Portfolio performance and warm-TTFT columns use the selected
single-stream profile; concurrent-serving results remain in model-specific evidence.

Qwen3.6's published FTW artifact includes its native BF16 MTP weights. The portfolio
row reports the opt-in MTP2 profile: 80.55 tok/s with 0.367 s warm TTFT on a 256-token
GB10 probe, matching the fresh eager target-only output on that prompt. Full MTP
certification remains pending; the certified target-only profile measured 67.79 tok/s
and 0.329 s warm TTFT.

GLM-5.3 Flash's selected opt-in MTP3 profile measured a three-trial median of 7.77 tok/s
and 6.395 s warm TTFT after eliminating rejection replay. All three trials reproduced
the fresh MTP3 baseline's output. Target-only remains the default, with its separate
6.27 tok/s, 5.681 s result; broader certification remains outstanding.

The previous Inferact Qwen3.8-Flash-Next checkpoint measured 85.72 aggregate tok/s
with native target-only batch-eight serving. That concurrency result does not transfer
to the current NVIDIA checkpoint. Its selected MTP3 metric is batch-one.

## Evidence and caveats

| Model | Current result | Evidence |
|---|---|---|
| Qwen3.6-35B-A3B | Fast-certified target-only profile, including exact 32K recall and a 60-minute zero-swap run. Replay-free native MTP2 reached 80.55 tok/s and matched the eager target-only output on the focused prompt; it remains opt-in pending full certification. | [GB10-QWEN36-FAST-002](../benchmarks/gb10/results/GB10-QWEN36-FAST-002.json), [original MTP sweep](../benchmarks/gb10/results/GB10-QWEN36-MTP-003.json), [optimized MTP](../benchmarks/gb10/results/GB10-QWEN36-MTP-004.json), [replay-free MTP](../benchmarks/gb10/results/GB10-QWEN36-MTP-005.json) |
| Qwen3.8-27B | The opt-in DFlash2-12 profile reached 45.88 tok/s with exact target-output parity on the 128-token probe. Longer math, coding, and prose probes improved; their complete traces can differ with verification grouping. Full Fast certification remains pending. | [target-only](../benchmarks/gb10/results/GB10-QWEN38-27B-001.json), [DFlash2-12](../benchmarks/gb10/results/GB10-QWEN38-DFLASH-004.json) |
| Qwen3.8-Flash-Next | NVIDIA MTP3 with accepted-prefix state commits measured 31.97 tok/s and 0.260 s warm TTFT, 13.9% above its baseline with zero rejection replay. Output differs from target-only; prior Inferact certification evidence does not transfer. | [NVIDIA baseline](../benchmarks/gb10/results/GB10-QWENNVIDIA-001.json), [accepted-prefix commits](../benchmarks/gb10/results/GB10-QWENNVIDIA-002.json) |
| DeepSeek V4 Flash | The selected three-trial DSpark5 burst profile reaches 14.02 tok/s with replay-free compressor-prefix commits. This fixes the previous first-rejection carry shortcut; target-only remains the default and full certification is pending. | [GB10-DSV4-PREFIX-006](../benchmarks/gb10/results/GB10-DSV4-PREFIX-006.json) |
| GLM-5.3 Flash | The opt-in MTP3 profile reached a median 7.77 tok/s with no rejection replay, 3.4% above its fresh baseline and identical output across three trials. NVMe sensitivity and the remaining certification gates keep it Experimental. | [target-only](../benchmarks/gb10/results/GB10-GLM53-MHC-003.json), [MTP sweep](../benchmarks/gb10/results/GB10-GLM53-MTP-004.json), [optimized MTP3](../benchmarks/gb10/results/GB10-GLM53-OPT-005.json), [KDA state commits](../benchmarks/gb10/results/GB10-GLM53-OPT-006.json) |
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
`--from-source` to reproduce conversion locally. FTW arranges weights for fast loading;
the recipe defines any precision conversions. Separately quantized artifacts do not
inherit the source recipe's certification.

## FTW and NVMe execution

FTW is SparkLab's self-contained fast-load checkpoint format. It stores model metadata,
resident tensors, and routed-expert banks in indexed shards. Preparation can repack
existing quantized weights or apply recipe-specific conversions, such as resident FP8
storage. Check the exact recipe and artifact fingerprint before reusing a prepared model.

In disk mode (`--moe-storage disk`), the runtime reads routed expert rows from local
NVMe into bounded host staging buffers and a GPU expert cache. Resident weights, expert
caches, KV and recurrent state, and workspaces still need to fit the unified-memory
budget. `--moe-preload-all` instead requires the entire routed-expert bank to fit in
memory. Swap is not counted as usable model capacity.

Use `sparklab plan <recipe> --prepare` to check storage and memory requirements before
downloading. Source conversion may need space for both the source checkpoint and the
prepared artifact; prebuilt FTW downloads avoid that conversion. Disk-backed throughput
and TTFT depend on NVMe performance and expert-cache residency.

SparkLab can load additional uncataloged architectures, but only recipes returned by
`sparklab models` are part of the supported GB10 portfolio. Direct conversion and serving
options are documented in [cli.md](cli.md).
