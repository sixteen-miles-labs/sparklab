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
| [Qwen3.6-35B-A3B](https://huggingface.co/oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW) | 35B total / 3B active | NVFP4 · FTW | `qwen3.6-35b-a3b` | Certified | 67.79 | 0.329 |
| **Frontier — hard coding, reasoning, and long agent work** |  |  |  |  |  |  |
| [Qwen3.8-Flash-Next](https://huggingface.co/oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW) | 125B LM + 55B auxiliary / 6B active | NVFP4 · FTW | `qwen3.8-flash-next` | Experimental | 16.06 | 0.434 |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 284B total / 13B active | DS-FP4 | `deepseek-v4` | Preview | 10.28 | 0.604 |
| [GLM-5.3 Flash](https://huggingface.co/oakmindai/GLM-5.3-Flash-NVFP4-FTW) | 320B total / 18B active | NVFP4 + KDA FP8 · FTW | `glm-5.3-flash` | Experimental | 6.27 | 5.681 |
| **Research — complete or novel models outside the interactive envelope** |  |  |  |  |  |  |
| [GLM-5.2](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | 753B total / 40B active | NVFP4 | `glm-5.2` | Experimental fallback | 0.80 | 2.570 |
| [GLM-5.3](https://huggingface.co/Inferact/GLM-5.3-NVFP4) | 753B total / 40B active | NVFP4 | `glm-5.3` | Experimental fallback | 0.81 | 2.530 |
| [Kimi K3](https://huggingface.co/oakmindai/Kimi-K3-NVFP4-FTW) | 2.8T total / 16 of 896 experts | ModelOpt NVFP4/FP8 · FTW | `kimi-k3` | Experimental | 0.16 | 395.405 |

Model links point to the selected source or published FTW checkpoints. Qwen3.6 and Kimi K3
use pinned prebuilt FTW artifacts; their source-conversion paths remain available for
reproducibility.

Parameter values use publisher-reported architecture counts. Qwen3.8's auxiliary total is
the 51B n-gram embedding plus its 4B MTP module. NVIDIA reports Kimi K3 activation as 16 of
896 experts rather than as an active-parameter count.

Measured values are copied from the evidence named by the recipe. “Not yet measured”
means no accepted complete-checkpoint GB10 performance evidence is attached.

The primary lineup is Qwen3.6 NVFP4 for Fast; Qwen3.8 Flash Next, GLM-5.3 Flash,
and DeepSeek V4 Flash for Frontier; and Kimi K3 for Research. The Beta Qwen3.8 recipe
preserves Inferact's publisher-quantized ModelOpt NVFP4 precision. Recipe 0.6.0 preloads
all routed experts into immutable unified-memory slots, caps KV capacity at 128K tokens,
and measured 16.06 decode tok/s with 0.434 s warm TTFT, passing the Frontier performance
thresholds. Exact 64K sparse-QSA recall, reasoning/tool parsing, the coding-agent probe,
and a five-problem quality sample also passed; the clean-revision 60-minute endurance
gate remains outstanding. Its historical FP8 and earlier
NVFP4 results do not transfer across checkpoint and recipe-version boundaries. See the
[optimization evidence](../benchmarks/gb10/results/GB10-QWEN38-NVFP4-OPT-001.json).
Qwen3.6 recipe 0.3.0 passed the Fast gate on its pinned FTW artifact: 67.79 decode
tok/s, 0.329 s warm TTFT, exact 32K recall, capability probes, and an uninterrupted
60.28-minute zero-swap run. The fixed five-problem AIME sample hit the 2,048-token cap
on every problem and scored 0/5, so the certification is an operational Fast-tier claim,
not a quality-benchmark claim. See the
[versioned evidence](../benchmarks/gb10/results/GB10-QWEN36-FAST-002.json).
GLM-5.3 Flash's fused-mHC complete-checkpoint probe measured 6.27 tok/s and 5.681 s
warm TTFT. Against an identical-geometry eager-mHC control, decode throughput improved
by 25.6% and per-token latency fell by 20.4%; a confirmation run reproduced the output
hash within 0.2% throughput. It now passes the Frontier performance thresholds, but
prompt-selected NVMe expert reads still dominate TTFT and the broader certification gates
remain outstanding. See the
[versioned evidence](../benchmarks/gb10/results/GB10-GLM53-MHC-003.json).
DeepSeek V4's optimized route-first sparse prefill measured 10.28 tok/s and 0.604 s
warm TTFT with an auto-sized 5,321-slot expert cache. The repeated fixed probe required
no physical expert reads during the measured request and preserved the established greedy
output hash. Longer prompts fall back to bounded full-layer streaming. See the
[full experiment](../exps/exp_dsv4_gb10.md).
The Kimi K3 FTW artifact preserves NVIDIA's mixed checkpoint: routed experts remain NVFP4,
supported attention projections remain block FP8, and other tensors retain their source
precision on disk. Its validated GB10 resident profile converts 188 dense/shared/embedding/
head matrices to per-row FP8 while loading so the minimum 896-slot expert cache fits. The
exact 256-token probe measured 0.1613 tok/s and 395.405 s warm TTFT with zero scoped runtime
OOM or swap-out. It stopped before a final AIME answer and diverged from shorter greedy
rungs, so correctness and cross-rung determinism are not established. See the
[versioned evidence](../benchmarks/gb10/results/GB10-KIMI-001.json) and
[full experiment](../exps/exp_kimik3_gb10.md).

GLM-5.3 uses the GLM-5.2 runtime recipe because the pinned checkpoint declares the same
`glm_moe_dsa` architecture and execution dimensions. Its own complete-checkpoint probe
measured 0.813 tok/s and 2.530 s warm TTFT with no OOM or swap-out growth. The 256-token
output hit its length cap before stating the expected answer, so correctness is not
established and the recipe remains Experimental. See the
[versioned evidence](../benchmarks/gb10/results/GB10-GLM53-RESEARCH-001.json) and
[full experiment](../exps/exp_glm5_3_full_gb10.md).

GLM-5.2 is listed in Research because its selected GB10 experiment sustained 0.802 tok/s,
below the 5 tok/s Frontier threshold. The displayed 2.57 s TTFT and throughput are measured,
but the result remains Experimental because the 256-token trial swapped out 680 KiB and did
not pass the strict Research gate. See the [full experiment](../exps/exp_glm5_2_gb10.md).

## Engine architecture support

SparkLab's native runtime can load additional checkpoints, including
GLM-4.7, Qwen3.x, GPT-OSS, Gemma-4, MiniMax, and Muse-Glimmer variants. That
compatibility is not a SparkLab support or performance claim. Only entries
returned by `sparklab models` participate in the GB10 product portfolio.

## FTW and NVMe execution

FTW is the engine's self-contained fast-load format. Conversion is optional for
resident checkpoints and required by recipes such as full Kimi K3 whose routed
experts must be independently addressable from NVMe:

```bash
sparklab checkpoint --model /path/to/hf-checkpoint --out /path/to/model-ftw
```

Recipe-backed `sparklab pull <recipe> --prepare` downloads a prebuilt FTW when
the recipe declares an immutable `runtime_artifact`; otherwise it performs this
conversion at the pinned source revision. A runtime artifact declaration records
its Hugging Face repository, full commit revision, byte size, and FTW source
fingerprint. `--from-source` forces conversion. Direct conversion and serving
flags remain an expert interface documented in [cli.md](cli.md).

For Beta 0.1 recipes, FTW preparation is precision-preserving: it may align, shard,
fuse, or repack tensors for the native backend, but it must retain the source
checkpoint's dtype and quantization. Any precision-changing transform is a separately
named experimental artifact and cannot inherit the source recipe's certification.
