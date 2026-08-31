# SparkLab GB10 benchmark evidence

This directory contains compact, reviewable summaries for SparkLab model
recipes. Raw logs and large result streams stay outside the source repository.

- `result.schema.json` defines the versioned summary contract.
- `results/GB10-BASELINE-001.json` records the measured DeepSeek V4 launch baseline.
- `results/GB10-DSV4-SPARSE-001.json` records the optimized DeepSeek V4 route-first
  sparse-prefill probe: 10.28 decode tok/s and 0.604 s warm TTFT, with the baseline
  output hash preserved.
- `results/GB10-QWEN36-FAST-001.json` records the Qwen3.6 NVFP4 Fast
  performance probe.
- `results/GB10-QWEN38-27B-001.json` records the dense Qwen3.8-27B ModelOpt
  NVFP4 result: 8.83 decode tok/s, 0.144 s warm TTFT, exact 64K recall, and
  passing reasoning, tool-call, and coding-agent probes.
- `results/GB10-QWEN38-FRONTIER-001.json` records the certified text-only
  Qwen3.8-Flash-Next result for the superseded NVFP4 recipe.
- `results/GB10-QWEN38-FP8-001.json` records the official FP8 recipe's measured
  complete-checkpoint performance probe; it does not pass Frontier admission and predates
  the current Inferact NVFP4 recipe.
- `results/GB10-QWEN38-NVFP4-001.json` records the current Inferact NVFP4 recipe's
  complete-checkpoint performance probe: 12.58 decode tok/s and 0.786 s warm TTFT.
- `results/GB10-GLM53-NVFP4-001.json` records the GLM-5.3 Flash NVFP4
  complete-checkpoint probe: 4.46 decode tok/s and 5.760 s warm TTFT. Its corrected
  greedy probe reaches the reference answer; the broader quality gates remain outstanding.
- `results/GB10-KIMI-001.json` records the full Kimi K3 ModelOpt NVFP4 FTW
  capacity experiment with SparkLab's GB10 resident FP8 profile: 0.161 decode tok/s,
  395.405 s warm TTFT, exact 256-token completion, and zero runtime OOM/swap-out.
  Cross-rung greedy determinism and answer correctness are explicitly not established.

The Qwen3.8 recipe version 0.5.0 passes the Frontier performance thresholds but remains
Experimental because its context, capability, quality, and endurance gates are outstanding.
Neither historical Qwen3.8 result transfers across its checkpoint revision and recipe version.

A model catalog entry may cite a result ID, but that evidence does not make the
recipe Certified by itself. Tier latency, context, correctness, agent, and
endurance gates must all pass on a release artifact.

Complete-checkpoint tier evidence follows `certification.schema.json` and can be
evaluated without changing catalog state:

```bash
sparklab gate <recipe> <evidence.json> --json
```

Reduced-layer, dummy-weight, or mismatched-recipe evidence is rejected even for
Research admission.
