# Spark Lab GB10 benchmark evidence

This directory contains compact, reviewable summaries for Spark Lab model
recipes. Raw logs and large result streams stay outside the source repository.

- `result.schema.json` defines the versioned summary contract.
- `results/GB10-BASELINE-001.json` records the measured DeepSeek V4 launch baseline.
- `results/GB10-QWEN36-FAST-001.json` records the Qwen3.6 NVFP4 Fast
  performance probe.
- `results/GB10-QWEN38-FRONTIER-001.json` records the certified text-only
  Qwen3.8-Flash-Next result for the superseded NVFP4 recipe.
- `results/GB10-QWEN38-FP8-001.json` records the official FP8 recipe's measured
  complete-checkpoint performance probe; it does not pass Frontier admission.

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
