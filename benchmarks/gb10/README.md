# Spark Lab GB10 benchmark evidence

This directory contains compact, reviewable summaries for Spark Lab model
recipes. Raw logs and large result streams stay outside the source repository.

- `result.schema.json` defines the versioned summary contract.
- `results/GB10-BASELINE-001.json` records the measured DeepSeek V4 launch baseline.

A model catalog entry may cite a result ID, but that evidence does not make the
recipe Certified by itself. Tier latency, context, correctness, agent, and
endurance gates must all pass on a release artifact.
