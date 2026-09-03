# SparkLab GB10 benchmark evidence

This directory contains compact, reviewable summaries for SparkLab model
recipes. Raw logs and large result streams stay outside the source repository.

- `result.schema.json` defines the versioned summary contract.
- `results/GB10-QWEN38-FP8-PLE-005.json` measures NVFP4 routed experts with a
  50%-smaller FP8 external PLE table.
- `results/GB10-QWEN38-HYBRID-006.json` adds physical FP8 resident projections:
  22.16 tok/s, 3.36 GiB lower server VRAM, exact 128-token output parity, and a
  passing 4096-token-cap W1 smoke gate.
- `results/GB10-DSV4-ADAPTIVE-004.json` records confidence-adaptive DSpark,
  replay-free first-draft rejection commits, residency profiling, true FP8
  window/compressed KV, and packed FP4 Lightning-Indexer storage.
- `results/GB10-DSV4-SMALLM-005.json` records the SM121 small-row FP8/BF16 kernels,
  the 6,550-slot residency and confidence-threshold sweeps, and the selected
  three-trial DSpark5 burst median of 13.15 tok/s. Its 256-token control records
  why target-only remains the sustained/default profile.
- `results/GB10-BASELINE-001.json` records the measured DeepSeek V4 launch baseline.
- `results/GB10-DSV4-SPARSE-001.json` records the optimized DeepSeek V4 route-first
  sparse-prefill probe: 10.28 decode tok/s and 0.604 s warm TTFT, with the baseline
  output hash preserved.
- `results/GB10-DSV4-DSPARK-002.json` records the fused 0731 checkpoint's fixed
  256-token DSpark matrix. The target-only control reached 7.41 tok/s; N=1/3/5/7
  reached 7.07/6.19/5.97/5.16 tok/s, so speculation remains opt-in for the
  single-GB10 disk-offload recipe.
- `results/GB10-DSV4-RESIDENCY-003.json` records the unified-memory residency
  optimization: removing the unused host expert LRU raised target-only decode
  from 7.41 to 8.67 tok/s and DSpark N=1 from 7.07 to 8.27 tok/s.
- `results/GB10-QWEN36-FAST-001.json` records the Qwen3.6 NVFP4 Fast
  performance probe.
- `results/GB10-QWEN36-MTP-003.json` records the native BF16 MTP sweep. Three drafts
  reproduced the target-only output hash and accepted 41/50 proposals, but measured only
  8.57 tok/s versus the 52.63 tok/s control before short-batch kernel optimization.
- `results/GB10-QWEN36-MTP-004.json` records the optimized native MTP path. Width two
  reached 74.42 tok/s versus its matched 45.38 tok/s eager control on the 256-token
  Triton-attention probe, a 64.0% gain and 10.60x the prior width-two implementation.
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
- `results/GB10-QWEN38-NVFP4-OPT-003.json` records the current default concurrent
  profile: 16.84 single-stream decode tok/s, 0.306 s repeated-prompt TTFT, and
  61.79 aggregate tok/s at concurrency four.
- `results/GB10-QWEN38-MTP-007.json` records the optimized native MTP sweep. The selected
  three-draft profile measured 30.67 decode tok/s and 0.258 s warm TTFT, a 55.35%
  improvement over its controlled target-only run; all three selected-profile trials
  reproduced the same 128-token output hash.
- `results/GB10-GLM53-NVFP4-001.json` records the GLM-5.3 Flash NVFP4
  complete-checkpoint probe: 4.46 decode tok/s and 5.760 s warm TTFT. Its corrected
  greedy probe reaches the reference answer; the broader quality gates remain outstanding.
- `results/GB10-KIMI-001.json` records the full Kimi K3 ModelOpt NVFP4 FTW
  capacity experiment with SparkLab's GB10 resident FP8 profile: 0.161 decode tok/s,
  395.405 s warm TTFT, exact 256-token completion, and zero runtime OOM/swap-out.
  Cross-rung greedy determinism and answer correctness are explicitly not established.

The Qwen3.8 recipe version 0.8.0 passes the Frontier performance, context, capability,
and focused quality gates but remains Experimental because its clean-revision 60-minute
endurance gate is outstanding. MTP is an opt-in batch-one greedy profile; its result does
not replace the separate default/concurrent-profile measurements. Historical Qwen3.8
results do not transfer across checkpoint revisions and recipe versions.

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
