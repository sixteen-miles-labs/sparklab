# Run Qwen3.8-27B

Qwen3.8-27B is an Experimental Fast-tier, text-only resident NVFP4 recipe for
one NVIDIA GB10. The target-only recipe remains the default; native DFlash2 is
the optional batch-one greedy Fast profile.

## Prepare

Follow the [installation guide](../install.md), then prepare the pinned target:

```bash
sparklab pull qwen3.8-27b --root /path/to/models --prepare
```

Download the pinned DFlash2 draft separately:

```bash
hf download maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal \
  --revision bd7a934213c47a9e7ef69eef36bb3325f47fd1f1 \
  --local-dir /path/to/models/qwen3.8-27b-dflash2
```

## Run

Target only:

```bash
sparklab run qwen3.8-27b --root /path/to/models
```

Opt-in DFlash2-12:

```bash
sparklab run qwen3.8-27b --root /path/to/models -- \
  --attention-backend triton \
  --speculative-method dflash2 \
  --speculative-tokens 12 \
  --speculative-draft-model /path/to/models/qwen3.8-27b-dflash2
```

The selected three-trial DGX Spark probe measured 45.88 decode tok/s and 0.152 s
warm TTFT, up 29.3% from the previous published 35.48 tok/s profile. All three
128-token trials reproduced the original target-only output hash. The default
target-only profile's recorded result remains 8.83 tok/s.
The fresh matched DFlash2-8 control measured 37.69 tok/s, making the gain against
that control 21.7%.

After rejecting a draft, SparkLab now drafts directly from the correction token
and the accepted target-feature prefix. This avoids an extra single-token target
forward. With Triton attention, fixed-width verification uses a CUDA graph with
multi-query attention metadata and per-token recurrent-state snapshots. Drafting
itself remains eager. Set `SPARKLAB_DFLASH2_REJECT_DRAFT=0` and
`SPARKLAB_DFLASH2_VERIFY_GRAPH=0` to reproduce the previous eager control.

The matched 512-token, thinking-off sweep measured these three-trial single-stream
medians on the same checkpoint and 16K KV allocation:

| Workload | Previous DFlash2-8 | Optimized DFlash2-12 |
|---|---:|---:|
| Math | 35.67 tok/s | 59.58 tok/s |
| Coding | 27.49 tok/s | 37.51 tok/s |
| Prose | 15.27 tok/s | 19.26 tok/s |

The 4/8/12/16 block-size sweep favored twelve overall; eight remained faster on
prose. These are fixed prompt/output workloads, not broad task-suite averages.
Verification grouping changes floating-point rounding, so complete long traces
need not match the previous profile. Arithmetic, reasoning-channel, tool-call,
executable coding, and a roughly 7K-token recall probe passed. The graph/eager
verification checks reproduced identical logits and recurrent states.

DFlash2 supports block sizes 2–16 and remains limited to batch-one greedy requests.
The optimized profile clears the Fast performance thresholds, while full Fast
certification, 64K speculative context, and endurance validation remain pending.
See the [benchmark evidence](../../benchmarks/gb10/results/GB10-QWEN38-DFLASH-004.json).

To reproduce the measurements from a source checkout, run the server above with
`--num-tokens 16384 --max-seq-len-override 16384`, then run:

```bash
python benchmarks/bench_single_stream.py --base-url http://127.0.0.1:1919 \
  --label dflash12 --tokens 512 --trials 3 --output results/dflash12.json
python benchmarks/bench_single_stream.py --base-url http://127.0.0.1:1919 \
  --label dflash12-aime --workloads aime --thinking --tokens 128 --trials 3 \
  --output results/dflash12-aime.json
```

Repeat with block size eight and both environment switches above set to zero for
the matched baseline. `benchmarks/bench_qwen27_nvfp4.py` reproduces the exploratory
small-row kernel sweep; its split-K changes were not selected for deployment.
