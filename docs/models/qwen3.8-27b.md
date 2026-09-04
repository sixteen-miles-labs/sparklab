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

Opt-in DFlash2-8:

```bash
sparklab run qwen3.8-27b --root /path/to/models -- \
  --attention-backend triton \
  --speculative-method dflash2 \
  --speculative-tokens 8 \
  --speculative-draft-model /path/to/models/qwen3.8-27b-dflash2
```

The selected three-trial DGX Spark probe measured 35.48 decode tok/s and 0.153 s
warm TTFT, with the same greedy output hash as target-only. DFlash2 supports block
sizes 2–16, but remains limited to batch-one greedy requests and runs without CUDA
graphs. It clears the Fast throughput and latency thresholds, while full Fast
certification remains pending. See the [benchmark evidence](../../benchmarks/gb10/results/GB10-QWEN38-DFLASH-003.json).
