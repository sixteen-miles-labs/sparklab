# Run Qwen3.6-35B-A3B

Recipe **0.6.0** makes the fast Marlin/MTP4 container the default on one 128 GiB
NVIDIA GB10. SparkLab still runs the model and scheduler; the container supplies
PyTorch 2.13 and vLLM 0.28's Marlin kernels. The host environment stays unchanged.

Two fixed 8K-input/256-output benchmark repeats averaged **112.08 decode tok/s**
versus vLLM **111.60**. Mean request latency was **3.735 versus 3.664 seconds**.
The profile passed 28 serving checks, 775 runtime/kernel tests, and a one-hour run
with 1,411 successful requests, zero container swap and no OOM events.

The default is marked **preview** because quality equivalence was not established:
the fixed core screen scored 17/20 versus the previous native profile's 18/20;
the five-problem AIME sample capped at 16K output scored 0/5 versus 1/5 for both
native SparkLab and vLLM. See [the qualification record](../../benchmarks/qwen36_marlin/QUALIFICATION.md).

## Build and prepare

Install SparkLab from this source checkout using the [installation guide](../install.md).
Docker with NVIDIA GPU access is required. Build the separate runtime image from
the repository root:

```bash
docker build -f benchmarks/qwen36_marlin/Dockerfile -t sparklab-qwen36:gb10-v1 .
sparklab pull qwen3.6-35b-a3b --root /path/to/models --prepare
```

Preparation downloads NVIDIA's pinned source checkpoint and converts it inside
the container into a separate Marlin FTW artifact under `prepared/0.6.0`.
The previous hosted Triton artifact cannot be used with this profile. Existing
`prepared/0.5.0` artifacts are retained.

## Run

```bash
sparklab run qwen3.6-35b-a3b --root /path/to/models
```

The normal command now launches the container with the recipe's MTP4, graph,
routing, and projection settings. `--dry-run` prints the complete Docker command;
`--json` includes the command and environment in the launch plan. Override the
port with `-- --port 1929`; the default remains 1919.

Wait until `/health` reports JSON `"status": "ok"`, then send requests using model
`nvidia/Qwen3.6-35B-A3B-NVFP4`. HTTP 200 alone may indicate that loading is ongoing.

The profile supports text, one actively decoded request, BF16 KV, FP32 recurrent
state, and **32,832 total sequence tokens**. Additional requests queue. Greedy
requests use MTP4; sampled requests use target-only decoding. Equal 96 GiB Docker
memory and memory-plus-swap limits disable swap for the serving container.
Unrelated host swap is reported by the planner but does not block this isolated
profile; available RAM and the safety reserve are still checked.

## Previous native profile

To run the retained target-only artifact in the host PyTorch 2.11 environment:

```bash
sparklab serve --model /path/to/models/models/qwen3.6-35b-a3b/prepared/0.5.0 \
  --served-model-name nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --moe-backend offload --moe-storage ram --nvfp4-backend triton \
  --moe-cache-rate 1.0 --num-tokens 32832 --moe-prefill-hit-d2d
```

Use a clean shell without the fast profile's environment flags. The previous
profile's short-prompt certification measured 67.79 tok/s and 0.329 s TTFT;
those numbers use a different workload from the 8K comparison above.
[Previous certification](../../benchmarks/gb10/results/GB10-QWEN36-FAST-002.json).

See [container build/conversion details](../../benchmarks/qwen36_marlin/README.md)
and [the full optimization report](../../exps/exp_qwen36_speedbench_vllm_gb10.md).
