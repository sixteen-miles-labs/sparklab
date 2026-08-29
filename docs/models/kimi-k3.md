# Run Kimi K3

Kimi K3 is SparkLab's 2.8T-parameter Research recipe for inference beyond physical memory.
It is Experimental. The validated text-only FTW artifact is published at
[`oakmindai/Kimi-K3-NVFP4-FTW`](https://huggingface.co/oakmindai/Kimi-K3-NVFP4-FTW).

## Install SparkLab

Follow the [full installation guide](../install.md). On NVIDIA DGX Spark, the recommended
package install is:

```bash
uv venv && source .venv/bin/activate
uv pip install "sparklab[accel]"
sparklab --version
```

The `sparklab` distribution provides the `sparklab` command.
See [Install from source](../install.md#method-2-install-from-source) for a development
checkout.

## Prepare

Use fast local NVMe storage. `pull --prepare` automatically selects a pinned Hugging Face
FTW artifact. The Kimi repository contains 1,610,936,311,808 bytes across 194 FTW shards,
so allow approximately 1.5 TiB for the artifact plus the planner's operational reserve.
Use `--from-source` only when you deliberately want to download NVIDIA's original
checkpoint and reproduce the conversion; that path requires roughly 3.36 TB.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan kimi-k3 --root /path/to/models --prepare
sparklab pull kimi-k3 --root /path/to/models --prepare
```

Do not continue unless `doctor` and `plan` pass. Preserve the generated FTW directory so
an interrupted download does not need to restart from zero. `pull` pins the public artifact
revision and validates its byte count and FTW fingerprint before recording it for launch.

## Run

```bash
sparklab run kimi-k3 --root /path/to/models
```

The recipe applies the measured bounded configuration automatically: disk-backed ModelOpt
NVFP4 experts, an 896-slot layer-aware cache, per-row FP8 resident weights, disabled startup
prefill warmup, eager execution, and one active request. The server can take several minutes
to become ready; the first request pays the prefill/JIT cost.

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

The promoted 256-token probe measured **0.1613 tok/s** and **395.405 s** warm TTFT with
17.68 GiB minimum `MemAvailable`, zero scoped runtime OOM kills, and zero runtime swap-out.
This is a bounded-capacity result, not an interactivity or correctness claim: the response
ended before stating the expected AIME answer, and the 256-token greedy text diverged from
the shorter promotion ladder. See
[`GB10-KIMI-001`](../../benchmarks/gb10/results/GB10-KIMI-001.json), the
[full experiment](../../exps/exp_kimik3_gb10.md), and the
[quick start](../quickstart.md) for API examples.
