# Run DeepSeek V4 Flash

DeepSeek V4 Flash is SparkLab's Preview Frontier recipe. It preserves the source DS-FP4
precision, includes the fused checkpoint's three DSpark draft blocks, and uses NVMe-backed
MoE execution on one NVIDIA DGX Spark.

For prompts up to 512 tokens, the recipe loads only the routed expert rows and preserves
the warmed GPU expert cache. Longer prompts automatically use bounded full-layer streaming.

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
FTW artifact when the recipe publishes one. DeepSeek V4 does not currently declare one,
so the same command downloads the pinned source checkpoint and prepares FTW locally;
allow time and space for both artifacts.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan deepseek-v4 --root /path/to/models --prepare
sparklab pull deepseek-v4 --root /path/to/models --prepare
```

Review the exact storage and runtime admission output from `plan` before continuing.
The pinned recipe auto-sizes expert residency from safe available unified memory.

## Run

```bash
sparklab run deepseek-v4 --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.

## Optional DSpark speculative decoding

The fused 0731 checkpoint supports one to seven speculative tokens. DSpark is deliberately
disabled in the default single-GB10 recipe because its extra three MoE draft layers and wider
target verification increase expert traffic when experts are disk-backed. For short,
batch-one greedy bursts, the selected high-residency profile is:

```bash
sparklab run deepseek-v4 --root /path/to/models -- \
  --speculative-method dspark \
  --speculative-tokens 5 \
  --dspark-confidence-threshold 0.45 \
  --moe-cache-size 6550
```

The 128-token threshold sweep used the same AIME-25 problem 0 prompt, one warmup request,
greedy sampling, 2,048 KV tokens, and a 6,550-slot expert cache:

| Profile | Decode tok/s | Acceptance | Outputs / target forward |
|---|---:|---:|---:|
| target-only | 10.73 | — | 1.00 |
| DSpark5, threshold 0.35 | 12.01 | 53.6% | 2.10 |
| DSpark5, threshold 0.40 | 12.65 | 60.0% | 2.06 |
| **DSpark5, threshold 0.45** | **13.15** | **68.0%** | **2.06** |
| DSpark5, threshold 0.47 | 12.77 | 70.9% | 1.78 |
| DSpark5, threshold 0.50 | 12.13 | 64.7% | 1.75 |

The selected 0.45 profile reproduced the same output hash and speculative counters in
three trials (13.21, 13.15, and 13.11 tok/s). The SM121 small-row FP8/BF16 kernels account
for a measured 3.7% of its gain; higher expert residency and the adaptive cutoff provide
the rest.

This is a burst profile, not a universal default. Over the matched 256-token sustained
probe, later routed-expert churn reversed the ordering:

| Profile | Decode tok/s | Warm TTFT |
|---|---:|---:|
| **target-only** | **10.52** | **0.607 s** |
| DSpark1 | 9.67 | 1.579 s |
| DSpark5, threshold 0.45 | 9.31 | 1.546 s |

These numbers are not directly comparable to vLLM's resident two-DGX-Spark setup. SparkLab's
single-GB10 path makes the 167 GB fused checkpoint fit by keeping most experts on NVMe, and the
additional draft/verification routes can cost more I/O than the accepted tokens save. Greedy
verification also uses multi-token sparse-prefill kernels, so its output need not be bitwise
identical to single-token decode even though speculative rejection preserves the target choice
at each verification call.
