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
target verification increase expert traffic when experts are disk-backed. To experiment with
the official probabilistic draft policy:

```bash
sparklab run deepseek-v4 --root /path/to/models -- \
  --speculative-method dspark \
  --speculative-tokens 1 \
  --draft-sample-method probabilistic
```

The measured 256-token, batch-one matrix on one GB10 was:

| DSpark tokens | Decode tok/s | Acceptance | Outputs / target forward | Physical expert I/O |
|---:|---:|---:|---:|---:|
| disabled | 7.41 | — | 1.00 | 38.44 GiB |
| 1 | 7.07 | 77.2% | 1.62 | 49.58 GiB |
| 3 | 6.19 | 42.3% | 1.68 | 52.64 GiB |
| 5 | 5.97 | 32.7% | 1.82 | 59.87 GiB |
| 7 | 5.16 | 17.5% | 1.60 | 61.56 GiB |

These numbers are not directly comparable to vLLM's resident two-DGX-Spark setup. SparkLab's
single-GB10 path makes the 167 GB fused checkpoint fit by keeping most experts on NVMe, and the
additional draft/verification routes can cost more I/O than the accepted tokens save. Greedy
verification also uses multi-token sparse-prefill kernels, so its output need not be bitwise
identical to single-token decode even though speculative rejection preserves the target choice
at each verification call.
