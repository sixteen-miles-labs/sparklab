# Run GLM-5.3 Flash

GLM-5.3 Flash is a text-only Frontier-tier recipe with NVFP4 routed experts, FP8 KDA
main projections, and NVMe-backed MoE execution. The recipe is Experimental.

## Install SparkLab

Follow the [full installation guide](../install.md). On NVIDIA DGX Spark, the recommended
package install is:

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
sparklab --version
```

The distribution is currently named `freetoken`; it installs the `sparklab` command.
See [Install from source](../install.md#method-2-install-from-source) for a development
checkout.

## Prepare

Use fast local NVMe storage. `pull --prepare` automatically downloads the pinned,
validated Hugging Face FTW artifact from
[`oakmindai/GLM-5.3-Flash-NVFP4-FTW`](https://huggingface.co/oakmindai/GLM-5.3-Flash-NVFP4-FTW).
Use `--from-source` only when you intentionally want to download the Red Hat AI source
checkpoint and reproduce the FTW conversion locally.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan glm-5.3-flash --root /path/to/models --prepare
sparklab pull glm-5.3-flash --root /path/to/models --prepare
```

Review the exact storage and runtime admission output from `plan` before continuing.

## Run

```bash
sparklab run glm-5.3-flash --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.

## Performance and TTFT

The fixed batch-one DGX Spark probe measured 4.98 decode tok/s and 5.379 s warm TTFT.
KDA FP8 reduced resident storage by about 4.25 GiB and improved decode throughput by
18.5% against the same-machine BF16-resident control. TTFT improved by 4.0% because its
main cost is different: prompt-selected expert rows that are absent from the GPU cache
must still be read from local NVMe. A first cold-cache request measured 10.745 s TTFT;
startup warmup removes compilation from that number but cannot pre-resident every routed
expert row.

See the [versioned GB10 evidence](../../benchmarks/gb10/results/GB10-GLM53-KDA-FP8-002.json)
for the complete configuration, comparison, and remaining validation gaps.
