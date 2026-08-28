# Run GLM-5.2

GLM-5.2 is a Research-tier NVFP4 recipe for complete-model NVMe-backed inference. It is
Experimental and is not an interactive-latency target.

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

Use fast local NVMe storage. `pull --prepare` automatically selects a pinned Hugging Face
FTW artifact when the recipe publishes one. GLM-5.2 does not currently declare one, so
the same command downloads the pinned source checkpoint and prepares FTW locally. The
catalog currently requires about 1.03 TB of free space.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan glm-5.2 --root /path/to/models --prepare
sparklab pull glm-5.2 --root /path/to/models --prepare
```

Review the exact storage and runtime admission output from `plan` before continuing.

## Run

```bash
sparklab run glm-5.2 --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

Expect Research-tier throughput. See the [GLM-5.2 experiment](../../exps/exp_glm5_2_gb10.md)
and the [quick start](../quickstart.md) for more detail.
