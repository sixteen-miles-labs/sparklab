# Run GLM-5.3

GLM-5.3 is an Experimental Research-tier NVFP4 recipe for complete-model,
NVMe-backed inference. The pinned Inferact checkpoint declares the same
`glm_moe_dsa` architecture and runtime dimensions as GLM-5.2, so SparkLab uses
the existing GLM-5.2 execution path.

This is an architecture-compatibility recipe, not a performance result. GLM-5.2
benchmark evidence does not transfer to the GLM-5.3 checkpoint, and GLM-5.3 has
not yet completed a full GB10 load or generation probe.

## Install SparkLab

Follow the [full installation guide](../install.md). On NVIDIA DGX Spark:

```bash
uv venv && source .venv/bin/activate
uv pip install "sparklab[accel]"
sparklab --version
```

## Prepare

Use fast local NVMe storage. The recipe downloads the immutable source revision
and prepares FTW locally. Budget about 1.03 TB of free space; the prepared size
is currently estimated from the same-shape GLM-5.2 conversion.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan glm-5.3 --root /path/to/models --prepare
sparklab pull glm-5.3 --root /path/to/models --prepare
```

Review the exact storage and runtime admission output from `plan` before continuing.

## Run

```bash
sparklab run glm-5.3 --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

Treat the first run as validation and capture a versioned benchmark result before
making performance or reliability claims. See the [quick start](../quickstart.md)
for more detail.
