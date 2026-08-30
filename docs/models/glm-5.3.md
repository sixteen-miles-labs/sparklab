# Run GLM-5.3

GLM-5.3 is an Experimental Research-tier NVFP4 recipe for complete-model,
NVMe-backed inference. The pinned Inferact checkpoint declares the same
`glm_moe_dsa` architecture and runtime dimensions as GLM-5.2, so SparkLab uses
the existing GLM-5.2 execution path.

The complete checkpoint has been measured on one NVIDIA GB10 at 0.813 decode
tok/s and 2.530 seconds warm TTFT. The selected output reached its 256-token cap
before stating the expected final answer, so this remains a bounded performance
result rather than a correctness or certification claim.

## Install SparkLab

Follow the [full installation guide](../install.md). On NVIDIA DGX Spark:

```bash
uv venv && source .venv/bin/activate
uv pip install "sparklab[accel]"
sparklab --version
```

## Prepare

Use fast local NVMe storage. By default, the recipe downloads the pinned, validated
FTW artifact directly, so no local conversion is required. The FTW payload is
428,713,099,264 bytes (about 399.3 GiB). Source conversion remains available for
reproducibility; budget about 1.03 TB when keeping both source and prepared artifacts.

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

Expect Research-tier throughput. See the
[GLM-5.3 experiment](../../exps/exp_glm5_3_full_gb10.md) and the
[quick start](../quickstart.md) for more detail.
