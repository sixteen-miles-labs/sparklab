# Run GLM-5.2

GLM-5.2 is a Research-tier NVFP4 recipe for complete-model NVMe-backed inference. It is
Experimental and is not an interactive-latency target.

## Prepare

Use fast local NVMe storage. The catalog currently requires about 1.03 TB of free space.
This recipe prepares FTW locally from the selected source checkpoint.

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
