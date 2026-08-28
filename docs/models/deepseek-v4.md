# Run DeepSeek V4 Flash

DeepSeek V4 Flash is SparkLab's Preview Frontier recipe. It preserves the source DS-FP4
precision and uses NVMe-backed MoE execution on one NVIDIA DGX Spark.

For prompts up to 512 tokens, the recipe loads only the routed expert rows and preserves
the warmed GPU expert cache. Longer prompts automatically use bounded full-layer streaming.

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
