# Run Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is a text-only Frontier-tier NVFP4 recipe with NVMe-backed MoE
execution. It uses a pinned, prebuilt FTW artifact. The recipe is Experimental.

## Prepare

Use fast local NVMe storage. The catalog currently requires about 503 GB of free space;
`plan` reports the authoritative requirement before downloading.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan qwen3.8-flash-next --root /path/to/models --prepare
sparklab pull qwen3.8-flash-next --root /path/to/models --prepare
```

`pull --prepare` downloads the pinned FTW artifact and preserves the publisher's ModelOpt
NVFP4 precision.

## Run

```bash
sparklab run qwen3.8-flash-next --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.
