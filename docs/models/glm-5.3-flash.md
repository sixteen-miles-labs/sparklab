# Run GLM-5.3 Flash

GLM-5.3 Flash is a text-only Frontier-tier NVFP4 recipe with NVMe-backed MoE execution.
The recipe is Experimental.

## Prepare

Use fast local NVMe storage. By default, `--prepare` downloads the pinned, validated FTW
artifact from
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
