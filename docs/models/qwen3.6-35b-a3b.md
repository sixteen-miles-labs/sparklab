# Run Qwen3.6-35B-A3B

Qwen3.6-35B-A3B is SparkLab's Fast-tier NVFP4 recipe. It uses a pinned, prebuilt FTW
artifact and runs resident on one NVIDIA DGX Spark. The recipe is Experimental.

## Prepare

Install SparkLab, then validate the host and inspect the storage plan:

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan qwen3.6-35b-a3b --root /path/to/models --prepare
sparklab pull qwen3.6-35b-a3b --root /path/to/models --prepare
```

`pull --prepare` downloads the pinned FTW artifact; it does not requantize the model.

## Run

```bash
sparklab run qwen3.6-35b-a3b --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.
