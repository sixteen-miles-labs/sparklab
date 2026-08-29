# Run Qwen3.6-35B-A3B

Qwen3.6-35B-A3B is SparkLab's Fast-tier NVFP4 recipe. It uses NVIDIA's pinned NVFP4
checkpoint, repacks it locally as FTW, and runs resident on one NVIDIA DGX Spark. The
recipe is Experimental.

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

Validate the host and inspect the storage plan:

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan qwen3.6-35b-a3b --root /path/to/models --prepare
sparklab pull qwen3.6-35b-a3b --root /path/to/models --prepare
```

`pull --prepare` downloads the pinned
[`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
checkpoint and repacks it locally into SparkLab's FTW runtime format. The conversion
preserves the source NVFP4 quantization; it does not requantize the model. Keep enough
free storage for both the 23.5 GB source and the approximately 20.9 GB prepared artifact.

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
