# Run Qwen3.6-35B-A3B

Qwen3.6-35B-A3B is SparkLab's Fast-tier NVFP4 recipe. It uses a pinned, prebuilt FTW
artifact and runs resident on one NVIDIA DGX Spark. The recipe is Experimental.

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

Validate the host and inspect the storage plan:

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan qwen3.6-35b-a3b --root /path/to/models --prepare
sparklab pull qwen3.6-35b-a3b --root /path/to/models --prepare
```

`pull --prepare` automatically downloads the pinned Hugging Face FTW artifact from
[`oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW`](https://huggingface.co/oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW),
then validates its immutable revision and fingerprint. It does not requantize the model.
Use `--from-source` only to reproduce the FTW conversion locally.

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
