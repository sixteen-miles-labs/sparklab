# Run Qwen3.6-35B-A3B

Qwen3.6-35B-A3B is SparkLab's Fast-tier NVFP4 recipe. It uses a pinned, prebuilt FTW
artifact and runs resident on one NVIDIA DGX Spark. Recipe 0.3.0 is Fast-certified on
one NVIDIA GB10: 67.79 decode tok/s, 0.329 s warm TTFT, exact 32K recall, and a stable
60-minute zero-swap run. See the
[versioned evidence](../../benchmarks/gb10/results/GB10-QWEN36-FAST-002.json).

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

`pull --prepare` downloads the immutable FTW revision from
[`oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW`](https://huggingface.co/oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW)
and validates its fingerprint before it can run. The artifact preserves the source NVFP4
quantization; it does not requantize the model. Use `--from-source` only when you want to
reproduce the FTW repack locally from NVIDIA's pinned source checkpoint.

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
