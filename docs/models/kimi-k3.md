# Run Kimi K3

Kimi K3 is SparkLab's 2.8T-parameter Research recipe for inference beyond physical memory.
It is Experimental, and its prebuilt FTW artifact is not yet published.

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

Use fast local NVMe storage. `pull --prepare` automatically selects a pinned Hugging Face
FTW artifact when the recipe publishes one. Kimi K3 does not have one yet, so the same
command downloads the source checkpoint and builds FTW locally. The catalog currently
requires about 3.36 TB of free space, making this a long, storage-intensive operation.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan kimi-k3 --root /path/to/models --prepare
sparklab pull kimi-k3 --root /path/to/models --prepare
```

Do not continue unless `doctor` and `plan` pass. Preserve the generated FTW directory so
an interrupted model download or preparation does not need to restart from zero.

## Run

```bash
sparklab run kimi-k3 --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

No accepted GB10 performance result is attached yet. See the
[quick start](../quickstart.md) for API examples.
