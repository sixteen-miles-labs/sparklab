# Run Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is a text-only Frontier-tier NVFP4 recipe for one NVIDIA GB10. It
uses a pinned, prebuilt FTW artifact and preloads every routed expert into immutable
unified-memory slots. Local NVMe still holds the artifact and the external n-gram bank.
The recipe is Experimental.

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

Use fast local NVMe storage. The catalog currently requires about 503 GB of free space;
`plan` reports the authoritative requirement before downloading.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan qwen3.8-flash-next --root /path/to/models --prepare
sparklab pull qwen3.8-flash-next --root /path/to/models --prepare
```

`pull --prepare` automatically downloads the pinned Hugging Face FTW artifact from
[`oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW`](https://huggingface.co/oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW),
then validates its immutable revision and fingerprint. The artifact preserves the
publisher's ModelOpt NVFP4 precision. Use `--from-source` only to reproduce the FTW
conversion locally.

## Run

```bash
sparklab run qwen3.8-flash-next --root /path/to/models
```

Startup includes a one-time full-expert preload (about 35 seconds in the measured run).
Steady-state decode then performs no routed-expert disk staging or LRU bookkeeping. On
GB10, SparkLab first advises Linux to release clean download page cache so a freshly
pulled artifact does not hide reclaimable unified memory from the cache planner.
The recipe caps KV capacity at 131,072 tokens, enough for the validated 64K context gate
while retaining substantially more operating-system headroom than the unconstrained
auto-allocation.

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.
