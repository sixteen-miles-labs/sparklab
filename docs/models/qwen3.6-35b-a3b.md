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
quantization and includes the upstream model's BF16 MTP layer as a fourth FTW shard; it
does not requantize either component. Use `--from-source` only when you want to reproduce
the complete FTW repack locally from NVIDIA's pinned source checkpoint.

## Run

```bash
sparklab run qwen3.6-35b-a3b --root /path/to/models
```

## Experimental speculative decoding

SparkLab can load Qwen3.6's native MTP layer, verify its draft tokens with the target,
and commit or roll back paged KV and GDN recurrent state at the accepted boundary. The
[upstream vLLM profile](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B) uses three drafts:

```bash
sparklab run qwen3.6-35b-a3b --root /path/to/models -- \
  --speculative-method mtp \
  --speculative-tokens 3
```

This path is available for experimentation, not recommended for production on GB10. In
a controlled 64-token greedy sweep, target-only decode reached 52.63 tok/s. Draft widths
one, two, and three reached 5.68, 7.02, and 8.57 tok/s respectively. Width three accepted
41/50 drafts and reduced the run to 23 target forwards, but the resident BF16 draft layer
cost more than the saved NVFP4 target work. It was also the only measured width that
reproduced the target-only output hash. Keep the default target-only recipe for normal
chat and agent use. See [GB10-QWEN36-MTP-003](../../benchmarks/gb10/results/GB10-QWEN36-MTP-003.json).

MTP currently applies to one running greedy request. Sampled requests fall back to target
decoding, and target verification runs eagerly.

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.
