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
plus the publisher's pinned 1.49 GiB MTP sidecar, then validates their immutable
revisions, sizes, and FTW fingerprint. The artifact preserves the publisher's ModelOpt
NVFP4 precision. Use `--from-source` only to reproduce the FTW conversion locally; that
path copies the MTP sidecar into the prepared checkpoint automatically.

## Run

```bash
sparklab run qwen3.8-flash-next --root /path/to/models
```

Startup includes a one-time full-expert preload (about 35 seconds in the measured run).
Steady-state decode then performs no routed-expert disk staging or LRU bookkeeping. On
GB10, SparkLab first advises Linux to release clean download page cache so a freshly
pulled artifact does not hide reclaimable unified memory from the cache planner.
QSA decode replays CUDA graphs at batch sizes 1, 2, and 4 while every request remains
inside the exact dense budget (up to 2,051 visible tokens). SparkLab pads a three-request
batch to the batch-four graph and automatically returns to eager sparse QSA for longer
contexts; no command-line switch is required. The recipe admits up to four concurrent
requests. On the measured GB10, aggregate decode throughput was 16.94, 36.78, and 61.79
tok/s at concurrency 1, 2, and 4 respectively.

The default hybrid radix cache also snapshots the complete recurrent state: GDN state,
PLE convolution history, paged QSA K/V, and pooled index keys. In a controlled repeated
96-token prompt, it reused 64 tokens and lowered warm TTFT from 404 ms to 306 ms without
changing the greedy output hash. Prefix reuse is most useful for repeated system prompts,
few-shot examples, and agent/tool schemas; it does not increase single-stream decode speed.
The recipe caps KV capacity at 131,072 tokens, enough for the validated 64K context gate
while retaining substantially more operating-system headroom than the unconstrained
auto-allocation.

## Speculative decoding

SparkLab can use the upstream model's native MTP layer. It verifies draft tokens with the
target model and transactionally commits or rolls back paged KV, GDN, PLE, and QSA state
at the accepted boundary. Enable the measured two-token setting with:

```bash
sparklab run qwen3.8-flash-next --root /path/to/models -- --speculative-tokens 2
```

On GB10, a controlled 64-token greedy decode measured 20.31 tok/s versus 15.13 tok/s
without speculative decoding, a 34.2% improvement. One and three draft tokens measured
18.21 and 17.10 tok/s respectively, so more drafts are not automatically faster. All four
settings reproduced the same output hash on fresh prompts and 64-token radix-prefix hits.

MTP is intentionally opt-in. The current transactional path supports one running greedy
request, runs eagerly with overlap scheduling disabled, and falls back to ordinary target
decoding for non-greedy sampling. Keep the default recipe for concurrent traffic and use
`--speculative-tokens 2` for single-stream deterministic chat or agent workloads.

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.
