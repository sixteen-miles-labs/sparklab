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
which now contains the pinned 1.49 GiB native MTP sidecar. SparkLab validates the
artifact's immutable revision, total size, and FTW fingerprint. The artifact preserves
the publisher's ModelOpt NVFP4 precision. Use `--from-source` only to reproduce the FTW
conversion locally; that path also copies the MTP sidecar into the prepared checkpoint.

## Run

```bash
sparklab run qwen3.8-flash-next --root /path/to/models
```

Startup includes a one-time full-expert preload (about 35 seconds in the measured run).
Steady-state decode then performs no routed-expert disk staging or LRU bookkeeping. On
GB10, SparkLab first advises Linux to release clean download page cache so a freshly
pulled artifact does not hide reclaimable unified memory from the cache planner.
QSA decode replays CUDA graphs at batch sizes 1, 2, 4, and 8 while every request remains
inside the exact dense budget (up to 2,051 visible tokens). SparkLab pads intermediate
batches to the next captured size and automatically returns to eager sparse QSA for
longer contexts; no command-line switch is required. The recipe admits up to eight
concurrent requests.

In a matched three-trial, 128-token HTTP benchmark, raising admission and graph capture
from batch four to eight increased concurrency-eight aggregate throughput from 55.64 to
85.72 tok/s (+54.1%). It reduced p95 TTFT from 9.709 to 0.867 seconds (-91.1%) and p95
end-to-end latency from 18.403 to 11.945 seconds (-35.1%). Batch-eight eager serving
reached 80.21 tok/s, so the batch-eight graph contributed a further 6.9% throughput.
The C1 and C4 medians changed by -3.4% and -2.2%; this is a burst-concurrency
optimization, not a single-stream speedup. See
[`GB10-QWEN38-CONC-009`](../../benchmarks/gb10/results/GB10-QWEN38-CONC-009.json).
The vLLM FP8-KV scale hoist does not transfer to this path: SparkLab's QSA K/V
cache is BF16 and has no per-tensor K/V scales. The portable optimization is the
wider request-admission and CUDA-graph profile.

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
at the accepted boundary. Enable the measured three-token setting with:

```bash
sparklab run qwen3.8-flash-next --root /path/to/models -- --speculative-tokens 3
```

On GB10, the selected three-draft profile measured a three-trial median 30.67 tok/s and
0.258 s warm TTFT on a 128-token greedy decode. One and two draft tokens measured 25.19
and 29.15 tok/s respectively. The three selected-profile trials reproduced the same
output hash.

MTP is intentionally opt-in. The current transactional path supports one running greedy
request and falls back to ordinary target decoding for non-greedy sampling. Dense-QSA
verification uses a dedicated CUDA graph; longer sparse-QSA requests remain eager. Use
`--speculative-tokens 3` for the measured single-stream greedy profile.

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.
