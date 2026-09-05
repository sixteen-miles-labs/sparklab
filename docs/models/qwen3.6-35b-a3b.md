# Run Qwen3.6-35B-A3B

Qwen3.6-35B-A3B is SparkLab's Fast-tier NVFP4 recipe. It uses a pinned, prebuilt FTW
artifact and runs resident on one NVIDIA DGX Spark. Recipe 0.5.0 retains the Fast
certification established by its target-only profile on
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

## Optional speculative decoding

SparkLab can load Qwen3.6's native MTP layer, verify its draft tokens with the target,
and commit or roll back paged KV and GDN recurrent state at the accepted boundary. The
[upstream vLLM profile](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B) uses three drafts.
On GB10, SparkLab's measured optimum is currently two drafts with Triton attention:

```bash
sparklab run qwen3.6-35b-a3b --root /path/to/models -- \
  --speculative-method mtp \
  --speculative-tokens 2 \
  --attention-backend triton
```

The optimized path sends verification and short rejection repairs through decode-sized
MoE and recurrent kernels instead of padded prompt kernels. In the controlled 64-token
FlashInfer sweep, target-only decode reached 49.07 tok/s and widths one, two, and three
reached 63.46, 66.31, and 53.64 tok/s. A longer matched Triton-attention probe reached
74.42 tok/s with width two versus 45.38 tok/s target-only: a 64.0% gain, 88.4% draft
acceptance, and 2.49 output tokens per target forward.

Keep this profile opt-in for now. It covers one greedy request, and the multi-token
numerical path selected a different close greedy continuation than single-token eager
decode. The complete context, quality, agent, and endurance certification suite has not
been rerun. See [the original sweep](../../benchmarks/gb10/results/GB10-QWEN36-MTP-003.json)
and [the optimized result](../../benchmarks/gb10/results/GB10-QWEN36-MTP-004.json).

The latest source-tree optimization saves the GDN state at each verified token and
commits the accepted prefix directly, eliminating rejection replay. Three 256-token
trials measured a median **80.55 tok/s** and **0.367 s warm TTFT**, versus a fresh
75.43 tok/s MTP2 control. All three matched the fresh eager target-only output hash
on this prompt; the old MTP control selected a different continuation. Reported
server memory increased from 21.75 to 21.93 GiB. This focused result does not extend
the target-only certification to MTP or establish general output parity. See
[the replay-free evidence](../../benchmarks/gb10/results/GB10-QWEN36-MTP-005.json).

MTP currently applies to one running greedy request. Sampled requests fall back to target
decoding, and target verification runs eagerly.

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.
