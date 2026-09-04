# Run GLM-5.3 Flash

GLM-5.3 Flash is a text-only Frontier-tier recipe with NVFP4 routed experts, FP8 KDA
main projections, and NVMe-backed MoE execution. The recipe is Experimental.

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

Use fast local NVMe storage. `pull --prepare` automatically downloads the pinned,
validated Hugging Face FTW artifact from
[`oakmindai/GLM-5.3-Flash-NVFP4-FTW`](https://huggingface.co/oakmindai/GLM-5.3-Flash-NVFP4-FTW).
Use `--from-source` only when you intentionally want to download the Red Hat AI source
checkpoint and reproduce the FTW conversion locally.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan glm-5.3-flash --root /path/to/models --prepare
sparklab pull glm-5.3-flash --root /path/to/models --prepare
```

Review the exact storage and runtime admission output from `plan` before continuing.

## Run

```bash
sparklab run glm-5.3-flash --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.

### Optional MTP speculative decoding

GLM-5.3 Flash includes a publisher-trained next-token prediction layer. SparkLab can
use it for batch-one greedy serving by placing `model_mtp.safetensors` beside the FTW
artifact, or by pointing `SPARKLAB_GLM5_MTP_PATH` at the sidecar. The current pinned
prebuilt FTW predates this support and therefore needs the explicit sidecar path:

```bash
SPARKLAB_GLM5_MTP_PATH=/path/to/model_mtp.safetensors \
  sparklab run glm-5.3-flash --root /path/to/models -- \
  --speculative-tokens 3
```

Source conversions now copy an available `model_mtp.safetensors` into the prepared
artifact automatically. MTP remains opt-in; target-only serving is unchanged when no
speculative token count is requested.

## Performance and TTFT

The fixed batch-one DGX Spark probe measured 6.27 decode tok/s and 5.681 s warm TTFT.
Fusing GLM-5.3 Flash's multi-stream hyper-connection path improved throughput by 25.6%
and reduced per-token latency by 20.4% against an identical-geometry eager control. A
second optimized run reproduced the greedy output hash with 6.28 tok/s. KDA FP8 remains
part of the artifact and saves about 4.25 GiB of resident storage.

TTFT is effectively unchanged because its main cost is different: prompt-selected expert
rows that are absent from the GPU cache must still be read from local NVMe. The measured
cold-cache request took 10.687 s; startup warmup removes compilation from that number but
cannot pre-resident every routed expert row.

See the [versioned GB10 evidence](../../benchmarks/gb10/results/GB10-GLM53-MHC-003.json)
for the complete configuration, comparison, and remaining validation gaps.

An opt-in width sweep on the same batch-one greedy workload selected three draft tokens:

| Profile | Decode tok/s | Draft acceptance | Output hash vs target |
|---|---:|---:|---|
| Target only | 5.516 | — | reference |
| MTP-1 | 6.624 | 96.9% | different |
| MTP-2 | 6.926 | 89.1% | exact |
| **MTP-3** | **7.232** | **87.9%** | **exact** |
| MTP-4 | 6.371 | 73.7% | different |

MTP-3 is 31.1% faster than the matched fixed-cache target-only control and 15.3%
faster than the published 6.271 tok/s target-only result. The comparison is optimization
evidence, not a replacement certification: it uses one problem, one measured request per
width, a smaller 6,149-slot expert cache required by the resident MTP layer, and no
quality, long-context, concurrent, sampled, or endurance suite. See
[`GB10-GLM53-MTP-004`](../../benchmarks/gb10/results/GB10-GLM53-MTP-004.json).

A follow-up GB10 pass adds a measured skinny BF16 kernel plan for the
154,880-by-4,096 LM head and overlaps shared-expert compute with routed-expert
fetches. Two MTP-3 trials measured 7.434 and 7.438 tok/s (7.436 tok/s mean),
2.82% above the earlier 7.232 tok/s result, while retaining 87.9% acceptance,
the same route/I/O counts, and the exact target-only output hash. Shared-expert
overlap is enabled by the recipe; MTP itself remains opt-in. See
[`GB10-GLM53-OPT-005`](../../benchmarks/gb10/results/GB10-GLM53-OPT-005.json).
