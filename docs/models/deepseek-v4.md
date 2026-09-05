# Run DeepSeek V4 Flash

DeepSeek V4 Flash is SparkLab's Preview Frontier recipe. It preserves the source DS-FP4
precision, includes the fused checkpoint's three DSpark draft blocks, and uses NVMe-backed
MoE execution on one NVIDIA DGX Spark.

For prompts up to 512 tokens, the recipe loads only the routed expert rows and preserves
the warmed GPU expert cache. Longer prompts automatically use bounded full-layer streaming.

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
FTW artifact when the recipe publishes one. DeepSeek V4 does not currently declare one,
so the same command downloads the pinned source checkpoint and prepares FTW locally;
allow time and space for both artifacts.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan deepseek-v4 --root /path/to/models --prepare
sparklab pull deepseek-v4 --root /path/to/models --prepare
```

Review the exact storage and runtime admission output from `plan` before continuing.
The pinned recipe auto-sizes expert residency from safe available unified memory.

## Run

```bash
sparklab run deepseek-v4 --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

See the [quick start](../quickstart.md) for API and agent examples.

## Optional DSpark speculative decoding

The fused 0731 checkpoint supports one to seven speculative tokens. DSpark remains
opt-in: its three MoE draft layers and wider target verification can increase disk
traffic. The selected batch-one greedy profile is:

```bash
sparklab run deepseek-v4 --root /path/to/models -- \
  --speculative-method dspark \
  --speculative-tokens 5 \
  --dspark-confidence-threshold 0.45 \
  --moe-cache-size 6304
```

The portfolio's original AIME-25 problem 0 probe (48 prompt tokens, 128 generated
tokens, thinking enabled) now measures **14.02 tok/s and 0.515 s warm TTFT**, using
three trials after warmup. This is 6.6% above the older published 13.15 tok/s.

Verification now captures the attention and indexer compressors' state after each
token, then commits only the accepted prefix. Shared slot metadata is resolved
once per verification block. This avoids additional target forwards on rejection,
including at compression and window-page boundaries. Target weights are unchanged.

This also fixes a correctness issue in the previous first-rejection shortcut: its
capture hook was attached to single-token decode, but verification uses prefill
kernels. Without a captured prefix, the safe fallback now replays the anchor too.
The older [threshold/residency results](../../benchmarks/gb10/results/GB10-DSV4-SMALLM-005.json)
remain historical evidence, not a correctness reference for the current decoder.

Fresh single-stream controls used the same checkpoint, 6,304 expert slots, and
2,048 KV tokens:

| Profile | 128-token decode | 256-token decode |
|---|---:|---:|
| Corrected DSpark5 replay, three-trial median | 11.50 tok/s | 8.27 tok/s |
| **DSpark5 prefix commits, three-trial median** | **14.02 tok/s** | **9.96 tok/s** |
| Target-only, one measured trial | 11.02 tok/s | 8.79 tok/s |

The prefix-commit gains over corrected replay are 22.0% and 20.5%, respectively.
The selected short trials share the legacy profile's output hash, but target-only,
corrected replay, and batched verification can produce different traces. Exact
target-only output parity and broad quality equivalence are not claimed.

The single-trial 512-token coding and prose checks reached 9.03 and 7.77 tok/s.
Plain target-only reached 8.69 tok/s on prose, so it remains the default. Basic
arithmetic, executable coding, reasoning, tool-call, and recall probes returned
correct results; two strict math-format checks failed (an equation instead of a
bare integer, and a boxed answer instead of the requested `FINAL=70`). Target-only
produced the same two strict failures and the same six passes.

An online cost-aware width/target-only policy and immediate re-drafting after
rejection were also tested. Neither was selected for deployment: the cost-aware
policy's prose result trailed plain target-only, and immediate re-drafting did not
improve the short probe. The existing confidence threshold is retained.

For a matched reproduction, add `--num-tokens 2048 --max-seq-len-override 8704` to
the server command above. From a source checkout, run:

```bash
python benchmarks/bench_single_stream.py --base-url http://127.0.0.1:1919 \
  --model deepseek-v4 --label dspark-prefix --workloads aime --thinking \
  --tokens 128 --trials 3 --output results/dspark-prefix.json
```

Use the model ID returned by `/v1/models` if serving a checkpoint path directly.
Repeat with `--tokens 256` for the longer probe. Start the server with
`SPARKLAB_DSPARK_PREFIX_COMMIT=0` to measure the corrected replay control.

These are native SparkLab results on one GB10, not resident two-Spark vLLM numbers.
The 167 GB fused checkpoint still requires NVMe-backed experts. Full 64K context,
agent, quality, and endurance certification remain pending; exploratory runs
showed swap activity and memory-guard aborts. See the
[versioned evidence](../../benchmarks/gb10/results/GB10-DSV4-PREFIX-006.json).
