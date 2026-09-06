# Run Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is an Experimental, text-only Frontier recipe for one NVIDIA
GB10. Recipe 0.9.0 uses NVIDIA's mixed-precision checkpoint: native NVFP4 target
experts, native block-FP8 MTP experts, BF16 resident projections, and scaled FP8
PLE n-gram embeddings.

## Install

Use a [source installation](../install.md#method-2-install-from-source) containing
the NVIDIA mixed-precision loader, scaled FP8 PLE support, complete indexed MTP
sidecar extraction, and accepted-prefix state commits. The previously released
0.1.2 wheel does not contain these changes.

## Prepare

Use local NVMe storage. `plan` reports the authoritative disk requirement,
including the safety margin.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan qwen3.8-flash-next --root /path/to/models --prepare
sparklab pull qwen3.8-flash-next --root /path/to/models --prepare
```

The prebuilt artifact is
[oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW](https://huggingface.co/oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW),
pinned to revision `f547c96e86d0e50908c1415f4525c4325555691e`.
It derives from NVIDIA source revision `fab0aecb760cec45227f6656abcaafa11abca87a`.
The weight payload is 131,931,279,080 bytes; fingerprint `94e1ee0daa442357`.

Preparation preserves the published weight bytes and scales. The 51.2 GB PLE
payload uses its global scale from `qwen4_ngram.json`. The legacy-named
`nvfp4_experts_mtp.safetensors` now contains the complete NVIDIA FP8/BF16 draft
module, assembled from three source shards. Use `--from-source` to reproduce
conversion locally.

The previous Inferact artifact remains available at revision
`5ab790b83f149a96594237a35905d84be24599a3`; old pinned recipes continue to resolve it.

## Run

```bash
sparklab run qwen3.8-flash-next --root /path/to/models
```

Startup preloads all 24,576 routed experts into immutable unified-memory slots,
eliminating steady-state expert disk reads and LRU management. The recipe retains
the 131,072-token KV capacity and batch-eight admission/capture configuration.
Dense QSA uses CUDA graphs; longer sparse-QSA requests fall back to eager
execution. These are runtime settings, not transferred NVIDIA concurrency or
long-context certification results.

## Native MTP

```bash
sparklab run qwen3.8-flash-next --root /path/to/models -- --speculative-tokens 3
```

MTP is opt-in, batch-one greedy, with one to three draft tokens. Accepted-prefix
commits retain intermediate GDN state and PLE convolution inputs, avoiding
rejection replay while keeping KV and recurrent state at the accepted boundary.
Dense-QSA verification uses a dedicated fixed-shape graph.

The pre-optimization NVIDIA MTP3 baseline was 28.08 tok/s and 0.268 s warm TTFT
(three-trial medians). See
[the checkpoint comparison](../../benchmarks/gb10/results/GB10-QWENNVIDIA-001.json).
The selected accepted-prefix profile measured **31.97 tok/s** and **0.260 s**
warm TTFT (three-trial medians), a 13.9% speed improvement with zero rejection
replay. All three runs produced the same output hash; that hash differs from
target-only and pre-optimization MTP. See
[accepted-prefix evidence](../../benchmarks/gb10/results/GB10-QWENNVIDIA-002.json).

## Experimental reduced draft vocabulary

An independent experiment inspired by
[MiaAI-Lab's single-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
limits only the draft output head. Target logits and target verification keep the
full vocabulary; checkpoint weights on disk are unchanged. Enable it with:

```bash
SPARKLAB_QWEN4_DRAFT_VOCAB_SIZE=65536 \
  sparklab run qwen3.8-flash-next --root /path/to/models -- --speculative-tokens 3
```

The default is `0` (full draft vocabulary). Positive budgets must fit the target
vocabulary and include all tokenizer-added tokens. This implementation uses a
deterministic low-token-ID prefix plus those added tokens, not MiaAI-Lab's
corpus-frequency vocabulary. It supports TP=1 and greedy native MTP only.

The 65,536-row draft head adds approximately 0.31 GiB of resident memory. On the
128-token math probe it reached a three-run median **35.28 tok/s**, versus a fresh
**32.11 tok/s** baseline, with the same generated output. However, the 256-token
code probe was effectively flat (25.41 to 25.36 tok/s), and Chinese regressed
from 22.93 to 17.12 tok/s as draft acceptance dropped from 46.3% to 15.1%.
Code and Chinese outputs differed. Do not enable this globally for multilingual
or coding workloads; a frequency-trained vocabulary needs separate evaluation.
Limited output parity
does not establish quality equivalence across languages or long contexts;
out-of-shortlist tokens may reduce acceptance. This remains opt-in and does not
replace the portfolio's default-profile metric. See
[the experiment](../../benchmarks/gb10/results/GB10-QWENNVIDIA-003.json).

BF16 recurrent state was also tested but was slower on this workload; FP32
remains the default. The other repository's prose/concurrency numbers use a
different checkpoint and workload and are not a direct comparison to this probe.

## Validation limits

The NVIDIA checkpoint does not inherit the old Inferact quality, 64K recall,
concurrency, or endurance evidence. Short greedy probes and state-comparison
tests do not establish general quality parity or an advantage over Inferact.
Saving verified intermediate states can change floating-point rounding relative
to replaying smaller prefixes; exact generated-text parity is not guaranteed.

See the [quick start](../quickstart.md) for API and agent examples.
