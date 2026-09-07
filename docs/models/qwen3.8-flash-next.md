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

## Experimental full-vocabulary draft optimization

The development profile below combines immediate drafting after a rejection,
a separate FP8 draft-only output head, and smaller verification snapshots.
It requires the corresponding source changes; it is not enabled by recipe 0.9.0.

```bash
SPARKLAB_QWEN4_DRAFT_VOCAB_SIZE=0 \
SPARKLAB_QWEN4_REJECT_DRAFT=1 \
SPARKLAB_QWEN4_DRAFT_HEAD=fp8 \
SPARKLAB_QWEN4_LIGHT_VERIFY_SNAPSHOT=1 \
SPARKLAB_QWEN4_FAST_QSA_METADATA=0 \
  sparklab run qwen3.8-flash-next --root /path/to/models -- --speculative-tokens 3
```

All three switches are off by default (`DRAFT_HEAD=bf16`). The FP8 head is a
separate full-vocabulary proposal copy, adding approximately 0.59 GiB. It never
replaces the BF16 target head or tied embeddings. Target weights on disk, target
precision, BF16 QSA KV and FP32 GDN state stay unchanged. Target verification
still checks every draft; draft-head quantization may change acceptance and
floating-point execution paths. This is not a promise of identical generated
text or general quality equivalence.

Fresh single-stream measurements use five trials after one warmup per workload:

| Workload | Current MTP3 | Optimized MTP3 | Change |
|---|---:|---:|---:|
| Math, 128 tokens, thinking on | 32.57 tok/s | 42.14 tok/s | +29.4% |
| English hash-map prose, 600 tokens | 27.58 tok/s | 34.21 tok/s | +24.0% |
| Python LRU code, 512 tokens | 33.67 tok/s | 38.64 tok/s | +14.8% |
| Chinese hash-map prose, 600 tokens | 29.28 tok/s | 36.18 tok/s | +23.6% |

These are greedy, fixed-length streaming probes, not quality scores. The math
output hash matches the baseline; the longer generated texts differ. The current
development evaluation has no new failures on the 52-case arithmetic/recall/JSON
set. However, the fixed 100-case English/Chinese MGSM subset fell from **94/100
to 92/100**: three new failures and one improved case. Two new failures exhausted
the identical 1,536-token allowance; one chose a different interpretation than
the dataset's answer. Gold answers, prompts, budgets and scoring were not changed.
The 15 extended coding/reasoning/recall/tool checks and both 15-case lifecycle
runs passed, with no OOM, swap-out, or memory-guard event. Actual long recall
was about 11,735 prompt tokens, not 32K/64K certification. This is the fastest
fully evaluated profile so far and is a reasonable opt-in when the observed
two-point loss on this subset is acceptable. It does **not** meet a
no-quality-drop requirement; two points here are not a bound on losses on other
tasks. Recipe defaults remain unchanged; the README portfolio reports this
opt-in profile.
This fixed subset was used during tuning, not as an independent final evaluation.

Restoring the BF16 draft head while retaining immediate redraft scored
**93/100**, with two new failures and one improvement, at smaller speed gains
of 5.6–13.9%. Keeping the baseline draft schedule with FP8 proposals, light
snapshots and `SPARKLAB_QWEN4_FAST_QSA_METADATA=1` scored **94/100** and reached
35.98/30.19/35.70/32.11 tok/s in the same workload order. That unchanged total
includes one new failure and one recovery, not identical answers. Its extended
and lifecycle checks also passed. These ablations do not isolate draft
quantization as the sole cause of changed outputs. See
[the measured evidence](../../benchmarks/gb10/results/GB10-QWENNVIDIA-004.json).

Further five-trial screens did not produce a better general replacement:
metadata reuse changed each median by less than 0.5%; MTP4 improved code by
6.4% but slowed math and Chinese, and has not run the MGSM subset. Existing
`--qwen4-dense-storage fp8` also slowed math by 14.1% and Chinese by 9.7%,
with prose/code nearly flat and one new smoke-test answer regression versus
the accepted fast profile. That target-precision experiment was not selected;
see its [separate results](../../benchmarks/gb10/results/GB10-QWENNVIDIA-005.json).

To reproduce the speed probes against each freshly started server, use the
same order and settings (one request at a time, no concurrent tests):

```bash
python benchmarks/bench_single_stream.py --base-url http://127.0.0.1:8000 \
  --model qwen3.8-flash-next --workloads aime --tokens 128 --thinking \
  --trials 5 --label candidate --output results/candidate-math.json
python benchmarks/bench_single_stream.py --base-url http://127.0.0.1:8000 \
  --model qwen3.8-flash-next --workloads hashmap --tokens 600 \
  --trials 5 --label candidate --output results/candidate-prose.json
python benchmarks/bench_single_stream.py --base-url http://127.0.0.1:8000 \
  --model qwen3.8-flash-next --workloads code --tokens 512 \
  --trials 5 --label candidate --output results/candidate-code.json
python benchmarks/bench_single_stream.py --base-url http://127.0.0.1:8000 \
  --model qwen3.8-flash-next --workloads chinese --tokens 600 \
  --trials 5 --label candidate --output results/candidate-chinese.json
```

Use the model ID returned by `/v1/models`. Repeat on a fresh baseline server
with `REJECT_DRAFT=0`, `DRAFT_HEAD=bf16`, `LIGHT_VERIFY_SNAPSHOT=0`, and
`FAST_QSA_METADATA=0` (all names
have the `SPARKLAB_QWEN4_` prefix). The measured speed runs additionally used
`--max-seq-len-override 8792`, `--max-running-requests 1`,
`--cuda-graph-max-bs 1`, `--disable-moe-prefill-overlap`, full expert preload,
131,072 KV tokens and Triton NVFP4. No larger-context performance is implied.

See the [focused quality runner](../../benchmarks/quality/README.md#focused-qwen4-optimization-regressions)
for the fixed paired evaluation protocols.

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
