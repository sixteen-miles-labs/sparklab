# Qwen3.8-27B fast native profile on GB10

Measured September 9, 2026 on one NVIDIA GB10. The integrated RadixArk + FP4
prefill + DFlash2-12 profile reduces mean request time by **21.9%** against the
previous Inferact DFlash profile on the full matched benchmark. Recipe 0.4.0
selects this configuration and automatically acquires its pinned draft.

## Single-client performance

Each configuration used three warmups and 30 fixed SPEED-Bench prompts, exactly
256 output tokens, temperature 0, default thinking enabled, and AIPerf 0.12.0.
Prompt identities and server token counts matched across all four configurations:
7,908–8,983 input tokens, mean 8,553.13. Every measured request returned HTTP 200
and exactly 256 tokens. Each profile has one measured trial.

| Configuration | Decode tok/s/user | Total output tok/s | Mean TTFT | Mean request |
|---|---:|---:|---:|---:|
| Previous native Inferact + DFlash2-12 | 22.19 | 13.30 | 7.53 s | 19.24 s |
| Native RadixArk target-only, FP4 prefill | 11.18 | 9.08 | 5.40 s | 28.20 s |
| **Native RadixArk + DFlash2-12, FP4 prefill** | **27.47** | **17.04** | **5.43 s** | **15.02 s** |
| vLLM 0.28, RadixArk target-only | 11.96 | 10.19 | 3.79 s | 25.11 s |

Decode throughput excludes time to first token; total output throughput includes
prefill and scheduling. The fast native profile improves decode by 23.8% over
the previous native control. Its request time is 40.2% below the recorded vLLM
target-only configuration, using speculative generation that vLLM does not use
in this comparison. vLLM still wins target-only decode and prompt processing.
No vLLM DFlash performance is claimed.

## What changed

The pinned RadixArk checkpoint stores more projections in reduced precision than
the previous Inferact checkpoint. Dynamic FP4 activation quantization and
FlashInfer CUTLASS GEMMs accelerate large NVFP4 MLP prefills; batches below 128
rows retain the native W4A16 path. The same option applies to the draft's NVFP4
projections, including context QKV. The selected prefill chunk size is 2,048.
DFlash2 uses width 12, correction-token drafting, and CUDA graphs for verification.

These changes are integrated into the normal runtime through
`--nvfp4-prefill-backend flashinfer` and the versioned recipe. The
[earlier six-request experiment](../qwen27_optimization/README.md) used 128 output
tokens and a wrapper; its timings are a separate workload.

## Quality and serving checks

| Profile | Regression + long-prompt + tools | GSM8K | HumanEval | AIME-25 | Long recall | Total |
|---|---:|---:|---:|---:|---:|---:|
| Previous native Inferact + DFlash2-12 | 23/26 | 20/20 | 18/20 | 4/5 | 3/3 | 68/74 |
| Fast native RadixArk + DFlash2-12 | 24/26 | 20/20 | 19/20 | 4/5 | 3/3 | 70/74 |
| vLLM RadixArk target-only | 24/26 | 20/20 | 18/20 | 3/5 | 3/3 | 68/74 |

The fast profile introduced no new failures against the previous native control;
it additionally passed JSON case 3 and HumanEval/127. It still failed arithmetic
case 5, JSON case 1, HumanEval/116, and AIME case 1. vLLM passed HumanEval/116 and
failed HumanEval/127, HumanEval/129, and AIME cases 1 and 4, in addition to the
same arithmetic and JSON cases. Budget exhaustion counts as failure. HumanEval
solutions were executed against the official assertions in a resource-limited,
network-isolated CPU container.

The three original long-recall inputs had a tokenizer-counting bug. Their vLLM
results were discarded and replaced by separately recorded, corrected 16,384-,
32,768-, and 64,512-token cases. Every other vLLM payload matched the final corpus.
The [corpus builder and dataset pins](../qwen27_optimization/README.md#expanded-verification-corpus)
reproduce the corrected inputs byte for byte.

The fast profile also passed all 29 operational checks, including streaming
output budgets, four queued clients, cancellation and recovery, sampled
fallback, exact long-prompt lengths, and overlength rejection. A separate final recipe launch check generated exactly
32 tokens after a 65,504-token prompt, reaching the 65,536-token limit. All three GPU
kernel tests passed, covering independently dequantized FP4 operands, fused
projection scales and bias, zero activations, small-row parity, and graph replay.

## Scope and reproduction

Use the [model instructions](../../docs/models/qwen3.8-27b.md) for acquisition and
serving. [Versioned evidence](results/GB10-QWEN38-27B-FAST-001.json) records exact
commands, immutable checkpoint revisions and file hashes, preparation validation,
source revision, per-case results, latency percentiles, and telemetry. Raw inputs,
responses, and logs remain in `results/qwen27-shipping-20260909/`.

Both engines use BF16 KV in this run. Their activation kernels, recurrent-state
implementations, framework versions, and allocated KV capacity differ. Native
uses BF16 KV and FP32 recurrent state; FP4 prefill changes activation numerics.
There is no claim of general quality parity or identical outputs.

Native keeps one active generation; multiple clients queue. Sampled requests
fall back to target-only generation, and the default model sampling temperature
remains 1.0. This is a single-client benchmark, not a new batched-throughput test.
The host had pre-existing swap and some swap-in activity, with zero measured
swap-out. GPU jobs ran sequentially; CPU activity was not isolated. The recipe
remains Experimental, with full Fast certification and endurance outstanding.
