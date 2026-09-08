# Qwen3.6 GB10 container qualification

This record separates serving reliability, speed, and model-quality results for
SparkLab's Marlin/MTP4 container. After reviewing these results, the user selected
the fast profile as the default in recipe 0.6.0. It retains a preview label and is
not certified as quality-equivalent. The versioned result records the qualification
decision before this later default promotion; its measurements are unchanged.

## Frozen quality comparison

All three engines used the same NVIDIA checkpoint, messages, greedy decoding,
thinking settings, and output budgets. The corpus contains 20 core cases, two
derived tool-result round trips, and the first five AIME-25 problems. Core thinking
cases allow 8,192 output tokens; AIME allows 16,384. Truncation is a failure, even
if an intermediate reasoning step contains the expected answer.

| Profile | Core cases | Tool round trips | Capped AIME sample | AIME truncated |
| --- | ---: | ---: | ---: | ---: |
| Native SparkLab, PyTorch 2.11, target only | 18/20 | 2/2 | 1/5 | 4/5 |
| SparkLab Marlin/MTP4, PyTorch 2.13 | 17/20 | 2/2 | 0/5 | 5/5 |
| vLLM 0.28 | 17/20 | 2/2 | 1/5 | 4/5 |

All profiles passed the four bounded Python-function tasks, four core reasoning
tasks, and two typed tool calls. They shared an arithmetic remainder error and a
JSON sorting response that omitted the requested object wrapper. The optimized
container and vLLM additionally returned `is_even` instead of the requested `even`
key. Native SparkLab and vLLM completed the first AIME problem correctly; the
optimized profile reached its output cap. This small sample does not establish
statistical non-inferiority or representative AIME accuracy.

Ten-case controls disabled MTP, restored native dense projections, or changed
routing. None removed the JSON-key difference. A further control used the original
FTW, native Triton experts and projections, and target-only decoding inside the
PyTorch 2.13 image; it also returned `is_even`. Thus that difference can occur
without the Marlin conversion or speculative decoding. These controls do not
isolate the cause of the AIME divergence.

## Cancellation repair

Validation reproduced a server crash after stream cancellation. An in-flight
verification block could advance `cached_len` beyond the host's client-visible
`input_ids`. Cache insertion then used a shorter key than cleanup's ownership
accounting, leaking KV pages and risking donation of mismatched recurrent state.
Cleanup now marks that state as overadvanced and clamps the cacheable prefix to
the visible tokens, while retaining the full allocation length for page release.

All eight new cancellation regression cases failed before the repair and passed
afterward. The expanded runtime/kernel suite passed 775 tests with one skip;
14 separate grader tests passed. The rebuilt image passed context recall at
exactly 1,024, 8,192, 16,384, and 32,768 input tokens, nine exact output budgets,
four queued clients, three stream cancellations with recovery, an early
disconnect, sampled fallback, and overlength rejection with recovery.

## One-hour reliability result

The isolated release container completed **3,600.10 seconds** of serving with
**1,411/1,411** repeated streaming requests passing. All 28 operational checks passed,
all health samples belonged to one healthy server instance, container swap stayed
at **zero bytes**, and scoped OOM and OOM-kill counters stayed unchanged at zero.
The server reported peak CUDA reserved memory of **24.66 GiB**; available host
memory never fell below **89.02 GiB**. These are different accounting measures.
The loop mixes unique short requests and 1K/8K recall prompts with 256 forced
output tokens. Context and cancellation checks run before that loop, within the
same hour. This is a reliability workload, not a throughput or quality score.

## Final isolated speed comparison

After endurance, each engine ran twice from a fresh server restart, sequentially
on the same GB10. Each pass used three distinct warmups and the same 30 fixed
SPEED-Bench prompts (mean 8,511 input tokens), exactly 256 output tokens, greedy
decoding, and concurrency one. Restarts reuse local compilation caches. All 120
measured requests returned HTTP 200 with identical messages and prompt counts;
every completion contained exactly 256 tokens. Telemetry brackets each measured
window and the four windows do not overlap.

| Run | Decode tok/s | Mean TTFT | Mean request latency |
| --- | ---: | ---: | ---: |
| release-1 | 111.93 | 1.460 s | 3.766 s |
| vllm-final-1 | 110.64 | 1.365 s | 3.683 s |
| release-2 | 112.24 | 1.412 s | 3.704 s |
| vllm-final-2 | 112.56 | 1.369 s | 3.644 s |

Across the two equal-sized repeats, SparkLab averaged **112.08 decode tok/s**
versus **111.60** for vLLM (+0.43%). Mean TTFT was **1.436 versus 1.367 seconds**
(+5.07%), and mean request latency was **3.735 versus 3.664 seconds** (+1.95%).
This supports comparable decode performance on this workload; vLLM retains the
lower TTFT and request latency. Two repeats do not establish statistical
equivalence or a consistent win. No swap-out pages were observed in any measured
window; host swap counters include unrelated processes.

[Versioned result](../gb10/results/GB10-QWEN36-MARLIN-001.json) records all four
runs, image/configuration verification, quality comparisons, and reliability
checks. The earlier speed image lacked the cancellation repair and remains a
separate historical result.

## Configuration and evidence

The container targets one 128 GiB GB10, text inputs, one actively decoded request,
BF16 KV, FP32 recurrent state, and 32,832 total sequence tokens. Greedy decoding
uses four speculative tokens; sampled requests use target-only decoding. The
launcher imposes equal 96 GiB memory and memory-plus-swap limits to prevent
container swap. Other clients queue; this is not concurrent decode or multimodal
qualification.

The earlier uncapped shared-load run swapped approximately 2.6 GB of CPU pages
when another model server started. That attempt is retained as failed evidence
for the no-swap requirement, not counted as an endurance pass.

The release image is
`sha256:a563337b36f29e11803737bca3c8833d6ed8d66a2efd244348cc9dc1bd53ba5e`.
All 411 recorded runtime source/configuration files matched the workspace.
The image runs packaged source with no source-tree mount. Its pinned base,
conversion command, environment flags, and launcher are in [README.md](README.md).

Raw evidence is retained in `results/qwen36-final-validation-20260908/`, including
failed attempts, request payloads and responses, server logs, telemetry, image
verification, test logs, and both quality comparisons. The frozen corpus SHA-256
is `989733d71627b09134c9e7c81c37262df9b5e4422f12350d6983d7723d89ff47`.
