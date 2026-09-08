# Qwen3.6 NVFP4: SparkLab versus vLLM on SPEED-Bench

Latest qualification: the cancellation-fixed SparkLab image averaged **112.08
decode tok/s** across two isolated restarts, versus **111.60** for vLLM on the
same 30 prompts. Mean request latency was **3.735 versus 3.664 seconds** (+2.0%),
and TTFT **1.436 versus 1.367 seconds** (+5.1%). All 120 measured requests passed
independent validation. The fixed image also passed 775 runtime/kernel tests,
14 grader tests, 28 serving checks, and a one-hour run with 1,411 successful
requests, no OOM events, and zero container swap.

Quality equivalence was **not** established: the fixed screen found capped
reasoning and JSON instruction regressions against native SparkLab. The faster
container was initially kept opt-in. The user subsequently selected it as the
default in recipe 0.6.0, retaining the preview label and quality limitations.
[Qualification and limitations](../benchmarks/qwen36_marlin/QUALIFICATION.md),
[versioned result](../benchmarks/gb10/results/GB10-QWEN36-MARLIN-001.json).

The sections below preserve the earlier optimization sequence and its historical
images and measurements. Their outstanding-validation statements describe those
stages; the qualification record above reports the later fixed image.

Initial comparison measured September 7, 2026, on one NVIDIA GB10. vLLM's GB10 MTP3 recipe
outperformed both the default SparkLab recipe and its documented optional MTP2
profile on the same 30 prompts.

| Profile | Decode tok/s | Mean TTFT | Mean request latency | Input tokens / TTFT |
|---|---:|---:|---:|---:|
| SparkLab default | 64.57 | 4.471 s | 8.420 s | 1,904 tok/s |
| SparkLab MTP2, Triton attention | 59.26 | 4.488 s | 8.801 s | 1,897 tok/s |
| vLLM 0.28.0 MTP3 | **112.28** | **1.444 s** | **3.729 s** | **5,895 tok/s** |

vLLM delivered **1.74x** the default SparkLab decode speed and **55.7% lower**
mean request latency. SparkLab MTP2 was **8.2% slower** at decoding than its
default profile. The earlier 80.55 tok/s MTP2 result on a 54-token prompt does
not carry over to this longer-context workload. The initial recipe comparison did not isolate the causes; the follow-up below
profiles the bottleneck and tests attention, residency and prefill tiling.

## Workload and method

- AIPerf 0.12.0, streaming `/v1/chat/completions`, concurrency one.
- The same pinned NVIDIA NVFP4 checkpoint, revision
  `491c2f1ea524c639598bf8fa787a93fed5a6fbce`. All three original safetensors
  shards passed SHA-256 verification. SparkLab used recipe 0.5.0's FTW artifact.
- `nvidia/SPEED-Bench`, `throughput_8k`: 30 measured prompts, ten from each
  entropy tier, plus three distinct warmup prompts. Selection seed: `20260907`.
  Gated `cais/hle` source rows were excluded because they were inaccessible.
  Mixed-entropy prompts therefore come from the needle-in-a-haystack category.
- Original prompt text was retained. With Qwen's chat template, inputs ranged
  from 7,866 to 8,941 tokens, averaging 8,511.13 tokens.
- Explicit `temperature=0`, `ignore_eos=true`, `max_completion_tokens=256`;
  thinking enabled by the checkpoint's default template. All 90 measured
  requests succeeded and reported exactly 256 completion tokens.
- Separate server processes, run sequentially: SparkLab default, SparkLab MTP2,
  then vLLM MTP3. Downloads and source-file verification finished before the
  corresponding measured runs. Initialization and warmup were excluded.

Decode is the arithmetic mean of AIPerf's `output_token_throughput_per_user`
and includes reasoning tokens. TTFT and request latency are measured by the
client. The last table column divides total input tokens by summed TTFT; it
includes API, scheduling, template and first-token overhead and is **not an
isolated prefill-kernel measurement**.

## Configurations

SparkLab ran source version 0.1.2 at commit
`8ee23cf7cd158118104999fa1a01fb8e5cfa631b`, with PyTorch 2.11.0+cu130,
Triton 3.6.0 and FlashInfer 0.6.15.post1. Both profiles used RAM expert backing,
cache capacity for all routed experts, Triton NVFP4, a 32,832-token KV pool
and prefill D2D hits. This RAM configuration still streams expert layers during
prefill; it does not enable the immutable fully-resident cache path.
The default selected FlashInfer attention and CUDA graphs; MTP2 used Triton
attention and eager verification as documented. The installed kernel-cache
wheel was older, so `SPARKLAB_DISABLE_KERNEL_CACHE=1` selected current-source
JIT kernels. No version-check bypass was used.

vLLM used the official `vllm/vllm-openai:v0.28.0` image, digest
`sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`,
with PyTorch 2.13.0+cu130, Triton 3.7.1 and FlashInfer 0.6.16.post3.
Its settings follow the [upstream NVFP4 GB10 recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B):
FlashInfer attention, Marlin MoE, FP8 KV, 0.5 GPU memory utilization, 262,144
maximum context, eight maximum sequences, an 8,192-token prefill batch budget,
chunked prefill, asynchronous scheduling, prefix caching, fastsafetensors,
and three MTP drafts with Triton draft MoE. The vision encoder remained enabled.
The model mount, loopback API address, port and served alias were local choices.

The host driver was 580.126.09, with CUDA 13.0. This is a comparison of the
serving recipes; KV precision, capacity, attention, scheduling and speculation
are intentionally different.

## Entropy groups and speculation

| Entropy tier | SparkLab default decode | SparkLab MTP2 decode | vLLM MTP3 decode |
|---|---:|---:|---:|
| Low | 64.54 tok/s | 59.78 tok/s | 113.40 tok/s |
| Mixed | 64.58 tok/s | 60.68 tok/s | 118.26 tok/s |
| High | 64.60 tok/s | 57.32 tok/s | 105.18 tok/s |

SparkLab accepted 4,284 of 5,302 draft tokens (80.8%) and produced 2.26 output
tokens per target forward, with zero rejection replay. vLLM's measured counter
deltas show 5,233 accepted draft tokens out of 7,356 (71.1%), across 2,452 draft
steps. vLLM reported **zero prefix-cache hit tokens** during measurement.

## Limits and memory observations

This is one measured pass over an accessible 30-prompt subset, not the complete
SPEED-Bench suite or a quality/endurance certification. The upstream guide
reports 97.7 decode tok/s, 6,265 prefill tok/s and 3.95 s latency, but does not
publish its exact prompt selection or full client command. Our workload,
software and host driver do not establish an exact reproduction of that result.

Neither SparkLab measured window had host swap I/O. vLLM initialization grew
host swap usage to roughly 2.2 GiB. Its measured window recorded 10,986 swap-in
pages (42.9 MiB), zero swap-out and at least 66.3 GiB available memory. These
are host-wide counters sampled approximately once per second; they do not
attribute activity to a specific process. The result is not a zero-swap vLLM
certification.

## Evidence and reproduction

- [Versioned result](../benchmarks/gb10/results/GB10-QWENSPEED-001.json)
- [Raw artifact directory](../results/qwen36-speedbench-20260907/): fixed inputs,
  question IDs, dataset revision, raw request/response exports, per-request
  metrics, telemetry, server logs, package versions, checkpoint hashes,
  commands and a SHA-256 manifest. This local directory is git-ignored.
- `*-server-command.json` records each server command; `*-command.json`
  records each AIPerf command. `run_profile.py`, `launch_vllm.py`,
  `prepare_subset.py`, `upstream-prepare.py` and `summarize.py` preserve the
  execution and analysis steps. Commands retain this host's original paths.

The AIPerf client environment was created separately with
`uv venv /tmp/qwen36-bench-venv` and
`uv pip install --python /tmp/qwen36-bench-venv/bin/python aiperf==0.12.0 datasets`.
Use the saved inputs for a repeat comparison; do not resample from a moving
dataset revision. Both benchmark servers were stopped after collection.


## Follow-up: why the gap exists

The same GB10 runs vLLM faster, so the measured gap is in the software serving
paths. At this stage parity had not been demonstrated. Three findings narrowed the work:

1. **Expert prefill dominates.** A CUDA profile of one 8,192-token target prefill
   chunk records 2.857 s across 80 native NVFP4 expert GEMM launches (gate/up and
   down for 40 layers). Full attention accounts for about 73 ms. HtoD transfers
   account for another 539 ms; transfers can overlap compute, so these times
   cannot simply be added as a latency prediction. The native kernel dequantizes
   NVFP4 inside a BF16 dot-product loop, with tiles originally tuned for MiniMax-M2.
2. **MTP verification is expensive.** Qwen3.5/3.6 MTP explicitly disables ordinary
   CUDA graphs and uses eager verification. The original optional recipe also
   uses Triton attention. Switching only attention to FlashInfer improved decode
   on the same six prompts from 58.80 to 71.44 tok/s. Our 80.8% draft acceptance
   in the original 30-prompt run is higher than vLLM's 71.1%, so acceptance alone
   does not explain vLLM's lead. vLLM uses a different three-draft serving path,
   Marlin experts and FP8 KV; we have not isolated each remaining contribution.
3. **Cache capacity is not full preload.** `--moe-cache-rate 1.0` reserves enough
   slots but does not set `OffloadMoeCache.fully_resident`. The RAM-backed recipe
   still streams layers for prefill. The existing `--moe-storage disk
   --moe-preload-all` path reads all experts once at startup, then serves from
   immutable GPU slots. It also disables prefill overlap and avoids retaining a
   second complete RAM bank. This removes recurring expert staging from that path.

The six-prompt controls use the first six fixed inputs of the original set,
two per entropy tier, with three separate warmups. Baseline and vLLM rows below
are those same six requests extracted from the original 30-request runs.

| Six-prompt profile | Decode tok/s | Mean TTFT | Mean request latency |
|---|---:|---:|---:|
| SparkLab default | 64.80 | 4.430 s | 8.365 s |
| SparkLab MTP2, Triton attention | 58.80 | 4.413 s | 8.769 s |
| MTP2, FlashInfer attention | 71.44 | 4.286 s | 7.908 s |
| MTP2, FlashInfer, full preload | 73.34 | 3.981 s | 7.504 s |
| vLLM MTP3 | 107.43 | 1.535 s | 3.927 s |

An attempted FlashInfer expert-backend switch is excluded: FTW's fast loader
returns the checkpoint's stored bank layout directly, so passing
`--nvfp4-backend flashinfer` with this native NVFP4 FTW artifact did not switch
its experts to b12x. A real comparison needs a repacked artifact or explicit
repacking at load; the CLI flag alone is insufficient.

### Experimental prefill tiling

A synthetic Qwen-shaped MoE microbenchmark used M=8192, H=2048, I=512,
E=256 and top-k=8, with BF16 activations and native NVFP4 banks. Changing
`(BLOCK_M, BLOCK_N, BLOCK_KB, warps, stages)` from `(32,64,32,8,4)` to
`(128,128,32,4,2)` reduced complete MoE time from 72.40 to 45.22 ms.
`GROUP_SIZE_M=8` was unchanged. Outputs were bitwise equal on this fixture;
this synthetic timing is not a model throughput claim. One larger shared-memory
configuration failed to launch and is recorded in the sweep output.

The follow-up server applies this tile only to M>=4096, keeps the original
small-M tiles, and combines FlashInfer attention, MTP2 and full preload. During
its first warmup it compares both kernel configurations on all 40 actual model
layers of an 8,192-token chunk; every output tensor matched bitwise. Those
extra correctness passes and compilation are excluded from measurement.
The experimental wrapper is `tuned_server.py` in the follow-up raw artifacts;
production kernels and the certified recipe are unchanged.

### Full 30-prompt follow-up result

| Profile | Decode tok/s | Mean TTFT | Mean request latency |
|---|---:|---:|---:|
| SparkLab default (original run) | 64.57 | 4.471 s | 8.420 s |
| SparkLab MTP2 + FI + preload + tuned tile | 74.17 | 3.019 s | 6.480 s |
| vLLM MTP3 (original run) | 112.28 | 1.444 s | 3.729 s |

The experimental profile improved decode by
14.9% and reduced TTFT by
32.5% versus the default recipe.
Mean request latency fell by 23.0%.
vLLM still delivered 1.51x decode
throughput and 42.5% lower mean request latency. This establishes a useful improvement, not parity.

All 30 requests returned HTTP 200 with exactly 256 completion tokens. Prompt
IDs, messages, sampling parameters and server-reported input counts match the
original comparison. Measured host swap I/O was
13 pages in / 0 pages out,
with at least 88.2 GiB available. vLLM's original
swap caveat still applies. This is a serving-speed experiment, not a replacement
for quality, 32K recall or endurance certification.

The strongest remaining candidates are a real Marlin/b12x expert comparison
with compatible repacked banks, CUDA-graph capture for Qwen3.5/3.6 multi-token
verification, and long-context attention/KV tuning. Their individual benefit is
unmeasured; matching vLLM cannot yet be promised.

- [Follow-up result](../benchmarks/gb10/results/GB10-QWENSPEED-002.json)
- [Follow-up raw artifacts](../results/qwen36-tuning-20260907/): exact inputs,
  commands, wrapper, profiler trace, kernel sweeps, telemetry and AIPerf exports.

## Integrating vLLM 0.28 Marlin into SparkLab

Measured September 8 UTC / September 7 Toronto, 2026. This experiment uses
SparkLab's model and scheduler with vLLM's compiled Marlin kernels and weight
transforms. It does not launch a vLLM server for the SparkLab measurements.

### Implementation

- Resolve both the legacy `fused_moe.fused_marlin_moe` import and vLLM 0.28's
  `fused_moe.experts.marlin_moe`. Preserve the donor's activation argument type
  and account for the removed `gating_output` parameter.
- Store vLLM 0.28 global scales as float32, as its compiled MoE operator requires.
  The legacy adapter keeps its BF16 global-scale contract.
- Pass immutable, per-layer views to Marlin for prefill and decode. Its sorting
  and GEMM geometry see 256 experts rather than all 10,240 cache slots. Weights
  are sliced as views, and the global scales use the same layer offset.
- Allow the full model cache under explicit full preload, while keeping the
  existing dynamic-cache slot cap. Automatic sizing also recognizes this policy.
- Reject explicit NVFP4 backend flags that disagree with an FTW artifact's
  stored layout. Log the actual expert bank layout at startup.

The original checkpoint was converted into a separate `nvfp4_marlin` FTW
artifact; the certified native artifact and recipe were not modified.
[Build and serving instructions](../benchmarks/qwen36_marlin/README.md) include
a pinned Dockerfile. The image built successfully and its CLI smoke check passed.

### Runtime and verification

The donor container uses the same official vLLM 0.28 image digest as the earlier
vLLM benchmark. It runs Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1,
FlashInfer 0.6.16.post3, Transformers 5.15.1 and TVM FFI 0.1.11. SparkLab's C++
extensions were built against that runtime. The normal PyTorch 2.11 environment
is unchanged; its sglang-kernel wheel is not installed in this container, so
those optional operations use SparkLab's existing fallback kernels. This is
an experimental donor environment outside the package's normal dependency pins.

Verification includes:

- **831 dense and speculative tensors are byte-identical** between the original
  FTW and the converted artifact, checked by shape, dtype and SHA-256 of each
  tensor's stored bytes.
- All 40 routed layers were checked during the first warmup on their first 16
  actual activation rows against the original native FTW experts. Maximum
  relative L2 error was **0.011347**; minimum cosine similarity was **0.999967**.
  The acceptance thresholds were relative L2 < 0.03 and cosine > 0.9995.
  These outputs are not bitwise identical; this check does not establish broad
  model-quality equivalence.
- GPU tests cover repacking, prefill, decode, overlap and full-resident layer
  routing with both logical and physical slot IDs. One-row and three-row
  Marlin calls also pass CUDA-graph capture/replay with changed input values.
  This validates the expert operation's capture support; **Qwen's complete MTP
  verification still runs eagerly** in this implementation.
- Arithmetic, exact-code retrieval and JSON smoke checks passed 3/3 with
  thinking disabled. These are smoke checks, not a reasoning benchmark.
- The selected regression suite passed 79 checks. One initial failure came
  from the previously documented stale host kernel-cache wheel; rerunning it
  with current-source JIT enabled passed. The donor-focused suite passed 22
  checks, and the additional graph checks passed all three parametrizations.

The first measured profile uses MTP2. A separate native-Triton MTP2 control
uses the same container, FlashInfer attention, full preload, KV capacity,
warmups and 30 fixed prompts. This distinguishes the Marlin change from the
runtime and residency changes relative to the original certified recipe.
The MTP3 screening run used the same first six prompts as the earlier screening
runs: it reached 74.97 tok/s versus 62.50 tok/s for MTP2 on those six inputs,
so MTP3 was selected for another complete 30-prompt run.

The container lost GPU access between the completed native-control run and
final MTP3 startup. The host GPU remained available. Restarting only the test
container restored access; GPU tests passed again before the final run. No
measured request window overlaps the failed startup or restart.

### Full Marlin comparison

| Profile | Decode tok/s | Mean TTFT | Mean request latency |
|---|---:|---:|---:|
| Original certified recipe (earlier runtime) | 64.57 | 4.471 s | 8.420 s |
| Native Triton MTP2 control (donor runtime) | 69.38 | 3.404 s | 7.087 s |
| Marlin MTP2 | 66.48 | 1.457 s | 5.307 s |
| Marlin MTP3 | 75.15 | 1.394 s | 4.809 s |
| vLLM MTP3 reference | 112.28 | 1.444 s | 3.729 s |

Within the same runtime, Marlin MTP2 reduced TTFT by
57.2% and request latency by
25.1%; decode changed by
-4.2%.
Marlin MTP3 delivered 13.0% higher
decode throughput than Marlin MTP2. Its measured TTFT is close to vLLM's,
but vLLM still delivers 1.49x decode
throughput. These results support reusing Marlin for prefill; they do not establish
complete serving parity. All 90 new measured requests returned HTTP 200, with
identical prompt counts and exactly 256 completion tokens each.

| Measured profile | Host swap-in pages | Host swap-out pages | Minimum available RAM |
|---|---:|---:|---:|
| native-mtp2-control | 2 | 0 | 88.0 GiB |
| marlin-mtp2 | 5 | 0 | 87.4 GiB |
| marlin-mtp3 | 0 | 0 | 86.4 GiB |
| vllm-mtp3-reference | 10986 | 0 | 66.3 GiB |

The experimental artifact is retained at
`~/.sparklab/models/qwen3.6-35b-a3b/prepared/experimental-marlin-vllm028`.
The benchmark server and test container were stopped after collection.

- [Versioned Marlin result](../benchmarks/gb10/results/GB10-QWENSPEED-003.json)
- [Raw artifacts](../results/qwen36-marlin-20260908/), including the implementation
  patch, numerical checks, commands, server logs and raw AIPerf responses.


## Decode pipeline follow-up (2026-09-08)

The next experiment keeps the Marlin artifact, fixed prompts, MTP3, full
residency, and AIPerf settings above. It changes execution in SparkLab:

- After a partial rejection, rebuild draft KV from only the accepted target
  features and the correction token, then immediately produce the next draft.
- Capture the four-row target verification with FlashInfer's causal prefill
  graph wrapper. Its fixed query dimension remains separate from request batch
  size; KV indices and lengths are replanned before replay.
- Avoid the fallback causal convolution's device-to-host length read for a
  single request: its length is already the input tensor's second dimension.
- Save convolution history alone for prefix commits; transactional GDN
  verification does not overwrite live recurrent state.
- Capture small MTP draft steps with independent FlashInfer planning buffers.
  Copy proposed tokens out of reusable graph output storage and destroy draft
  graphs before cache rebuilds and shutdown.

Completed intermediate measurements:

| Profile | Decode tok/s | Mean TTFT | Mean request latency |
|---|---:|---:|---:|
| Previous Marlin MTP3 | 75.15 | 1.394 s | 4.809 s |
| Continuous rejection recovery | 76.01 | 1.460 s | 4.838 s |
| Recovery + target graph + convolution sync fix | 105.12 | 1.386 s | 3.826 s |
| Above + small verification snapshot | 105.22 | 1.385 s | 3.820 s |
| Previous vLLM MTP3 reference | 112.28 | 1.444 s | 3.729 s |

Recovery reduced target forwards from 3,017 to 2,512 for the 7,680 measured
output tokens. Its acceptance was 5,168 / 7,380 (70.0%). The improvement in
forward count alone did not translate into a large throughput gain; graph
capture and eliminating per-layer host synchronization supplied the substantial
improvement. The snapshot optimization had little measured effect.

GPU tests compare recovered draft tokens and KV with clean accepted-prefix
execution, including graph replay. The FlashInfer verification test changes
prefix length and uses a non-identity page table. Full-model warmup validation
compares graph/eager target logits, hidden features, convolution state and
verification state on 12 blocks; maximum logit difference was zero. Draft
validation compares tokens, feedback and KV on three examples of each width
1, 2, 3 and 4; tokens and KV matched exactly and maximum feedback difference
was zero. Validation runs during warmup, before measured requests.

Working raw artifacts are in `/tmp/qwen36-decode`. Two failed warmup attempts
are retained separately: one validation checker initially compared float32
and BF16 logits without converting their dtypes, and one client started before
the replacement backend was ready. Neither is included in measured results.

### Remaining gap after draft graphs and scheduling changes

| Profile | Decode tok/s | Mean TTFT | Mean request latency |
|---|---:|---:|---:|
| Target + draft graphs, MTP3 | 106.51 | 1.380 s | 3.789 s |
| Above + donor dense kernels | 107.32 | 1.646 s | 4.039 s |
| Above + overlap and explicit Torch router | 107.55 | 1.403 s | 3.789 s |
| Same execution settings, MTP4 | 108.51 | 1.415 s | 3.799 s |
| Fresh vLLM MTP3 reference | 110.98 | 1.357 s | 3.668 s |

At this stage, the best completed SparkLab run was 2.22% below the refreshed vLLM decode
measurement (3.35% below the earlier 112.28 tok/s reference). MTP4 improves
decode slightly but has worse total latency than the MTP3 overlap run.
These single passes do not establish statistical parity.

Importing vLLM's dense helpers also made an optional Triton router discoverable.
Explicitly selecting the original Torch router removed the observed prefill
regression. That intermediate run used the Torch override; the final environment instead explicitly selects the tested local fused router. Eight exact output
budgets and two queued EOS requests passed with speculative scheduling overlap.
The MTP4 warmup validator checked target graph/eager execution and draft widths
1 through 5. Three arithmetic, retrieval, and JSON smoke cases passed for the
draft-graph and donor-dense configurations.

A six-round GPU profile of the draft-graph configuration measured about
28.79 ms of GPU kernels per target round. Marlin expert kernels used about
8.02 ms, FP8 rowwise projections 4.41 ms, and all full-attention calls 1.11 ms.
The draft expert microbenchmark showed essentially equal native and donor
performance, so those kernels were retained. The subsequent attention experiment
used FP8 KV and FlashInfer XQA with 16-token physical pages. Both engines retain
FP32 recurrent SSM state; the actual vLLM metrics confirm this precision.
FP8/XQA was experimental and is not part of the results above; its runtime changes were later removed.

### Attention follow-up

| Profile | Decode tok/s | Mean TTFT | Mean request latency |
|---|---:|---:|---:|
| XQA + FP8 KV, MTP4, page size 16 | 108.19 | 1.453 s | 3.839 s |
| XQA + BF16 KV, MTP4, page size 16 | 106.13 | 1.418 s | 3.846 s |

Neither attention change improved the best completed configuration. This rejects
the simple explanation that matching vLLM's KV precision and decode kernel would
by itself eliminate the residual gap. Both attention experiments were removed
from the final runtime; their source is retained in the raw archive under
`attention-experiment-source/`. The experimental FP8 storage used unit K/V scales,
changed numerics, and halved paged KV allocation from 0.69 to 0.34 GiB at this
capacity. Neither experiment changed the FP32 recurrent SSM state.

Nine GPU attention cases compare graph replay with eager XQA and independent
FA2 causal attention, using BF16/FP8 KV, one/four/five queries, shuffled physical
pages and changing prefix lengths. A scheduler-order regression test prepares
metadata before input tensors exist. Cache budget/rebuild tests check storage
precision and byte accounting. Together these checks passed 26 tests. The first
FP8 warmup failed because the new metadata path read an input tensor before the
scheduler attached it; the corrected rerun is the measured profile. Its failed
warmup and server log are retained separately.

The BF16 XQA full-model warmup passed 12 target graph/eager checks and draft
checks across widths 1 through 5. Both attention variants passed the three
arithmetic, retrieval and JSON smoke cases. All ten complete profiles in this
follow-up contain 30 HTTP-200 responses, identical messages and prompt counts,
and exactly 256 output tokens per request; raw validation is retained.

A paired prompt bootstrap (10,000 resamples, seed 41) for MTP4 versus refreshed
vLLM gives a mean decode difference of -2.47 tok/s and a 95% percentile interval
of [-5.81, 1.19] tok/s. This describes prompt-sampling uncertainty in these single
passes. It does not measure run-to-run variation or establish equivalence.

- [Versioned decode result](../benchmarks/gb10/results/GB10-QWENSPEED-004.json)
- [Raw decode artifacts](../results/qwen36-decode-20260908/), including per-request
  responses, profiler traces, validation, telemetry and the source patch.

The focused regression run at this stage passed **469 tests** across the changed model,
attention, donor kernels, cache budgeting/rebuild, and scheduler paths.

## Residual decode costs: native draft kernels and deferred state commits

The profile of the MTP4 configuration measured about 32.4 ms of GPU kernels
per target round. Marlin experts used 9.27 ms, FP8 projections 6.37 ms, vocabulary
projections 6.20 ms, and the recurrent-state commit copy about 0.52 ms.

The shared-expert Marlin microbenchmark was slower than the existing native
kernels and was not adopted. For the BF16 draft attention projection, a native
skinny kernel reduced the one-row microbenchmark from 0.203 to 0.148 ms. An
opt-in shape selection also covers the two 2048-by-4096 draft projections at
one or two rows. All weights and activation precision remain unchanged.

Deferred commits let the next verification read the previously accepted state
straight from the intermediate-state buffer. Each GDN CTA loads its disjoint
value tile before writing new intermediates, so reuse does not need a cross-CTA
barrier. The pool materializes canonical state before snapshots, other request
owners, non-verification forwards, clearing, freeing, or rebuilding slots. The
option preserves FP32 recurrent storage and arithmetic.

| Profile | Decode tok/s | Mean TTFT | Mean request latency |
|---|---:|---:|---:|
| Draft skinny kernels, MTP4 | 108.54 | 1.418 s | 3.791 s |
| Above + deferred recurrent-state commits | 108.82 | 1.408 s | 3.778 s |

The skinny change alone was effectively neutral. Deferred commits reduced the
cost per target round, but accepted draft counts also changed: target forwards
were 2,225 and 2,251 respectively for the 7,680 measured outputs. These results
still do not establish parity with vLLM.

The deferred-state GPU test compares outputs and all intermediate states exactly
against ordinary materialized commits across consecutive CUDA graph replays,
accepted lengths 1 through 5, request-owner changes, and a snapshot copy. CPU
lifetime tests cover clear, reset, view, free and rebuild consumers. The targeted
run passed 38 tests. The full-model warmup passed graph/eager checks; arithmetic,
retrieval, JSON, eight exact output budgets and two queued EOS requests passed.

A router microbenchmark found the local fused renormalized top-k kernel faster
than the Torch sequence: 2.03 versus 13.15 microseconds at five rows, and 43.81
versus 254.66 microseconds at 8,192 rows. vLLM's fused router measured 3.55 and
32.42 microseconds respectively. The opt-in local path accepts power-of-two
routing too; the existing explicit Torch override retains priority. Router,
draft skinny, deferred-state, engine and scheduler tests passed 74 checks before
the combined benchmark. Boundary ties may choose different equally scored
experts across router implementations; exact end-to-end output equivalence is
not claimed.


## Final image and repeated comparison

The final profile combines resident Marlin experts, target and draft CUDA graphs,
rejection recovery, four-token speculation, donor dense kernels, scheduling overlap,
native draft skinny kernels, deferred recurrent commits, and the local fused router.
It retains BF16 KV and FP32 recurrent state. The certified recipe is unchanged.

| Profile | Decode tok/s | Mean TTFT | Mean request latency |
|---|---:|---:|---:|
| Combined development source, fused router | 110.79 | 1.397 s | 3.713 s |
| Rebuilt image, first run | 111.35 | 1.466 s | 3.781 s |
| Fresh vLLM repeat | 112.42 | 1.374 s | 3.655 s |
| Rebuilt image with OMP/MKL threads limited to one | 110.27 | 1.398 s | 3.733 s |
| Rebuilt image, steady-state restart | 113.14 | 1.405 s | 3.684 s |

The first image run started with cold compilation caches; the restart reused
those caches. Both runs had three distinct warmup requests before measurement.
The final repeat's decode was 0.63% faster than the latest vLLM run, while TTFT
was 2.30% higher and request latency was 0.80% higher. The two unmodified final
image runs span 111.35–113.14 tok/s; this supports practical parity on this
workload without establishing statistical equivalence or a consistent win.
Thread limits did not improve the overall result and are not selected.

The separate MoE output-reduction microbenchmark saved only about 0.0006 ms at
five rows and 0.0074 ms at 8,192 rows; that donor substitution was not adopted.
These negative results and every intermediate full run remain in the archive.

The final image is
`sha256:06ac519342c002ef1e2649b973f49b1705e74117f3da5f2fb901f9d8f3ad0567`.
Its server used the normal entrypoint without a source bind mount or validation
monkeypatch. All 21 changed/new runtime Python files were hash-checked against
the image. Exact launch, restart, mounts and environment records are retained.

The cleaned final regression suite passed **483 tests**; an additional targeted
check in the normal host runtime passed **48 tests**. Full-model graph/eager
checks passed before the combined development measurement. The rebuilt image
passed arithmetic, retrieval and exact-JSON smoke checks, eight exact output
budgets, and two queued EOS requests. Independent validation checked all 510
measured requests across 17 follow-up profiles: identical messages and prompt
counts, HTTP 200, and exactly 256 completion tokens per request.

This remains an experimental, fully resident GB10 serving profile using the
pinned vLLM donor runtime. Kernel numeric checks and smoke tests do not certify
model quality, multimodal behavior, long contexts, concurrency, or endurance.
Equal-score routing ties may select different experts; identical generated text
across engines is not claimed. Our fixed accessible subset also differs from
the upstream guide's unspecified exact prompt selection.

- [Final versioned result](../benchmarks/gb10/results/GB10-QWENSPEED-005.json)
- [Reproducible build and serve recipe](../benchmarks/qwen36_marlin/README.md)
- [Raw artifacts](../results/qwen36-decode-20260908/): commands, responses,
  telemetry, tests, source verification, microbenchmarks and SHA-256 manifest.
  `final-implementation.patch` records the final working-tree implementation;
  `implementation.patch` is preserved as the historical source for result 004.
