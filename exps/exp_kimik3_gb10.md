# Frontier Models on One GB10: DSV4 Flash to GLM-5.2 to Kimi K3

Last updated: 2026-08-28

## Summary

This document defines one ordered FreeToken experiment on an NVIDIA GB10 system
with 128 GB of coherent unified memory. The work deliberately starts with the
smallest model whose behavior is already known, then increases model and storage
pressure only after the shared runtime has passed correctness and performance
gates:

1. reproduce `deepseek-ai/DeepSeek-V4-Flash-0731` on GB10;
2. run `nvidia/GLM-5.2-NVFP4` using the validated disk/cache stack;
3. attempt the full `nvidia/Kimi-K3-NVFP4` checkpoint at its pinned revision.

DSV4 Flash is the control experiment. It verifies SM121 kernels, FTW conversion,
explicit NVMe storage, unified-memory accounting, cache behavior, and benchmark
methodology against the existing RTX 3090 results in [exp_dsv4.md](exp_dsv4.md). GLM-5.2 is the
scale-up experiment: FreeToken already implements its model, DSA attention, and
NVFP4 expert path, but its approximately 380 GiB routed pool stresses GB10 memory
and storage far more heavily. Kimi K3 is attempted only after those shared systems
have been validated.

The final intended result remains a capacity-efficiency and systems result, not a
claim of performance parity with hosted Kimi K3:

> Starting from a reproduced DSV4 Flash control, demonstrate that the same bounded
> GB10 runtime can scale to GLM-5.2 and ultimately generate reference-validated
> output from a 2.8T-parameter, approximately 1.5 TB Kimi K3 checkpoint.

The following target table is the original pre-measurement hypothesis. Values in
the current execution record are measured only where their evidence is explicitly
identified; kernel and layer-probe timings are not full-model throughput:

| Program stage | Initial target | Stretch target |
|---|---:|---:|
| DSV4 Flash reproduction | 2-4 tok/s | 4-6 tok/s |
| GLM-5.2 official NVFP4 | 0.45-0.75 tok/s | 1.0 tok/s |
| Kimi K3 NVIDIA NVFP4 | 0.2-0.5 tok/s | 1.0 tok/s |
| Kimi K3 mixed cold tier | 0.5-1.0 tok/s | 2.0 tok/s |

Official-weight and modified low-bit results must always be reported separately.

## Current execution record: NVIDIA Kimi K3 NVFP4

This section is the authoritative record for the active full-checkpoint run. The
older Moonshot MXFP4 design material below remains useful background, but it does
not describe the artifact currently being tested.

### Artifact and inventory

| Field | Value |
|---|---|
| Repository | `nvidia/Kimi-K3-NVFP4` |
| Immutable revision | `f8c5234a0a880bcc6cbf779a315e7ee2f405b812` |
| Hugging Face files | 116 total, including 96 safetensors shards |
| Indexed tensors | 992,508 |
| Indexed logical tensor bytes | 1,609,777,087,808 |
| Validated safetensors physical bytes | 1,609,918,090,976 |
| Initial remote byte estimate | 1,610,038,482,254 |
| Whole repository bytes | 1,610,039,528,867 |
| ModelOpt producer | 0.45.0 |
| Quantized layer groups | 2,790 `FP8_PB_WO`; 552 `NVFP4` |

The resumable pull ran from `2026-08-28T17:33:38.979225+00:00` through
`2026-08-29T02:02:49.767688+00:00`, completing in 30,550.8 seconds (8.49 hours)
at an effective 52.70 MB/s. All 96 shards and 1,609,918,090,976 manifest-validated
physical bytes are present, with no partial files. The strict metadata/header audit
passed with complete 2,946/2,946 model-state mapping, 992,508 indexed tensors,
741,888 indexed expert entries, no cross-shard resident scales, and no errors.
The final acquisition records are
`$HOME/results/kimi-k3-gb10/acquisition-final.json` and
`$HOME/results/kimi-k3-gb10/source-audit-final.json`.

Acquisition recorded zero new OOM kills and one isolated swap-out page (4 KiB)
over the full 8.49-hour window; the counter did not continue increasing. This is
preserved in the final telemetry rather than rounded away. Conversion and runtime
promotion, when resumed, must establish fresh baselines and independently satisfy
their zero-delta safety gates.

The NVIDIA checkpoint differs materially from the original Moonshot artifact. It
uses ModelOpt mixed quantization: routed experts are NVFP4, aligned resident
attention/KDA projections are 128x128 block FP8, and the remaining tensors retain
their checkpoint precision. The FTW conversion preserves that representation; it
does not requantize the model into a second format.

The native bank mapping also matches NVIDIA's
[ModelOpt NVFP4 reference implementation](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py):
the low nibble stores the even E2M1 value, the high nibble stores the odd value,
and dequantization multiplies the E4M3 block scale directly by
`weight_scale_2`. Unlike compressed-tensors' quant-side
`weight_global_scale`, ModelOpt's `weight_scale_2` is already a dequantization
scale and must not be inverted.

Pre-conversion index audits passed:

- the text model constructed on the meta device expects 2,946 resident state keys;
  the mapped source index contains exactly 2,946, with no missing or unexpected keys;
- all 741,888 required expert `weight`, `weight_scale`, and `weight_scale_2`
  entries map exactly once, with no gaps or duplicates;
- completed-shard header probes confirm the packed weight and scale shapes used by
  the native NVFP4 banks;
- a real payload probe of shard 11, expert 0 (`w1`, `w2`, and `w3`) found finite
  E4M3 block scales in `[64, 256]`, a shared ModelOpt dequant scale of
  `0.0001220703125` (`2^-13`), and a resulting conservative absolute weight bound
  of `0.1875`; this magnitude also rules out accidentally inverting
  `weight_scale_2`;
- a real 128x128 block from layer 0 `self_attn.f_b_proj` passed the GB10 resident
  FP8 kernel probe with the checkpoint's FP32 scale: batch-one W8A16 decode was
  bit-exact to dense dequantization, while the W8A8 prefill path reached cosine
  `0.99964875` and relative L2 error `0.02653` for eight rows;
- the actual unaligned layer-0 `b_proj` (`96x7168`) and the absorbed layer-3
  `kv_b_proj` (`24576x512`) both normalized their singleton-rich ModelOpt scale
  grids to the expected 2-D block layouts and dequantized to finite BF16; sampled
  entries were bit-exact to literal FP8 x block-scale multiplication;
- a payload sanity sweep across the first 16 completed shards found no invalid
  values in 120 resident FP8 scale tensors or in the 45 block-scale and 45 global
  scale tensors for expert 0 across the represented MoE layers;
- a guarded production-loader probe of global layer 1 materialized all 896 experts
  and all 8,064 required packed/scale/global source entries into the exact
  16,664,625,152-byte (15.520 GiB) native bank. This probe caught and fixed a
  pre-conversion geometry bug: model-agnostic NVFP4 allocation had used the
  7,168-wide residual stream instead of Kimi's 3,584-wide latent expert stream;
- after that fix, 16 real experts spanning IDs 0 through 895 matched literal E2M1
  dequantization through the GB10 SiTU kernels. One-token decode reached cosine
  `0.99999523` and relative L2 error `0.003063`; eight-token prefill reached cosine
  `0.99999499` and relative L2 error `0.003152`. The sample-bank SHA-256 is
  `e18785bf7a882451dd1a22adfa0ebb54ecf4e6603ef854829dbf1f67279ab7f2`;
- the same genuine layer then passed a production streaming-FTW round trip. Its six
  banks produced exactly 16,664,625,152 logical bytes across an 8 GiB shard and a
  7.52 GiB shard; all 5,376 `(layer, expert, bank)` row descriptors indexed cleanly.
  O_DIRECT reads of the same 16 spread expert IDs reproduced the source SHA-256
  exactly, including FP8 storage bytes, and the temporary FTW returned disk usage to
  baseline after validation;
- the pre-probe targeted implementation suite passed 211 tests with 5
  hardware/environment skips. After the latent-width fix, the focused Kimi,
  offload, conversion, catalog, and promotion-gate subset passes 79 tests with 4
  optional-backend skips, including GB10 numerical coverage for SiTU through both
  NVFP4 prefill and decode paths.

The reproducible metadata-only auditor is
`benchmarks/audit_kimi_k3_checkpoint.py`. During acquisition it writes
`$HOME/results/kimi-k3-gb10/source-audit-partial.json` with
`--allow-partial`; the final evidence audit reruns it without that flag after all
96 indexed shards are present. The real-layer payload probe is
`benchmarks/probe_kimi_k3_expert_layer.py`; its payload and supervisor records are
`$HOME/results/kimi-k3-gb10/expert-layer1.json` and
`expert-layer1-supervision.json`.
The conversion-seam round trip is `benchmarks/probe_kimi_k3_ftw_layer.py`; its
payload and supervisor records are `ftw-layer1.json` and
`ftw-layer1-supervision.json` in the same results directory.

### Exact routed-pool geometry

One prepared NVFP4 expert row is 18,598,912 bytes (17.7373 MiB), including packed
gate/up/down weights, block scales, and expanded per-output-row global scales.

| Allocation or traffic | Exact size |
|---|---:|
| Full 92 x 896 routed pool | 1,427.853 GiB |
| One 896-expert layer | 15.520 GiB |
| Minimum 896-slot device cache | 15.520 GiB |
| 16 routed experts across 92 layers, before cache hits | 25.497 GiB/token |
| All non-expert source tensors, before dropping vision/calibration metadata | 72.861 GiB |

The FTW per-row global expansion adds approximately 1.492 GiB to the routed pool.
The full-layer staging path is deliberately avoided for the first request: an
896-slot cache is the runtime minimum and short prompts use route-first sparse
prefill.

### OOM and storage controls

The host had approximately 3.7 GiB of pre-existing swap occupancy before this run.
That absolute value is not attributed to Kimi; every stage records swap growth and
swap-I/O deltas. Reported runtime evidence requires zero new swap-out and zero OOM
kills. The workflow records Linux `oom_kill`, `pswpin`, `pswpout`, minimum
`MemAvailable`, and process-group peak RSS.

After the requested model cleanup, a byte-level preflight at 36 of 96
completed source shards observed 2,786,563,706,880 free bytes and credited the
resumable partial-download cache, which becomes source data
rather than permanent duplication. SparkLab's source-conversion plan requires
2,732,573,696,864 additional bytes, including the remaining pinned source, the
declared 1,610,038,482,254-byte FTW artifact, and a 128 GiB policy reserve. It
therefore projects 191,428,963,488 bytes (178.3 GiB) free after the durable source
and prepared artifacts, 38.3 GiB above the 140 GiB conversion stop floor.

The plan reports `artifacts.ready=true` and zero storage shortfall. It is a capacity
projection, not proof of the eventual prepared size; the live disk guard remains
authoritative. Runtime admission remains correctly closed because no full Kimi
runtime-memory budget has been measured and the machine has pre-existing swap
occupancy. The record is
`$HOME/results/kimi-k3-gb10/plan-partial.json`; artifact capacity is not
being misreported as runtime certification.

Conversion is guarded by:

- `MemAvailable >= 12 GiB`;
- free filesystem space `>= 140 GiB`;
- a validated pinned-source manifest before conversion starts;
- a passing strict audit of all 96 shards, including the complete mapped model
  state, 741,888 expert entries, resident FP8/FP32 scale geometry, and scale
  co-location, before conversion starts;
- an FTW structural validation before serving starts.

`benchmarks/validate_conversion_promotion.py` is the fail-closed handoff between
conversion and serving. In addition to successful exit and intact memory/disk
telemetry, it requires zero `oom_kill`, zero `pswpout`, and zero swap-occupancy
growth; missing counters fail the gate. Its record is
`$HOME/results/kimi-k3-gb10/conversion-gate.json`. The supervisor/gate
schema was exercised end to end with a short child process before queuing the full
conversion.

The successful real-layer probe reached a 34,177,593,344-byte process-group peak
RSS and 104,119,128,064-byte minimum `MemAvailable`, with zero OOM-kill and
swap-out deltas. Its kernel timings include first-use compilation and are numerical
probe timings, not end-to-end decode throughput. The preceding attempt successfully
loaded and released the layer but failed while hashing FP8 evidence through NumPy;
that non-runtime failure is preserved in `expert-layer1-attempt1-supervision.json`.

The FTW round trip reached a 34,584,514,560-byte process-group peak RSS and
103,983,493,120-byte minimum `MemAvailable`, again with zero OOM-kill and swap-out
deltas. Its 22.31-second load/write interval and 0.27-second sampled-row read are
conversion-integrity observations, not full-checkpoint conversion or decode rates.

The conversion command is:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
SPARKLAB_CONVERT_PROGRESS=1 PYTHONPATH=python \
.venv/bin/python benchmarks/supervise_process.py \
  --output $HOME/results/kimi-k3-gb10/conversion.json \
  --log $HOME/results/kimi-k3-gb10/conversion.log \
  --disk-path $HOME --min-memory-gib 12 --min-disk-gib 140 \
  -- .venv/bin/sparklab checkpoint \
  --model $HOME/.sparklab/models/kimi-k3/source/f8c5234a0a88 \
  --out $HOME/.sparklab/models/kimi-k3/prepared/0.2.0 \
  --dtype bfloat16 --moe-backend offload --nvfp4-backend triton \
  --shard-gib 8 --device cuda:0
```

### Staged runtime ladder

The first complete-checkpoint probe uses the minimum expert cache and a fail-closed
12 GiB memory floor:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/bench_decode_moe.py \
  --model $HOME/.sparklab/models/kimi-k3/prepared/0.2.0 \
  --recipe kimi-k3 --backend offload --storage disk --nvfp4-backend triton \
  --host-cache-gb 0 --cache 896 --cache-policy lru --mem-ratio 0.85 \
  --num-tokens 2048 --disable-prefill-overlap \
  --prefill-sparse-max-tokens 256 --decode 1 --no-graph \
  --collect-moe-stats --greedy --include-output --min-memory-gib 12 \
  --json $HOME/results/kimi-k3-gb10/probe-1.jsonl
```

Promotion is strictly one token, then 16, 64, and finally 256 tokens. A stage is
promoted only after the prior result shows a complete API response, no memory-guard
trigger, zero OOM-kill delta, zero swap-out delta, zero swap growth, and plausible
greedy output. Throughput and correctness remain **not measured** until those
full-checkpoint probes finish; analytical ceilings elsewhere in this report are not
benchmark results.

`benchmarks/validate_decode_promotion.py` enforces that contract against each raw
JSONL row before the next server launch. It also requires an identified FTW
checkpoint, complete token/event accounting, a nonempty output hash and payload,
and both lifecycle-wide and measured-request safety telemetry. Gate records are
saved as `$HOME/results/kimi-k3-gb10/gate-{1,16,64,256}.json`; a missing
counter is a failure rather than an assumed zero.

`benchmarks/finalize_kimi_k3_result.py` performs the publication handoff. It
independently rereads the acquisition, strict source audit, conversion telemetry,
conversion gate, four raw decode rows, and four decode gates; it also requires one
consistent FTW fingerprint/byte size and prefix-consistent greedy output across the
ladder. Only then does it write the schema-conforming compact result
`benchmarks/gb10/results/GB10-KIMI-001.json`.

Each raw rung also records the dataset's expected AIME answer, an extracted boxed or
explicit final answer when present, and the exact match result. Unfinished reasoning
therefore remains valid serving/safety evidence but is not mislabeled as task
correctness in the compact benchmark.

## Program order and promotion gates

| Stage | Purpose | Promotion gate |
|---|---|---|
| A: DSV4 Flash | Reproduce a known model and validate GB10/SM121 | 64 correct greedy tokens, bounded memory, no swap, complete I/O/cache telemetry |
| B: GLM-5.2 | Transfer the DSV4 storage features to a much larger supported model | 64-token correctness plus a stable 256-token run at least 0.5 tok/s, or a documented hardware bound |
| C: Kimi K3 | Add the new KDA/AttnRes/LatentMoE architecture and scale to 1.5 TB | Full load, 64-token correctness, reproducible bounded-memory run; 0.5 tok/s is the usability target |

Do not begin full Kimi K3 conversion merely because its files fit on disk. Stage C
starts only after Stage B demonstrates that GB10 memory accounting, FTW row reads,
scan-resistant caching, and long-run telemetry remain trustworthy at GLM scale.

## Stage A: reproduce DSV4 Flash on GB10

### Control configuration

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`.
- Use the official checkpoint and a freshly converted FTW checkpoint.
- Run batch one, greedy decoding, eager execution, and explicit disk storage.
- Preserve the existing RTX 3090 experiment's fixed AIME-25 prompt and output
  hashes as comparison evidence; treat a GB10 token mismatch as a numerical
  investigation, not automatically as a runtime failure.
- Record GB10 firmware, driver, CUDA, PyTorch, Triton, SSD, filesystem, thermals,
  and the exact FreeToken commit.

### Reproduction ladder

| ID | Configuration | Purpose |
|---|---|---|
| D0 | synchronous disk, 1 GiB host cache, no overlap | Correctness reference |
| D1 | parallel and coalesced reads | Verify GB10 SSD queue-depth scaling |
| D2 | safe cache-budget sweep | Measure expert locality without unified-memory oversubscription |
| D3 | borrowable per-layer LRU and combined hybrid staging | Reproduce the selected DSV4 storage policy |
| D4 | bounded two-buffer prefill | Measure TTFT tradeoff separately from sustained decode |

The RTX 3090 reference reached 1.982 tok/s over the matched 64-token run with a
40 GiB pageable host cache, 48.57% host hit rate, and 57.90-second warm TTFT. That
configuration is a comparison point, not a GB10 allocation prescription: GB10 host
and device allocations consume one 128 GB physical pool.

### Stage A exit gate

- Source and FTW checkpoints both boot on SM121.
- At least 64 greedy tokens agree with the reference, or any divergence is bounded
  with token/logit and operator-level evidence.
- Peak unified memory leaves at least 8-12 GB operational reserve.
- Swap-in and swap-out remain zero during reported runs.
- Logical/physical disk bytes, read rate, queue depth, cache hits, TTFT, and decode
  latency are present in the result JSON.
- A 256-token sustained run completes without memory growth, stale staging, or
  thermal collapse.

## Stage B: scale the validated path to GLM-5.2

### Model and storage facts

Use `nvidia/GLM-5.2-NVFP4`: 753B total parameters, approximately 40B activated,
78 transformer layers, 75 MoE layers, 256 routed experts per layer, and 8 selected
experts per token. FreeToken already implements the architecture, MLA/DSA latent
KV, IndexShare, native NVFP4 experts, and resident FP8 conversion.

One native NVFP4 expert row, including block and global scales, is approximately
20.27 MiB. Therefore:

```text
full routed pool   = 20.27 MiB x 256 experts x 75 layers ~= 380 GiB
uncached traffic   = 20.27 MiB x 8 experts x 75 layers   ~= 11.88 GiB/token
one full layer     = 20.27 MiB x 256 experts             ~= 5.07 GiB
```

### Feature-transfer ladder

| ID | Work | Required evidence |
|---|---|---|
| G0 | Memory-bounded native-NVFP4 FTW conversion | Bounded RSS, no swap, validated row descriptors and checksums |
| G1 | Synchronous disk correctness run | One, 16, then 64 reference-checked greedy tokens |
| G2 | Reuse parallel/coalesced FTW reads and pageable cache | Disk-rate and byte/token improvement without output change |
| G3 | Generalize DSV4's unique-expert short-prefill path | Lower short-prompt TTFT and no full-layer scan when selection is sparse |
| G4 | Launch the resident shared expert before routed-expert staging | Demonstrated compute/I/O overlap with identical output |
| G5 | Add borrowable per-layer protection to the GPU expert cache | Lower physical reads than equal-size global LRU on held-out prompts |
| G6 | Make cache planning GB10-unified-memory-aware | One budget covering OS reserve, weights, KV/DSA, staging, host LRU, and GPU slots |
| G7 | Compare GPU-only offload with native-NVFP4 hybrid | Select hybrid only if it wins end to end on Grace CPU plus shared LPDDR |

Double-buffered full-layer prefill consumes approximately 10.1 GiB for GLM and is
therefore optional. Start with one staging layer and spend the recovered memory on
persistent expert slots. Likewise, begin with little or no separate host LRU: on
GB10 the host LRU and GPU cache are duplicate logical tiers backed by the same
physical memory.

### Stage B exit gate

- The complete official checkpoint converts and loads without unbounded resident
  growth.
- At least 64 greedy tokens match a trusted reference, or numerical differences
  are explained and bounded.
- A 256-token run completes with zero swap and stable unified-memory usage.
- Report cold, warm-repeated, and warm-distinct prompts.
- Reach at least 0.5 tok/s sustained, or publish evidence that SSD bandwidth and
  measured cache locality impose a lower honest ceiling.
- Freeze the shared FTW/cache interfaces before beginning Kimi-specific model work.

## Stage C: run Kimi K3 after the shared stack passes

Stage C retains the detailed Kimi design below. It may add MXFP4 schemas, KDA,
Gated MLA, Attention Residuals, Stable LatentMoE, SiTU-GLU, vision support, and
DSPARK, but it should not fork a second storage engine. FTW extents, bounded read
queues, unified-memory budgeting, cache policy, telemetry, and correctness tooling
must come from Stages A and B.

## Stage C model and shared hardware facts

### Kimi K3

The official Kimi K3 model card reports:

| Property | Value |
|---|---:|
| Total parameters | 2.8T |
| Activated parameters | 104B |
| Transformer layers | 93 |
| Dense layers | 1 |
| Attention composition | 69 KDA + 24 Gated MLA |
| Routed experts | 896 |
| Selected experts per token | 16 |
| Shared experts | 2 |
| Latent MoE dimension | 3,584 |
| Expert hidden dimension | 3,072 |
| Weight / activation format | MXFP4 / MXFP8 |
| Maximum context | 1,048,576 tokens |
| Modalities | Text and image |

Sources:

- [Official Kimi K3 repository](https://github.com/MoonshotAI/Kimi-K3)
- [Official Hugging Face model card](https://huggingface.co/moonshotai/Kimi-K3)
- [Official report](https://arxiv.org/abs/2607.24653)

The published W4A8 deployment artifact is approximately 1.49 TB. The exact
download size, file count, tensor inventory, and checksum manifest must be
recorded locally before conversion. Standard vLLM guidance currently calls for
at least 8x GB300; the single-GB10 path in this document is deliberately an
out-of-core implementation outside that supported configuration.

### GB10

| Property | Value |
|---|---:|
| Architecture | Grace Blackwell |
| CPU | 20-core Arm |
| Unified system memory | 128 GB LPDDR5x |
| Aggregate memory bandwidth | 273 GB/s |
| FP4 tensor performance | Up to 1 PFLOP, sparse theoretical |
| Built-in storage | 4 TB NVMe M.2 |
| GB10 TDP | 140 W |

Source: [NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).

The 128 GB pool is shared by Linux, CPU, GPU, model state, KV/KDA state, staging,
and the routed-expert cache. Do not carry a fixed cache-size assumption from one
model to the next. For DSV4, GLM, and Kimi independently, measure the resident
floor and staging peak first, preserve an 8-12 GB operational reserve, then assign
only the remaining safe capacity to expert caching. Swap must remain disabled or
unused during reported runs.

## Stage C analytical I/O bound

A K3 routed expert contains approximately three matrices: gate, up, and down.
Ignoring scales and alignment, its payload is:

```text
3 x 3,584 x 3,072 = 33,030,144 parameters
33,030,144 x 0.5 byte ~= 16.5 MB at four bits
```

With 16 selected experts across 92 MoE layers, uncached routed-weight traffic is
approximately:

```text
16.5 MB x 16 x 92 ~= 24.3 GB/token
```

Scales, padding, shared experts, and implementation layout add overhead. Dense,
shared, attention, KDA, embedding, and output weights should remain resident and
must not be reread per token.

At 5 GB/s effective NVMe throughput, 24.3 GB requires about 4.9 seconds before
cache hits or overlap, giving an I/O-only ceiling near 0.2 tok/s. Approximate
bounds are:

| Effective routed-weight cache hit | Physical I/O/token | 5 GB/s I/O floor | I/O-only ceiling |
|---:|---:|---:|---:|
| 0% | 24.3 GB | 4.86 s | 0.21 tok/s |
| 25% | 18.2 GB | 3.65 s | 0.27 tok/s |
| 50% | 12.2 GB | 2.43 s | 0.41 tok/s |
| 75% | 6.1 GB | 1.22 s | 0.82 tok/s |
| 90% | 2.4 GB | 0.49 s | 2.06 tok/s |

The expert cache holds only a small fraction of the routed pool, so a 90% hit
rate cannot be assumed. Expert locality must be measured separately for coding,
reasoning, multimodal, repeated-session, and distinct-prompt workloads.

## Stage C scope and success criteria

### Primary scope

- One physical GB10 system.
- Full official Kimi K3 text weights in native MXFP4/MXFP8 semantics.
- Single-request, batch-one inference first.
- Dense and stateful components resident in unified memory.
- Routed experts streamed from local NVMe with a strict memory bound.
- Greedy output or logits validated against a supported reference deployment.
- Reproducible conversion and execution from a clean checkpoint.

### Secondary scope

- Image input after text correctness is stable.
- Prefix/KDA/MLA state reuse across persistent agent sessions.
- Multi-request vectorization for aggregate throughput.
- DSPARK speculative decoding.
- Optional mixed-precision cold experts, reported as a distinct accuracy tradeoff.

### Non-goals for the first result

- API-class 30-60 tok/s performance.
- One-million-token context on one GB10.
- Production concurrency or service-level objectives.
- Training or fine-tuning K3.
- Claiming official-weight equivalence for secondary 2-3-bit experts.

### Milestone definitions

| Level | Required result |
|---|---|
| Feasibility | Model inventory and memory plan prove that resident state fits below 128 GB |
| Bring-up | Full model loads and completes one forward pass without OOM or swap |
| Correctness | At least 64 greedy tokens match a trusted reference, or logit differences are explained and bounded |
| Reproducibility | Conversion and run complete from documented commands with checksums and peak-memory records |
| Usability | At least 0.5 tok/s sustained over 256 output tokens after warm-up |
| Stretch | At least 1.0 tok/s with official weights, or 1-2 tok/s with separately labeled mixed quantization |

## Shared system design

### Weight tiers

```text
128 GB unified memory
|-- permanently resident dense/attention/KDA/MLA/shared weights
|-- KV and recurrent state, activations, and runtime workspaces
|-- pinned/hot routed experts
|-- per-layer frequent and recent expert cache
`-- bounded asynchronous staging buffers

NVMe
`-- immutable expert-major FTW extents for cold routed experts
```

The exact resident-weight total must come from checkpoint inspection for every
stage. Conversion and serving must fail before allocation if the resident floor,
staging, caches, KV/state, and operational reserve cannot fit safely in the one
GB10 physical memory pool.

### Database execution model

Treat one token as a query over immutable routed-weight data:

```text
router output       -> predicate / partition selection
selected experts    -> indexed extents
unified memory      -> buffer pool
NVMe checkpoint     -> cold column store
MoE kernels         -> vectorized operators
```

The proposed runtime is not Spark or Flink embedded in the inference process.
It borrows their execution techniques in a native CUDA/C++/Python runtime.

#### Buffer pool

Use `(layer, expert, projection, block)` as the physical key. Maintain distinct
classes:

- `pinned`: resident and statistically hot data;
- `frequent`: repeatedly reused expert extents;
- `recent`: new decode admissions;
- `prefetch`: predicted data with lower eviction priority;
- `streaming`: scan data that bypasses admission.

Compare LRU against 2Q, ARC-like, and TinyLFU-style admission. Prefill is a scan
and must not pollute the decode cache. Give every MoE layer a guaranteed minimum
allocation, with a shared overflow pool redistributed by measured marginal value:

```text
value = predicted_hits x avoided_read_time / resident_bytes
```

#### Physical layout and late materialization

Convert raw checkpoint shards into expert-major, directly consumable extents:

```text
layer / expert / gate_up blocks and scales / down blocks and scales
```

Extents must be contiguous, aligned, checksummed, independently addressable, and
already arranged for the target kernel. Routing happens before materialization;
only missing blocks for the selected experts are read.

#### Read planning

For every layer:

1. Deduplicate requested extents.
2. Resolve buffer-pool hits.
3. Sort misses by file and physical offset.
4. Coalesce adjacent ranges.
5. Submit bounded high-queue-depth direct reads.
6. Dispatch ready morsels without waiting for all experts.
7. Release or admit extents according to the cache policy.

The first implementation should use expert-level cache entries and block-level
streaming. Pure block-level eviction adds metadata and fragmentation without
saving bytes when every block of a selected dense expert is required.

#### Pipeline and backpressure

Use at least two staging sets, preferably three:

```text
buffer A: GPU consuming current blocks
buffer B: NVMe filling later current-layer blocks
buffer C: bounded candidate prefetch
```

Every stage has a bounded queue. GPU slowdown reduces read submission; SSD
slowdown shrinks speculation and prefetch depth. Report queue occupancy and
backpressure time so apparent throughput gains cannot hide unbounded buffering.

#### Adaptive execution

An online cost model chooses among:

- cached whole-expert execution;
- direct block streaming;
- cache admission or bypass;
- GPU or Arm CPU execution, if the CPU kernel is competitive;
- reader count and coalescing width;
- prefetch depth;
- speculative verification width.

The Arm CPU path is a risk: current FreeToken MXFP4 CPU optimization primarily
targets x86 AVX2/AVX-512. Begin GPU-only on GB10 and benchmark any Arm kernel
before enabling hybrid scheduling.

## Historical Stage C implementation roadmap

### Archived implementation status (2026-08-26; superseded)

This section preserves the pre-acquisition design baseline for the original
Moonshot checkpoint. It is not the status of the current NVIDIA ModelOpt
experiment described at the start of this report. At that historical snapshot,
architecture work had begun in parallel with the GLM-5.2 checkpoint experiment,
but no full K3 download, conversion, or GPU run had started. The text-only code at
that point included:

- both K3 architecture registrations and strict parsing of the released 1-based
  69-KDA/24-MLA layer partition;
- NoPE gated MLA with latent-KV weight absorption and a hybrid MLA pool that stores
  only the 24 full-attention layers;
- KDA projections, three short convolutions, key-wise decay, the `-5` safe-gate
  floor, sigmoid output gating, and recurrent/conv state-pool integration. The
  released shard's 128-element `A_log` layout is handled explicitly despite the
  published reference constructor declaring a stale 96-element shape;
- Attention Residuals, SiTU, the leading dense MLP, sigmoid/bias routing, shared
  experts, and the 3,584-dimensional latent routed-expert bottleneck;
- native compressed-tensors MXFP4 expert mapping into FreeToken's transposed
  six-bank disk format, with the bias banks zeroed because K3 experts are biasless;
- the checkpoint's Python/tiktoken XTML renderer plus automatic K3 reasoning and
  typed tool-call parsing, including `reasoning_effort` to `thinking_effort` and
  thinking-toggle translation (vision placeholders remain disabled at the model layer);
- text-wrapper checkpoint name mapping and CPU references/tests for config, SiTU,
  Attention Residuals, KDA continuation, pool remapping, and registration.

A read-only audit of the official 96-shard index found 497,220 source tensors and
confirmed that all 2,367 non-expert text tensors map exactly onto the FreeToken
text model state. Header checks also confirm the expert matrices use the expected
MXFP4 shapes: gate/up `[3072, 1792]` and down `[3584, 1536]`, plus group-32 scales.

This was an architecture milestone, not a K3 inference claim. Open items before
K3-C1/C2 can pass are a full-checkpoint streaming-conversion rehearsal, GPU parity
for KDA/MLA/MXFP4, a chunk-efficient KDA prefill/snapshot path, tokenizer/parser
validation against official transcripts, and the deferred vision tower. Direct loading
of the 1.56 TB source checkpoint is deliberately rejected; GB10 serving requires FTW
conversion.

### K3-C0: checkpoint and hardware characterization

- Download all official K3 files and record SHA-256 hashes.
- Inventory tensor names, shapes, dtypes, compression metadata, and file extents.
- Separate resident tensors from routed-expert tensors.
- Measure built-in NVMe sequential, random-range, direct-read, and queue-depth curves.
- Measure GB10 unified-memory bandwidth and allocation headroom.
- Confirm CUDA/driver, SM121, FlashInfer, Triton, and MXFP4 kernel compatibility.
- Produce a static memory plan with at least 8-12 GB operational reserve.

Go/no-go: all non-routed weights and minimum state fit safely, and the selected
kernel path runs on SM121.

### K3-C1: architecture-only K3 model

- Validate tokenizer/chat-template and XTML tool/reasoning behavior against official transcripts.
- Validate the implemented embeddings, output head, normalization, Attention Residuals, routing,
  Stable LatentMoE, SiTU-GLU, KDA, and Gated MLA.
- Reuse official kernels where compatible; add explicit numerical references.
- Run reduced-shape CPU/GPU tests for every operator.
- Defer the vision encoder until text generation is correct.

Go/no-go: a tiny synthetic K3 configuration matches a PyTorch reference layer by layer.

### K3-C2: expert-major FTW conversion

- Extend FTW schemas for K3 MXFP4 blocks, scales, routing metadata, and projections.
- Stream conversion without materializing the full checkpoint.
- Transpose/repack once during conversion, never at inference time.
- Store dense/resident and routed/streamed inventories separately.
- Validate every extent's bounds, alignment, checksum, and logical tensor mapping.
- Keep conversion RSS bounded and evict completed output shards, following the
  DSV4 conversion lessons in [exp_dsv4.md](exp_dsv4.md).

Go/no-go: all expert rows reconstruct within the official MXFP4 numerical contract.

### K3-C3: synchronous correctness baseline

- Implement one-layer staging with direct reads of only routed experts.
- Disable overlap, prefetch, graphs, hybrid CPU execution, and cache admission.
- Run a minimal prompt and generate one token, then 16 and 64 tokens.
- Compare greedy tokens and selected intermediate logits with an official reference.
- Record every physical read and prove the peak memory bound.

This phase is expected to be slow. Its purpose is a trusted correctness baseline.

Go/no-go: 64-token text generation completes without OOM, swap, stale staging,
or unexplained output divergence.

### K3-C4: asynchronous and coalesced I/O

- Add a persistent bounded read pool or `io_uring` backend.
- Sort and coalesce extents across all selected experts within a layer.
- Read directly into target staging buffers.
- Sweep reader count, queue depth, extent size, and number of staging sets.
- Separate logical bytes, physical bytes, read service time, and overlap time.

Go/no-go: at least 70% of isolated sustainable NVMe bandwidth is reached end to end.

### K3-C5: scan-resistant per-layer cache

- Establish an uncached route trace corpus before tuning.
- Replay traces through LRU, 2Q, TinyLFU, and layer-partitioned policies offline.
- Add prefill admission bypass and per-layer minimum capacity.
- Split globally hot, session-hot, recent, and speculative entries.
- Sweep cache budgets without encroaching on the runtime reserve.

Go/no-go: the chosen policy reduces physical decode reads versus equal-size LRU
on both repeated and distinct prompts, not only on its tuning trace.

### K3-C6: block-streamed vectorized execution

- Split selected gate/up and down matrices into kernel-sized morsels.
- Start gate/up computation as soon as a morsel is ready.
- Preserve the gate/up -> SiTU-GLU -> down dependency exactly.
- Group tokens by expert during prefill and concurrent/speculative verification.
- Read each expert extent once for all matching tokens in the microbatch.
- Add bounded backpressure and queue telemetry.

Go/no-go: lower staging memory or better overlap without numerical divergence or
an increase in physical bytes.

### K3-C7: route prediction and prefetch

- Measure per-layer expert-transition predictability from prior hidden states,
  tokens, and session history.
- Prefetch only above a cost-model confidence threshold.
- Assign prefetched blocks lower cache priority until they are consumed.
- Report precision, recall, useful bytes, wasted bytes, and net latency.

Go/no-go: prefetch improves end-to-end decode on held-out prompts without raising
physical I/O enough to reduce throughput.

### K3-C8: DSPARK speculative decoding

- Validate the official draft checkpoint independently.
- Start with one proposed token, then sweep verification widths up to seven.
- Group verification routes by expert and coalesce the union of required extents.
- Measure acceptance, unique experts per verified token, reused weight bytes,
  wasted reads on rejection, and total speedup.
- Adapt verification width to acceptance and storage pressure.

Go/no-go: at least 1.2x end-to-end speedup on held-out agent and reasoning traces.

### K3-C9: optional mixed-precision cold tier

- Rank expert/projection/block sensitivity against a validation set.
- Retain dense, shared, hot, and sensitive weights in official MXFP4.
- Test 3-bit then 2-bit storage for cold routed experts.
- Promote repeatedly used experts to an MXFP4 cache where storage permits.
- Report model quality, output divergence, disk bytes, and throughput separately
  from the official-weight result.

This phase must never replace or be conflated with the official-MXFP4 baseline.

## Stage C experiment matrix

### Workloads

| Workload | Purpose |
|---|---|
| One-token deterministic smoke | Bring-up and crash detection |
| 64-token greedy fixed prompt | Correctness and comparable optimization ladder |
| 256-token greedy reasoning | Sustained batch-one decode |
| Repeated coding-agent session | Prefix, session cache, and expert locality |
| Distinct AIME/coding prompts | Cache-policy generalization |
| Long prompt sweep: 1K/8K/32K | Prefill and state scaling |
| Image plus text, later | Vision-path correctness and residency impact |

Do not use a single specially favorable prompt as the headline result. Publish
cold-start, warm-repeated, and warm-distinct numbers.

### Optimization ladder

| ID | I/O | Cache | Execution | Speculation | Precision |
|---|---|---|---|---|---|
| K0 | synchronous | none | whole selected expert | off | official MXFP4 |
| K1 | parallel | none | whole selected expert | off | official MXFP4 |
| K2 | parallel + coalesced | none | whole selected expert | off | official MXFP4 |
| K3 | coalesced | global LRU | whole selected expert | off | official MXFP4 |
| K4 | coalesced | per-layer scan-resistant | whole selected expert | off | official MXFP4 |
| K5 | triple-buffered | per-layer | block-streamed/vectorized | off | official MXFP4 |
| K6 | adaptive + prefetch | per-layer | block-streamed/vectorized | off | official MXFP4 |
| K7 | adaptive + prefetch | per-layer | vectorized verification | DSPARK | official MXFP4 |
| K8 | adaptive + prefetch | per-layer | vectorized verification | DSPARK | mixed cold tier |

## Required metrics

### User-visible

- Cold and warm TTFT.
- Prompt processing throughput.
- Decode tok/s and milliseconds/token.
- Inter-token latency p50/p90/p99.
- End-to-end time for fixed agent tasks.
- Output token count and deterministic hash.

### Correctness

- Greedy token agreement with the reference.
- Selected-layer logit absolute and relative error.
- MXFP4/MXFP8 operator error against official/reference kernels.
- Expert IDs, routing weights, and draft acceptance trace.
- Text, tool-call, reasoning-content, and later vision behavior.

### Memory and cache

- Peak and steady unified-memory use by category.
- Swap-in/out, page faults, and system reserve.
- Cache capacity, occupancy, admission, eviction, and hit rate.
- Hits and avoided bytes by layer and cache class.
- Prefill pollution and session reuse.
- KDA/MLA/KV state bytes per admitted request and token.

### Storage

- Logical and physical bytes per prompt and output token.
- Read count and request-size distribution.
- Effective GB/s, queue depth, coalescing ratio, and read amplification.
- SSD service time, GPU stall time, and achieved overlap.
- Prefetch useful and wasted bytes.

### Compute and scheduling

- Time per attention, router, fetch, gate/up, activation, down, and combine operator.
- GPU utilization and unified-memory bandwidth.
- Queue occupancy and backpressure time.
- CPU/GPU assignment if hybrid is enabled.
- Energy, average power, and joules/token.

## Reporting rules

- Clearly label measured values, analytical bounds, and projections.
- Preserve K0 as the correctness reference throughout optimization.
- Report official and mixed-precision results in separate tables.
- Include initialization and conversion time, not only steady-state decode.
- Report failures, OOMs, swap growth, and thermal throttling.
- Publish exact software revisions, container images, driver, firmware, kernel,
  model hashes, conversion fingerprint, and commands.
- Compare against the hosted API only for user experience and economics, not as
  equivalent hardware performance. Current API observations are roughly
  30-60 output tok/s and vary with provider load, prompt, and reasoning effort.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Resident model/state exceeds safe memory | Inventory first; cap context/concurrency; enforce an allocation reserve |
| Official Blackwell kernels require datacenter SM100-class features | Build SM121-compatible Triton/CUDA references before checkpoint conversion |
| Built-in NVMe cannot sustain projected range-read rate | Repack/coalesce, raise queue depth, benchmark external/striped NVMe as a separate configuration |
| Low expert locality makes caching ineffective | Report it; emphasize direct streaming, batching, and speculation rather than hiding misses |
| Route prediction wastes bandwidth | Confidence gate, bounded prefetch queue, low-priority admission, automatic disable |
| Arm CPU hybrid execution is slow | GPU-only default; enable hybrid only after a K3-shape bandwidth benchmark |
| Prefill destroys decode state | Scan bypass and separate admission classes |
| Mixed quantization damages capability | Keep official baseline; sensitivity tests and held-out quality evaluation |
| One-token demo overstates success | Require 64-token correctness and 256-token sustained runs |
| SSD endurance or capacity pressure | Read-only checkpoint, adequate free space, health/temperature telemetry |

## Decision points

1. **After Stage A:** do not proceed to GLM until DSV4 Flash has reference-checked
   output, stable 256-token memory behavior, and complete storage/cache telemetry.
2. **After Stage B conversion:** stop if the GLM resident floor plus one 5.07 GiB
   staging layer and the operational reserve cannot coexist safely.
3. **After Stage B sustained decode:** do not begin Kimi architecture work until
   GLM completes a stable 256-token run and the measured physical-byte model explains
   its throughput.
4. **After K3-C0:** stop if Kimi resident state cannot safely fit or no viable SM121
   MXFP4/KDA kernel path exists.
5. **After K3-C3:** continue only after reference-validated full-model output.
6. **After K3-C4:** if end-to-end reads remain below 70% of isolated SSD bandwidth,
   fix storage layout/scheduling before cache tuning.
7. **After K3-C5:** if the safe unified cache produces little reuse, prioritize
   speculation/vectorization and consider whether 0.2-0.4 tok/s is the honest
   official-weight ceiling.
8. **After K3-C8:** if official weights remain below 0.5 tok/s, publish the
   bounded-memory feasibility result and treat mixed quantization as a separate
   usability track.

## Headline result matrix

The final report shows the complete promotion ladder rather than selecting only
the best Kimi run:

| Model/configuration | Weight fidelity | Peak memory | TTFT | Decode | Physical GiB/token | Cache hit | Power |
|---|---|---:|---:|---:|---:|---:|---:|
| DSV4 no-overlap control | Official DS-FP4 | 87.34 GiB device | 16.219 s | 4.764 tok/s | 2.864 GiB | 78.6% | Not recorded |
| DSV4 selected GB10 run | Official DS-FP4 | 76.19 GiB device | 0.604 s | 10.282 tok/s | 0 GiB | 100% | 37.95 W avg |
| GLM G1 synchronous control | Official NVFP4 | 30.44 GiB device | 45.760 s | 0.619 tok/s | 17.909 GiB | 0% | 15.00 W avg |
| GLM selected GB10 run | Official NVFP4 | 33.70 GiB device | 2.569 s | 0.802 tok/s | 8.769 GiB | 26.5% | 16.57 W avg |
| Kimi 1-token admission probe | NVIDIA ModelOpt NVFP4 experts + FP8 resident | Pending gated run | Pending | N/A | Pending | Pending | Pending |
| Kimi 16-token promoted probe | NVIDIA ModelOpt NVFP4 experts + FP8 resident | Pending gated run | Pending | Pending | Pending | Pending | Pending |
| Kimi 64-token correctness probe | NVIDIA ModelOpt NVFP4 experts + FP8 resident | Pending gated run | Pending | Pending | Pending | Pending | Pending |
| Kimi 256-token sustained probe | NVIDIA ModelOpt NVFP4 experts + FP8 resident | Pending gated run | Pending | Pending | Pending | Pending | Pending |
| Kimi best mixed cold-tier run | Separate experiment | Not run | Not run | Not run | Not run | Not run | Not run |
| Hosted K3 API observation | Official service | N/A | Not measured | Not measured | N/A | N/A | Not measured |

Peak memory in this matrix is the server's reported device allocation. Physical
GiB/token divides measured-request physical I/O by returned completion tokens; it
is not an SSD bandwidth figure. The pending Kimi cells are populated only from
their promoted raw rows, never from layer probes or analytical projections.

## Deferred actions for the Kimi run

The pinned source acquisition and strict complete-source audit are finished. Per
the refactor pause, no usable FTW conversion, runtime probe, or benchmark result
was produced. A pre-queued conversion watcher raced the final source audit and ran
for 16.5 seconds before it was caught and stopped. Its two incomplete FTW shards
(16,621,034,424 bytes) were moved out of the active prepared path to
`$HOME/.sparklab/models/kimi-k3/prepared-interrupted-20260829T020606Z`;
the canonical prepared path and `conversion.json` are absent. The interruption is
recorded in
`$HOME/results/kimi-k3-gb10/conversion-interrupted-20260829T020606Z.json`.

1. After the refactor, run the supervised FTW conversion; stop on the disk or
   memory floor, any OOM or
   swap-out delta, conversion failure, or structural FTW validation error.
2. Promote the complete checkpoint through 1, 16, 64, and 256 greedy tokens only
   after each raw row passes safety, provenance, output, and accounting checks.
3. Run the fail-closed result finalizer, populate the pending Kimi matrix cells from
   the promoted rows, and add `GB10-KIMI-001` to the GB10 benchmark index.
4. Update recipe evidence and limitations only to the level proven by the full run,
   then rerun focused tests, JSON/schema checks, and the completion audit.
