# Qwen3.8-Flash-Next on one DGX Spark — 2026-09-09

This run compares single-client text generation on the same NVIDIA GB10 using
frozen prompts and fixed output budgets. SparkLab and SGLang use the same pinned
NVIDIA source checkpoint. **vLLM uses a different Mia quantization and the existing
Spark adaptation**, so its row compares deployable configurations and does not
isolate framework efficiency or establish quality parity.

## Measured results

Decode throughput, in output tokens/second:

| Configuration | Short / 128 output | 1K / 256 output | ~8K / 256 output |
|---|---:|---:|---:|
| SparkLab fast (NVIDIA) | 41.90 | 30.49 | 25.93 |
| vLLM + Spark adaptation (Mia) | 42.14 | 33.96 | 31.14 |
| SGLang (NVIDIA) | 35.07 | 28.83 | 26.46 |

Prompt and full-request latency:

| Configuration | 1K TTFT | ~8K TTFT | ~8K full request |
|---|---:|---:|---:|
| SparkLab fast (NVIDIA) | 2.213 s | 23.244 s | 33.172 s |
| vLLM + Spark adaptation (Mia) | 0.730 s | 4.527 s | 12.788 s |
| SGLang (NVIDIA) | 0.816 s | 10.536 s | 20.303 s |

The short column is the median of three repeats of the original AIME25 portfolio
prompt, after one warmup. The other columns are means over 30 distinct requests,
after three separate warmups each. All engines completed all 70 requests including
warmups. Input token counts match across engines: 96 for short, 1,024 for random1k,
and 7,908–8,983 for SPEED-Bench. Each response has the exact requested output count.

Decode rate is `(completion_tokens - 1) / (request_s - ttft_s)`: timing continues
until the stream completes, including generation that emits no visible text.
The original last-visible-token calculation inflated some SGLang random1k trials;
all rows use the corrected window, derived from the saved timings. Both versions
and trailing durations are retained in the result. TTFT ends at the first nonempty
content/reasoning delta; full-request latency includes prefill and generation. These are client-observed
streaming metrics. Do not mix the short probe with long-prompt means.

## Configurations

All runs use one client, temperature zero, `ignore_eos`, a 12,288-token context
limit, BF16 KV, and MTP with three speculative steps. Thinking is enabled for the
short and SPEED-Bench prompts and disabled for random1k. Servers run sequentially.
Framework KV pool capacities, page sizes, kernels, and MTP execution differ.

- **SparkLab:** clean source `3f259fb04d7d28a4d78b49f96502083e8586012f`;
  NVIDIA source `fab0aecb760cec45227f6656abcaafa11abca87a`, FTW fingerprint
  `94e1ee0daa442357`. Published opt-in fast profile: separate FP8 draft head,
  immediate redraft, light verification snapshots, full 24,576-expert preload,
  Triton NVFP4/W4A16 prefill, FP32 recurrent state, page size 16, 131,072 KV tokens.
  This is not recipe 0.9.0's target-only default.
- **vLLM:** `0.1.dev20073+g8e685d198`, dedicated image digest
  `sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`;
  Mia source `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`;
  [existing Spark adaptation](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  at local commit `5218435fcbfe567c58cc28668be5035ec137d77f`.
  Adaptations provide NVFP4/file-backed PLE, GB10 compatibility, and QSA handling;
  mounted-file hashes are retained in the evidence. Mia's PLE and attention
  precision differ from NVIDIA's export. Prefix caching is disabled, batch-token
  cap is 8,192, memory utilization is 0.80, and the runtime reports 216,760 KV tokens.
- **SGLang:** stock cookbook image `0.0.0.dev1+g4ccff141d`, commit
  `4ccff141dbe992794f9da6c3aa23535b4f72000d`, ARM64 image digest
  `sha256:a1e17bbf0e9618364dc6395a4de5f9c3e8dfc9f43d603995b7a6aebc83de6e06`.
  Same NVIDIA source revision as SparkLab; ModelOpt mixed auto-detection,
  FlashInfer CUTLASS MoE, page size 64, 4,096-token prefill chunks, FP32 recurrent
  state, NEXTN three steps/four draft tokens, maximum 8 requests/40 Mamba slots,
  memory fraction 0.85, and 97,792 KV tokens. PLE is a fresh file-backed FP8 table
  with an 8 GiB resident cap. Compared with the cookbook command, context is reduced to the matched
  workload limit and radix caching is disabled; precision is explicit and the
  model is loaded offline from the pinned snapshot.

The original NVIDIA safetensors were recovered from the existing native cache
plus source-only tensors and headers. All 11 complete weight files match the
publisher's SHA256 hashes; this did not introduce another quantization.

## SGLang cookbook reference

The [Qwen3.8-Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
was retrieved on 2026-09-09. The requested H200/BF16/low-latency/single-node cell
uses tensor parallelism of four and lists accuracy, but no speed measurements.

Its single-Spark NVIDIA NVFP4 random 1,024-input/256-output, concurrency-one entries
report:

| Published profile | TTFT | TPOT | Decode rate derived as 1000 / TPOT |
|---|---:|---:|---:|
| MTP | 650.03 ms | 33.26 ms | 30.07 tok/s |
| No MTP | 604.33 ms | 63.54 ms | 15.74 tok/s |

The published 136 and 78 tokens/s/GPU values include input plus output tokens;
they are not output decode rates. Our random1k workload matches the lengths, but
uses different frozen inputs and chat/streaming measurement, so it is not an exact
reproduction of the published benchmark.

## Limits and operational observations

- Fixed token budgets test speed, not completed-answer quality. The prior native
  fast-profile MGSM tuning subset scored 92/100 versus 94/100 for the original
  profile; no new quality comparison was performed. Identical NVIDIA source
  weights do not guarantee identical activation arithmetic or generated text.
- The three native short repeats generated identical text. vLLM and SGLang each
  produced three distinct outputs despite temperature zero; decode variation
  includes different generated continuations and speculation acceptance.
- Native short repeats reuse 64 of 96 prompt tokens, while vLLM and SGLang have
  prefix reuse disabled. Short TTFT is therefore omitted from the comparison
  table. All native 1K/8K prefill batches report zero cached tokens.
- The host was not isolated. SGLang image acquisition overlapped native short/1K
  and early 8K requests; background source acquisition resumed during native 8K.
  Bulk source recovery ended before vLLM measurement. These observations are
  practical measurements, not a fully isolated causal framework experiment.
- Pre-existing swap and additional host swap traffic occurred. Memory telemetry,
  guard status, and container exits are retained; these runs do not certify
  zero-swap serving or production endurance.
- SGLang emitted a startup compiler warning about concurrent writes to
  `Logits(bx, position)`. Its effect on output quality was not evaluated.
- Concurrency one does not measure multi-client throughput or capacity.

## Evidence

The [versioned result](../gb10/results/GB10-QWEN38-FLASH-NEXT-FRAMEWORKS-001.json)
contains exact commands, revisions, per-request timing/usage/output hashes,
validation, and memory summaries. Raw requests, generated text, logs, suite,
launch scripts, and checkpoint verification live in
`results/qwen38-flash-next-frameworks-20260909` at the repository root. The frozen
suite SHA256 is
`38316b0c391b4dc37c57923315389d69663a98aa90b20a5c8b1e04dc6f15f7fb`.

This supersedes the framework availability conclusions in the
[earlier comparison](QWEN38_FLASH_NEXT.md), which used older checkpoints/runtimes
and a different 256-output-token short probe. Historical measurements remain
unchanged.
