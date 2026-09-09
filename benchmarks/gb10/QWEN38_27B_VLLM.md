# Qwen3.8-27B: SparkLab and vLLM on GB10

For the current RadixArk default, see the
[September 9 fast-profile verification](QWEN38_27B_FAST.md).

Measured on 2026-09-08 using the same pinned Inferact NVFP4 target, 30 fixed SPEED-Bench prompts of roughly 8K input tokens, and exactly 256 generated tokens per request. Each profile has three warmups and one measured trial. All measured requests passed HTTP and token-count checks; prompt identities and counts match across engines.

| Engine/profile | Clients | Decode tok/s/user | Total output tok/s | Mean TTFT | Mean request time | p95 TTFT |
|---|---:|---:|---:|---:|---:|---:|
| SparkLab target-only | 1 | 8.92 | 7.17 | 7.10 s | 35.69 s | 7.41 s |
| SparkLab DFlash2-12 (opt-in) | 1 | 22.05 | 13.30 | 7.47 s | 19.24 s | 7.80 s |
| vLLM target-only | 1 | 9.43 | 8.24 | 4.01 s | 31.05 s | 4.19 s |
| SparkLab DFlash2-12 (queued) | 4 | 22.00 | 13.28 | 61.24 s | 73.04 s | 68.53 s |
| vLLM target-only (batched) | 4 | 7.13 | 22.12 | 8.36 s | 44.37 s | 14.95 s |

Decode tok/s/user excludes time to first token. Total output tok/s includes prompt processing and scheduling over the complete measured run. Four native clients queue behind one active generation; vLLM can batch them. Prefix-cache reuse is controlled as recorded in the evidence.

The vLLM 0.28.0 run with the same NVFP4 DFlash2 draft failed at startup. Its fused context-KV projection treated packed weights as a dense matrix, producing a `(8192×5120) × (2560×10240)` shape mismatch. No speed is reported for that configuration. A different draft or a compatibility fix would need a separate benchmark.

vLLM used FP8 KV cache, while SparkLab used BF16. This measures the recorded serving configurations, not precision-identical kernels or quality parity. The arithmetic smoke check passed, but it is not a general quality evaluation. Recipe defaults were unchanged.

[Versioned evidence and exact commands](results/GB10-QWEN38-27B-VLLM-001.json) include checkpoint SHA256 verification, metrics, tail latency, telemetry, raw-response hashes, and limitations. Raw responses and scripts remain in `results/qwen38-27b-vllm-20260908/`.
