# Bonsai 2: vLLM isolation check

Measured September 18, 2026 on the same DGX Spark. Stopping the idle vLLM server produced no observed decoding speed benefit. The native SparkLab gap against Prism remains. No inference implementation changed for these measurements.

## Controlled short-prompt comparison

PTQ1, Q8 vision loaded, 96 input / 128 output tokens, one warmup and three measured trials; median client-observed decode. Only one Bonsai backend was resident. SparkLab kept the **same process** before and after stopping vLLM. Prism used separately warmed instances with identical flags.

| Engine | vLLM resident | vLLM stopped | Change |
|---|---:|---:|---:|
| native | 19.73 tok/s | 19.75 tok/s | +0.10% |
| prism | 34.47 tok/s | 33.42 tok/s | -3.04% |

Inputs and generated output hashes were identical before/after for both engines. Prism was slightly slower in the later run; this sequence does not establish the cause of that small difference. It does not show vLLM slowing down decoding.

## Isolated long-prompt comparison

Same three frozen 8K requests as the initial development run, one warmup, 256 output tokens, medians. The earlier runs also had the other Bonsai backend resident; this part tests combined isolation, not vLLM alone.

| Engine | Earlier decode | Isolated decode | Earlier TTFT | Isolated TTFT |
|---|---:|---:|---:|---:|
| native | 18.95 tok/s | 18.73 tok/s | 9.116 s | 8.798 s |
| prism | 30.93 tok/s | 30.42 tok/s | 19.193 s | 19.062 s |

## Official-style microbenchmark

Prism llama-bench, no other model resident, no vision tower, depth 0, five repetitions. The [official throughput report](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#cross-platform-throughput) uses this test family but lists other hardware, not DGX Spark. These numbers are means from llama-bench, not chat-stream medians.

| Packing | Test | Mean tok/s | Standard deviation |
|---|---|---:|---:|
| PTQ1_0 | pp512 | 451.53 | 2.88 |
| PTQ1_0 | tg128 | 34.02 | 0.05 |
| PQ2_0 | pp512 | 1020.00 | 14.62 |
| PQ2_0 | tg128 | 29.19 | 0.06 |

The official-style decode values are close to our Prism chat results. Removing the API and vision tower does not uncover the much higher throughput reported on other GPUs. The remaining SparkLab-versus-Prism gap requires runtime profiling; this experiment does not quantify the contribution of individual kernels or CUDA graphs.

## Evidence and limits

- [Machine-readable summary](results/isolation/summary.json) includes per-phase available memory and swap deltas.
- [Commands](results/isolation/commands.json), [one-second host telemetry](results/isolation/telemetry.json), and all raw benchmark JSONs are saved alongside it.

The saved commands replace the machine-specific checkout path with `/path/to/sparklab`.
- Swap counters are host-wide; one-second samples do not identify a process and omit the intervals crossing phase boundaries.
- Single-client, short development runs; no endurance or general quality qualification.
- vLLM and all temporary Bonsai benchmark servers were left stopped.
