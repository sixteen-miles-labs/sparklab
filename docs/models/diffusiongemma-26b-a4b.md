# DiffusionGemma 26B A4B — NVIDIA NVFP4

SparkLab recipe `diffusiongemma-26b-a4b` serves
[NVIDIA's ModelOpt NVFP4 checkpoint](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)
through an isolated **vLLM 0.28.0 Docker backend**. Its weights and container image
are pinned to immutable revisions. SparkLab owns acquisition, artifact validation,
memory admission, and launch; vLLM owns model execution and the OpenAI API.
SparkLab's native autoregressive engine does not implement block diffusion.

DiffusionGemma shares Gemma 4's MoE backbone but uses causal prompt encoding,
bidirectional canvas attention, self-conditioning, and iterative denoising to emit
blocks of text. Aliasing it to SparkLab's Gemma 4 causal model would produce an
incorrect generation path.

## Run

Requires a source installation of SparkLab, Docker with NVIDIA Container Toolkit,
and one NVIDIA GB10. vLLM stays inside the container; do not install it into the
native engine's Python environment.

```bash
sparklab plan diffusiongemma-26b-a4b
sparklab pull diffusiongemma-26b-a4b
sparklab run diffusiongemma-26b-a4b --dry-run
sparklab run diffusiongemma-26b-a4b
```

The checkpoint is loaded directly from safetensors; omit `--prepare`.
The default endpoint is `http://127.0.0.1:18080/v1`. Override the listening port with
`sparklab run diffusiongemma-26b-a4b -- --port 18081`. Other trailing options are
vLLM options; changing memory, context, or concurrency settings requires a fresh
benchmark and memory budget.

The default GB10 profile uses a 256-token canvas, one concurrent sequence,
32,768-token maximum model length, Triton attention, prefix caching, and thinking.
It reduces memory allocation and concurrency from the
[upstream GB10 recipe](https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it?variant=nvidia_nvfp4).
The container has a 72 GiB memory limit and cannot use swap.
SparkLab reserves 72 GiB for admission, a conservative budget for this measured profile.
The model's published 256K context is not a verified capacity of this profile.

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/diffusiongemma-26B-A4B-it-NVFP4","messages":[{"role":"user","content":"Explain hash tables."}],"max_tokens":1024,"chat_template_kwargs":{"enable_thinking":false}}'
```

Readiness is exposed at `/health`; vLLM exposes Prometheus metrics at `/metrics`.
The native SparkLab daemon, `/v1/stats`, and Anthropic API are not provided by this
adapter. Stop the foreground server with Ctrl-C. The Qwen services stopped for
the benchmark can be restarted with `docker start fct-qwen36-vllm fct-qwen38-vllm`
after stopping DiffusionGemma.

## Benchmark

```bash
python benchmarks/frameworks/bench_diffusion.py \
  --base-url http://127.0.0.1:18080 --thinking both \
  --max-tokens 256 512 1024 --trials 3 \
  --output results/diffusiongemma-26b-a4b/benchmark.json
```

Each case has one warmup followed by three sequential measured requests, using a
fixed hash-table explanation prompt and the checkpoint's entropy-bound sampler
(`entropy_bound=0.1`, denoising schedule `t_min=0.4`, `t_max=0.8`, up to 48 steps).
vLLM 0.28 rejects ordinary `temperature` and `seed` request parameters for diffusion;
the harness omits them and does not claim greedy or seeded output parity. Prefix caching
is enabled and the repeated prompt is warm. Thinking and non-thinking are measured
separately. Output lengths are request caps; EOS can end a request sooner.

Throughput is the exact server-reported completion token count divided by the entire
request duration, **including TTFT**. TTFT ends at the first nonempty content or
reasoning SSE delta. Completion usage includes reasoning tokens in thinking mode.
Emission timings, usage, finish reasons, and output hashes are
retained in versioned evidence; full output text remains in local artifacts. SSE chunks can contain many tokens: counting chunks or timing only the
interval between first and last chunks can greatly inflate diffusion throughput.
The autoregressive portfolio `decode_tokens_per_second` field is intentionally empty.

## GB10 results — October 5, 2026

| Thinking | Output tokens | Median completion tok/s | Median TTFT | Median request time |
|---|---:|---:|---:|---:|
| Off | 256 | 100.73 | 2.541 s | 2.542 s |
| Off | 512 | 69.81 | 4.386 s | 7.334 s |
| Off | 1024 | 63.65 | 3.955 s | 16.089 s |
| On | 256 | 65.09 | 3.933 s | 3.933 s |
| On | 512 | 59.16 | 3.530 s | 8.655 s |
| On | 1024 | 66.29 | 4.172 s | 15.446 s |

All 18 measured requests reached their requested token cap. They emitted one,
two, or four chunks for 256, 512, or 1024 tokens respectively. Outputs varied between
requests; these are three-trial latency probes, not deterministic quality comparisons.
The default thinking-on 1024-token case measured **66.29 completion tok/s**.

Other GPU model servers were stopped for this run. vLLM logged 18.19 GiB for model
loading and approximately 49 GiB for the allocated runtime including KV cache.
Across the startup and request monitor, available host memory stayed above 60 GiB.
Container swap remained zero, and pre-existing host swap did not grow. The admission
reservation remains 72 GiB. The monitor covered roughly 13 minutes, rather than an
hour-long endurance qualification.

Three basic API smoke checks passed: `17 × 19 = 323`, separated reasoning output,
and a parsed `get_weather` tool call with `city="Toronto"`. The thinking arithmetic
response hit its 512-token cap; this checks arithmetic and parsing, not complete-answer
quality. No external weather tool was executed.

Evidence: [GB10-DIFFUSIONGEMMA-001](../../benchmarks/gb10/results/GB10-DIFFUSIONGEMMA-001.json),
[per-request timings and output hashes](../../benchmarks/frameworks/results/diffusiongemma-26b-a4b/benchmark.json),
[API smoke checks](../../benchmarks/frameworks/results/diffusiongemma-26b-a4b/smoke.json),
[memory monitor](../../benchmarks/frameworks/results/diffusiongemma-26b-a4b/memory.jsonl),
and [runtime diagnostics](../../benchmarks/frameworks/results/diffusiongemma-26b-a4b/runtime.txt).

The recipe remains Experimental. This throughput probe does not certify long context,
coding quality, tool use, image understanding, or hour-long stability. Audio is unsupported.
