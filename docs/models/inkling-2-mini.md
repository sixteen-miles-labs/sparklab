# Inkling 2 Mini

The experimental `inkling-2-mini` recipe serves
[Thinking Machines' checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
through an isolated vLLM Docker runtime. Both the checkpoint and ARM64 image are
pinned. Inkling has 24 physical layers, with the middle 12 visited three times,
and mixed NVFP4/MXFP8 experts. Its 26.4B stored parameters correspond to about
5.4B effective active parameters per token across all loop passes.

The native SparkLab engine does not implement `Inkling2LoopForConditionalGeneration`.
The [upstream vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini?hardware=dgx_spark_gb10)
requires the Inkling runtime image until support reaches a stable vLLM release.

Requires a source installation, Docker with NVIDIA Container Toolkit, and GB10.

```bash
sparklab plan inkling-2-mini
sparklab pull inkling-2-mini
sparklab run inkling-2-mini --dry-run
sparklab run inkling-2-mini
```

The profile serves one sequence at a time, with at most one image per request,
an 8,192-token context, and a 56 GiB
container memory limit, with swap disabled. Its KV/activation allocation fraction
is 0.3 and CUDA graph capture is limited to batch one. ModelOpt safetensors load
directly; omit `--prepare`. DSpark is disabled. The bundled MTP draft is downloaded
with the checkpoint but unused. The full advertised 1M context is unqualified.

The API listens at `http://127.0.0.1:18082/v1`, with readiness at `/health` and
Prometheus metrics at `/metrics`. It provides OpenAI chat, image inputs, and the
Inkling reasoning and tool parsers. The checkpoint defaults to high reasoning
effort; clients can select another level through `chat_template_kwargs`.

```bash
curl http://127.0.0.1:18082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"thinkingmachines/Inkling-2-Mini","messages":[{"role":"user","content":"What is 17 * 19?"}],"max_tokens":512,"chat_template_kwargs":{"reasoning_effort":"none"}}'
```

Reproduce the short probe with:

```bash
python benchmarks/frameworks/bench_inkling.py \
  --base-url http://127.0.0.1:18082 --trials 3 \
  --output results/inkling-2-mini/probe.json
```

The probe checks arithmetic, constrained JSON, automatic tool selection, and a
synthetic red-square image. It saves full API responses, then warms up and measures
three greedy hash-table explanation requests capped at 1,024 tokens with requested
reasoning effort `none`. This is an effort hint: the measured model still emitted
reasoning. Token counts include reasoning and come from the server's final usage object. Streaming
decode throughput is `(completion_tokens - 1) / (last nonempty delta - first nonempty
delta)`; TTFT ends at the first nonempty content or reasoning delta. Output token
caps may terminate before EOS. These smoke checks do not establish broad quality,
long-context capacity, isolated performance, or endurance certification.

## GB10 probe — November 22, 2026

All five smoke checks passed: integer arithmetic, constrained JSON, automatic tool
selection, a tool-result round trip using a deterministic fixture, and recognition
of a synthetic red square. The reasoning parser returned separate reasoning and
answer fields. The runtime was `0.1.dev25855+g7cb5dae22` from the pinned ARM64 image.

| Output cap | Trials | Median decode tok/s | Median warm TTFT | Finish reason |
|---|---:|---:|---:|---|
| 256 | 3 | 37.45 | 0.164 s | length |
| 1,024 | 3 | 36.65 | 0.165 s | length |

These rates include reasoning tokens. Despite the requested effort of `none`,
the 256-token trials used 254 reasoning tokens and did not reach answer text;
the 1,024-token trials used 795 reasoning tokens and emitted partial answers.
Clients should budget output tokens for reasoning as well as the visible answer.
A separate request with a 2,048-token budget completed normally at EOS after 1,951
tokens (795 reasoning and 1,156 answer tokens), taking 54.02 seconds.
Two existing retrieval processes shared the GPU; these are local operational
measurements, not isolated framework comparisons. DSpark was disabled.

Full responses, timings, and checkpoint/runtime identity are in the
[1,024-token probe](../../benchmarks/frameworks/results/inkling-2-mini/probe.json),
[256-token probe](../../benchmarks/frameworks/results/inkling-2-mini/probe-256.json),
and [provenance](../../benchmarks/frameworks/results/inkling-2-mini/provenance.json).
The [complete answer](../../benchmarks/frameworks/results/inkling-2-mini/complete-answer.json)
and [versioned evidence](../../benchmarks/gb10/results/GB10-INKLING2-MINI-001.json)
are also retained.
The portfolio performance fields remain empty because this short shared-GPU probe
does not qualify the model for a product tier.
