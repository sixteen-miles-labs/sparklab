# DeepSeek V4.1 Flash — native Research

`deepseek-v4.1-flash` is an **Experimental native SparkLab text implementation**.
It uses the shared serving API and scheduler with a model-specific eager decoder.
The [upstream vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
is the architecture reference; this SparkLab recipe does not launch vLLM or Docker.

The source is `deepseek-ai/DeepSeek-V4.1-Flash`, pinned to
`df42c109f1defefcbfcedbe7d905718a12266e40`. The complete snapshot is
510,313,345,254 bytes (about 475 GiB). Publisher parameter counts are 522B total,
8B active per prompt token and 16B active per generated token; these describe the
publisher model, including capabilities beyond this initial text path.

## Run

Use a source installation containing this implementation and local NVMe storage:

```bash
sparklab plan deepseek-v4.1-flash
sparklab pull deepseek-v4.1-flash
sparklab run deepseek-v4.1-flash --dry-run
sparklab run deepseek-v4.1-flash
```

Omit `--prepare`: the decoder reads the original safetensors snapshot directly.
FTW conversion is unsupported. The native engine fixes this path to TP=1, one
request, BF16 computation, eager execution, naive prefix caching and a 2,048-token
total context. Prompt-wide Engram, hyper-connection and MoE work runs
layer-by-layer; attention and its projections still advance causally. Vision, prefix reuse, CUDA
graphs and DSpark speculation are not implemented.

The model owns a 64 GiB packed-expert LRU plus a bounded 24 GiB device-weight LRU,
and reads Engram embedding rows from disk only when requested. MXFP4 experts run
directly from their packed checkpoint representation; MXFP8 dense projections use
a packed small-row kernel. Eight parallel bounded reads fill missing expert slots.
`SPARKLAB_DSV41_EXPERT_CACHE_GB` can lower the expert budget, though values below
the 64 GiB measured profile may reduce decode speed. The recipe reserves 96 GiB
for these caches, transient projections, request state and scheduler KV storage.
The engine's `fused` MoE setting prevents generic expert-bank allocation;
generic FTW cache and offload controls do not tune this model-owned cache.

## API example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash",
       "messages":[{"role":"user","content":"What is 17*19? Return only the integer."}],
       "temperature":0,"max_tokens":32,
       "chat_template_kwargs":{"thinking":false}}'
```

The pinned checkpoint's `encoding/encoding.py` renders chat prompts. Native
tool parsing uses V4.1's spaced DSML `calls`, `invoke` and `parameter` tags,
including incremental argument streaming. The existing DeepSeek reasoning parser
handles its `<think>` protocol. Integer `chat_template_kwargs.reasoning_effort`
values from 1 through 100 are preserved; named effort support is probed from the
pinned encoder. SparkLab defaults to chat mode unless thinking or tools are
requested. Keep both prompt and output within the 2,048-token total budget.

## Validation and limits

The reduced six-layer reference fixture checks window wraparound, ratios 1 and 2,
shared KV/index sources, candidate filtering, shifted hyper-connections, Engram
lookup, and request reset. Its sequential logits match the publisher reference
using common Torch arithmetic helpers; this is a decoder-wiring check, not an
independent kernel-parity result. A synthetic checkpoint also generated tokens
through SparkLab's native `/v1/completions` endpoint on GB10.

All 96,085 tensor headers in the real pinned checkpoint were inspected without
loading their payloads; required text-tower shapes and quantization scales passed
the loader's geometry validation. The real tokenizer produces the expected
99,092-entry compressed vocabulary and Engram table sizes of 384,006,168 and
384,016,682 rows. Its prompt renderer preserves a numeric reasoning budget.

## Measured performance

The optimized native GB10 profile measured **0.969 decode tok/s**, **74.387 s warm
TTFT**, and **205.420 s total request time**, medians of three trials after one
full-length warmup. The unchanged AIME prompt encodes to 74 input tokens and
generates 128 output tokens, with greedy sampling, thinking enabled and EOS ignored.
This is 3.98x the prior decode rate, with 4.61x faster TTFT and 4.21x lower total
time. See [optimized evidence](../../benchmarks/gb10/results/GB10-DSV41-OPT-002.json)
and the [prior baseline](../../benchmarks/gb10/results/GB10-DSV41-PORTFOLIO-001.json).

All optimized trials produced the same output hash. The packed/fused arithmetic
did not reproduce the prior decoder's output hash, and both 128-token traces end
before the answer. This remains a bounded performance probe rather than a quality
result. General numerical parity, completed answers, generated tool calls, agent
tasks and endurance remain unverified. No V4 certification transfers to V4.1.

## Architecture provenance

The implementation follows the [pinned DeepSeek inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40/inference).
V4.1 differs from V4 in its shared compressed KV, two-stage index selection,
Engram memory, shifted hyper-connection coefficients, and 32-element quantization.
The model therefore has a separate native registration and never aliases the V4
decoder. Adapted configuration and hashing code retain the publisher's MIT license.
