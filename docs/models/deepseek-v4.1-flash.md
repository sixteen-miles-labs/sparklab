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
total context. Prefill processes tokens sequentially. Vision, prefix reuse,
CUDA graphs and DSpark speculation are not implemented.

The model owns a 24 GiB device weight LRU and reads Engram embedding rows from disk
only when requested. Quantized projections are dequantized during computation;
the stored MXFP4 expert weights and MXFP8 dense weights remain packed in the cache.
The recipe's 32 GiB memory budget includes an estimated allowance for transient
projections, request state and scheduler KV storage. It is an engineering budget,
not a measured full-model memory result. The engine's `fused` MoE setting prevents
generic expert-bank allocation; execution still uses this model's disk reader.
Generic MoE cache and offload tuning flags do not tune this reader.

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

The native GB10 README probe measured **0.244 decode tok/s**, **343.131 s warm
TTFT**, and **864.682 s total request time**, medians of three trials after one
warmup. The unchanged AIME prompt encodes to 74 input tokens and generates 128
output tokens, with greedy sampling and thinking enabled. All three output hashes
match. See [versioned benchmark evidence](../../benchmarks/gb10/results/GB10-DSV41-PORTFOLIO-001.json).

This complete-checkpoint run uses the 24 GiB weight cache and eager disk decoder
above. Host swap-out was observed. It is a bounded performance probe, not a
completed-answer quality check. General numerical parity, generated tool calls,
agent tasks and endurance remain unverified. No V4 certification transfers to V4.1.

## Architecture provenance

The implementation follows the [pinned DeepSeek inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40/inference).
V4.1 differs from V4 in its shared compressed KV, two-stage index selection,
Engram memory, shifted hyper-connection coefficients, and 32-element quantization.
The model therefore has a separate native registration and never aliases the V4
decoder. Adapted configuration and hashing code retain the publisher's MIT license.
