# Native Qwen image inputs

Qwen3.6-35B-A3B, Qwen3.8-27B and Qwen3.8-Flash-Next have an **experimental, opt-in native image path**.
It loads the publisher's BF16 vision tower alongside the existing prepared text
checkpoint and accepts OpenAI `image_url` content parts containing base64 data URLs.
The text decoder stays in SparkLab; the image tower and preprocessing use the
installed Transformers Qwen implementation.

## Start a vision server

Install the vision dependencies in the source environment:

```bash
uv sync --extra vision
```

Keep the pinned source snapshot, including its safetensors, `config.json`,
`preprocessor_config.json` and tokenizer files. The source must correspond to the
prepared text checkpoint: NVIDIA revision `491c2f1ea524c639598bf8fa787a93fed5a6fbce`
for Qwen3.6, RadixArk revision `319f741cce68d7914884900c138a1fbb70a42f30` for Qwen3.8-27B,
or NVIDIA revision `fab0aecb760cec45227f6656abcaafa11abca87a` for Flash-Next.
`--vision-model` points to that local source directory. It reads only vision
weights from the source and adds their GPU allocation to the text model's memory.

For Qwen3.8-27B:

```bash
sparklab serve --model /path/to/qwen27/prepared \
  --vision-model /path/to/qwen27/source \
  --nvfp4-backend triton --attention-backend triton \
  --num-tokens 4096 --max-seq-len-override 4096 \
  --host 127.0.0.1 --port 8000
```

For Qwen3.6-35B-A3B, use the retained native Triton FTW (recipe 0.5.0), or
prepare a separate Triton artifact from the pinned NVIDIA source:

```bash
sparklab checkpoint --model /path/to/qwen36/source \
  --out /path/to/qwen36/vision-text-ftw --nvfp4-backend triton
sparklab serve --model /path/to/qwen36/vision-text-ftw \
  --vision-model /path/to/qwen36/source \
  --moe-backend offload --moe-storage disk --moe-host-cache-gb 0 \
  --moe-cache-size 10240 --moe-preload-all \
  --nvfp4-backend triton --attention-backend triton \
  --num-tokens 4096 --max-seq-len-override 4096 \
  --host 127.0.0.1 --port 8000
```

The released Qwen3.6 Marlin container needs a new build containing these source
changes before it can use this path. Its default MTP4 text profile and the
Qwen3.8 DFlash2 text profile have separate performance evidence.

For Qwen3.8-Flash-Next, use the prepared NVIDIA text artifact and its matching
source snapshot:

```bash
sparklab serve --model /path/to/flash-next/prepared \
  --vision-model /path/to/flash-next/source \
  --moe-backend offload --moe-storage disk --moe-host-cache-gb 0 \
  --moe-cache-size 24576 --moe-preload-all \
  --nvfp4-backend triton --attention-backend qsa \
  --num-tokens 8192 --max-seq-len-override 8192 \
  --host 127.0.0.1 --port 8000
```

Flash-Next inserts image features before its residual streams split. Both its
attention keys and pooled sparse-index keys use image spatial positions through
prefill and decode. The existing FTW stays usable; no Hugging Face weight update
is required when supplying the matching local source through `--vision-model`.

## Send an image

Wait until `/health` returns `"status": "ok"`. Then:

```python
import base64
import json
import urllib.request
from pathlib import Path

image = base64.b64encode(Path("example.png").read_bytes()).decode()
body = {
    "model": "qwen",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
        {"type": "text", "text": "Describe this image."},
    ]}],
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

## Bounds and evidence

This path supports still images, up to four per request, base64 data URLs only,
16 MiB encoded per image, and 16 megapixels before preprocessing. Images resize
to at most 1,048,576 pixels. Remote image URLs, video and Anthropic image inputs
are not implemented. Image prompts must fit in one prefill chunk and in the
configured prompt-plus-output context budget.

Enabling vision selects TP=1, one active request, eager execution and target-only
decoding. Image requests bypass shared prefix caching, including on repeated
prompts, and carry spatial positions through decode. Text-only servers retain
their existing execution profiles.

Qwen3.6-35B-A3B and Qwen3.8-27B passed five color/order/repeat image checks and four API checks
(streamed image follow-up, malformed-image rejection, remote-URL rejection and
text generation after images) on GB10. Both failed the single OCR probe: expected
`SPARK 42`, observed `SPARK2` on Qwen3.8 and `PSP 3000` on Qwen3.6. The same Qwen3.6 checkpoint in vLLM 0.28.0 also failed this exact OCR input,
returning `B E A U T Y`. This is bounded integration evidence, not general
vision-quality or full-model reference parity.
See [the exact results](../../benchmarks/gb10/results/GB10-QWENVISION-001.json).

Flash-Next passed the same five color/order/repeat checks and four API checks.
It also passed both image-order checks at 2,081 prompt tokens, exercising sparse
QSA. Its OCR response was `spark42`, failing the expected `SPARK 42` text.
See [Flash-Next results](../../benchmarks/gb10/results/GB10-QWENVISION-002.json).
These checks validate the native image path within the tested bounds; they do
not establish broad vision quality.
