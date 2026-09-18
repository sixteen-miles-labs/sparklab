# Ternary Bonsai 2 27B

Experimental **native SparkLab** text and still-image support for
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
pinned to `6ed5e12bf84b7a63069882c91dd9e9218647d17b`.
This implementation requires the source changes adding Bonsai support; it is not
included in the published SparkLab 0.1.3 wheels.

## Run

From a source installation containing this change:

```bash
pip install -e '.[vision]'
sparklab pull bonsai2-27b
sparklab run bonsai2-27b
```

The recipe downloads only `Ternary-Bonsai-2-27B-PTQ1_0.gguf` and
`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`, plus the model's notices. Their combined
weight size is 6,575,895,904 bytes. No FTW conversion or `--prepare` is needed.
Vision is enabled by default in this recipe.

To select the alternative PQ2_0 packing, download it at the same revision and use
the direct server interface (replace the paths below):

```bash
sparklab serve \
  --model /models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --vision-model /models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --served-model-name bonsai2 \
  --attention-backend triton --cuda-graph-max-bs 0 \
  --max-running-requests 1 --cache-type naive \
  --max-seq-len-override 12288 --num-tokens 12288 \
  --max-prefill-length 2048
```

For text-only serving, omit `--vision-model`. The same command accepts PTQ1_0.
The BF16 projector is also supported by the loader, but the integration tests
and published measurements use the shipped Q8_0 projector.

## Image requests

Use OpenAI `/v1/chat/completions` with `text` and `image_url` content parts.
Images must be base64 `data:image/...;base64,...` URLs. For example:

```python
import base64
import requests

image = base64.b64encode(open("receipt.png", "rb").read()).decode()
response = requests.post("http://127.0.0.1:1919/v1/chat/completions", json={
    "model": "prism-ml/Ternary-Bonsai-2-27B-gguf",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Read the total on this receipt."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
    ]}],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": False},
}, timeout=120)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

The existing Qwen image limits apply: at most four still images, no remote image
URLs or video, at most 16 megapixels per image and 16 MiB per encoded data URL.
Image prompts must fit in one prefill: 2,048 tokens including image placeholders
with the default recipe. Pure-text prompts may span multiple prefill chunks.

## Implementation and limits

- Qwen hybrid attention and request scheduling run in SparkLab. Prism's
  llama.cpp fork is used only as the validation reference.
- Both custom ternary formats stay packed while resident. Decode unpacks inside
  the GPU kernel. Large prefills temporarily dequantize one projection for a
  BF16 cuBLAS multiplication; there is no full-model BF16 copy.
- The loader restores linear-attention V-head ordering and applies the exact
  metadata-specified signed block Hadamard transforms, including inverse
  transforms after embedding lookup. Unknown rotation conventions fail closed.
- The Q8 vision tower is dequantized to BF16 at load and uses the native Qwen
  multimodal path. This adds approximately 0.9 GB of weights, plus activations.
- BF16 compute, TP=1, one active request, no speculative decoding. Multiple
  clients queue. Text decode uses CUDA graphs; image requests use the eager MRoPE path.
- The recipe caps context at 12,288 tokens. The model's advertised 262K limit has
  not been qualified here. No general quality-parity or certification claim.

The optimized PTQ1 default measured **26.95 tok/s** on the 96-input/128-output
single-client probe and **25.22 tok/s** on the three-request 8K/256 subset.
The FP32 reduction order changed, so generated outputs are not bit-identical
to the initial implementation. See the [optimization report](../../benchmarks/bonsai2/OPTIMIZATION.md)
for quality checks and the remaining gap to Prism.

Reproduction tools and measured results are in
[benchmarks/bonsai2](../../benchmarks/bonsai2/README.md).
