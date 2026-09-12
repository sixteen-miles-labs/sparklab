---
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4.1-Flash
license: mit
library_name: sparklab
tags:
- deepseek-v4.1
- mxfp4
- mxfp8
- sparklab
- ftw
- speculative-decoding
---

# DeepSeek V4.1 Flash — SparkLab disk-ready checkpoint

This repository is a byte-preserving mirror of
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
at revision `df42c109f1defefcbfcedbe7d905718a12266e40`, published for the
[SparkLab](https://github.com/sixteen-miles-labs/sparklab) native NVIDIA DGX Spark
research path.

The tensors retain DeepSeek's original mixed checkpoint representation: routed experts
are MXFP4, dense projections are MXFP8 with UE8M0 scales, and embeddings and the output
head are BF16. No training, tensor conversion, or additional quantization was performed.
Although the repository uses the `FTW` deployment label, DeepSeek V4.1 currently runs
through SparkLab's model-owned safetensors reader and packed expert cache rather than the
generic FTW container format.

## Validated scope

SparkLab currently supports text-only, TP=1, batch-one eager execution with a 2,048-token
total context on one 128 GB NVIDIA GB10. The model owns a 64 GiB packed expert LRU and a
bounded 24 GiB weight cache, reading the remaining checkpoint from local NVMe.

The default target-only profile measured 0.969 decode tokens/s and 74.387 seconds warm
TTFT on a fixed 74-input/128-output greedy probe. Opt-in native greedy DSpark-5 measured
1.053 decode tokens/s and 76.329 seconds warm TTFT, an 8.6% decode improvement. Both
DSpark trials reproduced the target-only output hash. These are narrow performance
measurements, not general quality, concurrency, long-context, or endurance certification.

The upstream vLLM recipe uses probabilistic draft sampling, block rejection, and adaptive
verification. SparkLab's current V4.1 path implements greedy drafting and exact
accepted-prefix state commits; probabilistic sampling and adaptive verification remain
out of scope.

## Download and run

The repository is approximately 480 GiB. Download it to local NVMe:

```bash
hf download oakmindai/DeepSeek-V4.1-Flash-FTW \
  --local-dir ~/models/DeepSeek-V4.1-Flash-FTW
```

Install SparkLab from source, then start the target-only server:

```bash
git clone https://github.com/sixteen-miles-labs/sparklab.git
cd sparklab
./install.sh

SPARKLAB_DSV41_EXPERT_CACHE_GB=64 sparklab serve \
  --model ~/models/DeepSeek-V4.1-Flash-FTW \
  --dtype bfloat16 \
  --max-running-requests 1 \
  --max-seq-len-override 2048 \
  --num-tokens 2048 \
  --attention-backend triton \
  --cache-type naive \
  --moe-backend fused \
  --cuda-graph-max-bs 0 \
  --disable-startup-prefill-warmup \
  --port 8000
```

Enable the measured DSpark profile by adding:

```bash
--speculative-method dspark \
--speculative-tokens 5 \
--draft-sample-method greedy
```

Send a request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4.1-Flash-FTW",
    "messages": [{"role": "user", "content": "What is 17*19?"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

## Provenance and license

DeepSeek AI developed and released the architecture, code, tokenizer, and weights. This
mirror preserves the upstream MIT license and source files. SparkLab supplies the native
runtime, direct packed MXFP4/MXFP8 kernels, bounded disk-backed execution, DSpark
integration, validation, and model recipe. Oakmind AI publishes this mirror.

- Upstream model: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- Exact upstream revision: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40
- SparkLab: https://github.com/sixteen-miles-labs/sparklab
- vLLM architecture recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash

