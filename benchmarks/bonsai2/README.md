# Native Ternary Bonsai 2 on DGX Spark

Initial implementation measured September 18, 2026. This is an experimental native implementation,
not a wrapper around llama.cpp. Text and still images use SparkLab's Qwen
hybrid decoder, multimodal path, and scheduler. Prism's fork is the reference.

## Sources and environment

- Model: `prism-ml/Ternary-Bonsai-2-27B-gguf`, revision
  `6ed5e12bf84b7a63069882c91dd9e9218647d17b`.
- Reference: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp),
  commit `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`.
- Native: changes on top of SparkLab `92d03c0`; see `results/provenance.json`
  for file hashes of the implementation being delivered.
- One NVIDIA GB10, DGX Spark, ARM64, driver 580.126.09; PyTorch
  2.11.0+cu130, Triton 3.6.0, Transformers 5.16.1.
- Both servers load the same Q8_0 image projector. Text remains packed on GPU;
  native vision weights expand to BF16. Large native prefills temporarily expand
  one text projection at a time for cuBLAS. Decode unpacks inside Triton kernels.
- Native: BF16 compute/KV, FP32 recurrent state, Triton attention, eager
  execution, 2,048-token prefill chunks, 12,288-token context, one active request,
  no speculation or cross-request prefix caching. Reference: all layers on GPU,
  flash attention, 12,288-token context, one slot, prompt caching disabled.
- GPU requests were serialized between engines. Both engines and a pre-existing
  idle vLLM server remained resident; the host had pre-existing swap. These are
  development-host measurements, not an isolated certification or memory soak.

The [vLLM isolation follow-up](ISOLATION.md) repeats the comparison with one
Bonsai backend resident at a time, tests before/after stopping vLLM, and adds
official-style `llama-bench` measurements. It found no decoding speed benefit
from stopping the idle vLLM server.

The current default includes the [native decode optimization](OPTIMIZATION.md):
26.95 tok/s on the short probe and 25.22 tok/s on the 8K subset. The table below
preserves the initial implementation baseline and its original launch flags.

## Measured performance

The table below is generated from the adjacent JSON reports. Decode is
`(completion_tokens - 1) / (last_text_chunk_time - first_text_chunk_time)`;
TTFT is client-observed time to the first text/reasoning chunk. Medians, one
client, temperature zero, thinking enabled, and exact output lengths with EOS
ignored. Initialization and one warmup are excluded.

| Packing / engine | Short decode tok/s | Short TTFT (s) | 8K decode tok/s | 8K TTFT (s) |
|---|---:|---:|---:|---:|
| PQ2 / native | 16.69 | 0.600 | 15.85 | 9.281 |
| PQ2 / prism | 29.49 | 0.318 | 27.39 | 8.885 |
| PTQ1 / native | 20.05 | 0.573 | 18.95 | 9.116 |
| PTQ1 / prism | 33.98 | 0.405 | 30.93 | 19.193 |

Short: original AIME-25 problem 0, 96 input / 128 output tokens, three measured
trials. This is the same short-prompt protocol used for the existing portfolio.
Long: the first **three** measured records and first warmup from the existing
30-request SPEED-Bench 8K corpus; 8,383 / 8,835 / 8,207 input tokens and exactly
256 output tokens. This is a development subset, not the full 30-request run.
Long-prompt JSONs record corpus hashes and question IDs. Reference reports show
zero cached prompt tokens. TTFT includes API, template, scheduling, prompt
processing, and the first token; it is not an isolated prefill-kernel time.

PTQ1 is the native default: it is smaller and faster in these tests. Native
decode still trails Prism substantially; this support change does not claim
performance parity. PQ2 remains supported as an alternative packing.

## Correctness checks and limits

- Both packings: **8/8** text/image smoke cases pass in both engines. Cases
  cover arithmetic, capital, JSON, code, simple logic, OCR, color, and an image
  change that must change the OCR answer.
- PTQ1: **3/3** additional checks pass in both engines: two images in one
  request, structured tool invocation, and the tool-result round trip.
- Both packings: five diagnostic prompts have identical tokenizer IDs and
  **5/5** matching next-token argmax against Prism. Top-100 log-probabilities
  are close but not identical; full values and errors are saved. This is not
  perplexity, benchmark accuracy, or general quality equivalence.
- Real packed blocks from five matrices (409,600 values per format) match
  Prism's compiled CPU dequantizer exactly after BF16 conversion. This caught
  and corrected PTQ1's scale-at-end layout during development.
- Kernel tests cover both packings, decode and prefill matmul, embeddings, and
  the signed Hadamard transform. Loader/tokenizer/recipe tests cover the new
  format and integration seams. See provenance for final test results.
- Single-request development profile only: four still images maximum, data
  URLs, no video, no qualified 262K context, no speculative decoding, no
  general quality-parity or certification claim. Image prompts must fit the
  2,048-token single-prefill budget. Multiple clients queue.

The small screen uses simple expected-answer checks; the raw responses are
included so they can be inspected. It does not certify instruction following,
reasoning accuracy, tool reliability, or vision accuracy on real documents.

## Reproduce

These are the original client and launch commands. Current source includes the
[optimized kernel](OPTIMIZATION.md); use that report and `--cuda-graph-max-bs 1`
to reproduce the current default, rather than the historical table above.

Install the SparkLab source with its vision extra. Acquire the recipe with
`sparklab pull bonsai2-27b`. This downloads PTQ1 and the Q8 projector. For PQ2,
download that file at the same model revision separately.

Build the pinned reference:

```bash
git clone https://github.com/PrismML-Eng/llama.cpp prism-llama
cd prism-llama
git checkout 1a07bfa5f4144274c8f1c9963821dd9d9a51854b
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j 8 --target llama-server llama-bench
```

For each packing, launch native on port 8093:

```bash
sparklab serve --model /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --vision-model /models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --served-model-name bonsai2 --host 127.0.0.1 --port 8093 \
  --attention-backend triton --cuda-graph-max-bs 0 \
  --max-running-requests 1 --cache-type naive \
  --max-seq-len-override 12288 --num-tokens 12288 --max-prefill-length 2048
```

And Prism on port 8092:

```bash
prism-llama/build/bin/llama-server \
  -m /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --mmproj /models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  -ngl 99 -fa on -c 12288 -np 1 --host 127.0.0.1 --port 8092 \
  --reasoning off --no-cache-prompt --cache-ram 0
```

`--reasoning off` prevents implicit reasoning on the small screen; the timed
clients explicitly override `enable_thinking=true`. Execute each engine's
requests serially. Substitute port 8092 to test Prism:

```bash
python benchmarks/bonsai2/compare.py --url http://127.0.0.1:8093 --output screen.json
python benchmarks/bonsai2/extra_checks.py --url http://127.0.0.1:8093 --output extra.json
python benchmarks/bench_single_stream.py --base-url http://127.0.0.1:8093 \
  --model bonsai2 --tokens 128 --trials 3 --workloads aime --thinking \
  --label native-ptq1 --output short.json
python benchmarks/bonsai2/latency.py --url http://127.0.0.1:8093 \
  --label native-ptq1 --dataset /path/to/measured.jsonl \
  --warmup /path/to/warmup.jsonl --warmups 1 --limit 3 --output long.json
```

The long probe requires the previously materialized SPEED-Bench corpus from
`results/qwen27-shipping-20260909/`; its SHA-256 hashes are recorded in every
long report. Source dataset is `nvidia/SPEED-Bench`, `throughput_8k`, revision
`487aa718444e816458d1a0a52bfce7a454285cf4`. Selection is documented in the
[earlier comparison](../../exps/exp_qwen36_speedbench_vllm_gb10.md).

For numerical diagnostics, run `benchmarks/bonsai2/capture_server.py` in place
of `sparklab serve`, with the same arguments and
`BONSAI_CAPTURE=/tmp/bonsai2-capture`. This diagnostic wrapper records one
prefill's last-token logits only when the client creates its request marker:

```bash
python benchmarks/bonsai2/logit_check.py \
  --model /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --output logits.json
python benchmarks/bonsai2/check_packing.py \
  --model /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --library prism-llama/build/bin/libggml-base.so --output packing.json
PYTHONPATH=python python -m pytest tests/bonsai2/test_kernels.py -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python -m pytest \
  tests/bonsai2/test_format.py tests/sparklab tests/models/test_models_registry.py \
  tests/multimodal -q
```
