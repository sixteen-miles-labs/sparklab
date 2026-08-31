# Qwen3.6 cross-framework benchmark

This benchmark compares batch-one, OpenAI-compatible serving of Qwen3.6-35B-A3B
on one NVIDIA DGX Spark. FreeToken and SparkLab share the exact FTW NVFP4 artifact;
vLLM and SGLang share NVIDIA's ModelOpt NVFP4 checkpoint; llama.cpp and Ollama share
an official Ollama Q4_K_M artifact. Results between groups are operational comparisons,
while results within each shared-weight group are directly controlled.

## Results

| Framework | Version | Checkpoint format | Status | Decode tok/s | Warm TTFT | Reason when not measured |
|---|---:|---|---|---:|---:|---|
| FreeToken | 0.1.2 (`4b94bdc`) | FTW NVFP4 | Measured | 68.72 | 0.286 s | Baseline |
| SparkLab | 0.1.0 | Same FTW NVFP4 | Measured | 70.11 | 0.279 s | +2.0% decode vs FreeToken |
| vLLM | 0.28.0 | ModelOpt NVFP4 safetensors | Measured | 69.96 | 0.079 s | — |
| SGLang | 0.5.18 | ModelOpt NVFP4 safetensors | Measured | 33.48 | 0.087 s | — |
| llama.cpp | 0.3.0-dev (9723942ad) | Official Ollama Q4_K_M GGUF | Measured | 78.39 | 0.058 s | — |
| KTransformers | N/A | Official Ollama Q4_K_M GGUF | Unavailable | — | — | `kt-kernel` 0.7.0.post1 publishes no ARM64 wheel |
| Ollama | 0.33.2 | Official registry Q4_K_M | Measured | 94.02 | 0.060 s | — |
| Ollama runner, MTP disabled | 0.33.2 | Same official Q4_K_M GGUF | Measured | 69.53 | 0.060 s | Controlled no-speculation run |

## Method

- Hardware: one NVIDIA DGX Spark, GB10, 128 GB coherent memory, ARM64 Linux.
- Workload: one fixed reasoning prompt, batch/concurrency 1, greedy sampling, a
  32-token warmup, then three 256-token measured requests.
- Warm TTFT: request start to the first non-empty streamed content or reasoning delta.
- Decode tok/s: `(completion_tokens - 1) / (last token time - first token time)`.
- Token count: the server's final streamed OpenAI usage object. A server that does not
  return exact usage is not assigned an estimated result.
- Each framework runs alone. A missing implementation, unsupported architecture or
  quantization, build failure, and OOM are distinct statuses with logs retained under
  `results/qwen3.6-35b-a3b/`.
- Ollama's native row includes its automatically selected Qwen MTP speculative
  decoding. The `ollama-no-mtp` control invokes Ollama's bundled CUDA runner with
  otherwise matched settings and deliberately omits `--spec-type draft-mtp`.

FreeToken and SparkLab were freshly run through this harness with identical flags:
full-resident expert cache, Triton NVFP4 expert kernels, one CUDA-graph batch, 32K
sequence capacity, and the same immutable FTW bytes. FreeToken is pinned to upstream
commit `4b94bdc38a46a4dfe534e8793126160d56904c44`.

## Reproduce

For vLLM and SGLang, download the immutable NVFP4 checkpoint:

```bash
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --revision 491c2f1ea524c639598bf8fa787a93fed5a6fbce \
  --local-dir "$HOME/models/frameworks/qwen3.6-35b-a3b/hf"
```

Install and run one framework:

```bash
benchmarks/frameworks/setup_framework.sh vllm
python benchmarks/frameworks/run_framework.py vllm
```

Run the controlled FreeToken/SparkLab pair:

```bash
benchmarks/frameworks/setup_framework.sh freetoken
python benchmarks/frameworks/run_framework.py freetoken
python benchmarks/frameworks/run_framework.py sparklab
```

For Ollama, pull `qwen3.6:35b-a3b`. Its verified GGUF weight layer can be passed to
llama.cpp and KTransformers with `--gguf`, keeping those native-weight runs on the
same bytes. Use `--model`, `--ftw`, `--gguf`, `--ollama-model`, `--port`, and `--output-dir`
to override local paths. KTransformers cannot currently be installed from its
official wheel release on this ARM64 host.
