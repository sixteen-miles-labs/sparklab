# DGX Spark model-serving demo

## Test system

| Component | Configuration |
|---|---|
| System | NVIDIA DGX Spark |
| Processor / GPU | NVIDIA GB10 |
| Memory | 128 GB coherent CPU/GPU memory (121.7 GiB reported usable) |
| CPU / OS | 20-core ARM64, Ubuntu Linux |
| CUDA | CUDA 13.0 |
| Workload | OpenAI-compatible chat completion, batch/concurrency 1 |
| Context | 32K sequence capacity |
| Measurement | One 32-token warmup, then three 256-token streamed requests |
| Decode metric | Median `(completion tokens - 1) / decode time` |
| Safety | Frameworks run one at a time with a 12 GiB host-memory reserve |

## Results

| Model | Parameters | Quantization used | vLLM | llama.cpp | SparkLab |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 35B total / 3B active | vLLM and SparkLab: NVFP4; llama.cpp: Q4_K_M GGUF | **69.96 tok/s** | **78.39 tok/s** | **70.11 tok/s** |
| Qwen3.8-Flash-Next | 125B language model + approximately 55B auxiliary / 6B active | vLLM: publisher ModelOpt NVFP4; llama.cpp: official Q8_0 GGUF; SparkLab: FTW NVFP4 | **Not run:** 135 GB checkpoint exceeds usable coherent memory before runtime and KV overhead | **Not run:** 163 GB official GGUF exceeds usable coherent memory | **16.42 tok/s** |

## Interpretation

- Qwen3.6 results are not fully quantization-controlled: vLLM and SparkLab use
  NVFP4, while llama.cpp uses Q4_K_M. They represent practical runtime choices,
  not a pure kernel comparison.
- SparkLab's Qwen3.8 result uses QSA, batch-one CUDA-graph replay, and a full
  immutable preload of all 24,576 routed experts. Decode-time expert disk staging
  is disabled.
- “Not run” is not a zero score. The corresponding official artifact could not be
  loaded while retaining the benchmark's operating-system and runtime memory reserve.
- The Qwen3.8 SparkLab startup emitted five recoverable NVIDIA
  `NV_ERR_NO_MEMORY` warnings. There was no Linux OOM kill, the server completed
  startup and all trials, but the configuration has a narrow device-memory margin.

## Evidence

- [Qwen3.6 cross-framework report](README.md)
- [Qwen3.6 vLLM result](results/qwen3.6-35b-a3b/vllm.json)
- [Qwen3.6 llama.cpp result](results/qwen3.6-35b-a3b/llama-cpp.json)
- [Qwen3.6 SparkLab result](results/qwen3.6-35b-a3b/sparklab.json)
- [Qwen3.8-Flash-Next report](QWEN38_FLASH_NEXT.md)
- [Qwen3.8 SparkLab result](results/qwen3.8-flash-next/sparklab.json)
