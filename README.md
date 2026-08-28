<h1 align="center">SparkLab</h1>

<p align="center">
  <a href="https://github.com/NVIDIA/Model-Optimizer">
    <img src="https://img.shields.io/badge/NVIDIA-Model%20Optimizer-76B900?logo=nvidia&amp;logoColor=white" alt="NVIDIA Model Optimizer">
  </a>
</p>

> Run frontier open-weight models privately on one NVIDIA DGX Spark.

SparkLab is a GB10-native inference product for local, single-system deployments. It
combines immutable model recipes, unified-memory admission, resumable checkpoint
acquisition, FTW preparation, NVMe-backed MoE execution, and OpenAI- and
Anthropic-compatible APIs.

## Supported production target

SparkLab deliberately supports one narrow hardware profile:

- NVIDIA GB10 Grace Blackwell Superchip (`SM121`)
- 128 GB coherent unified memory
- ARM64 Linux or DGX OS
- NVIDIA driver r580 or newer and CUDA 13 toolkit
- Local NVMe storage for checkpoints, FTW artifacts, and disk-backed experts
- One local DGX Spark; multi-node and high-concurrency serving are outside the Beta scope

Platform, memory, swap, dependency, and storage requirements fail closed before a recipe
launch. Unsupported hardware may still work through engine compatibility paths, but it is
not a SparkLab support claim.

## Why SparkLab

- **GB10 admission:** `sparklab doctor` validates architecture, CUDA, unified memory,
  swap, dependencies, NVMe backing, and free capacity with human and JSON output.
- **Immutable recipes:** acquisition pins model revisions, records manifests, validates
  prepared artifacts, and rejects mismatched provenance.
- **Unified-memory planning:** runtime admission budgets weights, expert cache, KV and
  recurrent state, workspaces, and an operating-system reserve without counting swap as
  capacity.
- **Frontier models beyond memory:** FTW expert banks and bounded NVMe-backed caching let
  selected MoE checkpoints exceed physical memory.
- **Stable local APIs:** OpenAI Chat Completions, Responses, and Anthropic Messages share
  one loopback endpoint with streaming, reasoning, and tool-call support.
- **Evidence-bound claims:** model status and performance point to versioned,
  complete-checkpoint GB10 records rather than inferred capability.

## Model portfolio

| Model | Parameters | Quantization | Status | tok/s | TTFT(s) |
|---|---|---|---|---:|---:|
| **Fast — routine chat, editing, and short agent loops** |  |  |  |  |  |
| [Qwen3.6-35B-A3B](https://huggingface.co/oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW) | 35B total / 3B active | NVFP4 · FTW | Experimental | 67.46 | 0.320 |
| **Frontier — quality-first coding, reasoning, and long agent work** |  |  |  |  |  |
| [Qwen3.8-Flash-Next](https://huggingface.co/oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW) | 125B LM + 55B auxiliary / 6B active | NVFP4 · FTW | Experimental | 12.58 | 0.786 |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 284B total / 13B active | DS-FP4 | Preview | 9.22 | 14.045 |
| [GLM-5.3 Flash](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) | 320B total / 18B active | NVFP4 · FTW | Experimental | 4.46 | 5.760 |
| **Research — bounded execution outside the interactive envelope** |  |  |  |  |  |
| [GLM-5.2](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | 753B total / 40B active | NVFP4 | Experimental | 0.80 | 2.570 |
| [Kimi K3](https://huggingface.co/nvidia/Kimi-K3-NVFP4) | 2.8T total / 16 of 896 experts | NVFP4 · FTW pending | Experimental | — | — |

Displayed values are fixed, batch-one probes—not certification or capacity promises:

- [Qwen3.6 evidence](benchmarks/gb10/results/GB10-QWEN36-FAST-001.json)
- [Qwen3.8 evidence](benchmarks/gb10/results/GB10-QWEN38-NVFP4-001.json)
- [DeepSeek V4 evidence](benchmarks/gb10/results/GB10-BASELINE-001.json)
- [GLM-5.3 evidence](benchmarks/gb10/results/GB10-GLM53-NVFP4-001.json)
- [GLM-5.2 experiment](exps/exp_glm5_2_gb10.md)

Status meanings:

- **Experimental:** implementation or measured evidence exists, but required gates remain
  incomplete or failed.
- **Preview:** a bounded supported path exists, but the full release promise is incomplete.
- **Certified:** the exact recipe, revision, artifact, and release environment passed all
  required correctness, parser, agent, context, latency, memory, NVMe, and endurance gates.

Run `sparklab models --json` for exact recipe versions, checkpoint revisions, artifact
fingerprints, implementation state, evidence IDs, and known constraints.

## Documentation

- [Installation](docs/install.md)
- [Quick start](docs/quickstart.md)
- [Model catalog and runtime behavior](docs/models.md)
- [CLI reference](docs/cli.md)
- [Migration from the legacy `ft` surface](docs/migration.md)
- [GB10 benchmark evidence](benchmarks/gb10/README.md)
- [Release scope and acceptance gates](docs/beta-0.1-release-plan.md)

## Credits and citation

SparkLab incorporates the [FreeToken](https://github.com/FlashML-org/FreeToken) research
work and builds on open inference projects including
[mini-sglang](https://github.com/sgl-project/mini-sglang),
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm), and
[llama.cpp](https://github.com/ggml-org/llama.cpp).

If you use the underlying engine in research, cite the
[FreeToken paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## License

[Apache License 2.0](LICENSE).
