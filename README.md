# Spark Lab

> Turn one NVIDIA GB10 into a private frontier-AI workstation.

Spark Lab focuses on optimizing frontier open-weight inference for the NVIDIA
DGX Spark and its GB10 Grace Blackwell Superchip. It packages tested model
recipes, unified-memory diagnostics, NVMe-backed MoE execution, and
OpenAI-/Anthropic-compatible APIs for a single DGX Spark.

The supported production target is intentionally narrow:

- NVIDIA GB10 Grace Blackwell Superchip, SM121
- 128 GB coherent unified memory
- ARM64 Linux / DGX OS
- CUDA 13
- Local NVMe for checkpoints and disk-backed experts

Spark Lab uses versioned recipes and fail-closed admission: no recipe is called
Certified until its complete checkpoint passes the published GB10 correctness,
latency, context, stability, and agent gates.

## Why Spark Lab

- **Optimized for DGX Spark:** platform checks, model recipes, and launch policies
  target the GB10 Grace Blackwell Superchip, CUDA 13, and SM121.
- **Made for unified memory:** capacity planning uses the physical 128 GB GB10 memory
  pool rather than treating it as discrete VRAM.
- **Frontier beyond memory:** FTW and NVMe-backed expert streaming can address MoE
  checkpoints larger than physical memory.
- **Agent ready:** reasoning, tool calls, prefix reuse, and compatible APIs are
  first-class validation targets.
- **Measured, not implied:** product claims point to versioned GB10 evidence.

See the [quick start](docs/quickstart.md), [model catalog](docs/models.md),
[CLI reference](docs/cli.md), and [installation guide](docs/install.md).

## Model portfolio

Spark Lab has three recipe tiers:

| Model | Parameter | Quantization | Status | tok/s | TTFT(s) |
|---|---|---|---|---:|---:|
| **Fast — routine chat, editing, and short agent loops** |  |  |  |  |  |
| [Qwen3.6-35B-A3B](https://huggingface.co/oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW) | 35B total / 3B active | NVFP4 · FTW | Experimental | 67.46 | 0.320 |
| **Frontier — quality-first coding, reasoning, and long agent work** |  |  |  |  |  |
| [Qwen3.8-Flash-Next](https://huggingface.co/oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW) | 125B LM + 55B auxiliary / 6B active | NVFP4 · FTW | Experimental | 12.58 | 0.786 |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 284B total / 13B active | DS-FP4 | Preview | 9.22 | 14.045 |
| [GLM-5.3 Flash](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) | 320B total / 18B active | NVFP4 · FTW | Experimental | 3.27 | 7.121 |
| **Research — correct, bounded execution beyond the interactive envelope** |  |  |  |  |  |
| [GLM-5.2](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | 753B total / 40B active | NVFP4 | Experimental | 0.80 | 2.570 |
| [Kimi K3](https://huggingface.co/nvidia/Kimi-K3-NVFP4) | 2.8T total / 16 of 896 experts | NVFP4 · FTW upload pending | Experimental | — | — |

Model links point to the selected runtime artifact when one is published, otherwise to
the pinned source checkpoint. The Kimi K3 FTW artifact remains pending.

The measured results link to compact, machine-readable evidence:

- Qwen3.6: [`GB10-QWEN36-FAST-001`](benchmarks/gb10/results/GB10-QWEN36-FAST-001.json)
- Qwen3.8: [`GB10-QWEN38-NVFP4-001`](benchmarks/gb10/results/GB10-QWEN38-NVFP4-001.json)
- DeepSeek V4: [`GB10-BASELINE-001`](benchmarks/gb10/results/GB10-BASELINE-001.json)
- GLM-5.3: [`GB10-GLM53-NVFP4-001`](benchmarks/gb10/results/GB10-GLM53-NVFP4-001.json)
- GLM-5.2: [`experiment and admission result`](exps/exp_glm5_2_gb10.md)

These are fixed-probe measurements, not certification. Qwen3.6 still requires context
and endurance gates; DeepSeek V4 remains Preview; GLM-5.2 remains an Experimental
fallback after its selected trial recorded 680 KiB of swap-out.

GLM-5.3's full-checkpoint probe loaded and served successfully at 3.27 tok/s and
7.121 s warm TTFT. Its corrected greedy probe reached the reference answer, but the
performance and remaining certification gates do not pass Frontier admission. See the
[experiment report](exps/exp_glm5_3_nvfp4_gb10.md) for the FTW and backend findings.

The Qwen3.8 recipe targets Inferact's publisher-quantized ModelOpt NVFP4 checkpoint and
preserves its published precision during FTW preparation. Its complete-checkpoint probe
measured 12.58 decode tok/s and 0.786 s warm TTFT, passing the Frontier performance
thresholds; context, capability, quality, and endurance gates remain outstanding. The archived
[`GB10-QWEN38-FP8-001`](benchmarks/gb10/results/GB10-QWEN38-FP8-001.json) and
[`GB10-QWEN38-FRONTIER-001`](benchmarks/gb10/results/GB10-QWEN38-FRONTIER-001.json)
results remain historical and do not certify it.

Run `sparklab models --json` for exact revisions, recipe versions, implementation state,
evidence IDs, and limitations.

## Credits and citation

Spark Lab incorporates the [FreeToken](https://github.com/FlashML-org/FreeToken)
research work and builds on ideas and code from
[mini-sglang](https://github.com/sgl-project/mini-sglang),
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm), and
[llama.cpp](https://github.com/ggml-org/llama.cpp). We thank their authors and
maintainers for their contributions to open inference.

If you use Spark Lab in research, cite the software:

```bibtex
@software{sparklab2026,
  title={Spark Lab: Frontier Open-Weight Inference for NVIDIA DGX Spark},
  author={{Spark Lab Contributors}},
  year={2026},
  note={GitHub repository URL forthcoming}
}
```

If you use its FreeToken components, also cite the
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
