# Qwen3.6 GB10 container profile

This default Qwen3.6 container profile runs SparkLab's scheduler and model with vLLM 0.28's
Marlin expert kernels and matching weight transforms on GB10. Recipe 0.6.0
selects this profile through `sparklab run qwen3.6-35b-a3b`. The preview label
retains the documented quality limitation. The container uses PyTorch 2.13 and source-built
SparkLab extensions; the normal SparkLab environment remains on PyTorch 2.11.
It does not install the incompatible PyTorch 2.11 sglang-kernel wheel.

Build from the repository root:

```sh
docker build -f benchmarks/qwen36_marlin/Dockerfile -t sparklab-qwen36:gb10-v1 .
```

Convert the original pinned NVIDIA checkpoint into a **separate** FTW directory.
Replace the two host paths below with the original safetensors directory and
an empty output directory. Do not point the output at the certified artifact.

```sh
docker run --rm --gpus all --shm-size 16g \
  -v /path/to/original-checkpoint:/model:ro \
  -v /path/to/new-marlin-ftw:/artifact \
  --entrypoint python3 sparklab-qwen36:gb10-v1 -c \
  "from sparklab.checkpoint.convert import convert_checkpoint; convert_checkpoint('/model', '/artifact', nvfp4_backend='marlin', device='cuda:0')"
```

Launch the converted artifact with the checked profile:

```sh
python benchmarks/qwen36_marlin/serve.py --model /path/to/new-marlin-ftw
```

`--dry-run` checks the artifact metadata and shard sizes and prints the command.
The launcher uses the fixed configuration below and caps the advertised total
sequence length at 32,832 tokens. This includes the prompt and generated output;
32,768 input tokens leave 64 tokens for generation. It targets text inference on
one 128 GB GB10, with one actively decoded request; additional clients queue.
Greedy requests use MTP4. Sampled requests use target-only decoding.

The equivalent Docker configuration is:

```sh
docker run --rm --gpus all --network host --shm-size 16g \
  --memory 96g --memory-swap 96g \
  --env-file benchmarks/qwen36_marlin/gb10.env \
  -v /path/to/new-marlin-ftw:/artifact:ro \
  sparklab-qwen36:gb10-v1 serve \
  --model /artifact --served-model-name qwen36-bench \
  --moe-backend offload --moe-storage disk --moe-preload-all \
  --moe-cache-rate 1.0 --nvfp4-backend marlin --num-tokens 32832 \
  --max-seq-len-override 32832 \
  --attention-backend fi --speculative-method mtp --speculative-tokens 4 \
  --port 1929
```

The startup log must show `Resolved expert bank layout: nvfp4_marlin` and a
completed preload. `/health` must report JSON `"status": "ok"` before sending
requests; an HTTP 200 response alone can describe a server that is still loading.
An explicit backend that disagrees with an FTW artifact's
stored layout now fails instead of silently using the stored backend.

The GB10 environment file enables target and draft CUDA graphs, immediate
drafting after partial rejection, and small transactional state snapshots.
It also selects vLLM's Marlin vocabulary projection and skinny CUTLASS FP8
projections, preserving the checkpoint's quantized weights and per-row scales.
The startup log must report a captured five-row MTP verification graph;
draft graphs are captured on demand during warmup. The ordinary request-batch
graph runner still reports disabled because verification uses its own runner.
These options require this donor image and full expert preload.

See [the experiment report](../../exps/exp_qwen36_speedbench_vllm_gb10.md)
for the fixed SPEED-Bench inputs, AIPerf command, numerical checks and results.
The measured runs retain their exact commands and runtime versions in the raw
artifact directory. Historical speed experiments are separate from the completed
[quality and one-hour reliability checks](QUALIFICATION.md). The fast profile
is now the default by explicit user choice; quality equivalence remains unestablished.

The environment also enables the measured native BF16 draft projections, a
fused renormalized top-k router, and deferred recurrent-state commits. Deferred
commits reuse the accepted verification state and materialize it before other
state consumers; recurrent precision stays FP32. The explicit Torch-router
override is disabled so the local fused kernel is selected.

The cancellation-fixed image's two final runs measured **111.93 and 112.24
decode tok/s**, versus **110.64 and 112.56** for vLLM. Their equal-weight means
were **112.08 versus 111.60 tok/s**. Mean request latency was **3.735 versus
3.664 seconds** (2.0% higher), and mean TTFT was **1.436 versus 1.367 seconds**.
All four runs used the same 30-prompt, 8K-input/256-output, concurrency-one workload.
See [the qualification record](QUALIFICATION.md) for every repeat and limitation.
The fixed image ran packaged source with no source-tree mount.

The unsuccessful FP8 KV/XQA experiments were removed from the runtime patch.
Their source snapshots and measurements remain in the raw artifact directory.
The selected recipe uses BF16 KV and FA2 attention. The previous native artifact
remains available for explicit target-only serving.

The launcher sets equal Docker memory and memory-plus-swap limits to disable
swap for the serving container. The 96 GiB cgroup ceiling is separate from CUDA's
reserved-memory reporting. An earlier uncapped run swapped cold CPU pages during
another server's startup despite ample available host memory.


## Qualification and reproduction

The [qualification record](QUALIFICATION.md) includes the frozen three-engine
quality comparison, 775 passing runtime/kernel tests, 14 grader tests, and the
completed one-hour run with 1,411 passing requests and zero container swap.

The PyTorch 2.13 container is now the default; the previous native PyTorch 2.11
profile remains available as an explicit fallback. The donor runtime
can produce different greedy answers: the quality comparison found a changed
JSON key even with the original FTW and native Triton kernels inside the newer
container. Throughput parity does not establish quality equivalence.

The serving validation covers reasoning-channel output, executable bounded
Python functions, JSON instructions, typed tool calls and tool-result round trips,
exact 1K/8K/16K/32K recall, output budgets, queued clients, cancellation, sampled
fallback, overlength rejection, and sustained serving. The first five AIME-25
problems are a separate capped greedy sample, not a representative quality score.

`prepare_validation.py` freezes the same cases and token-counted contexts for
both profiles. Run it inside the pinned image with a local AIME-25 JSONL file:

```sh
docker run --rm --entrypoint python3 \
  -v /path/to/new-marlin-ftw:/artifact:ro \
  -v /path/to/aime25.jsonl:/aime25.jsonl:ro \
  -v "$PWD/benchmarks/qwen36_marlin:/validation:ro" \
  -v /path/to/evidence:/evidence \
  sparklab-qwen36:gb10-v1 /validation/prepare_validation.py \
  --model /artifact --aime /aime25.jsonl --output /evidence/corpus.json

python benchmarks/qwen36_marlin/validate_serving.py \
  --corpus /path/to/evidence/corpus.json \
  --output /path/to/evidence/candidate --duration-seconds 3600
```

Use a fresh output directory for every run. `--quality-only` runs the same quality
cases against a separately launched reference; `--server-kind vllm` supports its
health and reasoning-field conventions. `--cgroup` optionally records the serving
container's scoped swap and OOM counters. `summarize_validation.py` rescores both
profiles with one grader and reports introduced failures without hiding existing
model errors or truncated answers.

The final validation also adds a cancellation regression fix: a discarded
verification block can advance device state beyond the client-visible tokens.
Cleanup now avoids donating that mismatched recurrent state and returns every
reserved KV page. The earlier speed image did not contain this fix.
