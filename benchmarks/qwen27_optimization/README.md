# Qwen3.8-27B optimization probes on DGX Spark

The original single-client comparison measured 8.92 decode tok/s with SparkLab
target-only, 22.05 with DFlash2-12, and 9.43 with vLLM target-only. Its roughly 8K
prompts took 7.10 seconds to reach the first token in SparkLab target-only versus
4.01 seconds in vLLM. See the [matched comparison](../gb10/QWEN38_27B_VLLM.md).

Profiling the native target found 95.9% of decode GPU time in linear kernels:
45.1% BF16 GEMV and 50.8% NVFP4 GEMV. Attention accounted for 2.3%, and recurrent
updates for 0.9%. The checkpoint leaves large GDN input projections and the output
head in BF16. Dense 27B generation therefore moves substantial weight data for
every token. DFlash acceptance also depends on the prompt; the earlier 45.88 tok/s
short-prompt probe is not the same workload as this long-prompt comparison.

## Exploratory results

One warmup and six fixed prompts, 8,207–8,835 input tokens, exactly 128 output
tokens, greedy generation, thinking enabled, one client. These are **not** the
30-request, 256-output-token measurements above. All profiles used the same prompt
text and reported the same prompt-token counts.

| Profile | Decode tok/s | First token | Complete request |
|---|---:|---:|---:|
| Inferact target, native | 9.12 | 7.07 s | 20.99 s |
| Inferact target, 512 MiB scratch | 9.02 | 6.62 s | 20.69 s |
| Inferact target, FP4 prefill, 2K chunks | 9.22 | 5.54 s | 19.32 s |
| RadixArk target, native | 11.23 | 7.56 s | 18.87 s |
| Inferact DFlash2-12 control | 22.23 | 7.33 s | 13.13 s |
| RadixArk DFlash2-12 | 26.26 | 8.09 s | 13.00 s |
| RadixArk DFlash2-12, FP4 prefill, 2K chunks | 25.66 | 5.35 s | 10.32 s |

The combined experiment reduced mean request time by 21.5% against the DFlash
control. Changing the checkpoint alone improved decode but barely changed total
request time. Kernel probes also found that replacing all native NVFP4 linears
with Marlin was not consistently faster, especially during large prefills.

The combined profile changes both weight quantization and activation numerics.
FP4 prefill uses dynamic activation scaling, not the checkpoint's calibrated
NVFP4 input scales. It retains an extra packed weight layout. Native FP32
recurrent state and BF16 KV cache remain in use. Recipe defaults are unchanged.

The reviewable launcher reproduced 10.31 s mean request time and all six
experimental output hashes. A separate regression screen passed 24/26 checks,
versus 23/26 for the Inferact DFlash control. Both failed the same arithmetic and
JSON-key cases; the candidate additionally passed the strict JSON-format case.
Both passed all reasoning, coding, tool, and four long-prompt checks. These are
small capability checks, not evidence of general quality equivalence with the
original checkpoint or vLLM. Neither profile passed every check.

## Reproduce the experimental server

Use the measured native environment: Torch 2.11.0+cu130, Triton 3.6.0,
FlashInfer 0.6.15.post1 on NVIDIA GB10. From a SparkLab source checkout:

```bash
hf download RadixArk/Qwen3.8-27B-NVFP4 \
  --revision 319f741cce68d7914884900c138a1fbb70a42f30 \
  --local-dir /path/to/radix-source
sparklab checkpoint --model /path/to/radix-source \
  --out /path/to/radix-prepared --nvfp4-backend triton --device cuda:0
hf download maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal \
  --revision bd7a934213c47a9e7ef69eef36bb3325f47fd1f1 \
  --local-dir /path/to/dflash-draft

Q27_W4A4_PREFILL=1 \
SPARKLAB_DISABLE_KERNEL_CACHE=1 \
SPARKLAB_DFLASH2_REJECT_DRAFT=1 \
SPARKLAB_DFLASH2_VERIFY_GRAPH=1 \
python benchmarks/qwen27_optimization/serve_probe.py \
  --model /path/to/radix-prepared --served-model-name qwen27-opt \
  --nvfp4-backend triton --cuda-graph-max-bs 1 \
  --cache-type radix --page-size 16 --max-running-requests 1 \
  --max-seq-len-override 65536 --num-tokens 65536 \
  --attention-backend triton --speculative-method dflash2 \
  --speculative-tokens 12 --speculative-draft-model /path/to/dflash-draft \
  --max-prefill-length 2048 --host 127.0.0.1 --port 1938
```

With that fresh server running, replay the frozen local benchmark inputs:

```bash
python benchmarks/qwen27_optimization/probe.py \
  --input-export results/qwen38-27b-vllm-20260908/sparklab-target-c1/profile_export_raw.jsonl \
  --output results/qwen27-reproduction.json
```

The client warms up with the seventh input and measures the first six in recorded
request order. The frozen input export is a local artifact from the earlier
comparison; its SHA256 is recorded in the evidence. The client saves full response
text under `results/`; do not commit those raw responses.

The experimental wrapper does not alter the normal server or the model recipe.
Long-context capacity, sampled generation, concurrent serving, and endurance have
not been qualified for this combination.

[Evidence, exact measured commands, source revisions, and hashes](../gb10/results/GB10-QWEN38-27B-OPT-001.json).
Raw probes remain under `results/qwen27-optimization-20260908/`.

## Expanded verification corpus

`prepare_quality.py` reconstructs the expanded screen from pinned datasets:
20 regression cases, four long-prompt variants, 20 GSM8K samples, 20 HumanEval
tasks, five AIME-25 problems, and three exact-length recall prompts. Successful
tool cases also receive a follow-up request, for 74 checks when both tool calls
succeed. GSM8K and HumanEval use seed `20260909`. Reasoning token budgets are
fixed; a response that exhausts its budget fails its answer check.

Download the GSM8K and compressed HumanEval files from the immutable URLs in
[quality_sources.json](quality_sources.json), decompress HumanEval to JSONL,
and obtain the pinned AIME input:

```bash
hf download math-ai/aime25 test.jsonl --repo-type dataset \
  --revision 563bb8404243c5f09de6ec262f2db674fe5bce9b \
  --local-dir /path/to/quality-data/aime

python benchmarks/qwen27_optimization/prepare_quality.py \
  --tokenizer /path/to/radix-source \
  --aime /path/to/quality-data/aime/test.jsonl \
  --gsm8k /path/to/quality-data/gsm8k.jsonl \
  --humaneval /path/to/quality-data/humaneval.jsonl \
  --output /path/to/quality-corpus.json \
  --operational-output /path/to/operational-corpus.json
```

The builder checks dataset hashes and counts tokenizer `input_ids`. With the
recorded RadixArk tokenizer, the corrected quality corpus SHA256 is
`00ec1857dd4c934161f54f0326168aca5c8a8da7193e92cb223c221dc28f1e6f`.
It contains recall prompts of exactly 16,384, 32,768, and 64,512 tokens.

```bash
python benchmarks/qwen27_optimization/verify_quality.py \
  --base-url http://127.0.0.1:1938 --model qwen38-bench \
  --server-kind sparklab --corpus /path/to/quality-corpus.json \
  --output /path/to/quality-results --quality-only
```

Use `--server-kind vllm` for the comparison server. To run protocol, context,
queueing, cancellation, and sampled-generation checks, use the operational
corpus with `--skip-quality` in place of `--quality-only`.

HumanEval solutions run against the official task assertions in a CPU-only Docker
container with network access disabled, a read-only filesystem, and resource
limits. The recorded interpreter image is
`sha256:a563337b36f29e11803737bca3c8833d6ed8d66a2efd244348cc9dc1bd53ba5e`,
from the [Qwen3.6 container build](../qwen36_marlin/Dockerfile). Another host can
set `SPARKLAB_QUALITY_IMAGE` to a pinned image containing `python3`; record that
image change with its results. A failed assertion or a model token-budget limit
is a scored failure, so the checker may exit with status 1 after completing the
screen. Inspect `summary.json` and the recorded responses before interpreting it.

These sampled checks do not establish general model-quality parity. Store full
prompts and responses outside the public source tree.
