# Run Qwen3.8-27B

An experimental native image-input path is available from source; see [vision setup and limits](qwen-vision.md).

Recipe 0.4.0 uses the pinned RadixArk NVFP4 checkpoint, FP4 prefill in 2K-token
chunks, and DFlash2-12 by default. It is an Experimental, text-only recipe for one
NVIDIA GB10. DFlash accelerates greedy requests; model-default sampling remains
unchanged and uses target-only generation.

## Prepare and run

Use the current source checkout from the [installation guide](../install.md),
including the `accel` dependencies. The measured environment used Torch
2.11.0+cu130, Triton 3.6.0, and FlashInfer 0.6.15.post1.
The measurements used `SPARKLAB_DISABLE_KERNEL_CACHE=1`. Use this setting with
a source checkout if an installed optional kernel-cache package has a different
version from SparkLab; otherwise startup rejects the mismatched cache.

```bash
sparklab pull qwen3.8-27b --root /path/to/models --prepare
sparklab run qwen3.8-27b --root /path/to/models
```

The pull command acquires and validates both immutable snapshots:

- Target: `RadixArk/Qwen3.8-27B-NVFP4`, revision
  `319f741cce68d7914884900c138a1fbb70a42f30`.
- Draft: `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`, revision
  `bd7a934213c47a9e7ef69eef36bb3325f47fd1f1`.

When upgrading, rerun `pull --prepare`. The new recipe does not reuse an old
Inferact checkpoint under the RadixArk model name. Previous prepared files remain
available; see the [previous Inferact profile](qwen3.8-27b-inferact.md) to reproduce it.

Use `temperature: 0` to exercise DFlash:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"RadixArk/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"Explain binary search."}],"temperature":0,"max_completion_tokens":512}'
```

The checkpoint defaults are temperature 1.0, top-k 20, and top-p 0.95. Sampled
requests use target-only generation. Multiple clients queue behind one active
request. To explicitly disable speculation, append `-- --speculative-tokens 0`
to `sparklab run`.

## Measured performance

The portfolio uses the original short AIME-25 problem 0 probe: 96 input tokens,
exactly 128 output tokens, thinking enabled, temperature 0, one client, one warmup
and three measured trials. The new default measured **49.52 tok/s** and
**0.125 s warm TTFT** (medians); individual decode trials were 49.47–49.54 tok/s.
This is 7.9% above the historical 45.88 tok/s result on the same short workload.
All three new outputs matched each other, but differed from the previous Inferact
output. This truncated speed probe does not measure completed-answer quality.
See the [short-probe evidence](../../benchmarks/gb10/results/GB10-QWEN38-27B-PORTFOLIO-001.json).

### Long-prompt comparison

Thirty fixed SPEED-Bench requests, roughly 8K input tokens and exactly 256 output
tokens, three warmups, thinking enabled, temperature 0, one client:

| Configuration | Decode tok/s | Mean first token | Mean request time |
|---|---:|---:|---:|
| Previous Inferact + DFlash2-12 | 22.19 | 7.53 s | 19.24 s |
| RadixArk target-only, FP4 prefill | 11.18 | 5.40 s | 28.20 s |
| **Default RadixArk + DFlash2-12, FP4 prefill** | **27.47** | **5.43 s** | **15.02 s** |
| vLLM 0.28, RadixArk target-only | 11.96 | 3.79 s | 25.11 s |

The default profile reduces request time by 21.9% against the previous native
profile. vLLM remains faster at prompt processing and target-only decode.
The speculative comparison uses native DFlash against vLLM target-only; it does
not measure vLLM with DFlash. The earlier 45.88 tok/s result used a short prompt
and is not comparable to this workload.

See the [full comparison and evidence](../../benchmarks/gb10/QWEN38_27B_FAST.md).

## Quality and limits

The expanded screen passed 70/74 checks for the fast native profile, versus
68/74 for the previous native profile and 68/74 for vLLM. There were no new
failures against the previous native control. The fast profile failed one
arithmetic check, one JSON check, HumanEval/116, and one AIME problem that exhausted
its token budget. vLLM passed HumanEval/116 and failed other cases. These sampled
results do not establish general quality equivalence.

Changing the checkpoint changes weight quantization. FP4 prefill also dynamically
quantizes activations, which changes numerics from native W4A16 and from the
checkpoint's calibrated activation scales. It retains an extra packed weight
layout. Native BF16 KV cache and FP32 recurrent state remain in use. To keep the
RadixArk checkpoint but restore W4A16 prefill, append
`-- --nvfp4-prefill-backend w4a16`; this is a separate performance profile.

All 29 serving checks passed, covering streaming budgets,
queueing, cancellation recovery, sampled generation, long context, and
overlength rejection. Normal EOS and tool roundtrips also passed in the quality
screen. Recall passed at 16K, 32K, and near 64K tokens. The
full 65,536-token boundary also passed with a 65,504-token prompt and exactly
32 generated tokens. Larger contexts and a 60-minute
endurance run remain unqualified. Full Fast certification is outstanding.

The recipe reserves a conservative 64 GiB runtime budget. Current recipe
admission requires zero host swap. The benchmark host had pre-existing swap and
some swap-in activity, with no measured swap-out; hardware verification used the
advanced `sparklab serve` command. Do not interpret these results as a zero-swap
certification run.
