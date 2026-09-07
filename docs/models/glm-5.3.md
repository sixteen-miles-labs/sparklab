# Run GLM-5.3

GLM-5.3 is an Experimental Research-tier NVFP4 recipe for complete-model,
NVMe-backed inference. The pinned Inferact checkpoint declares the same
`glm_moe_dsa` architecture and runtime dimensions as GLM-5.2, so SparkLab uses
the existing GLM-5.2 execution path.

The portfolio reports the **opt-in DFlash2-4** profile: **1.29 decode tok/s** and
**2.085 seconds warm TTFT** on a 256-token GB10 probe. This is 16.4% faster than
the saved same-length target-only baseline, not a fresh paired long-run comparison.
Exact greedy parity and completed-answer quality are not established.

**Target-only remains the default.** It measured **1.11 decode tok/s** and
**1.958 seconds warm TTFT** with a 2,850-slot expert cache: about 36% faster than
the previous target-only 256-token result. First-request TTFT was 25.858 seconds.

In separate target-only normal-EOS checks, AIME problem 0 completed correctly at 1.091 tok/s
(675-slot control: 0.815 tok/s). Problem 1 reached its 1,024-token cap without a
final answer. Cache layouts can change the generated text; no bit-exact parity
or broad quality certification is claimed. The recipe remains Experimental.

The target-only run retained at least 29.7 GiB available memory, with no scoped
OOM or swap-out. It used one request at a time, 2,048 KV tokens, FlashInfer b12x,
layer-LRU caching, route-only sparse prefill, and eager execution. The published
weights and recipe artifact version are unchanged; no checkpoint conversion is
needed for the cache optimization. Runtime admission reserves a conservative
96 GiB for the profile, separate from the planner's safety reserve.

## Install SparkLab

Follow the [full installation guide](../install.md). On NVIDIA DGX Spark:

```bash
uv venv && source .venv/bin/activate
uv pip install "sparklab[accel]"
sparklab --version
```

## Prepare

Use fast local NVMe storage. By default, the recipe downloads the pinned, validated
FTW artifact directly, so no local conversion is required. The FTW payload is
428,713,099,264 bytes (about 399.3 GiB). Source conversion remains available for
reproducibility; budget about 1.03 TB when keeping both source and prepared artifacts.

```bash
sparklab doctor --storage-path /path/to/models
sparklab plan glm-5.3 --root /path/to/models --prepare
sparklab pull glm-5.3 --root /path/to/models --prepare
```

Review the exact storage and runtime admission output from `plan` before continuing.

## Run

```bash
SPARKLAB_DISK_READ_WORKERS=20 sparklab run glm-5.3 --root /path/to/models
```

Wait for the API to listen on `127.0.0.1:1919`, then verify it:

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/models
```

The measured profile used 20 disk readers, set explicitly above. Expect
Research-tier throughput. See the
[cache optimization evidence](../../exps/exp_research_cache_optimization.md),
[original GLM-5.3 experiment](../../exps/exp_glm5_3_full_gb10.md), and the
[quick start](../quickstart.md) for more detail.

## Experimental DFlash2-4

Requires a [source installation](../install.md#method-2-install-from-source);
the released 0.1.2 wheel does not include full GLM speculative decoding.
Acquire the separate original BF16 draft from
[incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2), pinned to
`425aa615ce320caac34400208b30808c8f14f76c`. It adds approximately 4.58 GiB of
weights and is not bundled with or automatically downloaded by the target recipe.
The model card specifies **CC BY-NC-ND 4.0**, research/evaluation use and separate
commercial licensing; review these restrictions before use.

See the [speculative decoding report](../../exps/exp_glm53_speculative_decoding.md#reproduction)
for pinned acquisition and the complete 256-token benchmark command. Use
`--speculative-method dflash2 --speculative-tokens 4` and
`--speculative-draft-model /path/to/original-bf16-draft` with the report's settings.
Block 4 means three proposed tokens plus the target anchor. Full GLM speculation
is currently greedy-only, batch-one, TP=1 and eager.

The measured DFlash2-4 run retained at least 23.375 GiB available memory, with no
OOM, memory-guard trigger, swap growth or swap-out. Its output differs from the
saved target-only control, and neither capped run completed the answer. These
single-prompt results do not establish general quality, long-context stability,
or certification. Native BF16 MTP was also implemented and tested, but did not
improve throughput on the measured disk-offloaded profile.
