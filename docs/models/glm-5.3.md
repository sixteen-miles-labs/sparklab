# Run GLM-5.3

GLM-5.3 is an Experimental Research-tier NVFP4 recipe for complete-model,
NVMe-backed inference. The pinned Inferact checkpoint declares the same
`glm_moe_dsa` architecture and runtime dimensions as GLM-5.2, so SparkLab uses
the existing GLM-5.2 execution path.

The complete checkpoint measured **1.11 decode tok/s** and **1.958 seconds warm
TTFT** on one NVIDIA GB10 with a 2,850-slot expert cache: about 36% faster than
the previous 256-token result. First-request TTFT was 25.858 seconds.

In separate normal-EOS checks, AIME problem 0 completed correctly at 1.091 tok/s
(675-slot control: 0.815 tok/s). Problem 1 reached its 1,024-token cap without a
final answer. Cache layouts can change the generated text; no bit-exact parity
or broad quality certification is claimed. The recipe remains Experimental.

The selected run retained at least 29.7 GiB available memory, with no scoped
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
