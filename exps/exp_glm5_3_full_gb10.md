# GLM-5.3 NVFP4 on NVIDIA GB10

Last updated: 2026-08-29

## Result

The complete pinned `Inferact/GLM-5.3-NVFP4` checkpoint was downloaded,
converted to SparkLab's FlashInfer b12x FTW layout, and served through the
OpenAI-compatible API on one NVIDIA GB10. The selected 256-token request decoded
at **0.813 tok/s** with **2.530 s warm TTFT**. It completed without an OOM or
swap-out growth.

This is successful complete-checkpoint loading and a bounded performance result,
not a correctness or certification result. The output hit its 256-token limit
after deriving the correct divisibility reduction and first valid base, but
before stating the expected final answer `70`. Compact evidence:
[`GB10-GLM53-RESEARCH-001`](../benchmarks/gb10/results/GB10-GLM53-RESEARCH-001.json).

## Artifact and system

| Component | Value |
|---|---|
| Source | `Inferact/GLM-5.3-NVFP4` |
| Revision | `ce67b36f3669192b5bb233819f0fda6c8a9837f8` |
| Source size | 464,867,183,339 bytes |
| Prepared artifact | FTW NVFP4 b12x, 428,713,099,264 bytes |
| FTW layout | 77 shards, 2,043 tensor entries, fingerprint `a0e799b03bceb4bf` |
| Engine revision | `039bc348f30608b08103d877c6d2a225617db136`, clean tracked worktree |
| Hardware | NVIDIA GB10, SM121, 128 GB unified memory, NVMe |
| Software | Linux 6.17.0-1014-nvidia; CUDA 13.0; PyTorch 2.11.0+cu130; Triton 3.6.0 |

The checkpoint declares `GlmMoeDsaForCausalLM` with the same execution geometry
as GLM-5.2: 78 transformer layers, 75 MoE layers, 256 routed experts, top-8
routing, hidden size 6,144, expert intermediate size 2,048, MLA latent KV, and
DSA sparse attention. SparkLab therefore reused the established GLM-5.2 runtime
path while keeping checkpoint-specific evidence separate.

## Conversion

The pinned snapshot finished downloading at 16:50 local time. Conversion used
one-layer bounded expert staging and completed in about 16 minutes:

```bash
CUDA_HOME=/usr/local/cuda \
SPARKLAB_CONVERT_PROGRESS=1 \
sparklab checkpoint \
  --model /path/to/glm-5.3/source/ce67b36f3669 \
  --out /path/to/glm-5.3/prepared/0.1.0-b12x \
  --dtype bfloat16 \
  --moe-backend offload \
  --nvfp4-backend flashinfer \
  --shard-gib 8 \
  --device cuda:0
```

Routed experts remain in the checkpoint's NVFP4 precision and are repacked for
the SM12x FlashInfer b12x backend. The GLM resident-weight policy converts large
BF16 attention, dense-MLP, and output projections to per-row W8A16 FP8.

## Method

Both runs used AIME-25 problem 0, batch size one, greedy sampling, one warm
request, and one measured request. The configuration exactly follows the
selected GLM-5.2 b12x recipe: 675 layer-LRU expert slots, 20 disk readers, no
pageable host expert LRU, one staging layer, sparse prefill through 256 tokens,
shared-expert overlap, a 2,048-token KV allocation, and eager decode.

```bash
SPARKLAB_DISK_READ_WORKERS=20 \
python benchmarks/bench_decode_moe.py \
  --model /path/to/glm-5.3/prepared/0.1.0-b12x \
  --recipe glm-5.3 \
  --backend offload --storage disk --nvfp4-backend flashinfer \
  --host-cache-gb 0 --cache 675 --cache-policy layer_lru \
  --decode 256 --num-tokens 2048 --mem-ratio 0.90 \
  --disable-prefill-overlap --prefill-sparse-max-tokens 256 \
  --shared-expert-overlap --collect-moe-stats \
  --greedy --no-graph --include-output --server-timeout 1200
```

## Measurements

| Requested decode | tok/s | ms/token | Warm TTFT | p50 / p99 | Miss rate | Physical I/O | Output hash |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.832 | 1,202.5 | 2.962 s | 1,223.8 / 1,420.7 ms | 74.47% | 568.03 GiB | `e19ef21a678b` |
| 256 | 0.813 | 1,230.6 | 2.530 s | 1,233.3 / 1,447.3 ms | 75.49% | 2,294.97 GiB | `bdbf1a17f134` |

The selected request returned 256 completion tokens and 255 timed decode steps.
Minimum `MemAvailable` was 71.85 GiB; board power averaged 16.64 W and peaked at
17.77 W; GPU and NVMe temperature peaks were 47 C and 52.85 C. The measured
request read 7,882 pages from pre-existing swap but wrote zero pages out, had
zero scoped cgroup OOM kills, and contained no OOM markers.

## Output validation

The 64-token output ended after converting the two base-`b` numbers. The
256-token output correctly reduced the condition to `(b + 7) | 56`, selected
divisors 28 and 56, and derived `b = 21`; it ended while beginning the second
case. Because it did not state both bases or the requested sum `70`, SparkLab
records correctness as not evaluated rather than inferring a pass from the
partial reasoning. No native GLM-5.3 control hash was produced.

## Verdict

- Complete source acquisition, FTW conversion, model loading, DSA execution,
  sparse prefill, and 256-token decoding succeeded.
- The measured 0.813 tok/s clears the experiment's 0.5 tok/s usability floor
  but remains far below the 5 tok/s Frontier threshold.
- The request had no swap-out growth, but the machine began with roughly
  3.8 GiB of swap in use and performed swap-in reads, so this is not a clean
  zero-swap certification environment.
- Correctness, context, endurance, API/parser, quality, and coding-agent gates
  remain outstanding. The recipe remains Experimental Research fallback.

Raw local results are
`$HOME/results/glm5.3-gb10/selected-b12x-cache675-decode64.jsonl` and
`$HOME/results/glm5.3-gb10/selected-b12x-cache675-decode256.jsonl`.
