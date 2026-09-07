# Full GLM-5.3 speculative decoding on one GB10

Status: implementation and bounded local evaluation complete; **experimental,
not promoted**. Both paths are opt-in,
greedy, batch-one, TP=1, eager only. This is **full GLM-5.3**, not GLM-5.3 Flash.
The portfolio reports the opt-in DFlash2-4 256-token measurement (1.29 tok/s,
2.085 s warm TTFT). Recipe defaults, target weights, and published weight artifacts
are unchanged; this is not a default-profile or certification promotion.

## Outcome

**DFlash2 block 4 is the best measured candidate**, but exact greedy parity has
not been established. Keep target-only as the default. Native BF16 MTP did not
improve throughput on the tested disk-offloaded profile.

| Profile | Expert slots | Output tokens | Decode tok/s | Warm TTFT |
| --- | ---: | ---: | ---: | ---: |
| Target-only, matched MTP control | 2,100 | 64 | 1.011 | 2.166 s |
| Native MTP1 | 2,100 | 64 | 0.992 | 1.183 s |
| Native MTP3 | 2,100 | 64 | 0.819 | 1.179 s |
| Target-only, fresh selected-cache control | 2,850 | 64 | 1.137 | 2.058 s |
| DFlash2 block 8 | 2,850 | 64 | 1.124 | 2.075 s |
| DFlash2 block 4 | 2,850 | 64 | 1.360 | 2.012 s |
| Target-only, saved selected profile | 2,850 | 256 | 1.108 | 1.958 s |
| DFlash2 block 4 | 2,850 | 256 | 1.290 | 2.085 s |

DFlash2-4 gained **19.6%** against the fresh, same-length 64-token control.
The longer probe gained **16.4%** against the previously saved 256-token control
in `GB10-GLM53-RESEARCH-OPT-002-candidate256.json` (not a fresh paired 256-token
control). These are single-prompt probes, not cross-workload performance claims.

The 256-token candidate accepted 187/202 drafts (**92.6%**), emitted 3.71 outputs
per target forward, and performed zero target replays. It read 1,306.90 GiB
versus the saved target's 1,356.89 GiB: the I/O reduction is modest even though
the number of target forwards falls sharply. Minimum available memory was
23.375 GiB; no guard trigger, OOM, swap growth, or swap-out was observed (three
system-wide swap-in pages were recorded).

The candidate's 256-token hash `af173145b58e` differs from the saved control's
`5b69d85cc226`. Both visible continuations reach the divisor-of-56 argument, but
wording and the truncation point differ. Neither capped run completes the answer;
this is **not** a quality pass. Resolve target/block numerical consistency and
run completed-answer and broader workload checks before any promotion. The draft
license also needs consideration for deployment beyond research/evaluation.

Evidence index: `benchmarks/gb10/results/GB10-GLM53-SPEC-003.json`.

## Fixed target and separate drafts

Target: `glm-5.3` recipe, NVFP4 b12x FTW fingerprint `a0e799b03bceb4bf`.
The selected target-only profile remains 2,850 expert-cache slots / 20 disk
readers, previously measured at 1.108 tok/s for 256 output tokens.

| Path | Pinned source | Local representation |
| --- | --- | --- |
| Native MTP | `Inferact/GLM-5.3-NVFP4@ce67b36f3669192b5bb233819f0fda6c8a9837f8` | Original BF16 layer 78, five `mtp_bf16-*.safetensors` shards |
| DFlash2 | `incoai/GLM-5.3-DFlash2@425aa615ce320caac34400208b30808c8f14f76c` | Original BF16 six-layer drafter, `model.safetensors` |

The target FTW omits MTP weights. Acquisition downloads only the source index,
config, and five MTP shards, not the full upstream target checkpoint again.
The MTP tensors remain resident and do not participate in target expert-cache
admission. The first MTP runs use 2,100 target cache slots to accommodate the
roughly 18.54 GiB BF16 draft while preserving the 12 GiB available-memory guard.
DFlash2 weights are about 4.58 GiB and are evaluated with the selected target cache.

The [DFlash2 model card](https://huggingface.co/incoai/GLM-5.3-DFlash2)
specifies **CC BY-NC-ND 4.0**, research/evaluation use and separate commercial
licensing. These tests load the original weights locally; no draft or converted
weights are published. Its four-GB300 benchmark is not a single-Spark prediction.

## Runtime implementation

- Native MTP uses normalized token embeddings and target hidden states, the
  checkpoint's `eh_proj`, one resident MoE/MLA draft layer, and its final norm.
  Recursive proposals reuse that layer. It owns separate latent and index-K rows;
  the target still has 78 layers and 75 disk-backed MoE layers.
- DFlash2 captures six post-layer target features, projects them to draft context,
  and predicts a noncausal masked block. Its BF16 projections reuse the existing
  SparkLab DFlash2 convolution/selector implementation. The target's FP8 language
  head is projected with its row scales; it is not mistaken for a raw BF16 head.
- DFlash2's separate GQA cache shares physical token locations with target MLA
  rows. Allocation, byte budgeting, and rebuild cover target latent, index, and
  draft K/V together. Its direct SDPA does not require an MLA backend to run GQA.
- Verification is causal. Full GLM has no recurrent state to restore: rejected
  suffix rows are hidden by the accepted length and overwritten on continuation.
  Native MTP refreshes the retained shifted-token draft prefix with the correction;
  DFlash2 uses only retained target features. Neither path replays the target.

## Reproduction

Acquire to a separate local directory (accept the upstream license before use):

```python
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path.home() / "models/speculative/glm-5.3"
snapshot_download(
    "Inferact/GLM-5.3-NVFP4",
    revision="ce67b36f3669192b5bb233819f0fda6c8a9837f8",
    allow_patterns=["config.json", "model.safetensors.index.json", "mtp_bf16-*.safetensors"],
    local_dir=root / "mtp-bf16-ce67b36f3669",
)
snapshot_download(
    "incoai/GLM-5.3-DFlash2",
    revision="425aa615ce320caac34400208b30808c8f14f76c",
    allow_patterns=["config.json", "README.md", "model.safetensors"],
    local_dir=root / "dflash2-425aa615ce32",
)
```

Common benchmark settings: disk offload, b12x/FlashInfer, layer-LRU, no host
expert LRU, 20 readers, 2,048 KV tokens, memory ratio 0.90, no CUDA graph,
no prefill overlap, sparse prefill threshold 256, shared-expert overlap,
greedy AIME25 #0 prompt, 64-token capped warm throughput probe, output capture,
MoE statistics, and `--min-memory-gib 12`. Run benchmarks sequentially, with no
concurrent downloads or other GPU work.

Native MTP additions:

```bash
export SPARKLAB_GLM_DSA_MTP_PATH="$HOME/models/speculative/glm-5.3/mtp-bf16-ce67b36f3669"
# Add to benchmarks/bench_decode_moe.py's common target options:
# --cache 2100 --speculative-method mtp --speculative-tokens 1
```

DFlash2 additions:

```bash
# --cache 2850 --speculative-method dflash2 --speculative-tokens 8 \
# --speculative-draft-model "$HOME/models/speculative/glm-5.3/dflash2-425aa615ce32"
```

For MTP, `speculative-tokens` counts proposed tokens. For DFlash2 it is the
whole block, including the one target anchor: block 8 proposes 7 tokens.
Full GLM currently permits MTP 1–7 and DFlash2 blocks 2–8.

Complete DFlash2-4 probe command (after acquisition):

```bash
SPARKLAB_DISABLE_KERNEL_CACHE_VERSION_CHECK=1 SPARKLAB_DISK_READ_WORKERS=20 \
.venv/bin/python benchmarks/bench_decode_moe.py \
  --model "$HOME/models/models/glm-5.3/prepared/0.1.0-b12x" \
  --recipe glm-5.3 --backend offload --storage disk --nvfp4-backend flashinfer \
  --host-cache-gb 0 --cache-policy layer_lru --cache 2850 \
  --decode 256 --num-tokens 2048 --mem-ratio 0.90 \
  --disable-prefill-overlap --prefill-sparse-max-tokens 256 \
  --shared-expert-overlap --collect-moe-stats --greedy --no-graph --include-output \
  --server-timeout 1200 --request-timeout 1800 --min-memory-gib 12 \
  --speculative-method dflash2 --speculative-tokens 4 \
  --speculative-draft-model "$HOME/models/speculative/glm-5.3/dflash2-425aa615ce32" \
  --json /tmp/glm53-dflash4-256.json
```

## Evaluation boundaries

First matched-cache measurement (64 output tokens): target-only at 2,100 slots
was **1.011 tok/s**, versus **0.992 tok/s** for MTP1. MTP1's 77.1% acceptance
therefore did **not** produce a speedup. Physical reads increased from 396.36
to 431.04 GiB for the measured request. Warm TTFT fell from 2.166 to 1.183 s.
MTP1's output hash `e19ef21a678b` matches a saved selected-cache control, but
differs from this matched-cache target-only run (`e04fe1589d9a`). Do not claim
universal exact parity from the earlier matching control.

MTP1 lifecycle minimum available memory was 25.456 GiB, with no guard trigger
or OOM. System-wide swap grew 16 KiB (five pages out, one page in); the benchmark
cgroup's swap declined slightly. Preserve these counters rather than describing
the run as literally zero-swap. The existing system swap is not model capacity.

Initial evidence:

- `benchmarks/gb10/results/GB10-GLM53-SPEC-003-target-c2100-64.json`
- `benchmarks/gb10/results/GB10-GLM53-SPEC-003-mtp1-c2100-64.json`
- `benchmarks/gb10/results/GB10-GLM53-SPEC-003-mtp3-c2100-64.json`
- `benchmarks/gb10/results/GB10-GLM53-SPEC-003-sources.json`

MTP3 at the same 2,100 slots measured **0.819 tok/s**, 1.179 s warm TTFT,
52.8% acceptance, and 2.46 outputs per target forward. It performed no replays,
but read **540.15 GiB** for 64 output tokens. Its output hash was
`e19ef21a678b`. Increasing depth did not overcome the disk traffic penalty.

DFlash2 block 8 at the selected 2,850 slots measured **1.124 tok/s**, 2.075 s
warm TTFT, 81.5% acceptance, and 5.82 outputs per target forward. It read
**341.58 GiB** for 64 output tokens, with no replays. Output hash was
`e19ef21a678b`. Lifecycle minimum available memory was 22.389 GiB; no swap-in,
swap-out, guard trigger, or OOM was observed. This is effectively level with the
saved 1.114 tok/s selected-cache 64-token control, not a demonstrated speedup.
Evidence: `benchmarks/gb10/results/GB10-GLM53-SPEC-003-dflash8-c2850-64.json`.

DFlash2 block 4 at 2,850 slots measured **1.360 tok/s**, 2.012 s warm TTFT,
100% acceptance (47/47), and 3.76 outputs per target forward. It read **322.83
GiB** for 64 output tokens, with no replays and output hash `e19ef21a678b`.
The fresh target-only 2,850-slot control measured **1.137 tok/s**, so the short
probe gain is **19.6%**. The 256-token result is summarized above. Evidence:
`benchmarks/gb10/results/GB10-GLM53-SPEC-003-dflash4-c2850-64.json`.
Fresh control: `benchmarks/gb10/results/GB10-GLM53-SPEC-003-target-c2850-64.json`.
The control's output hash was `e04fe1589d9a`, not the speculative run's hash.
Thus **exact greedy parity is not established**, despite the match with an older
saved target-only control. Block-shape numerical differences are a risk with the
current kernels; the end-to-end difference has not been fully isolated. Do not
promote this path or call it lossless on the strength of speed/acceptance alone.

Compare MTP both against target-only at the **same 2,100-slot cache**, to isolate
speculation, and against the selected 2,850-slot deployment. Compare DFlash2 at
the selected 2,850-slot cache. Record acceptance, outputs per target forward,
replay count, physical disk bytes, memory minimum, swap delta, TTFT, and output
hash alongside speed. A high acceptance rate alone is not a speedup.

These short capped probes are not completed AIME answers or quality certification.
Greedy block verification may differ numerically from single-token kernels;
exact text parity is a measured property of each run, not a universal promise.
Any promotion requires broader correctness, long-context, and completed-answer
evaluation. The target-only profile remains the default pending those checks.

## Implementation checks

- Tracked CPU suite plus new model and portfolio tests: 1,135 passed, 78 skipped. An unrelated
  untracked document-demo test references a missing demo file and was not included.
- Focused engine/cache regression suite: 378 passed, 9 skipped.
- Focused GPU suite: 19 passed, covering native MTP prefill, three recursive steps,
  rejection recovery below and above the sparse top-k boundary, DFlash2 proposal
  execution and poisoned-suffix exclusion, and DSA cache allocation/rebuild.
- CPU engine tests cover zero, partial, and full acceptance for both new paths,
  assert no target replay, and verify retained lengths and correction proposals.
- Draft loaders are strict about keys and shapes, and private draft state is
  excluded from target checkpoint state. DFlash2 target/draft cache bytes match
  the startup cost model before and after rebuild.

The installed kernel-cache wheel has a pre-existing 0.1.0 versus runtime 0.1.2
version mismatch; local checks use `SPARKLAB_DISABLE_KERNEL_CACHE_VERSION_CHECK=1`,
as do the existing GB10 baseline commands. No wheel/package replacement is part
of this experiment.
