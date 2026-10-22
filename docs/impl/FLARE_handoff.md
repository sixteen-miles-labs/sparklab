# FLARE implementation handoff and resumption

The original plan is **not complete**. This is a reviewable local implementation
on `feat/flare-runtime`, based on `715af2b3058f0b642bbb04c8d8a449e7816bba37`.
The original checkout's unrelated changes are preserved. Commit, push, PR and
merge are user-authorized; live default promotion and model certification remain
outside this change.

Latest artifact: `$HOME/results/flare-runtime/wheels-partial-render/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl`,
SHA-256 `5a70962d6d84554029da8982b634947cc812be5bbb16f349bce07504ea42e46e`.
All 436 production Python modules match source. Issue 99 is published and the
API rejection fix is applied with user authorization. Latest native CPU/MoE:
2339 passed, 241 skipped, six deselected; local JIT-disabled product: 1558 passed,
95 skipped, plus the later wire-order check in that mode. Paired 12-case 8K/32K
factual recall, eight long state cases, 32K restart and four-protocol stage
reconciliation pass. A supported-auto tool workflow still fails its baseline's
second required-call observation; no candidate session pass is claimed.
`qwen36-validation-v2.json` reconciles artifact/source and evidence hashes.
All earlier evidence retains its recorded artifact scope; performance, soaks,
hosted CI/DCO and rollout authorization remain open.

Subsequent user-authorized Qwen3.6 testing has produced additional complete-model
evidence: 32 state cases, 16 HTTP branch cases and a four-boot restart/disable
pass. The 20-turn application baseline fails its first required tool call;
candidate remains unrun. See [the Qwen3.6 handoff](FLARE_qwen36.md) and
`$HOME/results/flare-runtime/qwen36-validation.json`. Original DFlash, broad
quality, performance and soak gates remain open.

## Latest authorized performance and pressure results

The user stopped the primary owner; all six owned servers exited 143 afterward.
The first complete performance pair fails strict 32K branch output parity,
with 13/14 matching outputs and no failed/incomplete request. Warmup and measured
branch outputs agree within each mode. Zero pairs qualify; the remaining four
are not run. Pure controls were also contaminated by shared-prefix reuse.

Independent finite baseline, memory and disk pressure collections pass 63
completed requests and 21 deliberate cancellations including warmups. Each
profile performs six timed cycles and six idle rebuilds. Disk telemetry confirms
13 completed restores, six coalesced restores and ten completed writes. The
initial disk attempt exposed HTTP 503 busy handling in the collector; its fix
passes 37 focused checks and the fresh-store disk rerun passes. Production code
and the wheel are unchanged. See [full results](FLARE_qwen36_performance.md) and
`$HOME/results/flare-runtime/qwen36-validation-v3.json` for scopes and hashes.

Next diagnose the 32K branch divergence, correct the pure controls, and rerun
full paired trials. No release speed qualification is available. Cancellation
while a restore is pending, longer pressure/endurance, tracing overhead and
expert NVMe contention remain open. Keep adaptive DFlash disabled and checkpoint
reuse opt-in. Primary model owner is inactive; unrelated retrieval is preserved.

## Behavior and review units

Review the new and modified files together; `git diff --stat` excludes untracked
implementation files. The [design](FLARE_design.md) describes the contracts and
the [gate ledger](FLARE_gates.md) separates local proof from missing qualification.

| Review unit | Main locations | Behavior and eligibility |
| --- | --- | --- |
| M1 observations and harness | `core.py`, `message/`, `serving/`, `tokenizer/`, scheduler timing/export modules, `benchmarks/` | Bounded opt-in stage, queue, resource, cost and delivery observations; paired completed-work evidence |
| M2 scheduling | `runtime/scheduler/` policy, prefill and scheduler integration | Baseline retained; bounded prefill/decode choice, admission and TP phase agreement |
| M3 exact continuation | `runtime/engine/`, `runtime/kvcache/`, scheduler checkpoint modules | Target-only TP1 MHA+GDN state, immutable snapshots and private partial tails; existing multimodal exclusion retained |
| M4 disk retention | Prefix record/store modules and scheduler restore integration | Checksummed, versioned, bounded asynchronous storage, recovery, coalescing and conservative measured-cost restore |
| M5 speculation | Scheduler policy/feedback, graph/engine, `models/qwen3_5_moe/dflash2.py` | Opt-in prepared depth control, bounded acceptance/cost observations; fixed control retained |
| M6 profiling | `FLARE_memory.md` and external profile reports | Measured 4.51% upper bound misses the preselected 5% gate; no allocation optimization applied |
| M7 ownership and integration | `serving/launch.py`, scheduler shutdown, `utils/mp.py`, tests, CI workflow | Worker registration/reap, prefix drain before IPC closure, TP local failure cleanup, HTTP failure/backpressure and executable session/restart/soak checks |

These are native-runtime changes, not isolated vLLM/container optimizations.
The initial complete-state contract is the local Qwen3.8-27B MHA+GDN target.
Grouped target/draft checkpoints, multimodal reuse and other physical state
contracts remain excluded. No weights, precision, sampling policy or support
claims were changed. The pending Responses constraint rejection patch is an
external preview, not part of the production implementation.

## Artifact and evidence

The current wheel is
`$HOME/results/flare-runtime/wheels-spawn-gap/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl`,
SHA-256 `f4cdfb32eb66ce63bbf7f658bb6b4f87c379750cc7394db9e46c1305b20c7542`.
All 436 production Python modules match the source used to build it.
The combined native-JIT CPU selection passed **2250**, skipped **241**, and
deselected **6**. Source stop-gap checks passed 28; the prior artifact failed
seven negative controls and passed two controls. These overlapping counts must
not be summed. Earlier GPU evidence belongs to its recorded artifact; the latest
launcher-only change has no new CUDA claim.

Raw evidence is under `$HOME/results/flare-runtime/`. Start with
`m7-spawn-gap-validation.json`, `m7-spawn-gap-artifact.json`,
`m7-spawn-gap-wheel-cpu.log` and [validation history](FLARE_validation_history.md).
The latest guarded full-model preflight, `qualification-preflight.log`, has
**three memory-guard skips**, not three passes. `local-ci-product-wheel.log`
records the separate JIT-disabled local product selection: **1469 passed,
95 skipped**, 228.92 seconds. It is not hosted x86/CPU-torch CI evidence.

The cleanup report `derived-extraction-cleanup.json` records removal of 21
obsolete wheel extraction directories. Every wheel and raw log was retained;
the current extraction remains available. Recreate a historical extraction
before replaying an old-artifact command:

```sh
python -m zipfile -e PATH_TO_PRESERVED_WHEEL PATH_TO_OLD_EXTRACTION
```

## Resume complete-model correctness first

Use fresh Python processes and fresh evidence destinations. The **30 GiB free
physical GPU memory floor before every boot is mandatory**; larger profiles may
need a higher floor. Do not count host available memory or swap as GPU capacity.
Do not stop unrelated VLLM/retrieval services. Keep at least 3 GiB disk free for
the restart fixture. The current host is near that disk limit.

From the worktree, set the common environment:

```sh
export PYTHONPATH="$HOME/results/flare-runtime/wheels-partial-render/extracted"
export SPARKLAB_DISABLE_JIT=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLARE_TARGET="$HOME/.sparklab/models/qwen3.8-27b/prepared/experimental-radix-319f741cce68"
export FLARE_DRAFT="$HOME/.sparklab/models/qwen3.8-27b/draft/dflash2-bd7a934213c4"
export FLARE_RUN="$HOME/results/flare-runtime/resume-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$FLARE_RUN"
```

Run the strict quality gate first. The prior 12 workload/mode parity failures
remain unresolved; do not weaken equality/tolerances or proceed to speed claims
on a failing path. The fixture includes 35 natural workload/mode controls,
repeated target-only and first-divergence diagnostics, plus 24 forced endpoints.

```sh
SPARKLAB_DFLASH_TEST_MODEL="$FLARE_TARGET" \
SPARKLAB_DFLASH_TEST_DRAFT="$FLARE_DRAFT" \
SPARKLAB_DFLASH_TEST_MIN_FREE_GIB=30 \
SPARKLAB_DFLASH_TEST_OUTPUT="$FLARE_RUN/dflash.json" \
"$HOME/projects/sparklab/.venv/bin/python" -m pytest -q -rs tests/e2e/test_adaptive_dflash.py
```

The independent target-only paired application gate uses 20–50 turns, tool
returns, immutable-parent branches, structured summaries and a fictional legal
case. Do not enable draft speculation with exact checkpoints.

```sh
SPARKLAB_SESSION_TEST_MODEL="$FLARE_TARGET" \
SPARKLAB_SESSION_PACKAGE_PATH="$PYTHONPATH" \
SPARKLAB_SESSION_EVIDENCE_ROOT="$FLARE_RUN/sessions" \
SPARKLAB_SESSION_MIN_FREE_GIB=30 SPARKLAB_SESSION_LEGAL_CASE=1 \
"$HOME/projects/sparklab/.venv/bin/python" -m pytest -q -rs tests/e2e/test_runtime_sessions.py
```

For a separate 8K/32K recall run, also set
`SPARKLAB_SESSION_MAX_CONTEXT=65536`,
`SPARKLAB_SESSION_RECALL_LENGTHS=8192,32768`, and
`SPARKLAB_SESSION_SERVER_ARGS='["--num-pages","8192"]'`, with a fresh evidence
root. The default 512-page profile cannot hold that work. Rebudget measured
memory before attempting the larger profile.

Run the four-boot cold/publish/restart/disabled control initially at its default
65-token unaligned prefix. Inspect raw dispatch, output, usage and resource
evidence rather than treating process exit alone as restart proof.

```sh
SPARKLAB_RESTART_TEST_MODEL="$FLARE_TARGET" \
SPARKLAB_RESTART_PACKAGE_PATH="$PYTHONPATH" \
SPARKLAB_RESTART_EVIDENCE_ROOT="$FLARE_RUN/restart" \
SPARKLAB_RESTART_MIN_FREE_GIB=30 \
"$HOME/projects/sparklab/.venv/bin/python" -m pytest -q -rs tests/e2e/test_prefix_restart.py
```

## Remaining qualification and rollback

There are **no qualifying absolute matched baseline/candidate performance
metrics** for the release gates. Instrumented or confounded timings are not
benefit evidence. After correctness, follow [benchmark commands and matched
profiles](FLARE_benchmarks.md): warmup plus five balanced trials, at most 2%
tracing overhead, 25% mixed-load gap improvement with at most 5% throughput and
10% TTFT regression, 20% restore benefit with at most 5% interference, and 5%
adaptive gain over the best preselected fixed control. Preserve the plan's
per-stratum/pure controls, complete work, variance and adequate tail samples.
The 60-minute profiles and four-hour proposed-default soak remain unrun.

CPU/file faults and small CUDA conservation tests prove their scoped ownership
contracts. They do not qualify full-model state/answers after restart, long
recall, HTTP cancel/restore pressure, expert/prefix contention, NCCL interruption,
allocator drift or arbitrary spawn failure. Hosted CI and DCO review remain.
Issue 99 and the API rejection patch are completed with user authorization;
forced-tool generation remains unsupported pending constrained decoding.

Feature rollback is `SPARKLAB_SCHEDULER_POLICY=baseline`,
`SPARKLAB_EXACT_PREFIX=0`, empty `SPARKLAB_PREFIX_STORE`, and
`SPARKLAB_DFLASH_POLICY=fixed`. Keep disk records while disabled if needed for
inspection; do not claim full restart/disable validation before its real gate.
No portfolio/default promotion is authorized. The next hardware-dependent step
requires a quiet host with at least 30 GiB measured free GPU memory, followed by
strict quality diagnosis rather than further synthetic evidence for that gate.
