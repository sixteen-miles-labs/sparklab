# Native runtime implementation progress

The scope remains [the supplied plan](SparkLab_FLARE_Implementation_Test_Validation_Plan.md).
Implementation is isolated on `feat/flare-runtime` in `.worktrees/flare-runtime`,
based on `715af2b3058f0b642bbb04c8d8a449e7816bba37`. Existing user work in the
original checkout is preserved. The objective is **not complete**: qualification
and strict quality gates remain open. No live default or certification changed.

| Phase | Current status | Remaining gate |
| --- | --- | --- |
| M0 | Source/backend/state audit and complete-artifact inventory; 1175-pass CPU baseline and unchanged native smoke | Controlled paired release baseline and hardware conditions |
| M1 | Bounded telemetry, stage stamps, resource/cost export, live expert counters, mixed harness and Qwen3.6 four-stream stage reconciliation checked | <=2% trace overhead, contended-host and broader workload qualification |
| M2 | Opt-in bounded latency-aware scheduler with baseline selector; independent fairness tests | Balanced 8K/32K delivery gap/TTFT/throughput gates, controls and sustained overload |
| M3 | Exact target-only TP1 MHA+GDN checkpoints; private tails, complete conv/recurrent/aux state, safe full-prompt suffix; short/8K/32K same-partition branches checked | Strict 32K branch output parity failed; long tool/session and other state contracts; target/draft interoperability excluded |
| M4 | Bounded asynchronous store, quotas, private restore/coalescing, CPU/CUDA faults and Qwen3.6 short/32K four-boot restart/rollback checked | Natural recall after disk restart, HTTP restore/cancel, benefit/interference and expert contention |
| M5 | Opt-in prepared-depth policy and state transitions; all 24 forced accepted prefixes checked; scheduler feedback drain repaired and regression-tested | Prior strict greedy parity failed in 12 workload/mode combinations; full-model API execution, quality and sustained matched benefit remain |
| M6 | Closed without allocation change for the measured 512-token control | 4.51% upper savings bound misses preselected 5%; no broader optimization claim |
| M7 | Built-artifact regressions, real HTTP lifecycle/backpressure and executable session/restart fixtures | Full-model application/API gates, hosted CI/DCO, integrated rollback and 60-minute/four-hour soaks |

## Latest concrete work and evidence

Raw logs, reports, model-state records and wheels are under
`$HOME/results/flare-runtime/`; none are release certification.

- The authorized first paired performance attempt **fails strict 32K branch
  output parity**: 13/14 outputs match, both arms complete all budgets, but no
  pair qualifies and the remaining four are not run. Within each mode, warmup
  and measured failing-branch text agree. Pure controls also had shared-prefix
  cache contamination. No release speed claim follows. Independent finite
  baseline/memory/disk pressure collectors pass **63 completed requests and 21
  cancellations**, including warmups; disk telemetry reaches 13 restores and
  six coalesced restores. A real HTTP 503 busy rebuild exposed a collector bug,
  now fixed with **37 focused passes**; the failed attempt remains preserved.
  All six owned servers exit 143; primary owner is inactive, retrieval preserved.
  See [current performance/pressure findings](FLARE_qwen36_performance.md) and
  `qwen36-validation-v3.json`. Pending-restore cancellation and long soaks remain.
- The user stopped the VLLM supervisor for the long-test window. It was active
  again at the final audit; no owned test/server remained. User-authorized
  [design issue 99](https://github.com/sixteen-miles-labs/sparklab/issues/99)
  is published, and forced-tool/Responses-format rejection is applied to all
  three adapters. Source adapter checks: **162 passes**; the API wheel CPU
  selection: **2284 passes, 142 skips** (its selection omitted the separate MoE
  directory). The actual model server rejects **18 HTTP constraints** with no
  request-ID consumption. Auto/none remain available; hard forced decoding is
  explicitly unsupported rather than qualified by a direct answer.
- Qwen3.6 **eight 8K/32K complete-state cases pass** in 224.24 seconds. The
  **32K four-boot restart passes**, with computed prefill **32769/32768/1/32769**,
  32768 reused tokens, strict output equality and four ordinary exits (143).
  Factual recall is independently collected; the original tool-session failure
  remains visible and is not counted as a completed 20-turn application run.
- Recall integration exposed a valid-request failure in optional boundary
  rendering: Qwen rejects a system-only partial history. Unsupported partial
  template boundaries are now omitted after full-request validation. Prior
  wheel: **two failures, two controls pass**; focused source/verifier: **17
  passes**. The latest wheel matches 436 production modules and differs from
  the API wheel only in tokenizer/tokenize.py. Full CPU/MoE selection: **2339
  passed, 241 skipped, six deselected**, 246.33 seconds. Paired factual recall
  passes all **12 seed/repeat cases** at 8K/32K, with strict output equality,
  six exact hits, and zero readers/private restore resources after completion.
  Repeats reuse N-1 (8191/32767) tokens. Artifact:
  `exact-partial-render-artifact.json`.
- Latest-wheel real four-protocol streaming/stage test passes in 26.60 seconds:
  four completed/admitted/generated requests plus one validation error reconcile,
  no invalid intervals or active request remains, ordinary exit 143. This is
  plumbing/accounting proof, not trace-overhead or sustained-pressure proof.
- A supported-auto arithmetic workflow completed one actual call/result round,
  then its baseline answered the next arithmetic request without calling the
  tool. Strict required-call observation failed; candidate did not run. An
  earlier attempt first exposed a client oracle bug: reference tool-schema
  order differed from canonical HTTP wire order. Reference rendering now
  matches the wire; **49 focused harness checks pass**. All 24 prior factual
  recall renderings are independently rechecked unchanged. Full mandatory-tool
  sessions remain unqualified; no oracle is weakened to count a direct answer
  as a call.
- Local CI's JIT-disabled product selection on the latest wheel: **1558 passed,
  95 skipped**, 233.16 seconds. The later wire-order regression separately passes
  in CI mode. This is aarch64/GPU-torch local evidence, not hosted x86 CPU-torch
  or DCO proof. Counts overlap the native selection and are not summed.
- Resumed qualification with the retained native **Qwen3.6-35B-A3B Triton**
  artifact after authorized VLLM container stop. Target-only baseline returned
  `ready`; complete-model same-partition state/output checks passed **32 cases**,
  and real HTTP exact-prefix/cold controls passed **16 cases**. Shared MoE
  profiles now use streamed immutable expert preload and canonical parsed CLI
  flags; **78 CPU checks pass**. Four-boot restart passed, with computed-prefill
  counts **66 / 65 / 1 / 66** for cold/publish/restore/disabled rollback and strict
  output equality. The 20-turn application gate failed on the baseline's first
  tool turn: plain fact JSON replaced the required call; candidate was not run.
  Cold diagnostics reproduced that context failure and passed minimal lookup
  and multiplication controls. All owned Qwen3.6 tests/servers are terminal.
  These are scoped Qwen3.6 results, not DFlash parity or performance proof.
  See [Qwen3.6 evidence and commands](FLARE_qwen36.md). The systemd model owner
  can recreate VLLM; stopping that supervisor requires user-side privileges.
- The CI product selection with JIT disabled passed **1469**, skipped **95**
  in 228.92 seconds on the current wheel (`local-ci-product-wheel.log`). This
  overlaps the broad selection and does not substitute for hosted CPU-torch CI.
  All three complete-model preflights skipped at the unchanged 30 GiB floor
  (`qualification-preflight.log`). That blocked audit measured **11.44 GiB** free.
  Read-only task-file cache advice preserved model bytes; removing 21 obsolete
  extracted wheel copies preserved every wheel/log and left about **3.08 GiB**
  disk free. See [the concrete handoff](FLARE_handoff.md) for resume commands.
- Closed the child-registration stop gap inside Process.start. Callable parent
  stops now defer until the child handle is visible, then re-deliver through the
  original handlers. Pending state is three signal numbers with no frames; child
  stop masks and measured frontend exit behavior are preserved. Prior-wheel
  **seven negatives fail, two controls pass**; focused source **28 passes**.
  Current wheel: **2250 CPU passes, 241 skips, six deselections**, including both
  actual pre-registration stop cases. `m7-spawn-gap-validation.json`. Stalled/OS
  spawn failures, default-fatal signals and model/GPU startup remain unqualified.
- Added real parent-stop controls at first worker entry before Torch/stop
  handlers, during partial launch and after full handle return. **Two focused
  passes; 613 serving regressions pass** on the startup-race wheel. Removing
  the ownership guard in an isolated fault control leaves parent/child alive until
  explicit harness recovery. Both normal cases reap the child and release all
  three queue semaphores. `m7-pre-wrapper-validation.json`. Arbitrary spawn/import
  interleavings and GPU/model startup remain unqualified.
- Corrected a stop racing with startup-error reporting that skipped a fully
  constructed scheduler's local drain. The original fatal cause remains exact,
  including when cleanup fails. Prior-wheel **four negatives fail, two no-owner
  controls pass**; focused source **17 passes**. Whole CPU selection: **2239 passes,
  241 skips, six deselections**. Actual SIGINT/SIGTERM cases complete a gated
  prefix write, retain exit code one and independently validate the record.
  `m7-startup-race-validation.json`. Partial-constructor and GPU/model proof remain.
- Corrected TP signal stops that submitted a shutdown barrier behind unmatched
  rank work. TP>1 stops now drain local owners and IPC, leaving interrupted
  device/communicator teardown to process exit. Two real Gloo stop cases fail on
  the prior wheel; source plus fatal-cleanup controls: **11 passes**. TP-stop
  wheel: **2233 CPU passes, 241 skips, six deselections**. Both packaged Gloo
  scenarios close owners and exit successfully within the stop bound.
  `m7-tp-stop-validation.json`. NCCL/model/active-GPU shutdown remains unqualified.
- Corrected mixed-batch image admission: a fitting image defers when earlier
  admissions spend the remaining token budget. Prior-wheel **five CPU negatives
  fail**; the atomic-image wheel passes **2231 CPU cases and four small CUDA
  admission cases**. Synthetic conditioning, page sizes one/four and text/image
  predecessors are covered. Vision inference and SWA-specific pressure remain
  unqualified. `m2-atomic-image-validation.json`.
- Corrected byte-fallback Unicode stop lookback and speculative burst truncation.
  Prior-wheel **six negatives fail, four controls pass**; focused source **33
  passes, one skip**. The Unicode-stop wheel passes **2226 CPU cases**; actual
  local Qwen tokenizer controls improve from five/seven to seven/seven without
  loading model weights. `m5-unicode-stop-validation.json`.
- The earlier Responses-only preview and its 24 failing negative controls are
  retained in `responses-constraint-preview/`. Issue 99 and the applied combined
  fix above supersede its pending-permission status; the original failure
  evidence remains `m5-responses-constraints-preview-validation.json`.
- Earlier exact-tail/private-branch accounting, seeded ownership, pending-DMA
  cancellation, expert sampling, adaptive phase/drain and lifecycle repairs retain
  their original scoped evidence. Session/restart/soak fixtures are executable;
  full-model execution and benefit/endurance gates remain open. Prior **12 strict
  speculative parity failures remain unresolved**. Detailed results, failures,
  artifacts and commands are preserved in [validation history](FLARE_validation_history.md),
  [gate ledger](FLARE_gates.md) and [benchmarks](FLARE_benchmarks.md).

The earlier spawn-gap wheel matches its **436 recorded production Python modules**; only launch.py
changed from the startup-race wheel. SHA-256:
`f4cdfb32eb66ce63bbf7f658bb6b4f87c379750cc7394db9e46c1305b20c7542`.
No new CUDA execution is claimed for this parent-launch change. Earlier device
and complete-model evidence retains its recorded artifact scope. Focused runs
overlap and are not summed. Ruff F and `git diff --check` pass; all owned checks
are terminal. The Qwen3.6 checks above are subsequent work. Changes
remain uncommitted, with no live default promotion.

## Configuration, review and rollback

See [design/eligibility](FLARE_design.md), [commands and profiles](FLARE_benchmarks.md),
[workspace negative result](FLARE_memory.md) and the requirement-by-requirement
[open-gate ledger](FLARE_gates.md). Design discussions are issues
[96](https://github.com/sixteen-miles-labs/sparklab/issues/96),
[97](https://github.com/sixteen-miles-labs/sparklab/issues/97) and
[98](https://github.com/sixteen-miles-labs/sparklab/issues/98).

Keep `SPARKLAB_SCHEDULER_POLICY=baseline`, `SPARKLAB_EXACT_PREFIX=0`, an empty
`SPARKLAB_PREFIX_STORE`, and `SPARKLAB_DFLASH_POLICY=fixed` for feature rollback.
Existing prepared fixed-depth execution remains available. Disk records may be
retained while disabled; full restart/disable validation remains a gate, not a
claim. Exact prefixes reject unsupported multimodal/speculative/grouped state.
No model weights, precision or public support claims were changed.

## Next work and current constraints

The complete FTW Qwen control is available (artifact `1ad54cc34ff10c49`), but its
full-server fixtures require at least **30 GiB free physical GPU memory before
each boot**. The user has stopped the VLLM supervisor for this window. Do not lower the guard, stop
unrelated workloads, count swap as capacity or download a replacement portfolio.
M3/M5 failures must be resolved or left disabled; narrow transfer passes cannot
qualify model answers or speed. No balanced performance or soak gate has passed.

The previous blocked audit is superseded by the user's Qwen3.6 test request and
authorization to stop VLLM. Measured free memory after container stop was 69.9 GiB;
new full-model correctness evidence is being collected. The systemd supervisor
was inactive during the long-test window and active again at the final audit;
issue 99/API application are authorized and completed. The original objective remains
incomplete; these additional results do not defer or waive any gate in the plan.

Detailed earlier results, failures, intermediate artifacts, baseline findings and
compatibility decisions are preserved in
[the chronological validation history](FLARE_validation_history.md). Historical
capacity/process observations there must not be treated as current live state.
