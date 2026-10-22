# Chronological FLARE validation history

Older observations and process descriptions below are historical evidence, not
current live-handle assertions. Use [the progress record](FLARE_progress.md) and
[open gates](FLARE_gates.md) for current status; revalidate every process handle
before waiting or restarting. Raw artifacts remain outside git.

# Native runtime implementation progress

The governing scope is [the implementation and validation plan](SparkLab_FLARE_Implementation_Test_Validation_Plan.md).
Work is isolated on `feat/flare-runtime` in `.worktrees/flare-runtime`, based on
`715af2b3058f0b642bbb04c8d8a449e7816bba37`. The original checkout is behind that
reference and contains unrelated uncommitted work; it has not been reset or copied
over the newer runtime. No live defaults, model weights, or precision have changed.

| Phase | Status | Evidence / next action |
|---|---|---|
| M0 | Implemented / baseline smoke validated | CPU baseline 1175 passed, six skipped; complete artifacts validated; unchanged native dense-weight hybrid control completed 128 output tokens. Release baseline statistics remain unrun. |
| M1 | Implemented / partial validation | Mixed SSE harness, exact text generator, bounded stage/dispatch/queue/host/GPU/output-ready metrics and opt-in trace ring. Frontend stage telemetry and native resource export are implemented; full-model stage reconciliation and trace-overhead gate remain. |
| M2 | Implemented / experimental | Independent fairness/service tests and real-server smoke pass. Shared CPU regression after M1/M2: 1190 passed, six skipped. Three-request timings are confounded and do not qualify benefit gates. |
| M3 | Implemented / validation in progress | Pool-backed immutable full pages, private tails, whole GDN/conv/aux snapshots, eligible message/prompt/safe-suffix/terminal endpoints. CPU branching/pressure/cancel/rebuild tests pass. Real CUDA byte-copy test passed after releasing owned server memory. Matched-partition full-model continuation passed; cross-partition quality remains unqualified. |
| M4 | Implemented / partial validation | Versioned CPU store, bounded I/O tickets, CUDA staging and scheduler-owned restore admission integrated. CPU and CUDA lifecycle checks pass; real-model memory/disk continuation passed on the complete local artifact. Benefit/API/long-context/restart/contention gates remain. Off by default. |
| M5 | Implemented opt-in / quality gate failed | Request-local policy, finite verification shapes and round ownership integrated for batch-one native Qwen DFlash2. CPU checks, real state-transition oracles and all 24 forced accepted-prefix cases pass; strict cross-shape greedy parity fails. Remains off; sustained benefit and exact-cache interoperability gates remain. |
| M6 | Closed without optimization for control profile | Complete-artifact allocation and 512-token operator profiles collected. Graph reuse upper bound is 4.51% of fixed-workspace lower bound, below the preselected 5% gate; existing allocation layout retained. See FLARE_memory.md. |
| M7 | Partial validation | Built-wheel CPU/CUDA ownership checks, short/8K/32K exact-state branches and public cache rebuild have scoped evidence below. Native process cleanup adds real signal/semaphore tests. Full API/quality/restart/soak gates and integrated rollback remain. |

## M0 findings

- GB10/SM121, 121 GiB physical host memory; NVIDIA driver 580.126.09, CUDA 13.0;
  Ubuntu 24.04.4, kernel 6.17.0-1014-nvidia, aarch64.
- Existing environment: Torch 2.11.0+cu130, Triton 3.6.0, Transformers 5.16.1,
  FlashInfer 0.6.15.post1, huggingface-hub 1.29.0. No upgrades performed.
- At audit: approximately 69 GiB available RAM, 3.8 GiB pre-existing swap, 203 GiB
  disk free. Other vLLM/retrieval services actively use the GPU. They are not stopped.
  Shared-machine smoke measurements cannot establish release benefit or zero-swap gates.
- The NVIDIA Flash-Next artifact `94e1ee0daa442357` is complete, including the
  external PLE and MTP artifacts. Its 100 GiB recipe reservation cannot safely fit
  alongside the active services. Hybrid full-model validation needs more free memory.
- Dense RadixArk Qwen3.8-27B artifact `1ad54cc34ff10c49` is complete. The control
  uses target-only native execution, Triton attention/NVFP4, BF16 computation,
  8192 cache tokens, 4096 context, page size 16, batch limit 2, graph limit 2,
  and 512-token chunks. This is an explicit control profile, not the default
  DFlash2-12 recipe. It does not inherit DFlash performance evidence.
- DeepSeek V4.1 checkpoint storage was removed at the user's request. It is not
  downloaded again. Availability of a separate V4 expert-offload control is unproven.
- Qwen3.6's selected recipe uses a native runtime inside an isolated Marlin
  container. DiffusionGemma uses the vLLM backend. Host scheduler changes cannot
  improve either running image automatically. Other native recipes need individual
  state and qualification gates, not a portfolio-wide claim.
- Prefix state: plain radix stores full KV pages. Hybrid adds GDN recurrent,
  convolution and registered auxiliary PLE state; snapshots currently use aligned
  boundaries and donation. SWA/DSV4 add window, compressed/index and carry state.
  V4.1 has a separate model-owned eager continuation path. Multimodal exclusion stays.
- `GB10-DSV4-ADAPTIVE-004` improved a short confidence-adaptive probe; sustained
  `GB10-DSV4-SMALLM-005` trailed target-only. `GB10-DSV4-PREFIX-006` removed an online
  0/1/3/5 cost controller: 12.50/9.01/8.20 tok/s on short/long/prose, with prose
  below 8.69 target-only. M5 must include proposal/commit/I/O cost, unobserved-depth
  censoring, diverse completed work, and target-only controls rather than repeat it.

## Baseline tests and environment failure

Initial CPU run with `SPARKLAB_DISABLE_JIT=1`: 153 failed, 1022 passed, six skipped.
The installed kernel-cache wheel is 0.1.0+cu130 while source is 0.1.3; the production
version guard correctly refused it and JIT fallback was prohibited. This is an
unchanged environment incompatibility, not a regression. Retesting with normal
runtime JIT (without bypassing the version guard) yielded **1175 passed, six skipped**.
The skipped cases require CUDA or reject page-equals-window reference geometry.
Raw output: `$HOME/results/flare-runtime/m0-tests.log`.

Raw server/workload logs and pinned FLARE reference copies are under
`$HOME/results/flare-runtime/`, outside git. Reference source is inspected for
contracts; no FLARE source has been copied into SparkLab.

## Compatibility and proposed review units

M1 adds an independent benchmark companion and bounded internal measurements.
M2 will be internal opt-in and retain prefill-first selection. M3/M4 require a
concrete versioned layout and design issue before introducing a stored format.
No public model-support, release certification, sampler, or artifact format changes
are authorized by a passing CPU suite. A measured benefit gate failure leaves the
corresponding feature disabled and does not block independent workstreams.


## Current M3 investigation and live handles

Exact reuse is internal opt-in: `SPARKLAB_EXACT_PREFIX=1`, qualified model-owned
state contract, TP=1, target-only, MHA+GDN pool. It forces normal (non-overlap)
scheduling to snapshot drained consumed endpoints. Full prompt hits explicitly
recompute a safe suffix from the `prompt_length-1` checkpoint; no stale sampled
output token or RNG state is restored. Records are capped at eight by default
(max 128); private tail pages and complete state slots come from existing pools.
Admission can evict ready unpinned records to avoid snapshot-induced starvation.
Source readers and unfinished producer events prevent eviction. Rebuild invalidates
physical IDs after the existing idle GPU fence. Multimodal reuse stays excluded.

CPU checks: initial shared M3 regression 399 passed, six skipped; broader integrated
checkpoint/scheduler/radix/tokenizer/serving run 898 passed, two skipped. Subsequent
new pressure/accounting tests pass. CI now runs CPU-safe runtime directories in a
separate step allowing the one CPU C++ radix helper to build; it needs no CUDA.

Real API pressure/cold-reference matrix passed five cases, then found a different
four-token output for a 16-token-prefix compass branch (raw:
`$HOME/results/flare-runtime/exact-api-churn.json`). Do not treat this as passed.
External diagnostic instrumentation (outside git) verified that the parent GDN/conv
hashes remained unchanged, the live restore matched every byte, and the actual new
input token was correct. The differing first logits tied navigation and compass at
10.75. Source prefix computation used 15+1 rows; the cold comparison used 16 rows.
A dedicated real-model test now compares identical partitions and every complete
state tensor to distinguish transfer correctness from partition-dependent numerics.
No tolerance was weakened and no quality gate is claimed. Cross-partition quality
and repeatability remain a required gate before promotion.

A first matched-state test reached endpoint 31 and selected a ready ancestor at 30
because the deeper GPU publication was still pending. It now waits for producer
completion in the test before demanding the deepest hit; production never waits
and may safely use a ready ancestor. Add explicit pending-fence lifecycle tests.

Same-size public cache rebuild was rejected by the existing admission guard with
a negative budget on the shared host; the server remained serving. This is recorded
in `$HOME/results/flare-runtime/exact-api.json` and logs, not bypassed. Controlled
cache pressure obtained independent zero-hit API references instead. Internal
idle logical-cache reset is used by the matched-state test without reallocating
or bypassing the public rebuild guard.

Owned HTTP test servers on port 19291 have been stopped. Original full-model
matched-partition test passed (one test, 33.23s). The strengthened test verifies
pool/slot conservation before clearing, then exercises memory and disk reuse
against cold runs in a single engine (Engine intentionally rejects a second
initialization after CUDA is initialized). No unrelated GPU services were stopped.

## M4 implementation and current validation

- Design discussion: [issue 96](https://github.com/sixteen-miles-labs/sparklab/issues/96),
  opened before introducing the stored format. No remote branch/PR/default change.
- Version-one records are self-contained logical state, streaming checksummed,
  bounded before allocation, atomically fsynced/renamed, and scoped by artifact,
  effective configuration, tokenizer/templates, precision and concrete state layout.
  Live page and slot IDs are never serialized.
- One I/O worker, bounded queued/running/undrained tickets, bounded per-ticket
  pinned staging, one private inbound restore, bounded coalesced waiters and index.
  A finished-but-unpolled or unfenced ticket retains its admission credit and source
  lease/private destination. Cancellation and timeout detach waiters only.
- The scheduler polls completions; no filesystem operation or transfer wait runs on
  the model worker. Unrelated requests bypass disk waiters. A 1ms idle backoff
  applies while I/O is outstanding. Cache rebuild refuses outstanding prefix I/O
  before touching engine allocations. Shutdown retains pools if draining fails.
- Default restore policy uses observed costs and recomputes unknown shapes. An
  explicit idle calibration mode and unconditional correctness-test mode exist.
  Comparison uses only the extra prefix work beyond a safe in-memory hit.
- Physical-memory reservations include bounded decoded metadata, staging, maps
  and parser storage in the same GB10 pool, frozen at engine startup. Best-effort
  page-cache advice targets only owned records; no global cache flush occurs.
- Latest broad CPU run: **1179 passed, four skipped** in 12.63s, including affected
  checkpoint/scheduler/radix/engine/tokenizer/serving/catalog and harness checks.
  Raw: `$HOME/results/flare-runtime/m4-integrated-cpu.log`.
- Real-file failure tests now cover permission denial, zero/short writes, file and
  directory fsync failure, ENOSPC, killed writer, corruption, namespace locks,
  quota pins, worker death, cancellation and undrained credits. Expanded record
  suite: **28 passed** (`m4-record-map.log`). Lifecycle+fault suite before adding
  sparse layer-map coverage: **33 passed** (`m4-faults.log`).
- CUDA inference-tensor transfer plus full service adoption/eviction: **two passed**
  in 2.22s (`m4-service-cuda.log`). These are physical-state transfer tests, not
  real-model quality or speed certification.
- An initial real-model disk run correctly failed on a cache miss: nonempty sparse
  layer maps were tuples in the writer's pre-serialization header but the validator
  required lists. All writes failed closed. The encoder now canonicalizes this
  field, with an independent real-file restart regression. Keep failed raw runs
  `m4-real-model-state*.log`; no missing hit is counted as a correctness pass.
- Earlier first focused command omitted this worktree's PYTHONPATH and failed
  collection. A subsequent unmasked CPU fixture encountered CUDA pinning OOM on
  the shared host; hiding CUDA for CPU tests yielded **197 passed**. These are
  distinct from the production JIT-version mismatch recorded above.

The corrected real-model matrix passed **one test in 66.24s**
(`m4-real-model-state-v4.log`): eight endpoint lengths (1, 15, 16, 17, 31, 63,
64, 65), two suffix branches, memory and disk sources, every consumed KV/GDN/conv/
auxiliary tensor compared byte-for-byte, identical complete greedy outputs versus
cold references using identical execution partitions. Disk cases require an
observed completed restore and exact admission hit, not a counter-only substitute
for state validation. The strengthened pre-reset page/slot conservation gate passed.
This does not qualify cross-partition quality, long-context/API/restart performance,
expert contention, or endurance. The test uses logical in-memory eviction without
model reload; CPU recovery tests prove file restart independently.

Offline native wheel build passed using existing CUDA 13.0 and the configured
Torch environment, with no dependency installation. Output is outside git under
`$HOME/results/flare-runtime/wheels/`. The final source CPU regression passed **1185 tests, four skipped** in 12.16s
(`m4-final-cpu.log`). The final wheel SHA-256 is
`ba15c0ade8c2e57b704bf365718790aa06bf42c13f2a267fae97d590db153c1e`.
All checkpoint modules and both native extensions imported from its extracted
package, with import paths asserted outside the editable checkout. The same
CPU regression against that package passed **1185 tests, four skipped** in 14.14s
(`m4-wheel-cpu.log`). Built-wheel real-model validation passed **one test in 77.71s**
(`m4-wheel-real-model.log`), importing the engine and cache code from the extracted
wheel. This is the same complete-state matrix, not a speed comparison. All M4
model validation processes and build sessions are terminal; no server was left running.
M5/M6/M7 remain incomplete; no default is promoted.

## M5 preparation rationale

The complete local Qwen DFlash2 draft is the selected first native path. Existing
verification captures one width; the opt-in integration explicitly prepares legal
widths instead of sending different sizes to an unmeasured eager fallback. Draft
context is append-only KV materialized by target forwards. Target-only rounds
must keep those writes, and positive proposals must hide rejected suffix rows.
The implementation does not slice already-paid proposals or mutate a global
request depth. Fixed-depth mode and exact-prefix speculation exclusion remain.

## M5 current implementation and failed gate

`SPARKLAB_DFLASH_POLICY=cost` is an internal opt-in. The default `fixed` path retains
its original proposal flow. Cost mode is restricted to native graph-prepared Qwen
DFlash2, TP=1, batch one, Triton, block at least four. Default prepared proposal
depths are 0/3/7/11 for DFlash2-12. Positive depths use exact-width verification
graphs; zero uses a prepared target graph that also updates draft context.
`SPARKLAB_DFLASH_DEPTHS` accepts at most four legal prepared depths including zero
and the configured maximum. `SPARKLAB_DFLASH_FIXED_DEPTH` selects a fixed control
inside the same prepared/resource envelope. Mode/depth changes require restart in
serving; live transitions in the GPU test are explicit internal test operations.

Adaptive proposals run only after selecting an existing decoder, before that
round's verification. This separates their cost from the prior output handoff.
Whole-round CPU time includes proposal, verification, commit, scheduler token
writes and output handoff; CUDA events separate proposal and complete GPU work
without a new per-step synchronization. The existing drain fence is recorded after
adaptive scheduler writes. Timing training and resource release happen at drain.
M2's service envelope limits depth choice when latency policy is selected.
A 2 GiB physical reservation bounds prepared graph workspace; retained graph-pool
reservation is checked at startup/rebuild, using reserved rather than allocated
bytes so private CUDA graph pools are counted.

Focused controller/preparation checks: **19 passed**. Integrated CPU model/runtime
checks: **365 passed, 17 skipped** (`m5-cpu-integrated.log`); the later scheduler/
engine/model slice after extending the output fence: **285 passed, 14 skipped**
(`m5-cpu-rounds.log`). Skipped hardware cases are not state qualification.

The real complete target plus complete local DFlash2 draft loaded and captured
0/3/7/11 execution shapes. The full five-stratum, six-mode, 64-output matrix
completed (`m5-real-model-v3.json`/`.log`), including in-request depth transitions.
It passed **897 immutable consumed-KV checks** across all target/draft geometries
and **479 exact recurrent/conv/aux accepted-prefix commit checks**. The first
instrumentation attempt assumed one KV geometry and failed before inference;
the actual model uses GroupedMHAKVCache and the oracle now checks every pool.

The strict greedy-parity gate **failed**: 12 mode/stratum combinations diverged
from target-only, including prose instrument/compass at output token 21 and code
at token 1. Full outputs and first-difference positions are retained. All five
strata and all six modes were collected before asserting failure; no difficult
case was omitted and no equality/tolerance was weakened. This does not yet prove
whether each divergence is partition-dependent native numerics or a remaining
continuation defect. Per-round transaction and immutable-state checks are stronger
than output resemblance but do not establish whole-model quality. No adaptive
benefit or speed claim follows from these instrumented, unbalanced runs.

Exact checkpoints still reject speculation: the grouped target/draft portable
state contract has not passed interoperability gates. The current pure adapter's
MHA-only rejection is preserved. Do not bypass it to make a combined profile run.
Adaptive mode remains disabled by default, and existing fixed mode is available.

## M6 measured workspace candidate

The offline native inventory command is implemented in
`benchmarks/profile_runtime_memory.py`. The first complete-artifact run
(`m6-memory-profile.json`/`.log`) records CUDA allocator segments/pool identities,
verification transaction tensors, graph input/logit storage, and warm reuse of
all depths. It also exercised the latest complete-round GPU timing fence.
Observed: 98,566,144 bytes added for the prepared graph set, including separate
private CUDA graph pools; verification recurrent intermediates retain
1,811,939,328 bytes, convolution intermediates 11,796,480 bytes, and indices four
bytes. Those per-prefix intermediates remain live until the target decides the
accepted prefix; aliasing layers/steps would lose rollback state. No memory
optimization is claimed. The 512-token allocation/operator profile completed successfully
(`m6-memory-operators.json`/`.log`). The selected graph-workspace reuse candidate
fails the preselected materiality gate even under a conservative upper saving
bound; see [the measured memory decision](FLARE_memory.md). No allocation change
was made and no scratch latency claim is made. There are no owned HTTP servers. M7 and earlier missing release/API/
long-context/soak/benefit gates remain open.


## M7 current artifact and shared-path evidence

Latest broader source CPU check after M5 integration: **1216 passed, 18 skipped**
in 12.42s (`m5-broad-cpu.log`). Expanded shared-cache/expert CPU checks: **306
passed, 105 skipped, six deselected**. Skip reasons are preserved in
`m7-shared-cache-moe-skips.log`; none is counted as a hardware pass. The hosted
CPU/DCO checks have not run. See [the working gate ledger](FLARE_gates.md) for
requirements that still lack direct evidence.

The current M5 native wheel built without dependency installation. SHA-256:
`b3a86ed2356d547af1318acb2661fc3c549bf484c8630a19e69b4f52a7f259e1`.
M5 modules resolve to the extracted wheel, and its expanded CPU suite passed
**1963 tests, 200 skipped, six deselected** in 21.44s (`m7-wheel-cpu.log`). This
covers the hosted CPU directories plus all runtime cache/engine/checkpoint and
MoE CPU-safe modules, with local JIT fallback for the pre-existing incompatible
kernel-cache package. It is local artifact evidence, not a hosted CI result.

The corrected extended GPU run completed (`m5-real-model-v5.json`/`.log`, 147.05s).
It checked **1144 consumed target/draft KV immutability observations**, **666 exact
recurrent/conv/aux commits**, poisoned inactive verification lanes, and **all 24
accepted-prefix endpoints** at depths 3/7/11. The 30 natural-output cases and all
forced commit cases completed before the unchanged strict parity assertion failed
on the same 12 combinations. The forced verifier is test-only state-transition
injection, not a modified production sampler or quality result. This run includes
the current complete proposal/target/commit/output GPU fence. No tolerance changed.

The built-wheel native allocation/execution profile also completed successfully
(`m7-wheel-native.json`/`.log`), executing every prepared depth and a 512-token
prefill, including the current adaptive GPU fence. This is a functionality/profile
check, not quality or speed certification. Source engine code has not changed
since the wheel was built. Tests/benchmark documentation were subsequently refined.

The short built-wheel memory/disk continuation regression completed as recorded
below (`m7-wheel-exact-state.log`). There are no owned HTTP servers. Real experimental-path
API, restart, long-context, balanced benefit and integrated soak gates remain.


The current M5 wheel passed the unchanged short M4 memory/disk continuation gate
again: **one passed in 81.72s** (`m7-wheel-exact-state.log`). The independent
bounded byte-digest observation path then passed a short endpoint 1/65 smoke:
**one passed in 27.80s** (`m7-wheel-streamed-state-smoke.log`). This test-only path
hashes every actual logical KV and recurrent/conv/aux byte through independent
8 MiB direct tensor reads, avoiding two retained multi-GiB state images for long
contexts. It does not reuse the record encoder/decoder or GPU staging transport
as its equality oracle. Production code is unchanged.

The completed built-wheel long-state matrix used **8192 and
32768 token endpoints**, two branches each, memory and disk sources, cold controls
with identical partitions, a declared **40960-token cache / 32784 context / 512
chunk** profile. Former tool session **27088** is terminal; raw
live dispatch output `m7-wheel-long-state.log` and compact result
`m7-wheel-long-state.json`. Inspect the live handle before restarting. These are
instrumented complete-state correctness comparisons, not matched TTFT or speed
measurements. Long-context factual recall, clean-server restart, public API and
all performance/soak gates remain separate.


Retired prefix records from completed test stores (pytest roots 36 and 37 only)
were removed after verifying their namespace locks were unheld: 54 records,
8,441,679,872 physical bytes freed. The active long-test store, source checkpoint,
all logs and compact evidence were preserved. Other disk users were not modified.

Code review found a calibration-order defect: applying a rotating cursor to a
changing combined depth list could keep skipping an unmeasured depth. The pure
controller now probes each unknown legal depth before refreshing a measured one.
The service bound, refresh interval and hysteresis thresholds are unchanged.
An independent fixture checks coverage of the unknown depths. This source-only
policy change postdates wheel `b3a86e...`; that artifact remains valid for the
prior controller and unchanged engine/checkpoint paths, but the final policy
requires a new artifact and validation. The current long-state run deliberately
continues on its pinned built artifact, with adaptive speculation disabled.


The built-wheel 8K/32K long-state matrix completed successfully: **one test passed
in 680.38s**, covering all eight source/endpoint/branch combinations. Compact
`m7-wheel-long-state.json` records actual artifact identity, the declared 40960
cache / 32784 context / 512 chunk profile, exact hits and state/output equality.
The raw live dispatch log is `m7-wheel-long-state.log`. This is instrumented byte
correctness, not a latency or throughput measurement.

The real public rebuild gate also passed on wheel b3a86e...: **one test in 32.32s**
(`m7-public-rebuild.log`, `m7-rebuild-server.log`). This separate declared profile
uses memory_ratio=0.85, page=16, context=4096, two request/graph rows, 512 prefill
chunks, and grows KV from 512 to 768 pages and the recurrent pool by eight slots.
Exact publication is enabled; disk and speculation are off. Actual generation
works before and after each resize. An impossible target is rejected before
teardown and generation continues. This does not bypass the guard or retroactively
qualify the earlier memory_ratio=0.4 rejection. The test was corrected to use the
current `sparklab serve` CLI and now accepts explicit profile/package/log overrides.
The owned server shut down. Its resource tracker reported four leaked semaphores;
no claim of a clean process-lifecycle/soak gate is made from this single run.

Tokenizer compatibility review against installed Transformers source found the
current `chat_templates/` directory was omitted. Identity discovery now includes
both that spelling and the legacy `additional_chat_templates/`, and bounds the
filename list before materialization/sorting. A named-template edit/add/remove
and combined-directory limit regression protects this. Existing artifacts without
named templates retain the same identity. This source-only identity change and
the calibration-order correction require the next built-artifact check.

Resource audit also identified a remaining M4 limit: quotas/index bounds currently
apply **per compatibility namespace**, not across all stale namespaces under a
reused store root. Test runs use fresh dedicated roots, but repeated identity
changes could accumulate old namespaces. Root-wide accounting/reclamation and
concurrent-writer ownership still need implementation and lifecycle tests before
claiming the full disk-bound gate. This is recorded explicitly in FLARE_gates.md;
it is not hidden by the passing per-namespace tests.


Current-source artifact refresh completed after the calibration and named-template
fixes. Wheel SHA-256:
`3715a7c8b512d4fe920a34af1b0d5d2b9ee5dd15583ba3508d38310d14e476be`.
The packaged identity/policy files were compared byte-for-byte with source before
execution. Expanded CPU suite from that extracted wheel: **1965 passed, 200
skipped, six deselected** in 18.91s (`m7-current-wheel-cpu.log`). Native execution
also completed (`m7-current-wheel-native.json`/`.log`): every prepared depth, a
512-token prefill and a 128-output adaptive smoke. The latter completed 65 full
rounds, with measured samples at depths 0/3/7/11 of 4/18/12/31. This proves
execution/measurement coverage of the corrected policy, not benefit or quality.
The prior strict quality failure remains open; the full matrix was not relabeled
as passed. The focused source identity/record/lifecycle/policy regression passed
**59 tests** (`m7-identity-refresh-cpu.log`).

All owned GPU/server/build/pytest sessions in this turn are terminal. The long
test's retired store (pytest root 38 only) was also locked and cleaned after its
process exited: three records, **6,905,094,144 physical bytes** freed. All compact
reports, dispatch logs and source model checkpoints remain. There are no owned
HTTP servers or active GPU validation jobs to restart.

Next implementation priority is **root-wide prefix-store quota/index accounting
and inactive-namespace reclamation**, preserving active namespace/read ownership
and model-worker nonblocking behavior. Current quotas are per namespace only.
Then complete frontend-stage observability, real protocol/agent/recall and
restart/cancellation/rollback matrices, balanced performance/control trials and
required soaks. The gate ledger preserves those outstanding requirements. The
goal is active; no default, recipe, certification or remote branch was promoted.


## Root-wide retention and shutdown ownership follow-through

The previous root-quota gap is now implemented. `checkpoint/quota.py` bounds
serialized record/staging bytes, record count and namespace count across all
compatibility identities under one root. `.quota.lock` serializes mutations on
the I/O worker without waiting for another writer. Version-one `.quota.json`
persists the root limits; malformed/future metadata fails closed. A limit change
requires other namespaces to drain. Active namespace contents cannot be reclaimed
by another owner; inactive oldest records and incomplete temporary files may be
reclaimed. Discovery has a fixed 32768-entry bound and retains one eviction
candidate rather than an unbounded prompt index. Physical block rounding and
filesystem metadata are outside the serialized-byte quota; record/namespace
counts separately bound owned metadata. Unknown files/symlinks are preserved.
Large legacy-root scan performance remains unqualified.

Shutdown ownership now extends through *undrained* tickets and GPU fences, even
when the I/O loop has stopped. The scheduler polls/releases metadata; only the
I/O worker closes files and releases the namespace lock. A bounded close can
return incomplete while retaining ownership. Known filesystem/store-unavailable
failures disable retention and recompute through ordinary admission. Unexpected
post-publication device/invariant failures remain visible. Actor snapshots use
the last cached root observation and do not scan files.

Root lifecycle/failure tests: **51 passed** (`m4-root-failure-bounds.log`). These
cover 40 identity changes, empty-namespace churn, protected active owners,
serialized concurrent publication, quota changes, undrained fake GPU fences,
actual child-process death midway through a write, corrupt/oversized/future root
metadata, symlink/unknown-file preservation, scan exhaustion, fsync failure,
orphan cleanup failure and repeated failed startups. The last case checks bounds
before creating a new namespace, preventing failed cleanup from growing roots.

Complete native source validation: **one test passed in 66.91s**, covering all
**32 memory/disk x 1/15/16/17/31/63/64/65 x two-branch cases**. Independent bounded
byte hashing checks actual KV, convolution, recurrent and auxiliary state, plus
same-partition greedy continuation. Evidence: `m4-root-native-state.json/.log`.

The current root-quota wheel was built offline, without dependency changes:
`wheels-root/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl`, SHA-256
`df18ba48970e908fa12503047596713a1b1733644660e6d4d083bd52658e7785`.
Every packaged checkpoint module was compared byte-for-byte with current source.
Actual extracted-wheel checks:

- Expanded required/affected CPU suites: **1982 passed, 200 skipped, six
  deselected** in 20.45s (`m7-root-wheel-cpu-clean.log`). Skips remain unavailable
  CUDA/model/optional-backend checks, not passes.
- CUDA staged real-file round trip and private scheduler adoption: **two passed**
  in 2.29s (`m7-root-wheel-cuda.log`).
- Complete-model memory/disk branches at endpoints 1 and 65: **one test passed**
  in 33.22s, all **eight cases** byte/state/output equal
  (`m7-root-wheel-native-state.json/.log`).

The first expanded CPU ordering exposed an existing environment leak: the
serving CLI test used `monkeypatch.delenv` on absent flags, then `parse_args`
assigned them directly, so teardown did not restore absence. This enabled Kimi
MLP FP8 in a later loader test. The failure is preserved in
`m7-root-wheel-cpu.log`; the isolated loader passed, the cleanup was corrected in
the serving test, and the reproducing two-test ordering passed before the full
rerun. No Kimi loader arithmetic or runtime defaults changed. The kernel-cache
0.1.0/source 0.1.3 mismatch warning still intentionally selects runtime JIT.

These checks supersede the prior open root-wide-bound item, but not full-server
restart/rollback, sustained pressure/swap or interference/performance gates. The
prior strict M5 quality failure and M3 cross-partition qualification remain open.
No defaults, recipes, model-support claims or remote branches were promoted.
Next work is frontend-stage observability and the real protocol/agent/recall and
restart/cancellation matrices, followed by balanced performance and required
soaks. The full goal remains active.


## Frontend-stage observability follow-through

Implemented opt-in host-stage plumbing in the native serving path, using a
stdlib-only `sparklab/request_timing.py` collector. It imports neither tensors
nor runtime modules. With `SPARKLAB_RUNTIME_TIMING=1` (also enabled by latency
scheduling), fixed-name stamps cross the real internal message codecs through
HTTP ingress, frontend submission, tokenization, scheduler receipt/admission,
first dispatch, host-observed first generated output, detokenization and frontend
receipt. Chunked prefill carries the original stamps; speculative bursts emit
only one first-output observation after the existing fence. No synchronization
was added. Parsed semantic readiness ignores empty deltas; role/metadata bodies
and downstream ASGI send start/return are measured separately. HTTP completion
is final body send return, not handler return. Errors and incomplete HTTP streams
remain in counters. No client-arrival claim comes from server timestamps.

The pure ASGI wrapper must sit outside the existing buffering HTTP middleware.
A real installed-stack CPU test blocks the downstream send and verifies the
reported send-return timestamp advances only when that send finishes. Concurrent
HTTP contexts, message round trips, empty parser deltas, first-only burst
observations, preserved chunk admission/dispatch, usage conservation, invalid
intervals, bounded history and disabled measurement are independently checked.
There is no new per-request global map. Histograms have fixed labels; the
optional trace ring has a default capacity of zero and a maximum of 100000.
When enabled, `/v1/stats` adds `runtime_timing`; otherwise its document stays as
before. Scheduler GPU/cost/resource snapshots and this frontend export are not
yet one coherent contention export.

Source expanded CPU run: **1992 passed, 200 skipped, six deselected** in 18.68s
(`m1-frontend-full-cpu.log`), before the final context/empty-delta/installed-stack
fixtures. The offline rebuilt wheel is
`wheels-frontend/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl`, SHA-256
`3561297434fe2b28d02f8323f275ab05c56c7011c81ba5f3635d413c960a744b`.
All 13 stage-plumbing modules were compared byte-for-byte with source. Extracted
wheel expanded CPU checks: **1994 passed, 200 skipped, six deselected** in 20.04s
(`m1-frontend-wheel-cpu.log`). Focused final fixtures: **13 passed** in 2.23s
(`m1-frontend-wheel-focused.log`); the installed-stack test was added after the
expanded run collected, so its pass is separate rather than invented in that
expanded count. Earlier development failures remain in the `first`/`stage-cpu`
logs; they exposed lightweight fixture compatibility and a mismatched test tensor
dtype, both fixed without changing runtime math or weakening assertions.

`tests/e2e/test_request_timing.py` now implements an actual four-protocol stream
and invalid-budget stage/accounting reconciliation, with optional extracted-wheel
and evidence destinations. Its attempted source run **skipped** before launching
a server: the shared host had about 24.3 GiB free physical GPU memory, below the
preselected 30 GiB guard. This is not a native/API pass. A scoped advisory to
release only clean pages from the three complete-model FTW shards preserved all
files and did not increase free memory (`m1-model-page-cache.json`). No global
cache drop, unrelated process stop or reduced guard was used. Complete-model
validation of the new stages, overhead <=2%, full API/tool/stop/cancel matrices,
paired performance trials, quality qualification and required soaks remain open.
The prior root-wheel GPU/complete-model proofs do not qualify this later code.

All owned test/build handles are terminal and no owned HTTP server was launched.
The full goal stays active. Next work should reconcile the stage timestamps on
the complete-model server when the declared memory guard is satisfied, and
continue independent protocol/recall/rollback and quality diagnostics meanwhile.


## Bounded native-runtime export and component freshness

The former scheduler/prefix/frontend export gap now has a bounded common read
path. With existing runtime timing active (explicit timing, latency mode or disk
retention), the scheduler piggybacks at most one version-one observation per
second on an ordinary output reply. The detokenizer preserves it through the
actual internal codec, and the frontend retains one `native_runtime` document in
`/v1/stats`, with observation time and age. No new metrics queue, file scan,
device wait or synchronizing GPU expert-counter read was added. Trace rings stay
out of this IPC payload. Idle/final observations can remain stale; this is not a
new drain-completion oracle. The optional read-only stats compatibility design
is recorded in [issue 97](https://github.com/sixteen-miles-labs/sparklab/issues/97),
following the repository's issue/PR search and design-discussion requirement.
No code/default/release was published by that design issue.

The export includes scheduler histograms and bounded GPU shape costs, resource
occupancy, exact-prefix metadata/readers, private restore resources, tickets and
staging, and bounded prefix read/write service-cost summaries. Production shape
history is 64 x 128 samples; prefix cost history is 32 x 32. A pressure fixture
fills all 64 shape slots plus 32 prefix buckets and keeps the combined serialized
snapshot below 32 KiB while excluding a large diagnostic ring. Named retained
verification tensors use shape metadata for byte counts, alongside existing
prefix/adaptive graph reservations. This is not a total workspace/allocator
census or a new allocation optimization.

The prefix I/O worker replaces a completed, immutable-by-ownership metadata
observation after boot and each operation. File counters, index totals and root
quotas reach the actor without actor file calls or copying a concurrently-mutated
Counter. Root quota observations have their own monotonic timestamp; a later
read can update the worker observation without pretending to refresh root usage.
This changes no `.quota.json` or checkpoint record format. Observation failures
still complete/poll/drain their tickets and release namespace ownership through
the worker; startup failure also releases its namespace.

Expert disk fields come from the existing CPU-side source stats; aligned read
bytes are not independent NVMe-device traffic. A real-file FTW expert fixture
and a real prefix record exercise those counters together without actor open or
scandir calls. Device hit/miss counters are cached only from existing terminal
observations, with their own timestamp. No live per-round device counter claim
is made. These components have explicit freshness and are not an atomic
cross-subsystem snapshot. Full-model expert/prefix contention remains unqualified.

Validation evidence, under `$HOME/results/flare-runtime/`:

- Initial affected CPU suites: **844 passed, three skipped**
  (`m1-runtime-export-first.log`).
- Real-file/codec/burst/ASGI integration: **34 passed**
  (`m1-runtime-export-integration.log`).
- Source expanded CPU before the final workspace/cost/freshness additions:
  **2004 passed, 200 skipped, six deselected** in 17.46s
  (`m1-runtime-export-source-cpu.log`).
- Focused final root/cost/observation/export matrix: **27 passed** in 3.80s
  (`m1-runtime-export-cost-root.log`).
- The final wheel's **17 relevant source/package modules** match byte-for-byte.
  Wheel SHA-256:
  `98e536197b0dd3ccd12aff5a491c7496f868cee4b467c7ec6f830d2cde3847c9`.
  Artifact: `wheels-export/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl`.
- Final extracted-wheel expanded CPU: **2006 passed, 200 skipped, six deselected**
  in 19.37s (`m1-runtime-export-wheel-complete-cpu.log`). Hardware/optional-backend
  skips are not passes; the existing incompatible kernel-cache wheel still
  intentionally falls back to JIT.
- Final extracted-wheel real CUDA staged file transfer and private scheduler
  adoption: **two passed** in 3.25s
  (`m1-runtime-export-wheel-complete-cuda.log`).

One initial source CUDA run failed on its first **4096-byte allocation**, before
any checkpoint worker/export path; its second test passed. That failure is
preserved in `m1-runtime-export-cuda.log` and is not reclassified as a pass.
An isolated allocation then succeeded, the prior frontend wheel passed both
checks (`m1-runtime-export-prior-cuda.log`), source rerun passed both
(`m1-runtime-export-cuda-retry.log`), and both intermediate/final export wheels
passed. The initial transient allocation failure's cause remains unexplained.
No assertion, memory guard or implementation arithmetic was weakened.

The shared host still had only roughly 22–23 GiB free physical GPU memory;
complete-model stage/API/export validation remains unrun below the 30 GiB guard.
The implemented real-server fixture now checks the common export as well. Actual
protocol/tool/cancel/recall/restart, live GPU expert sampling, overhead, balanced
performance and required soaks remain open. Prior M3 cross-partition and M5 strict
quality failures are not resolved by these metadata/copy tests. All owned
validation/build processes are terminal; no owned full-model server was launched.
The full goal remains active and no runtime default or recipe was promoted.

## M7 native shutdown and process ownership

The installed Uvicorn replays captured stop signals after lifespan shutdown.
With a default-fatal SIGTERM, that replay bypassed Python finalizers. In addition,
the parent's worker `terminate()` used SIGTERM before the scheduler's existing
KeyboardInterrupt cleanup. The same real CPU process probe against the previous
export wheel exits with `Popen.returncode=-15`, does not reach parent finalization
or worker drain, and reports seven leaked semaphores. This reproduces a cleanup
mechanism; it does not establish the exact origin of all four semaphore warnings
from the earlier complete-model rebuild run.

The compatibility design is recorded in
[issue 98](https://github.com/sixteen-miles-labs/sparklab/issues/98).
The foreground default SIGTERM now unwinds Python cleanup: shell status remains
143, while `Popen.returncode` becomes positive 143 rather than negative 15.
Explicit embedding handlers remain authoritative. Worker stop handlers are
one-shot, startup cancellation handles incomplete construction, and a stop racing
a real startup error cannot silently turn that error into success. Partial
launches retain ownership until a backend handle is returned. A shared ten-second
worker grace, followed by kill/join, avoids multiplying the grace by worker count.
That deadline does not bound Uvicorn's active-HTTP drain.

Frontend shutdown seals admission before stopping owned workers; owner-loop IPC
closure is idempotent and uses zero linger. Scheduler shutdown drains prefix/GPU
users before engine teardown, then attempts all owned transport closures even if
one close fails. The actual tokenizer worker also closes its owned transports.
Ordinary queue utility defaults and shell Ctrl+C turn cancellation are preserved;
explicitly ignored shell stop signals remain ignored.

Validation is scoped to production ownership transitions, with model execution
replaced in CPU process probes. Nine real subprocess scenarios cover ready stop,
repeated signals, loading cancellation, completed startup failure, failure/stop
race, three forced survivors, paused real prefix I/O, partial launch and the
actual local tokenizer. They verify final worker codes, no surviving owned PIDs,
removed named semaphore files and no graceful resource-tracker warnings. The
forced case exits all three workers with SIGKILL within the shared bound; it does
not prove a successful transfer drain. Two real ZMQ absent-peer cases verify
nonblocking zero-linger closure. An actual owner-loop test closes frontend async
IPC from another thread. A CUDA test pauses a real prefix file writer and checks
that tickets/read pins release before engine teardown and IPC close.

Final evidence under `$HOME/results/flare-runtime/`:

- Artifact `wheels-lifecycle/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl`,
  SHA-256 `9bef7c88fd54ad25221564d77768ef48fdb6a558e650f9ac03b60cf9132c7d98`.
  All **436 Python package modules** match current source byte-for-byte; hashes
  for the 21 lifecycle/telemetry modules are recorded in the manifest.
- Expanded extracted-wheel CPU, including hosted selections and wider native
  cache/MoE regressions: **2022 passed, 201 skipped, six deselected**, 74.81s
  (`m7-lifecycle-wheel-complete-cpu.log`). Skips remain unqualified hardware or
  optional-backend coverage.
- Extracted-wheel CUDA: **three passed**, 3.08s
  (`m7-lifecycle-wheel-race-cuda.log`). This is transfer/adoption/shutdown ordering,
  not complete-model inference.
- Current probe versus prior artifact: `m7-lifecycle-baseline-current-probe/`.
  Final nine-scenario logs: `m7-lifecycle-wheel-complete-matrix/`.
  Summary/source hashes/capacity/failures: `m7-lifecycle-validation.json`.
- Source startup-error and stop-race selection: **two passed**, 9.29s
  (`m7-lifecycle-error-race-source.log`). Earlier intermediate lifecycle wheel:
  2021 CPU passes and three CUDA passes, superseded by the final artifact above.

Initial failures are retained. Two process-test failures exposed an ack-versus-exit
race in the test and an incorrect stdout/stderr expectation. The completed-error
case now observes actual process exit; a separate failure/stop race remains in the
matrix. Low shared-host disk space also exposed mocked DJEV and Bonsai acquisition
tests that required real checkpoint storage despite replacing their I/O. Their
fixtures now budget the fake data; production storage guards and recipe sizes
are unchanged. The first expanded final-wheel run recorded **one failure, 1858
passes, 97 skips** from the Bonsai fixture before its correction. Earlier failed
source/log runs remain in the manifest. Focused Ruff F checks pass for the new
and lifecycle-edited modules; an existing unused `io` import in the DJEV test is
unchanged. `git diff --check` passes.

During validation, free disk fell below 0.5 GiB and later recovered to 14 GiB
without deleting unrelated data. Free physical GPU memory was 16.29 GiB at the
low-capacity observation and 22.52 GiB on the final check, below the preselected
30 GiB complete-server guard. No full-model server or unrelated stop was issued.
All owned build/test/probe processes are terminal. Complete-model/multi-rank,
active-HTTP/group-signal, earliest-spawn interruption, restart/rollback and soak
qualification remain open. M3 cross-partition and M5 strict quality failures,
balanced benefit/overhead measurements and hosted CI/DCO remain open as well.
No feature default, recipe, model-support or certification claim was promoted.

## M7 active HTTP drain and terminal process groups

The next real CPU reproducer exposed an additional lifecycle defect. Normal
`/generate` SSE completion, client disconnect and parent-only SIGTERM passed,
but terminal-group SIGINT stopped the scheduler before it sent a final reply.
Uvicorn then waited for the stranded active connection while the supervisor
reported the expected worker stop as a crash. The initial **three passed, one
failed** result and both process logs are preserved in
`$HOME/results/flare-runtime/m7-http-lifecycle-initial.log` and its sibling run
directory. This uses actual HTTP/IPC/worker wrappers with synthetic model replies;
it is not evidence about model quality or GPU execution.

The follow-up compatibility decision is recorded in the existing
[design issue 98](https://github.com/sixteen-miles-labs/sparklab/issues/98).
Foreground native scheduler and tokenizer/detokenizer workers now leave the
terminal process group, as shell workers already did. The frontend owns terminal
signals and retains Uvicorn's existing active-request drain. Lifespan/finally then
explicitly signals and reaps its workers. This changes foreground POSIX worker
process-group membership; direct worker signals and the shared forced-stop
backstop remain. No overall HTTP timeout, model execution, sampler, cache format,
feature-selection default or container backend changed. Earliest interruptions
before the worker reaches its detach wrapper remain unqualified.

The new actual `/generate` stream checks normal completion, disconnect/retry,
parent SIGTERM, group SIGINT and group SIGTERM. Stop cases observe Uvicorn's
active-connection drain before releasing the final synthetic reply, then verify
both data frames and the terminal marker, no surviving owned workers/semaphores,
and no false supervisor-crash report. The disconnect case observes AbortMsg at
the worker, empty frontend acknowledgement/event maps, and independent retry
completion. A tenth process-only scenario stops the actual local tokenizer by
signalling the frontend group. Test failure cleanup also explicitly targets
recorded detached children; the HTTP reader join guards incomplete startup.

Final extracted-wheel evidence under `$HOME/results/flare-runtime/`:

- `wheels-http-lifecycle/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl`, SHA-256
  `86a1770eb232d38593044eb409f688fdce249328c11ca562923071020c319152`.
  All **436 Python package modules** match current source byte-for-byte.
- Expanded CPU with the hosted selections and shared cache/MoE regressions:
  **2028 passed, 201 skipped, six deselected**, 106.21s
  (`m7-http-lifecycle-wheel-cpu.log`). Hardware/optional-backend skips remain
  unqualified.
- Actual CUDA transfer/adoption/paused-writer shutdown: **three passed**, 3.17s
  (`m7-http-lifecycle-wheel-cuda.log`), without a full model.
- Source active-HTTP matrix: **five passed**, 26.94s
  (`m7-http-lifecycle-source-complete.log`). Initial post-fix four-case run:
  **four passed**, 20.86s (`m7-http-lifecycle-source.log`).
- Final fixture cleanup guards: **seven passed, 15 deselected**, 34.18s
  (`m7-http-lifecycle-final-fixtures.log`), covering all five HTTP cases plus
  actual tokenizer group-stop and partial launch on the same wheel.
- Ten main process cases: `m7-http-lifecycle-wheel-process/`; five main HTTP
  cases: `m7-http-lifecycle-wheel-http/`. Final focused reruns have separate
  `m7-http-lifecycle-wheel-final-{http,process}/` directories.
- Summary, test hashes, capacity and limits: `m7-http-lifecycle-validation.json`;
  all package module hashes: `m7-http-lifecycle-artifact.json`.

Ruff F checks on changed/new lifecycle files and `git diff --check` pass. All
owned build, test and probe processes are terminal. Final capacity was 24.87 GiB
free physical GPU memory and 25.71 GiB free disk. The full-model server remains
below the preselected 30 GiB guard; unrelated services/files were not stopped or
deleted. These finite synthetic-output checks do not qualify full-model/public
protocol/tool behavior, GPU cancellation, real backpressure, multi-rank,
restart/rollback or integrated soaks. M3 cross-partition and M5 strict quality
failures, balanced benefit/overhead gates and hosted CI/DCO remain open. The goal
remains active and no feature or recipe has been promoted.

## M7 backend failure and admitted HTTP requests

A real worker exception left admitted generation callers waiting even though the
supervisor had closed admission and woken cache-rebuild callers. The same actual
HTTP/IPC probe admits nine buffered/streaming requests across Completions, Chat,
Responses, Anthropic and `/generate`. The previous HTTP-lifecycle wheel leaves
**nine pending, zero completed HTTP requests** at the shared five-second wait
bound. The new backend-failure wheel wakes all nine with explicit errors.
[Design issue 98](https://github.com/sixteen-miles-labs/sparklab/issues/98) records
the public failure/compatibility decision before implementation.

Frontend failure notification runs on the owning event loop, preserves queued
genuine terminal replies, handles failure racing the first send, and is
idempotent. Late real replies still update accounting without reopening failed
HTTP delivery. Admitted buffered callers receive 503; committed streams use
protocol error events and omit invented final usage totals. Public messages omit
worker exception details. Local HTTP errors never masquerade as scheduler
acknowledgements: the nine requests remain active with zero completed requests,
and the actual prepare-stop endpoint refuses to seal uncertain accounting.
No model math, feature selector, checkpoint format or container backend changed.

The source focused API/accounting checks passed **123 tests** in 8.14s. The final
wheel `wheels-backend-failure/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl` has
SHA-256 `67f4772f0e07cc2ca1fc03cadbecfdc0022750afceceb855179f4935f2d28037`;
all **436 Python package modules** match source. Its expanded CPU regression
passed **2031 tests**, with **201 skipped and six deselected**, in 112.34s
(`m7-backend-failure-wheel-cpu.log`). Its real CUDA transfer/adoption/shutdown
checks passed **three tests** in 3.19s (`m7-backend-failure-wheel-cuda.log`).
The new three-test module also passed with `SPARKLAB_DISABLE_JIT=1` in 9.48s;
this is a local CI-mode check, not hosted CI execution.

Raw evidence remains under `$HOME/results/flare-runtime/`. Initial source
failure: `m7-backend-failure-initial.log`; matched prior-wheel failure and exact
waiter counts: `m7-backend-failure-prior-wheel-final/` and its `.log`; candidate
protocol bodies/accounting: `m7-backend-failure-wheel-protocols/result.json`.
The failed-probe cleanup had killed the resource tracker along with the frontend,
leaving parent queue semaphores. Tests now kill their recorded children and the
frontend individually so the tracker can finish. Nine stale semaphore names from
three known failed probes were unlinked only after their recorded parent/worker
PIDs were absent; `m7-backend-failure-stale-probe-cleanup.json` records them.
The final prior-wheel failed probe leaves no named semaphores or owned PIDs.
This fixture correction does not change production signal handling.

Full-model public-protocol/GPU-failure/cancellation/restart/endurance qualification
is still open: execution and the tiny chat template here are synthetic, while
HTTP, IPC, supervision and accounting are real. The latest capacity observation
was 18.39 GiB free physical GPU memory, below the unchanged 30 GiB complete-server
guard, and 25.68 GiB free disk. M3 cross-partition and M5 strict quality failures,
balanced benefit/overhead gates and hosted CI/DCO remain unresolved. No unrelated
service or file was stopped or deleted, and no feature or recipe was promoted.

After correcting failed-test cleanup, a final JIT-disabled wheel selection passed
**10 tests, 15 deselected** in 47.52s (`m7-backend-failure-final-fixtures.log`):
the three backend-failure/race cases, all five HTTP lifecycle cases, actual
tokenizer group-stop and partial launch. The same final HTTP failure probe against
the prior wheel fails as expected with **nine pending/zero completed**; the new
wheel has **zero pending/nine terminal HTTP results**, without transport errors.
The prior-wheel negative controls and their expected assertion failures remain
recorded, not counted as candidate passes. Summary/source/test hashes, all run
results, 26 directories of PID/semaphore cleanup evidence and limits are in
`m7-backend-failure-validation.json`; `m7-backend-failure-artifact.json` records
all package hashes. Final source/package reconciliation, Ruff F checks and
`git diff --check` pass. All owned build/test/probe processes are terminal.

## M7 actual TCP backpressure across protocol adapters

Added ten real HTTP/IPC cases: resume and disconnect for `/generate`, Completions,
Chat, Responses and Anthropic. A paused socket with a small receive buffer must
leave an actual ASGI body send waiting while undelivered replies accumulate.
A second request through the same protocol must still finish. Resume then
verifies all 8 MiB of synthetic text, protocol termination and usage; disconnect
must reach the worker abort path and receive a genuine terminal accounting
acknowledgement. Both retire frontend request maps/queues and owned processes and
semaphores. The test worker now emits that abort acknowledgement rather than only
recording the abort marker. Model execution and the tiny chat template remain
synthetic; TCP, ASGI, adapters, ZMQ and accounting are production paths.

The extracted backend-failure wheel is unchanged, SHA-256
`67f4772f0e07cc2ca1fc03cadbecfdc0022750afceceb855179f4935f2d28037`.
All **436 production Python modules** still match source. No rebuild, dependency
change, model download or repeated CUDA run was needed for these test-only edits.
The same wheel retains its previous three CUDA transfer/adoption/shutdown passes;
those are not new full-model or backpressure CUDA evidence.

Evidence under `$HOME/results/flare-runtime/`:

- Initial `/generate` checks: **two passed**, 16.46s
  (`m7-http-pressure-initial.log`).
- All five protocols, JIT disabled as in the ordinary hosted CPU selection:
  **ten passed**, 82.11s (`m7-http-pressure-protocols.log`). This is local execution,
  not proof that hosted CI ran.
- Expanded wheel CPU, including prior lifecycle/backend-failure and wider
  cache/MoE regressions: **2041 passed, 201 skipped, six deselected**, 207.13s
  (`m7-http-pressure-wheel-cpu.log`). Hardware/optional-backend skips remain open.
- Final ten case records: `m7-http-pressure-wheel-protocols/`. Each observes one
  waiting send and at least 131072 queued characters, then a successful second
  request. Every resume delivers 8388608 characters and ends with 66 observed
  completion counts across its 64 synthetic chunks and the two-chunk fast
  request. Disconnect cases retain only the fast request as completed, observe
  the actual sampled prefix counts and finish with zero active requests.
- Compact results, test hashes, unchanged-package identity, capacity and 38 run
  directories checked for absent owned PIDs/semaphores:
  `m7-http-pressure-validation.json`. Large response bodies are not retained;
  each resumed wire body has a recorded hash and an independent all-`x` oracle.

Ruff F checks and `git diff --check` pass. All owned test/probe processes are
terminal. Final capacity was 9.82 GiB free physical GPU memory, below the unchanged
30 GiB complete-server guard, and 15.61 GiB free disk. A quiet qualification window
has been requested; no existing workload was stopped or global cache dropped.
These finite checks establish frontend delivery/cancellation behavior, not an
8K/32K context result, global memory bound, sustained overload, full-model GPU
cancellation/restore, restart or soak qualification. The queue observation excludes
active iterator/ASGI/transport buffers. M3/M5 quality, matched performance/overhead,
full-model application gates and hosted CI/DCO remain unresolved; the goal stays
active and no feature, recipe or certification was promoted.

## M7 executable application sessions and strict oracles

Added `benchmarks/validate_runtime_sessions.py` and a paired native-server fixture
in `tests/e2e/test_runtime_sessions.py`. The runner executes 20–50 turns with a
real parsed `lookup_fact` call/return every fourth turn, two branches from a
shared parent, and a structured fact/source summary. Twenty turns plan 28
requests; fifty plan 65. Optional recall adds three facts at each exact rendered
8192/32768-token length. User messages, tool results and branch expectations use
only fixed synthetic records. The parent branch must retain the original fact
after a sibling amendment. Every answer checks strict JSON, value types and source
attribution; tool names/arguments/IDs and usage must reconcile.

The native tokenizer renders every actual request. Baseline comparison rejects
changed token-ID hashes, output signatures, output counts, cache/model identity,
missing observations and incomplete runs. Tool wire IDs stay linked in actual
assistant/tool history; generated IDs are excluded only from the output signature,
not from rendered input identity. The selected Qwen template does not render those
IDs, confirmed by the real tokenizer. Partial evidence records an in-flight call
before sending, then replaces the report after each step; interruption remains
unfinished rather than counting an unvalidated request as success. The CLI refuses
to overwrite its baseline and preserves mismatch evidence before returning failure.

The paired full-model fixture clears inherited feature settings, launches a
baseline then an explicit target-only candidate, captures actual cache capacities,
checks observed feature activity, and performs ordinary SIGTERM cleanup. Its
small-profile guard has a **30 GiB floor before each boot**. The default candidate
uses exact prefixes alone; disk writes, if explicitly selected, target an owned
test directory. Geometry mismatches and forced-stop/semaphore-warning cleanup fail
the gate. This source fixture has not executed model inference on this host.

Evidence under `$HOME/results/flare-runtime/`:

- Final extracted-wheel focused runner/HTTP/CLI/fault/comparison controls plus the
  previous mixed-harness checks: **32 passed**, 9.43s
  (`m7-sessions-final-review.log`). This includes actual 20- and 50-turn CPU
  HTTP fixtures, five/twelve executed tool returns, interruption, input drift and
  wrong fact/source/tool/usage/truncation rejection. Their model responses are
  synthetic, so these are runner controls, not model validation.
- CLI control/candidate/mismatched-input and interrupted reports:
  `m7-sessions-final-review-fixtures/`. The valid pair passes; the independently valid
  run with changed rendered inputs fails comparison; an interrupted call stays
  non-finalized with zero passes.
- Using the complete local Qwen artifact's tokenizer through the extracted wheel,
  **six prompts** have exactly 8192/32768 rendered tokens and CUDA remains
  uninitialized (`m7-session-tokenizer-final-tool.json`). These are length/identity
  checks, not factual recall answers. The first CPU search probed an oversized
  upper bound and emitted a tokenizer warning; the measured final search bracket
  avoids that probe and preserves all six final token-ID hashes.
- The actual paired model fixture reports **one skip** at the fixed physical-memory
  guard (`m7-real-sessions-guard-final.log`). No model server was launched. Final
  capacity was 8.77 GiB free GPU memory and 14.43 GiB free disk.
- Summary, source hashes, all intermediate runs and limits:
  `m7-sessions-validation.json`. Commands and environment variables are implemented
  and documented in `FLARE_benchmarks.md`.

All 436 production Python modules remain identical to the prior backend-failure
wheel (`67f4772f0e07cc2ca1fc03cadbecfdc0022750afceceb855179f4935f2d28037`).
The previous 2041-pass expanded CPU regression and three CUDA ownership checks
remain scoped to that unchanged artifact; they were not rerun or arithmetically
combined into a new full-suite count. Ruff F and `git diff --check` pass. All owned
checks are terminal. Full-model application answers/state, legal-summary quality,
restore/restart/cancel integration, matched performance/overhead, hosted CI/DCO and
soaks remain open. M3 cross-partition and M5 strict parity failures are unchanged;
no feature/default/recipe/certification promotion or unrelated service stop occurred.

## M4/M7 executable restart and rollback dispatch evidence

Added a four-boot native fixture in `tests/e2e/test_prefix_restart.py` and the
standalone `benchmarks/validate_prefix_restart.py` verifier. Cold query,
publication, disk restart/extended query and disabled rollback use identical
target-only capacities. The default prefix is 65 tokens, exercising a private
partial page; its extended query has 66 tokens and requests four output tokens.
Verification requires raw prefill work of 66/65/1/66 tokens, the corresponding
decode work, exact cached usage, matching strict output, recovered records before
the first restored query, and zero outstanding transfer resources. Disabled
rollback must leave retained files unchanged. Full work counters must reconcile
with raw traces, so a hit counter cannot mask full recomputation or truncation.

The fixture records actual geometry, effective token identities, metadata hashes,
commands, requests/replies and ordinary shutdown. Existing empty cache-rebuild
control messages wake the idle scheduler while asynchronous recovery completes;
they do not rebuild pools or dispatch model work. Geometry and empty startup
traces are checked. The diagnostic restore policy is `always`, not a cost/benefit
qualification. The fixed 30 GiB free physical-memory floor applies before each
boot; the owned 2 GiB store requires 3 GiB free disk initially. Larger context/page
profiles must be explicit. No production source changed for this fixture.

Evidence under `$HOME/results/flare-runtime/`:

- **44 passed**, 2.63s: 32 independent restart-evidence checks plus 12 production
  checkpoint service/observation checks (`m7-restart-order-focused-final.log`). Faults
  include fake hits, truncated/missing dispatches, startup warming, output/usage
  drift, geometry/identity changes, resource leaks, out-of-order dispatch and failed cleanup.
- The 32 evidence checks also pass with JIT disabled, 1.80s
  (`m7-restart-order-nojit.log`). Actual `RuntimeTelemetry` serialization is
  covered. These runs overlap and are not added into a new full-suite count.
- Standalone CLI accepts the synthetic valid report and rejects a fabricated hit
  that recomputes the full query (`m7-restart-cli-{valid,false-hit}.{json,log}`).
  These are checker controls, not model inference.
- The actual Qwen tokenizer verifies exact prefix/query pairs 65/66, 8192/8193
  and 32768/32769 with matching prefix hashes and CUDA uninitialized
  (`m7-restart-tokenizer.json`). No model state or recall answer is tested.
- Real full-model fixture: **one skip**, 0.85s at the unchanged memory guard
  (`m7-restart-guard-final.log`). No model server launched. Final observed capacity
  was 20.35 GiB free GPU memory and 7.62 GiB free disk, still below the GPU floor.
- Source hashes, unchanged wheel identity, intermediate logs, CLI results and
  limitations: `m7-restart-validation.json`. All 436 production Python modules
  still match wheel `67f4772f0e07cc2ca1fc03cadbecfdc0022750afceceb855179f4935f2d28037`.
  The prior 2041-pass CPU regression and three CUDA ownership checks remain
  scoped to that artifact; neither was rerun for these benchmark/test-only edits.

Commands and profile limits are in `FLARE_benchmarks.md`. Full-server restart,
8K/32K continuation/recall, cancellation during restore, measured benefit,
contention, hosted CI/DCO and soaks remain open. M3/M5 quality findings remain
unresolved. Ruff F and `git diff --check` pass; owned checks are terminal. An initial
collection command mistyped the observation test filename and ran no tests; its
corrected command produced the 44-pass result above, with both logs retained.
No default, recipe, certification or unrelated service changed.

## M4 real CUDA staged-read cancellation and poisoned retry

Added `tests/runtime/checkpoint/test_transfer_cancel.py`: an independently seeded
CPU record is recovered through the production asynchronous store and real
`GPUStateSink`. Six cases combine three paused read stages (before copying, after
the first completed CUDA copy, after checksum validation) with cancelling all
waiters or keeping one coalesced waiter. File pins, three private pages and one
state slot remain owned while the worker is paused. A cancelled read publishes
nothing and returns its allocations after completion. Retry into poisoned KV,
convolution, recurrent and auxiliary storage restores all four components exactly.
The surviving-waiter cases publish one complete restore. Every case ends with
zero outstanding ticket/staging/waiter/private-restore resources; explicit cache
eviction returns all 128 pages and 31 allocatable recurrent slots.

This is real-file/CUDA ownership evidence with synthetic state. It does not run a
model, HTTP cancellation, or a cancellation while a CUDA operation remains
pending: copies complete normally around the gated I/O worker. Those broader
gates remain open. The tiny transfer fixture does not require loading model
weights or lowering the separate 30 GiB complete-server memory guard.

Evidence under `$HOME/results/flare-runtime/`:

- Final combined extracted-wheel CUDA suite: **nine passed**, 2.95s
  (`m7-transfer-cancel-combined.log`), comprising three existing transfer/adoption/
  shutdown checks and six new cases. Six unique case reports under
  `m7-transfer-cancel-combined/` retain stage observations and component hashes.
  Pytest's `current` symlinks are excluded from case counts.
- Initial corrected six-case run: **six passed**, 2.67s
  (`m7-real-transfer-cancel-pins.log`). An earlier assertion incorrectly expected
  one pin; the ticket and synchronous read each correctly own one. Its failed log
  remains in `m7-real-transfer-cancel.log`.
- CPU service/observation checks with the native CI job's JIT setting:
  **12 passed, six skipped**, 2.64s (`m7-transfer-cancel-cpu-jit.log`). The six
  skips are the CUDA cases. An earlier JIT-disabled invocation had 11 passes,
  one failure and six skips because the incompatible installed kernel cache
  requires runtime fallback; the rejection was preserved, not bypassed.
- `m7-transfer-cancel-validation.json` reconciles six reports, exact component
  hashes, test source identity and all 436 unchanged production modules against
  wheel `67f4772f0e07cc2ca1fc03cadbecfdc0022750afceceb855179f4935f2d28037`.
  Final observed free capacity was 20.63 GiB GPU and 7.78 GiB disk, so full-server
  qualification still cannot start at its 30 GiB floor.

The prior 2041-pass CPU suite was not rerun or combined with these focused counts.
Hosted CI already selects this new file through `tests/runtime/checkpoint`; its
actual hosted execution remains unrun. Ruff F and `git diff --check` pass. No
production code, recipe, default, dependency or unrelated service changed.
Full-model restore/cancel/restart, M3/M5 quality, balanced benefit/interference
trials, hosted CI/DCO and soaks remain open.


## M7 fictional case summary and chronology workload

The optional `--legal-case` session workload adds two independently checked
requests over four wholly fictional case documents: a structured, attributed
summary and date-sorted chronology. It distinguishes original and corrective
deadlines, preserves both party statements as unverified, and requires JSON null
for an absent court finding. Fixed oracles reject missing/extra facts, wrong
sources, invented rulings, changed types and document ordering errors. Case record
and oracle hashes enter workload identity; existing strict rendered-input, output
and completed-work comparison applies. Default session work remains 28/65 requests;
selecting the case makes it 30/67 before optional recall. The paired full-model
fixture accepts `SPARKLAB_SESSION_LEGAL_CASE=1` for both control and candidate.

Evidence under `$HOME/results/flare-runtime/`:

- **42 passed**, 15.13s (`m7-legal-session-controls.log`): 35 session/CLI/fault
  controls plus seven prior mixed-harness checks. HTTP model replies are synthetic.
  Eight new fault cases reject established allegations, wrong sources/deadlines,
  missing positions, invented rulings/fields, wrong null types and reversed dates.
- CLI control/candidate reports each complete all 30 requests; their matched
  comparison passes. Another independently correct run with changed rendered
  inputs fails comparison. Reports live under `m7-legal-session-controls/`.
- Actual native Qwen serving tokenizer: **two prompts**, 344/309 tokens; canonical
  JSON answer budgets are 132/84 tokens within 256 outputs and 4096 context
  (`m7-legal-tokenizer.json`). CUDA stays uninitialized. No answers were generated.
- Full-model paired fixture: **one skip**, 0.91s at the unchanged 30 GiB floor
  (`m7-legal-model-guard.log`). Latest capacity: 19.97 GiB free GPU and 7.72 GiB disk.
- `m7-legal-session-validation.json` records source hashes, reports and all 436
  production Python modules unchanged from wheel
  `67f4772f0e07cc2ca1fc03cadbecfdc0022750afceceb855179f4935f2d28037`.

Ruff F and `git diff --check` pass; owned checks are terminal. No native code,
weights, precision, recipe, default or dependency changed. Full-model case answer
preservation, state/recall, M3/M5 quality, benefit/interference, hosted CI/DCO and
soaks remain open. This workload tests runtime preservation, not legal correctness
or general legal-summary quality. Earlier 2041 CPU and nine CUDA passes remain
scoped to the unchanged artifact and are not summed with these focused checks.


## M4 completed file read with a real pending final CUDA DMA

Added `tests/runtime/checkpoint/test_transfer_fence.py`. It recovers the independent
synthetic record through the actual store/transport and injects one idempotent
24-byte auxiliary-state DMA onto the restore stream after the ordinary staged
copies and checksum validation. A CUDA driver wait on four requested bytes of
mapped pinned host memory holds that final copy. No transfer-complete/event-query
result is mocked. The payload's additional 24-byte pinned test buffer and driver
allocation overhead are outside production staging counters; no total-memory-bound
claim comes from them.

Four cases observe a completed I/O ticket and pending real event/stream, then
cancel all waiters, retain one coalesced waiter, expire the restore timeout or
attempt a short shutdown. The completed ticket, 4096-byte staging window, file pin,
three private pages and one state slot remain owned. Another primed stream
completes actual work during the hold. After host release the event completes;
cancelled paths publish nothing, the surviving waiter restores every component
byte-exactly, and all cases return 128 pages and 31 allocatable state slots after
cache eviction. A ten-second host rescue prevents an indefinite held stream;
rescue firing fails the gate.

Raw evidence under `$HOME/results/flare-runtime/`:

- Combined extracted-wheel CUDA: **13 passed**, 3.32s
  (`m7-pending-fence-combined.log`), including the previous nine cases and these
  four. Four unique reports record actual event/stream queries, I/O completion,
  retained/drained resources, independent progress and ordinary close.
- Focused four-case run: **four passed**, 2.53s (`m7-pending-fence-owned.log`).
- CPU service/observation checks: **12 passed, four skipped**, 2.64s
  (`m7-pending-fence-cpu.log`). New CUDA cases skip explicitly; JIT remains enabled
  as in the native CPU CI step. The incompatible kernel cache is still rejected.
- Setup failures remain in `m7-pending-fence-initial.log` (one failure: test callback
  missing thread-local inference mode) and `m7-pending-fence-corrected.log` (one
  failure: idle-only integrity reductions invoked during the held transfer).
  Both triggered the rescue and are not passes. Cleanup now finalizes even on
  callback failure; full integrity checks run after the transfer drains.
- `m7-pending-fence-probe.log` records a primed real-event/independent-stream probe.
  An earlier ad hoc unprimed probe assertion failed and is not passing evidence.
- `m7-pending-fence-validation.json` reconciles four reports, test hash, capacity
  and all 436 unchanged production modules against wheel
  `67f4772f0e07cc2ca1fc03cadbecfdc0022750afceceb855179f4935f2d28037`.
  Latest free capacity: 17.16 GiB GPU and 8.43 GiB disk. Full-server memory floor
  remains 30 GiB; no model server was launched or existing service stopped.

Ruff F and `git diff --check` pass; owned processes are terminal. The new file is
selected by the existing native CPU CI directory, but hosted CI has not run. No
production module, model, precision, default, recipe or dependency changed. This
is scoped pending-GPU ownership evidence with an explicitly injected final write;
it does not prove an ordinary file read naturally finishes before staged copies,
full-model HTTP cancellation/state, performance/interference, sustained pressure
or soaks. M3/M5 quality and all broader release gates remain open.


## M5 scheduler feedback reachability repair and rebuilt artifact

A source audit found the adaptive feedback loop below the unconditional return
of `_runtime_snapshot()`. The loop never retired `req.adaptive_round` or trained
its request-local acceptance/cost history. A continuing request could therefore
fail its next preparation with the previous round still attached. Direct-engine
state/policy tests had not covered this real scheduler transition; their passes
did not establish a working adaptive server loop.

Moved feedback to the end of `_process_last_data()`, after the existing output
copy fence and reply handoff, where its complete service cost belongs. Continuing
and EOS-completed rounds train once; aborted rounds clear ownership without
training. This is independent of telemetry. The existing last-observed IPC
snapshot can precede the just-handed-off round's feedback counters; it does not
claim a synchronous final census. No new device wait or default change was added.

`tests/runtime/scheduler/test_adaptive_round_drain.py` drives actual page/slot,
engine round preparation, policy observation and scheduler drain logic. Twelve
CPU cases cover depth zero/three, continuing/EOS/aborted work and telemetry off/on;
they require the output fence and handoff before training, censored acceptance,
clear ownership and successful next preparation. CPU forward outputs are fixture
inputs, not model inference. One CUDA case uses actual round/copy events and a
small tensor operation to check the GPU timing observation after drain.

Evidence under `$HOME/results/flare-runtime/`:

- Prior wheel fails the first regression as expected with the round still attached:
  **one failed**, 1.77s (`m7-adaptive-drain-prior-wheel.log`, stopped at first
  failure). This is a negative control, not a pass.
- Corrected source focused selection: **38 passed, one skipped**, 2.08s
  (`m7-adaptive-drain-source-final.log`). An earlier test run had 36 passes/two
  failures because the test requested depth three inside a two-output budget;
  the safe depth guard correctly selected zero. The corrected fixture uses a
  legal larger budget and EOS completion. Both logs are retained.
- Offline build succeeds (`m7-adaptive-drain-build.log`). Artifact
  `wheels-adaptive-drain/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl`, SHA-256
  `bfb05f400039f7b3c41c5bfbc53f251d0bfa7b25002baaace6b02ddf913280fc`.
  All **436 Python modules** match source. Only scheduler.py differs from the
  prior backend-failure wheel; module hashes and the scheduler-only diff are
  `m7-adaptive-drain-artifact.json` and `m7-adaptive-drain-vs-prior.patch`.
- Expanded extracted-wheel CPU: **2120 passed, 212 skipped, six deselected**,
  211.61s (`m7-adaptive-drain-wheel-cpu.log`). Selection covers existing hosted
  package/API/harness areas plus scheduler, all native cache/engine/checkpoint
  and MoE tests. CUDA is hidden; native CPU JIT is enabled. Optional/hardware
  skips remain open. No model download or dependency upgrade was needed.
- Extracted-wheel CUDA: **14 passed**, 4.31s
  (`m7-adaptive-drain-wheel-cuda.log`): the new real-event feedback case and the
  previous 13 transfer/adoption/shutdown/cancellation/pending-fence cases.
  No full model, adaptive API or quality/benefit test is implied.
- Bounded source audit for terminal statements followed by sibling code:
  `m7-adaptive-drain-unreachable-audit.json`. The remaining finding is pre-existing
  FLA code after its head_first deprecation raise; it was inspected and untouched.
  This is not a general reachability proof.
- `m7-adaptive-drain-validation.json` records source/packaged identity, failures,
  selections and capacity. Latest free capacity was 16.19 GiB GPU and 8.41 GiB disk,
  below the unchanged 30 GiB complete-server floor.

Ruff F and `git diff --check` pass; owned test/probe processes are terminal.
Current broad counts are actual results, not sums with earlier focused suites.
The new CPU/GPU-marked file is included through the existing native CI directory;
hosted CI/DCO remains unrun. No recipe, precision, weights, sampler, live default,
public support claim or unrelated service changed. Fixed-depth rollback remains
available. M3 cross-partition and M5 strict parity failures remain unresolved;
earlier direct-engine state evidence stays scoped to its original artifact.
Full-model repaired serving execution, stage reconciliation, periodic live expert
observations, matched benefit/interference and soaks remain open.


## M1 bounded asynchronous live expert counters

Implemented `ExpertCounterSampler` alongside existing telemetry and initialized
it only with runtime timing plus the existing MoE stats collection option. It
aggregates the offload cache's three LRU columns (or hybrid scalar counters) and
fetched count on the engine stream, then copies four int64 values to one pinned
host window. One event and one pending sample are reused, with submissions at
least one second apart. The actor only queries readiness and converts CPU values
after completion; it never invokes synchronizing terminal counter readers.
Source caches/weights are not retained by the sampler. Existing rebuild/shutdown
fences order and drain these operations. No new configuration flag or default
was introduced.

The existing expert_decode shape is preserved. The timestamp is the sample's
host submission time, conservatively older than completion; newer terminal
observations take precedence. Host-only prefill fields are captured at submission.
One cached completed result and one pending result remain bounded and immutable.
The optional retained_workspaces.expert_counter_sampling object reports 32-byte
device and 32-byte pinned payloads and an unconsumed-sample flag. These counts
exclude allocator/driver overhead. Idle observations may remain stale; no
per-round or one-second delivery guarantee, atomic cross-subsystem snapshot or
measurement-overhead qualification follows.

Evidence under `$HOME/results/flare-runtime/`:

- New source controls: **eight passed**, 1.90s
  (`m1-live-expert-source-complete.log`), comprising two CPU precedence/disabled
  controls and six actual CUDA cases. Source focused CPU selection: **23 passed,
  seven skipped**, 1.85s (`m1-live-expert-source-cpu.log`).
- GPU cases cover ordinary/hybrid independent count oracles, zero activity,
  rate limits, immutable observations, actual scheduler export and a real held
  copy with independent-stream progress. Guards forbid explicit synchronization,
  CPU transfer or device scalar conversion inside the production poll. The
  pending window is not reused even across repeated overdue polls.
- Initial seven-case run passed; adding workspace reporting exposed one incomplete
  CPU observation stub (seven passes/one failure in
  `m1-live-expert-source-final.log`). Supplying the stub's buffer metadata fixed
  the test; no production fallback was added. Intermediate logs are retained.
- Offline rebuilt wheel: `wheels-live-expert/sparklab-0.1.3-cp311-cp311-linux_aarch64.whl`,
  SHA-256 `5dc3bdbf54b368f71f6d471ae1051b16546ee0c9a08c6f34dd477d76007596ab`.
  All **436 Python modules** match source. Only scheduler and telemetry differ
  from the adaptive-drain wheel (`m1-live-expert-artifact.json`).
- Expanded extracted-wheel CPU: **2122 passed, 218 skipped, six deselected**,
  218.15s (`m1-live-expert-wheel-cpu.log`). Selection includes hosted areas and
  all native cache/engine/checkpoint, scheduler and MoE suites. Native CPU JIT
  stays enabled; incompatible kernel-cache rejection is preserved.
- Extracted-wheel CUDA: **20 passed**, 4.23s (`m1-live-expert-wheel-cuda.log`):
  six new sampling cases and the prior fourteen adaptive/transfer/ownership/fence
  cases. These are synthetic counters/state, not complete-model inference.
- Raw independent numeric samples: `m1-live-expert-values.json`. Nonhybrid:
  active=50, missing=10, calls=3, fetched=7, miss_rate=0.2. Hybrid: active=60,
  missing=15, calls=4, fetched=8, miss_rate=0.25. Their submission times are fixture
  clock values, not model latency observations.
- Existing design issue 97 records the bounded asynchronous extension,
  compatibility and unqualified gates; its published body is verified against
  `m1-live-expert-design-body.md`, with the previous body retained separately.
- `m1-live-expert-validation.json` records source/artifact identity, selections,
  failures, design verification and capacity. Latest free physical GPU memory
  was 11.46 GiB, disk 4.93 GiB, below the unchanged 30 GiB server floor.

Ruff F and `git diff --check` pass; all owned checks are terminal. New CPU and
CUDA-marked controls are selected through the existing native CI directory;
hosted execution/DCO is still unrun. No model/precision/sampler/recipe/default or
dependency change, download or unrelated service stop occurred. M3 cross-partition
and M5 strict parity findings remain unresolved. Full-model counter ordering,
protocol/stage reconciliation, <=2% measurement overhead, matched benefit and
expert/prefix interference, sustained pressure and soaks remain open. Earlier
artifact evidence is not silently transferred into these broader gates.


## Private-tail admission and publication capacity correction

The prior live-expert wheel overadmitted immutable partial-tail restores and
fractional per-request page budgets. Meaningful prior-wheel controls failed six
cases (`m4-tail-capacity-prior-complete.log`), with two CUDA cases explicitly
skipped in that CPU run. The earlier four-case negative is retained separately.
A later source control also reproduced allocation failure after partial snapshot
publication consumed future decode capacity (`m4-tail-capacity-before-reclaim.log`).

Admission now rounds each private branch separately, includes occupied partial-tail
rows, counts snapshot plus live pages on disk restore, and retries lookup once
following eviction of ready unread exact records when a hit cannot fit. Allocation
can reclaim such tails. Publication preserves the complete pending prompt/output
budget for all active request tables, including intermediate chunks; finish and
abort release reservations, and rebuild asserts no live reservations. No record
format, public flag, dependency, checkpoint, precision or default changed.

The 14 new physical-page oracles include ten CPU cases and four actual CUDA
cases using synthetic state. The corrected source selection passed 45 CPU cases
with four CUDA skips (`m4-tail-capacity-source-corrected.log`), and the CUDA
selection passed four. Initial fixture errors are retained in source-final and
source-verified logs: intermediate chunks must be advanced before consuming the
full declared work, and idle-only whole-cache conservation must wait until all
active requests finish. Neither failed oracle was waived.

The rebuilt tail-capacity wheel matches all 436 production Python modules; five
changed relative to the live-expert wheel: core, prefill, cache, exact, service.
Its SHA-256 is bb13d0d12069266a67c6df5438aff45cb594ac7be7945f8115756a3040de6d05.
The combined actual CUDA suite passed 24 cases in 4.37 seconds, including the four
new admission/publication/reclamation cases and the prior 20 scalar/transfer/fence/
adaptive-event cases. Full-model continuation and performance do not follow from
these controls. Resource observation: 8.48 GiB GPU free, 3.58 GiB disk free;
full-server testing remains below the unchanged 30 GiB floor. Final wheel CPU
results and the exact artifact manifest are recorded in m4-tail-capacity-validation.json
and m4-tail-capacity-artifact.json outside git. M3/M5 strict quality findings,
full-server restart/API tests, balanced performance, hosted CI and soaks remain open.

Final expanded tail-capacity wheel CPU run: **2132 passed, 222 skipped, six
deselected**, 209.12 seconds (`m4-tail-capacity-wheel-cpu.log`). The 24-case CUDA
run and this CPU run both terminated with exit zero. Ruff F and diff whitespace
checks pass. Focused counts overlap and must not be summed with this selection.
All owned jobs are terminal; no full-server boot, default promotion, commit,
push, PR or merge was performed in this correction.


## Speculative first-divergence numerical diagnostics

The prior five-stratum/six-mode strict matrix still fails 12 combinations; no
quality criterion was weakened. The complete-model fixture now repeats depth
zero after each target-only control, then retains the existing depths 3/7/11,
transitions and adaptive modes (35 natural cases total). Its original 24 forced
accepted-prefix state cases remain separate and do not train this diagnostic.

Test-scoped wrappers call the original greedy sampler/verifier, then record only
emitted rows, excluding discarded suffix rows and intermediate prompt chunks.
Each case is capped at 64 records of eight top logits, exact tie count/margin and
full FP32-vector SHA-256. Selected tokens must agree with the real argmax, and
recorded token IDs must equal completed output IDs. The first divergent token
includes both aligned records after the common output prefix. This does not
prove state equality or establish the reason for the earlier numerical failures.
No complete vector, GPU tensor or asynchronous owner is retained. Synchronous
CPU observations mean these instrumented timings cannot qualify performance.

`m5-logit-diagnostic-cpu.log`: 30 passed, three CUDA skips, 1.85 seconds.
`m5-logit-diagnostic-cuda.log`: three passed, ten deselected, 2.11 seconds on
the extracted tail-capacity wheel. These controls use synthetic logits through
the real greedy verifier for all-accept, late-reject and first-reject prefixes,
check immutable source bytes and detached observations, and reject accounting,
nonfinite, selection and bound violations. All 436 production module hashes
still match that wheel. No production sampler, model arithmetic or flag changed.

The complete checkpoint fixture skips before loading at its fixed 30 GiB free
physical GPU floor (`m5-logit-diagnostic-full-model.log`: one skip, 0.82 seconds).
The new minimum-free environment option may raise that floor, never lower it.
The expanded matrix and its wrappers have not executed on the complete model;
no numerical diagnosis or quality pass is claimed from the small controls.
Compact manifest: m5-logit-diagnostic-validation.json. Earlier M3/M5 failures,
full-model API/restart, matched performance/interference and soak gates remain.


## Executable bounded endurance evidence collector

Added benchmarks/validate_runtime_soak.py using the existing mixed streaming
client on an explicitly owned server. A matching baseline identity/workload is
required; every declared completed output, token count and finish reason is
compared exactly. Both completed and intentional socket-cancelled work must be
present. A full-path warmup precedes the timed interval; complete cycle work is
finished before stopping. No case is dropped to improve successful-work counts.
Attempted but unobserved requests remain explicit after a control-path failure.

Resource snapshots include raw worker age and the host's global pswpout page
counter. Instance changes fail rather than silently resuming a restarted server.
Frontend drain is checked after closed sockets. Optional idle rebuilds hold
geometry constant and retry busy prefix I/O within the declared timeout. They
are disabled by default. The collector never launches, kills or restarts a server.

Current-cycle memory is bounded by 64 cases and 16 client workers. Resource
responses are capped at 2 MiB, duration at eight hours, and JSONL evidence by an
explicit budget (default 256 MiB, maximum 4096 MiB). Progress/failure reports are
atomically replaced; existing output directories and oversized evidence writes
are rejected while retaining earlier records. This bounds collector evidence,
not a claim that runtime disk/staging/resource bounds passed an endurance run.

The final focused CPU run passed 17 cases in 5.83 seconds
(m7-soak-collector-final-verified.log), covering the real standalone CLI and HTTP
streams, intended cancellation, strict output differences, HTTP failure,
instance restart, geometry changes, busy rebuild retries, evidence exhaustion,
changed baseline and preserved previous reports. Earlier 13/16-case intermediate
logs remain. Raw controls live in m7-soak-collector-final-tests. The standalone
CLI completed one warmup and timed cycles, explicitly reporting qualification
false. Its replies are synthetic; no model or CUDA state is exercised here.

All 436 production modules still match the tail-capacity wheel. Hosted CI's
existing tests/test_*.py selection includes this new CPU-safe test file. Local
checks do not count as hosted execution. Compact manifest:
m7-soak-collector-validation.json. Commands and scope are in FLARE_benchmarks.md.

The actual 60-minute and four-hour model soaks remain unrun. Stale worker counters
cannot prove drained GPU/I/O ownership, global swap-out cannot establish runtime
attribution, and the collector does not automatically qualify memory/latency
drift, temperatures, full state, dynamic tools, factual answers, expert
contention or restart/disable behavior. Reports remain qualification:false even
when duration is satisfied. M3/M5 quality failures and all broader model/release
requirements remain open. Current-turn resource observation was 9.59 GiB GPU
free and 3.56 GiB disk free, below the unchanged 30 GiB full-server GPU floor.

Follow-up arrival-order correction: the collector now submits cases in arrival
order, matching the existing mixed client command. A real HTTP one-worker
control supplies the later request first and requires the immediate cancelled
request to start earlier. Final selection: **18 passed**, 6.42 seconds
(m7-soak-collector-arrival-verified.log). The final source/CLI hashes and canonical
report paths replace the intermediate manifest entries. All 436 production
modules still match the existing wheel; no new runtime build was required.
Ruff F and diff whitespace checks pass; all owned jobs are terminal. No model
soak or release claim follows from this collector check.


## Separate live and reserved allocator observations

The previous worker export reported reserved GPU memory only. Added separate
current gpu_allocated_bytes/gpu_reserved_bytes from one host PyTorch memory_stats
read after the once-per-second export gate. No GPU wait, tensor copy, allocator
flush, file read or new retained buffer is involved. Missing/non-CUDA counters
remain null, and caller resource dictionaries and prior observations are not
mutated. The legacy reply/frontend reserved-memory gauge is unchanged. The
existing issue 97 bounded optional resource design was inspected; no CLI,
model-support, checkpoint, precision, dependency or default changed.

The prior tail-capacity wheel fails the new controlled export check because the
live-allocation field is absent (m1-allocator-prior-wheel.log, exit one). Focused
source: 14 passed, seven hardware skips in 1.88 seconds; actual CUDA lifetime
control: one passed in 1.91 seconds. The new wheel matches all 436 source Python
modules; scheduler and telemetry are the only changes from the tail-capacity
wheel. SHA-256: 55cd6b160ef451f8066427f67934ae3018eb60d16a84d776f9aa9274509ffee5.
Expanded wheel CPU: **2156 passed, 226 skipped, six deselected**, 217.51 seconds
(m1-allocator-wheel-cpu.log). Combined CUDA: **25 passed**, 4.39 seconds
(m1-allocator-wheel-cuda.log), including prior state/transfer/event/counter gates.

The small real allocation record (m1-allocator-values.json) observes live bytes
0 -> 1048576 -> 0 while reserved bytes stay 2097152. Its export clock is explicitly
synthetic; this is not a timing or model result. An exploratory control initially
subtracted an uninitialized null value and raised TypeError. The corrected
control preserves the initial null, primes and releases one owned byte, and
compares known values. Null was never coerced to zero in production. The real
CUDA test also forbids public synchronization and tensor host-readback methods
during observation. Three CPU controls check disabled/CPU behavior, rate limits,
coherent pairing, input/observation immutability and unavailable statistics.

These counters have the selected PyTorch allocator backend's semantics; they
exclude driver/non-PyTorch owners. Allocated is contained in reserved. Named
buffer budgets overlap these values, and host RSS can overlap the same physical
GB10 pool. This is not a total physical-memory census, whole-model conservation,
host/swap attribution or a <=2% overhead result. Idle exports remain stale.
Complete-model M1 reconciliation, M3/M5 quality failures, full-server API/restart,
balanced benefit/contention, soaks and hosted CI/DCO remain open. Scoped manifest:
m1-allocator-validation.json. Ruff F and diff whitespace checks pass; all owned
jobs are terminal. No complete-model boot or promotion was performed.


## Parallel branch admission: charge already-owned pages once

Found two sources of unnecessary exact-cache admission failure. Fresh unaligned
hits copy a private tail during _try_allocate_one, which already debits the free
list; the batch accumulator previously charged that page again. A chunked
continuation also charged the unused rows of its already-owned partial page as
if it needed another physical page. Initial fresh-hit admission still checks
the full private footprint before allocation. Subsequent batch booking now
credits the copied page, and continuations reserve only rounded pages beyond
allocated_len through the complete pending prompt/output budget. Exact-off
baseline admission, resource limits, locks/fences and all model arithmetic are
unchanged. No sampler, record format, dependency, flag or live default changed.

The independent physical oracle uses P=4 prefixes 3/7/11, one immutable source
and exactly four spare pages for two two-page private branches. It covers two
fresh hits and an existing unaligned chunk continuation plus a new sibling.
Both branches must be admitted together, consume their full declared work,
retain disjoint private mappings, preserve every parent KV/conv/recurrent/aux
byte, and return every page/state/table/future reservation. This is tiny
synthetic state through real production cache/I/O paths, not model inference.

Prior allocator-observation wheel: **six failed, six CUDA skips**, 14 deselected,
2.34 seconds (m3-parallel-tail-prior-final.log). An intermediate fresh-only fix
still fails the three continuation controls (m3-parallel-tail-continuation-before.log).
Final focused source: **53 passed, ten CUDA skips**, 3.43 seconds; actual CUDA:
**six passed**, 20 deselected, 2.60 seconds. The CPU selection retains all prior
one-page-short/overcommit/pinned-reader/publication conservation controls.

Intermediate failure evidence is preserved. The initial CUDA oracle compared a
CPU inspection tensor to a CUDA expected tensor; correcting both to CPU retained
exact equality. A subsequent first case failed in a 768-byte MHA pool allocation
before changed admission, while two cases passed (m3-parallel-tail-source-cuda-corrected.log).
Its cause remains unresolved, like the earlier first-allocation OOM; final
source/wheel success must not be relabeled as an OOM fix or host-stability proof.
An intermediate fresh-only wheel remains explicitly unqualified. The final v2
artifact matches all 436 source Python modules, with only prefill changed from
the allocator-observation wheel. SHA-256:
33b716a12ca354db997a89e5bd94f42e3926f488e5292f988ad339b4a0b8fbc7.
Combined built-wheel CUDA regression: **31 passed**, 4.88 seconds, including the
six new cases and prior 25 allocator/state/transfer/fence/counter/adaptive cases.
Expanded CPU result remains in its authoritative process until terminal.
Compact manifest: m3-parallel-tail-validation.json.

No full-model quality, performance, HTTP restore/cancel, long-context recall,
restart, contention, hosted CI/DCO or soak gate follows from these page controls.
M3 cross-partition quality and the 12 M5 parity failures remain open; features
remain experimental and no default has been promoted.

Final broad parallel-tail v2 CPU result: **2162 passed, 232 skipped, six
deselected**, 217.02 seconds (m3-parallel-tail-wheel-cpu.log), exit zero. The
31-case CUDA run also exited zero. Source/module hashes still match the final
wheel, F lint and diff whitespace checks pass, and all owned jobs are terminal.
Focused counts overlap this suite and are not summed. No complete-model boot,
commit, push, PR, merge or default change occurred in this correction.


## Seeded exact-KV and aligned snapshot ownership in the shared harness

Extended the existing Session/page-trie/SlotLedger harness, not a parallel
cache model. Full-only pins use base RefModel KV leases independently of aligned
GDN readers. Four seeds (7/73/319/901), four page sizes (1/4/16/64), and 256 action
attempts per case give 4096 scheduled attempts. Inserts, matches/splits, full-only
pins, ordinary readers, releases and both evictions are interleaved. Model,
counter, structural and slot-ledger checks run after every attempt. At most eight
pins per kind are retained; final release/eviction requires every slot returned.
Failures retain seed, geometry and operation prefix in pytest-owned JSON traces.

Final all-radix plus exact/admission selection: **207 passed, 11 skipped**, 2.87
seconds (m3-anchor-seeded-shared-final.log). Skips comprise ten explicit CUDA
cases and the existing SWA page==window geometry where that named boundary cannot
be isolated. Earlier seeded+hybrid focused selection: 39 passed, 1.84 seconds.
Initial test-only construction errors (misplaced old test tail and a missing
in_use() call) are retained in initial/corrected logs; neither is a runtime defect.

The mutation process substitutes ordinary snapshot unlock for full-only unlock
without modifying source files. All **16 seeded cases fail** with a model-node
mamba-ref mismatch (m3-anchor-seeded-mutation.log). Saved seed-73/P1 prefix:
74 actions -> ten via task-local deletion reduction, 137 replay attempts. The
reduced valid source replay also drains/reclaims all slots. This is deletion
minimal with recorded lock indices, not a claim of global semantic minimality.
The four-operation semantic core (attach snapshot, lock its reader, lock full-KV,
unlock full-KV) is retained as a named test across all four page sizes. All four
pass in the source selection and **all four fail** under the same mutant
(m3-anchor-seeded-minimal-mutation.log). No difficult case or equality was waived.

All 436 production modules still match wheel
33b716a12ca354db997a89e5bd94f42e3926f488e5292f988ad339b4a0b8fbc7.
Only CPU test/harness files and documentation changed; no new production build
was needed. Existing hosted CPU radix selection includes these tests, but hosted
execution/DCO remain unrun. Compact evidence: m3-anchor-seeded-validation.json;
full mutation traces and the task-local shrinker remain outside git. Ruff F and
diff whitespace checks pass; all owned checks are terminal.

This addresses randomized CPU full-KV/snapshot ownership coverage only. It does
not establish model byte/state quality, device-fence behavior, mixed disk
lifecycle, native application answers, performance/interference or endurance.
M3/M5 quality findings, full-model API/restart/recall, balanced benefit/overhead,
expert contention, hosted CI and 60-minute/four-hour soaks remain open. The last
physical-capacity observation in this turn was 8.81 GiB GPU free and 2.88 GiB disk
free; the unchanged complete-server guard remains 30 GiB GPU free.


## Adaptive proposal / verification / engine-commit / GPU-handoff intervals

The existing cost path exported proposal and total GPU spans but omitted the
required verification/commit separation. Added two optional timing events per
round under existing runtime telemetry: five events in diagnostic cost rounds,
three in ordinary cost rounds, no adaptive events outside cost mode. Markers are
recorded after real sampling/verification and after engine state commit/replay
but before nonblocking output copy. The existing scheduler round-end fence
remains after the GPU token-pool write. Four fixed histograms and bounded trace
fields expose proposal, verification, commit and handoff; no separate queue or
retained tensor buffer is added. Whole host service cost still trains the policy.
No sampler, weights, precision, checkpoint format, prepared shape, dependency,
public CLI flag or default changed. The existing optional telemetry design covers
these fixed-name spans; interval definitions are in FLARE_design.md.

These are ordered GPU-timeline spans, not exclusive kernel busy time. Dispatch,
state snapshots, stream waits and host decisions can contribute. GPU handoff
ends before CPU reply delivery/client receipt. Missing/out-of-order/duplicate
phase markers cannot pass as zero measurements, and phase sums are not used as
a substitute for full service cost.

New controls drive actual Engine.forward_batch, Scheduler._forward and drain
with synthetic model/graph callbacks. CPU independent-clock proposal/snapshot/
target/decision/commit costs check exact marker placement. Four cases cover
zero depth, first rejection, later rejection and all accept. Actual CUDA cases
use small tensors/events; output copy must follow engine commit, phase spans
must reconcile to round time, histograms receive one observation and state/output
bookkeeping still drains. Separate event controls check optional 3/5 ownership,
marker order and output-fence readiness. A process-local incorrect marker mapping
(verification-end replaced by proposal-end) fails all four CPU pipeline cases
(m1-phase-mutation.log). No source mutation persists after that negative process.

Focused source: 29 passed, five skips, 1.86 seconds; CUDA: four passed, five
deselected, 2.11 seconds. The initial combined wheel CUDA selection found a test
isolation issue (m1-phase-wheel-cuda.log: one failed, 34 passed). The new synthetic
scheduler shell retained a self-capturing cleanup callback; later cyclic collection
released its tiny GPU buffers inside the next allocator control, changing the
baseline by 3584 bytes (9216 -> 5632). A test finalizer clears those owners. This
is not a production allocator leak fix or an explanation of earlier setup OOMs.
Isolation follow-up: five passed, 1.94 seconds. Final combined CUDA: **35 passed**,
3.79 seconds (m1-phase-wheel-cuda-final.log).

The built artifact matches all 436 production Python modules; engine, adaptive_dflash,
scheduler and telemetry changed from the parallel-tail wheel. SHA-256:
9c03ca7a885eb0ab5fa0be0f6e76ddb5ea03d70ee64289c33453f6befadd769d.
Expanded CPU: **2187 passed, 236 skipped, six deselected**, 209.41 seconds
(m1-phase-wheel-cpu.log). Test-only cleanup/preflight changes followed CPU
collection; final focused CPU covers those latest helpers (five passed, four
hardware skips, 1.76 seconds). Production code did not change after wheel build.

The complete-model fixture now enables runtime timing and records all phase
fields after actual drain, retaining all strict state/output assertions. Its
memory preflight previously initialized CUDA in the owning process, violating
Engine.__init__'s fresh-CUDA requirement when resources became available. The
probe now runs in a short-lived child. A fresh-parent control reports 8.23 GiB
free and parent_cuda_initialized:false (m1-phase-engine-guard.log). The complete
checkpoint matrix still skips at the unchanged 30 GiB floor (one skip, 1.81 seconds,
m1-phase-full-model.log); no new complete-model phase/quality claim follows.

Manifest: m1-phase-validation.json. Ruff F and diff whitespace checks pass; all
owned jobs are terminal. The prior M3/M5 quality findings, full-model stream and
stage reconciliation, <=2% diagnostic overhead, native API/restart/recall,
balanced benefit/contention, hosted CI/DCO and required endurance remain open.
Earlier first-allocation OOMs remain unexplained and preserved. No commit, push,
PR, merge, release, server promotion or live-default change occurred.


## Truthful dispatch outcomes, selection ages and first-rejection positions

The prior dispatch trace retained the prefill policy reason even when actual
admission failed and decode/verification ran instead. Queue peak was measured
only after successful admission, which could erase a backlog. Dispatch telemetry
now preserves policy_reason separately, records prefill_unavailable only for an
actual fallback, and observes queue depth before and after admission. Existing
baseline/latency selection and resource admission flow are preserved. The output
age calculation also honors valid timestamp zero rather than treating it as absent.

Selection timestamps and maximum known active-output/pending-queue ages use the
same monotonic selection clock. Trace time remains after batch preparation;
unknown output clocks and empty queues report null/no histogram sample. Two
fixed age histograms, padded/active request counts, logical work and selected
prefill budget support interpretation. They are worker selection observations,
not client-visible latency, GPU busy time or a universal queue watermark. A
prefill-unavailable fallback can reflect capacity, tables or restore readiness;
it is not a specific resource-failure classifier.

Completed non-aborted adaptive rounds now expose bounded first-rejection counters
(position 0..15), rejected/all-accepted positive-depth totals and existing
proposal/acceptance/target-only totals. Unverified deeper drafts supply no
rejection observation. Scalar validation precedes partial updates; a 16-position
sweep keeps this counter family at 22 keys. No acceptance learner, sampler,
precision, checkpoint, public flag, dependency or live default changed.

The new dispatch controls use real six-page admission and DecodeManager selection
with CPU preparation and an independent clock. Baseline and latency fallback,
queue-before/after, preparation-time separation and zero/unknown clock cases all
fail on the prior phase wheel (eight failures, 1.66 seconds,
m1-dispatch-prior-wheel.log). Final focused source: **40 passed, five skipped**,
1.79 seconds (m1-dispatch-source-final.log). Existing drain tests additionally
require no rejection counters for aborts/target-only rounds. Censoring, finite
keys and invalid-observation atomicity have independent scalar controls.

The rebuilt wheel matches all 436 source Python modules; only scheduler and
telemetry changed from the phase wheel. SHA-256:
5eddd5300ef1f55d4fbec4eb2c49b802f994529791df265ae99f613ce88db543.
Expanded CPU: **2202 passed, 236 skipped, six deselected**, 211.90 seconds
(m1-dispatch-wheel-cpu.log). Combined actual CUDA/state controls: **35 passed**,
4.64 seconds (m1-dispatch-wheel-cuda.log). Focused counts overlap the expanded
selection and are not summed. Manifest: m1-dispatch-validation.json. Ruff F and
whitespace checks pass; all owned jobs are terminal.

No complete-model/API responsiveness, stage/stream reconciliation, <=2% overhead,
balanced scheduling benefit, sustained overload, expert contention or endurance
follows from these controls. M3/M5 quality findings and full-server restart/recall,
hosted CI/DCO and 60-minute/four-hour gates remain open. The unchanged full-server
physical GPU floor remains 30 GiB; no model boot/default promotion/commit/push/PR
or merge occurred in this correction.


## Tensor-parallel latency scheduling decision agreement

Source review found that latency mode made phase/chunk/service-budget decisions
on every rank using local host ages and local completed-cost samples. Different
clock/cost observations could therefore select different model work. The internal
selector now uses rank zero's decision through the existing CPU process group
for latency TP>1: five float64 scalar fields (40 bytes) and one int64 agreement
vote per rank. Baseline and TP1 take the existing local path. A primary exception
is broadcast before every rank raises; invalid or incompatible runnable envelopes
fail collectively before preparation. Existing queue/cache/request identity and
group timeout remain prerequisites. This is not a state-repair protocol.

A real two-process Gloo control deliberately gives peers conflicting ages and
costs, covering prefill/decode/idle, a reduced aligned chunk, and service-budget
authority. Peer selection is forbidden. It also checks shared failure for source
exceptions, a smaller peer budget and unavailable peer prefill, then verifies the
group remains usable. Fast paths prohibit collectives. Actual Scheduler wiring
must follow the received choice. That wiring control fails against the preceding
5eddd530 dispatch wheel; the focused source suite passes 18 tests in 4.01 seconds.
Raw logs: `m2-tp-prior-wheel.log`, `m2-tp-source.log`; matching five-decision
per-rank source records: `m2-tp-gloo-source-decisions.json`.

The new wheel matches all 436 production modules; only policy.py and scheduler.py
changed relative to the preceding wheel. Its SHA-256 is
`9a2653ec7ad68b1e1e57baa196ae3e8d9ee0aee41abd5d5c73116d3b7d8a8e1c`;
module hashes and exact artifact path are in `m2-tp-artifact.json`. The selected
small CUDA/state regression suite passes 36 tests (37 deselected) in 4.57 seconds.
It includes the previous 35 CUDA ownership, admission, allocator, live-expert,
adaptive-phase/drain and transfer controls plus the CPU allocator-disable control;
these are synthetic state/model fixtures, not complete-model TP execution.

Full-model stage reconciliation, TP model/NCCL behavior, communication overhead,
matched mixed-load benefit, fairness under sustained overload, existing strict
quality failures and endurance gates remain open. No exact/adaptive TP1
eligibility, public option, live default, model weight or precision changed.
Latest capacity: `m2-tp-capacity.json` (8.99 GiB GPU free, 2.78 GiB disk free),
below the unchanged 30 GiB pre-boot floor. No model server was launched.

Final built-wheel broad CPU regression: **2207 passed, 236 skipped, six
deselected**, 210.20 seconds, exit zero (`m2-tp-wheel-cpu.log`). Both real Gloo
workers completed and their five decisions match (`m2-tp-gloo-wheel-decisions.json`).
The selected device suite is **35 CUDA cases plus one CPU control**, all passed,
with 37 deselections. `m2-tp-validation.json` joins artifact/test hashes, exact
selection, source negative, final logs and capacity. Ruff F and diff whitespace
checks pass; all owned jobs are terminal. No commit, default promotion or release
qualification follows from these local regressions.


## Unexpected worker exception: local prefix/IPC cleanup

Review found that `_run_scheduler` called scheduler.shutdown only for
KeyboardInterrupt. A fully constructed worker failing in synchronization or
inference skipped prefix-ticket and IPC cleanup. The worker now invokes local
failure cleanup, then re-raises the exact original fatal cause. Cleanup errors
are logged explicitly and cannot report successful drain or replace that cause.
Scheduler failure cleanup drains prefix owners before IPC closure; final device
teardown belongs to process exit, so this branch requires neither healthy peers
to enter a shutdown barrier nor a usable device for ordinary graph teardown.
Normal signal-driven shutdown keeps its prior order and behavior. A partial
constructor remains outside this fully constructed-owner path.

Nine controls fail against the preceding TP-policy wheel. Six check exact error
identity for inference/sync/SystemExit failures with successful or failing cleanup;
two exercise actual Scheduler ordering with forbidden device/peer calls; one
spawns a real CPU worker and fails it during a gated prefix-file write. Only
cleanup opens the gate. The test independently reopens, checksum-validates and
reads the committed record after the failed worker exits, then requires parent
reap and semaphore release. Corrected focused source: **9 passed**, 7.32 seconds.
The first two source runs retained an incorrect expected event order for sync
errors: startup reporting already flushes the ack writer before cleanup. That
test assertion was corrected; the original runs remain as source/source-final
logs, not as production failures. Logs are `m7-failure-cleanup-*`.

The rebuilt wheel matches all 436 production modules, with only launch.py and
scheduler.py changed from the TP-policy wheel. SHA-256:
`2a78b95cb4ad2460f846332ef4ed753033e5fbe03e5ce22a8c4b2d7bc5bdcf21`.
Artifact/module hashes are in `m7-failure-cleanup-artifact.json`.
The pre-existing selected 35 CUDA cases plus one CPU allocator control pass
(37 deselected), 4.67 seconds. A supplemental normal/failure CUDA shutdown pair
passes (two deselected), 2.26 seconds. These use real small GPU state and prefix
files with a gate before transfer; failure mode is explicitly injected on a
usable device. No broken-context, full-model, pending-DMA failure recovery or
multi-rank cleanup qualification follows. Supplemental pair overlaps the earlier
normal-shutdown case; do not sum overlapping runs as unique cases.

Final built-wheel CPU selection: **2216 passed, 236 skipped, six deselected**,
216.31 seconds, exit zero. It was collected before the supplemental CUDA failure
case was added; that case is covered by the separately executed actual GPU pair.
The real spawned-worker evidence (16 files including the committed record and
logs) is preserved in `m7-failure-cleanup-real-worker/` outside Git. Capacity at
03:32 UTC: 8.07 GiB GPU free and 2.74 GiB disk free; unchanged 30 GiB boot floor.
`m7-failure-cleanup-validation.json` joins exact logs, source-test hashes and
scope limits. Ruff F and diff whitespace checks pass. All owned checks are
terminal; changes remain uncommitted and the full goal remains incomplete.


## Byte-fallback Unicode stop boundaries

Review of M5's stop restrictions found that the common scheduler matcher used
`max(len(stop)) + 1` tokens, with an incorrect comment that N characters need at
most N tokens. Byte-fallback tokenization can use three or four tokens for one
Unicode character. The matcher now bounds suffix decoding by longest-stop UTF-8
byte length plus one context token, clipped to generated output. The change is
host-only and preserves existing stop-list precedence. No sampler, acceptance,
model arithmetic, request option or live default changed.

A real ByteLevel BPE tokenizer with one token per UTF-8 byte supplies independent
encoding/decoding for ten controls: five exact-completion/bounded-lookback cases
and five actual speculative-burst drain→codec→detokenizer cases. Stops cover
ASCII, two/three/four-byte characters and a combined Unicode sequence. Burst
controls require no suffix leak, exact terminal metadata, rejection of
state-overadvanced donation and complete request resource release. The prior
failure-cleanup wheel fails six tests and passes four; corrected source passes
all ten. The combined focused source selection passes **33 tests, one skip**,
2.08 seconds. Logs: `m5-unicode-stop-prior-wheel.log` and
`m5-unicode-stop-source.log`.

The actual local Qwen complete artifact tokenizer was also loaded without model
weights. Its usual emoji/CJK/ASCII controls use one or two tokens and already
match; U+10FFFD uses four tokens and U+1FAE9 uses three, and both miss on the
preceding wheel. All seven match after the fix. Raw IDs, round-trip decoding,
matched values and tokenizer SHA are in `m5-unicode-stop-tokenizer-{prior,source,wheel}.json`.
The standalone tokenizer-only script is preserved beside those reports. This is
not a full-model answer, API or quality qualification.

The new wheel matches all 436 production modules. Only scheduler.py differs from
the preceding wheel, and an AST comparison confirms `_match_stop_str` is its only
changed method. SHA-256:
`c311169c0511feb6aa492e17ab6c20657ffe675d072a2bd74cdbe3b8526281e5`.
Exact artifact/module hashes: `m5-unicode-stop-artifact.json`. Earlier CUDA
regressions belong to their recorded artifacts; this host-matcher change does not
relabel them as a new CUDA run. Existing M3/M5 strict parity failures, complete
server stop/tool/grammar behavior, matched performance and endurance remain open.

Final built-wheel broad CPU selection: **2226 passed, 237 skipped, six deselected**,
216.02 seconds, exit zero. The additional skip relative to the preceding broad
run is the supplemental CUDA failure-drain case added after that earlier run's
collection; it was separately GPU-checked in the preceding turn. All seven
actual local-tokenizer controls match on the built wheel, with unchanged token
IDs and tokenizer SHA; the prior wheel matches five. Final record:
`m5-unicode-stop-validation.json`. Capacity: 8.63 GiB GPU free and 2.72 GiB disk
free, below the unchanged 30 GiB pre-boot guard. Ruff F and diff whitespace checks
pass. All owned jobs are terminal; no complete-model boot, commit, release or
default promotion occurred. Full original qualification scope remains open.


## Responses constraint admission: verified defect and isolated preview

The Responses request model accepts extra fields, but conversion silently drops
`text.format` constraints. Unlike Chat Completions, JSON/schema constraints can
enter unconstrained generation. Eight unsupported/malformed text configurations
were checked through streaming/nonstreaming ASGI routes and direct conversion:
**24 negatives fail** on current production, while **eight plain-text controls
pass**, 2.29 seconds. This is a defect, not a supported constrained decoder.

A candidate patch is prepared outside the worktree in
`$HOME/results/flare-runtime/responses-constraint-preview/`. External pytest
conftest loads only that candidate Responses module; production source remains
unchanged. The candidate rejects constraints/malformed values before request-ID
allocation, tokenizer submission or SSE creation, and applies the same rejection
to direct GenSpec conversion. No constraint contents enter error text. The
preview passes all **32 focused controls**, 0.29 seconds, and **99 combined
controls** including existing Responses and Chat Completions tests, 0.47 seconds.
Generation replies are synthetic; no model or GPU qualification follows.

CONTRIBUTING.md requires a design issue before changing public APIs. Searches of
existing issues and PRs found no matching constraint-rejection design issue.
The exact issue body and patch are reviewable beside the preview tests. Permission
to publish the GitHub issue has been requested; no external issue was posted and
no public behavior changed. All 436 production module hashes still match the
Unicode-stop wheel. Logs/hashes/scope are joined in
`m5-responses-constraints-preview-validation.json`. Apply and package the fix
only after the required design-issue step is authorized and completed. Full
original FLARE gates remain open; this is not an implementation-completion claim.


## Atomic image prompts under a partially spent prefill budget

Review found that a fitting image prompt could terminate the scheduler if an
earlier request left too little token budget in the batch. `_add_one_req` treated
that transient condition as an individually oversized image, after token-budget
and input-copy mutation. PrefillAdder now records the pass's initial budget and
defers an image that exceeds only the remaining budget. Existing tentative
admission rollback releases table/cache/recurrent owners; the full prompt and
conditioning remain pending. An individually oversized prompt keeps the existing
single-chunk error, raised before copy/sizing mutation.

Five CPU controls fail against the preceding Unicode-stop wheel: four mixed
text/image or image/image cases across page sizes one/four, plus the oversized
prompt's no-mutation requirement. Corrected focused source: **25 passed, one
skip**, 1.69 seconds. Initial development logs include one underspecified fake
manager in the oversized test and four incorrect calls to the idle-only cache
conservation oracle while requests were live. The final tests instead count
independent live pages and run full conservation after release; these test setup
errors are preserved, not reported as production failures. Final counterfactual:
`m2-atomic-image-prior-wheel.log` (five failures); corrected source:
`m2-atomic-image-source-final.log`.

The shared CPU lifecycle fixture gained an optional device argument, default CPU,
to run the same production owners on CUDA. The four additional CUDA cases pass
on source (1.95 seconds) and on the built wheel (1.96 seconds), each with five
deselections. No model weights or vision tower are loaded; image embeddings and
positions are synthetic. Tests check deferral, unchanged conditioning, table and
recurrent-slot rollback, zero cache reuse, whole prompt admission next pass,
page conservation and complete cleanup. The CUDA completion wait is test-only.

The new wheel matches all 436 production modules; only prefill.py changed from
the Unicode-stop wheel. SHA-256:
`55c00a964b1748660a0ab7554a44b69994ff3c3984081ad5e5caaeade68d9882`.
Exact artifact/module hashes: `m2-atomic-image-artifact.json`. This does not qualify
SWA-specific pressure, oversized-image HTTP behavior, full-model vision or
performance. Multimodal prefix reuse remains excluded. The separate Responses
constraint preview remains pending permission to publish its required design
issue; no Responses production behavior changed.

Final built-wheel broad CPU selection: **2231 passed, 241 skipped, six deselected**,
215.95 seconds, exit zero. The extra four skips are the new CUDA admission
variants. Final exact-test prior-wheel counterfactual: **five failures, four
deselections**, 1.67 seconds (`m2-atomic-image-prior-wheel-final.log`). Current
wheel CUDA: **four passes, five deselections**, 1.96 seconds. Raw logs, artifact
and test hashes, development-oracle corrections, capacity and scope are joined
in `m2-atomic-image-validation.json`. Capacity at 03:54 UTC: 8.95 GiB GPU free,
2.76 GiB disk free; unchanged 30 GiB full-server floor. Ruff F and diff whitespace
checks pass; all owned jobs are terminal. No public Responses change or design
issue publication occurred; that preview's approval remains pending. Full goal
and qualification scope remain active, with no default promotion or commit.


## TP signal stop must not submit a mismatched shutdown barrier

Real two-rank Gloo controls reproduce a shutdown failure: one peer submits an
unmatched broadcast while the primary is between operations. Operator stop
interrupts different collective positions, and normal Scheduler.shutdown then
adds a barrier behind mismatched work. Both parent-helper SIGINT and direct
SIGTERM cases time out and exit unsuccessfully. The fixture uses a six-second
Gloo timeout for bounded negative runs; production's configured distributed
wait may exceed the parent's shared ten-second reap grace. That production
consequence is inferred from the configured bounds, not a measured model run.

The worker wrapper now uses local cleanup for TP>1 signal stops, reusing the
existing unsafe-device/failure cleanup path. Local owners and IPC close; final
communicator/device teardown belongs to process exit, with no new barrier or
coordinated-GPU-drain claim. TP1 signal cleanup and fatal error identity remain
unchanged. The existing lifecycle design issue 98 covers this internal protocol.
No new public request behavior or model/precision/default promotion is added.

Both exact controls fail against the preceding atomic-image wheel (18.65 seconds,
`m7-tp-stop-prior-wheel.log`). Corrected source plus nine fatal-cleanup controls:
**11 passed**, 12.52 seconds (`m7-tp-stop-source-final.log`). The first corrected
source run passed worker-exit/IPC assertions but incorrectly checked the
parent-owned queue semaphores while retaining that queue; the fixture now drops
its own reference before asserting final resource conservation. The original
`m7-tp-stop-source.log` is preserved as a test-lifetime error, not a production
leak. Source stop records report approximately 0.61/0.64 seconds, both ranks zero
exit status, CUDA uninitialized. Raw per-rank records and reports are preserved
in `m7-tp-stop-source-evidence/`.

The new wheel matches all 436 production Python modules; only launch.py differs
from the preceding atomic-image wheel. SHA-256:
`9197c9b23fd573842ae002764dec79e17b64b2b123d7709149b654bd35f7bd74`.
Artifact/module hashes: `m7-tp-stop-artifact.json`. No new CUDA execution is claimed
for this process-wrapper change. CPU Gloo shutdown does not qualify NCCL, model
execution, active GPU work, partial constructor or earliest-spawn interruption.
Full original quality/API/performance/soak gates remain open; the separate
Responses design-issue publication still awaits user approval.

Final built-wheel broad CPU selection: **2233 passed, 241 skipped, six deselected**,
222.15 seconds, exit zero. Both packaged Gloo stop cases report zero worker exit
codes and completed local cleanup; their exact reports and per-rank markers are
preserved in `m7-tp-stop-wheel-evidence/`. `m7-tp-stop-validation.json` joins
artifact/test hashes, negative controls, development-oracle correction and final
scope. Source instruction and full-plan files still match the original checkout;
its FLARE_WORKTREE.md points to this isolated implementation as intended. The
progress page was condensed without removing the detailed history or open gates.
Capacity: 8.83 GiB GPU free and 2.93 GiB disk free; unchanged 30 GiB boot floor.
Ruff F and diff whitespace checks pass, all owned jobs are terminal, and the
full objective remains active. No Responses change, issue publication, commit,
release or default promotion occurred.


## Startup-report stop race retains constructed-owner cleanup

Review found that a KeyboardInterrupt racing with startup-error reporting entered
the original-error preservation branch and re-raised before cleaning a constructed
scheduler. A synchronization error after successful construction can still own
prefix I/O. Local failure cleanup is now shared by the ordinary fatal branch and
this racing-stop branch. It drains only constructed owners and never replaces
the original exception if cleanup fails. No-owner constructor errors retain their
existing behavior; this is not a claim to clean partially constructed objects.

Four of six exact controls fail on the preceding TP-stop wheel; two no-owner
controls pass (5.89 seconds, `m7-startup-race-prior-wheel.log`). Corrected source
plus existing fatal-cleanup and TP stop cases: **17 passed**, 16.71 seconds
(`m7-startup-race-source.log`). Unit cases check exact original exception identity,
constructed/no-owner behavior and a failing drain. The two process cases deliver
actual SIGINT/SIGTERM at the startup-report boundary while a real file worker is
paused. The sync error is injected in a synthetic scheduler; no model is loaded.
Only cleanup opens the write gate. The original ValueError acknowledgement and
exit code one remain, while the parent independently reopens and validates one
complete record and checks semaphore release/process reap. Source reports and
committed records are preserved in `m7-startup-race-source-evidence/`.

The new wheel matches all 436 production Python modules; only launch.py differs
from the preceding TP-stop wheel. SHA-256:
`6858ab0abb4d57a5e47e5435aa956d3177d4b26cebe02a0682e103fbbd38d910`.
Artifact/module hashes: `m7-startup-race-artifact.json`. No new CUDA execution is
claimed for this launcher-only change. Arbitrary constructor interleavings,
partial owner construction, model/GPU failure, quality/API/performance/soak gates
and the separate Responses design-issue publication remain open.

Final built-wheel broad CPU selection: **2239 passed, 241 skipped, six deselected**,
226.80 seconds, exit zero. Both packaged real-signal cases retain the original
ValueError and exit code one, with one independently validated complete record.
Raw reports and committed records are in `m7-startup-race-wheel-evidence/`;
`m7-startup-race-validation.json` joins exact logs, hashes and scope. Capacity:
8.16 GiB GPU free and 2.92 GiB disk free, below the unchanged 30 GiB full-server
boot guard. Ruff F and diff whitespace checks pass; all owned jobs are terminal.
No Responses behavior change, issue publication, commit, release or live default
promotion occurred. Full original implementation and qualification scope remains
active; this does not certify arbitrary constructor or GPU/model failure recovery.


## Controlled pre-wrapper stop and partial-launch ownership proof

No production change was needed. The existing startup ownership context now has
real stop coverage with a child held at first worker entry before explicit Torch
import and worker-handler/group-detach setup. Two cases stop the frontend while
the launch callback owns one of three intended workers, or after the complete
one-worker handle returns with health loading. Both use real frontend, supervisor,
spawned process, POSIX signals and named queue semaphores. They require parent
status 143, expected child KeyboardInterrupt exit, reap, three queue-semaphore
releases, no force-kill and no supervisor false alarm. Focused built artifact:
**two passed**, 6.56 seconds (`m7-pre-wrapper-wheel-focused.log`).

An isolated copy of the test probe removes only the startup-ownership guard.
After parent stop, both parent and child remain alive and parent.finalized records
an unfinished child exit. Explicit harness SIGINT to that child is required to
recover/reap the test processes. This is a detected fault, not a production pass.
Two earlier fault-control launcher attempts failed before worker entry because a
context factory was replaced with an instance; stderr capture and the factory
shape were corrected. Those failed harness runs are preserved and excluded from
ownership evidence. Valid fault report: `m7-pre-wrapper-fault-control.json`;
logs/script and recovered raw state are outside Git beside it.

The broader built-wheel serving regression passes **613 tests**, 193.57 seconds,
exit zero (`m7-pre-wrapper-wheel-serving.log`). This overlaps prior whole-suite
CPU results and is not added to them. Both new cases also pass in that serving
run; exact reports/logs are in `m7-pre-wrapper-wheel-evidence/`. All 436 production
module hashes remain identical to the startup-race wheel
`6858ab0abb4d57a5e47e5435aa956d3177d4b26cebe02a0682e103fbbd38d910`.
Only the test probe and new test changed. This qualifies the controlled first-entry
window, not arbitrary spawn-bootstrap/import interruptions, model/GPU construction
or full-model quality/performance/soaks. The Responses preview remains unapplied
pending permission to publish its required design issue.

Final serving-run reports show parent status 143, expected child interrupt exit
one, one child reaped and three named queue semaphores released in each case
(approximately 0.68 seconds partial launch, 1.72 seconds returned handle).
`m7-pre-wrapper-validation.json` joins reports, test/artifact hashes, the excluded
fault-launcher setup failures, valid missing-guard counterexample and exact scope.
Capacity: 10.35 GiB GPU free and 2.90 GiB disk free; unchanged 30 GiB complete-server
boot floor. Ruff F and diff whitespace checks pass; all owned jobs are terminal.
Production still matches the startup-race wheel, and the original whole-suite
2239 passes remain scoped to that run, not relabeled as a 2241-case rerun. No
model boot, Responses application, issue publication, commit or default promotion
occurred. Full original goal remains active.


## Stop inside Process.start before child-handle registration

Local multiprocessing source review and real-process controls found a gap in
partial ownership: Process.start assigns _popen only after its Popen constructor
returns. A stop after OS child launch could unwind while Process.pid was still
unavailable, so the existing owner guard could not stop/reap that child. Both
real SIGINT/SIGTERM controls reproduce a live unregistered child on the prior
wheel; an isolated Linux subreaper controller recovers all test descendants.

All three native launcher starts now use a parent-only guard that defers callable
stop handlers until Process.start completes, restores handlers, then re-delivers
pending signals through the OS. Signal blocking is used only for post-spawn
handler restoration; children inherit no added blocked-stop mask. Pending state
is at most three signal numbers with no retained frames. Default/ignored handlers
and background-thread/non-POSIX behavior stay authoritative. Custom callbacks are
replayed, while an original start error survives a callback exception. There is
no new guarantee for a nonreturning OS spawn or default-fatal unhandled signals.

Corrected focused source: **28 passed**, 28.65 seconds
(`m7-spawn-gap-source-validated.log`). Final prior-wheel counterfactual:
**seven failed, two passed**, 7.37 seconds (`m7-spawn-gap-prior-wheel-final.log`).
Controls include actual pre-registration signals, handler/callback authority,
coalescing, original error identity, no frame retention and background fallback,
plus the preceding lifecycle/TP tests. Early development runs used an incorrect
assumption that this frontend's early SIGINT exit was -SIGINT. Independent old
artifact measurement shows it is one (SIGTERM is 143), with the child still alive
and finalized exit code null. That incorrect oracle was corrected without
weakening ownership checks. The baseline raw state is preserved in
`m7-spawn-gap-baseline-exit-evidence/`; development logs remain separate. OS
redelivery is retained for actual signal-handler semantics rather than directly
calling a saved handler.

The new wheel matches all 436 production modules; only launch.py differs from
the startup-race wheel. SHA-256:
`f4cdfb32eb66ce63bbf7f658bb6b4f87c379750cc7394db9e46c1305b20c7542`.
Artifact/module hashes: `m7-spawn-gap-artifact.json`. This does not qualify model
construction, active GPU/NCCL work, arbitrary OS failures or full-model quality,
API, performance and endurance gates. The separate Responses design-issue
publication remains pending; no public Responses behavior changed.

Final built-wheel broad CPU selection: **2250 passed, 241 skipped, six deselected**,
238.67 seconds, exit zero. Both actual pre-registration signals retain the measured
frontend exit status and produce a registered/reaped child. The worker is already
gone before the isolated controller's recovery cleanup; queue semaphores are
released. Reports and raw state are in `m7-spawn-gap-wheel-evidence/`;
`m7-spawn-gap-validation.json` joins artifact/test hashes, original baseline
ownership/exit observations, corrected-oracle disclosure and precise scope.
Capacity: 10.17 GiB GPU free and 2.85 GiB disk free, below the unchanged 30 GiB
full-server floor. Ruff F and diff whitespace checks pass; all owned checks are
terminal. No Responses application, issue publication, model boot, commit, release
or live default promotion occurred. Full original goal remains active.

## Local CI environment check and resource-blocked handoff

The current spawn-gap wheel also completed the CI product selection with
`CUDA_VISIBLE_DEVICES=''` and `SPARKLAB_DISABLE_JIT=1`: **1469 passed, 95 skipped**,
two warnings, 228.92 seconds (`local-ci-product-wheel.log`). The PID was confirmed
live during execution and absent after the complete pytest summary. No owned
pytest/preflight process remains. This local aarch64 GPU-torch environment does
not prove the hosted x86/CPU-torch workflow. Counts overlap previous selections.

The combined adaptive, session and restart preflight completed with **three
skips in 2.02 seconds**, each at the unchanged 30 GiB physical GPU memory guard
(`qualification-preflight.log`). No complete target/server boot occurred. Latest
independent Torch measurement is **11.441345 GiB** free. VLLM PID 3493304 and
retrieval PIDs 3245901/3250768 are unrelated services and were left running.
This repeats the same resource blocker documented at 8.16, 10.35 and 10.17 GiB
across prior goal turns; it is not a new first-occurrence blocker.

Read-only POSIX_FADV_DONTNEED advice covered seven explicit task target/draft
files (22,585,608,698 bytes), checking unchanged size/mtime. It released some
reclaimable file cache but did not satisfy the GPU floor; no global cache drop,
weight modification, swap assumption or unrelated process stop was used.
`task-artifact-page-cache-advice.json` records the operation. Separately,
`derived-extraction-cleanup.json` records removal of 21 obsolete derived wheel
extractions after archive-content/holder checks. Every original wheel hash and
raw log was preserved, along with the current extraction. Historical extraction
paths need recreation from their preserved archives before replay. Disk free
after the checks is 3,294,392,320 bytes, only just above the restart fixture's
3 GiB disk admission requirement.

`FLARE_handoff.md` supplies review units, eligible state contracts, exact
quality/session/restart commands, larger recall resource settings, rollback and
the full outstanding gate boundaries. There are no qualifying matched absolute
performance metrics. Strict parity's prior 12 failures remain unresolved, and
the Responses design issue/patch remains an external preview pending permission.
No publication, commit, merge, release or default promotion occurred. With local
jobs terminal and the available independent checks/handoff complete, the next
required work needs external GPU capacity or the separately pending permission.
The full goal remains incomplete and is eligible for blocked status; no gate is
waived. Git whitespace checks pass.

## Qwen3.6 native complete-model validation after authorized VLLM stop

The user requested the retained Qwen3.6 alternative and explicitly authorized
stopping VLLM. Artifact metadata validation confirms `prepared/0.5.0`, fingerprint
`bda7268c3e0afd7b`, four shards, 1071 indexed tensors and 22,580,867,072 physical
bytes, including unused MTP weights. This is host Triton NVFP4, not the released
Marlin container. No download, weight modification or production source change
was made. All 436 production Python modules still match the spawn-gap wheel.

Added a shared internal test profile for the Qwen MoE architecture: disk-backed
full immutable GPU expert preload avoids complete pinned-RAM duplication on
GB10, with Triton, graphs off and speculation zero. Exact-state, session and
restart fixtures use it explicitly; their default model profile remains.
The exact-state fixture now checks a finite >=30 GiB floor in a child before
Engine boot, preserving the CUDA-fresh startup condition and rechecking after
imports. Initial CPU profile/session/restart selection: 74 passed. The first
actual CLI test used an invalid tool-parser control name (`qwen3` instead of
`qwen`); correcting that test yielded 43 passes. The first quiet server launch
then caught an existing fixture bug, `--max-extend-tokens` instead of the public
CLI's `--max-prefill-length`, before model load. Shared full-profile rendering
now uses canonical CLI flags and is checked through the real parser for both
model profiles and both concurrency geometries. Final CPU selection: **78 passed
in 16.40 seconds**, `qwen36-fixture-cli-final.log`. Counts overlap; they are not
summed. No product API was changed.

Three guarded attempts before the authorized stop declined to boot as free GPU
memory fluctuated below 30 GiB; these are preserved preflight blocks, not model
passes. After VLLM container stop, free memory rose to 69.9 GiB. The corrected
baseline (`qwen36-baseline-quiet-2/`) reached ready in 30.06 seconds, returned
`ready`, and exited 143. GPU free was 68.26 GiB before boot and 42.04 GiB ready;
this is not a peak allocation or throughput measurement.

The real exact-state fixture passed **32 cases in 58.68 seconds**, spanning
endpoints 1/15/16/17/31/63/64/65, map/compass branches and memory/disk restore.
Every case reports complete state/output equality against identically consumed
partitions (`qwen36-exact-state-retry.log`, `qwen36-exact-state.json`). An earlier
invocation skipped without displaying its reason; no pass was credited to it.
The HTTP exact-prefix/cold-rebuild controls separately passed **16 cases** with
expected cached-token accounting and strict output equality
(`qwen36-api-exact-1/branches.json`); its server exited 143.

The four-boot native disk restart/disable fixture passed **in 109.45 seconds**.
Independent raw dispatch verification records computed prefill **66/65/1/66**
for cold/publish/restore/rollback; 65 tokens were reused after restart, outputs
match strictly, rollback left record inventory unchanged, and all four server
exit codes are 143 (`qwen36-restart-1/evidence.json`). This is a short Qwen3.6
restart experiment, not long recall or qualification of another model.

The full application gate fails **on the baseline** at `turn-3.tool`: the model
returns fact JSON instead of the required lookup_fact call. Of 30 planned cases,
six were attempted, five passed and one failed; 24 were not run. Candidate did
not execute. The five passing cases include three fact answers and the separate
fictional legal summary/chronology. Baseline server exit was 143. The log and
full request/response are preserved in `qwen36-sessions-1/` and its pytest log.
Cold diagnostics with required/auto choice and original/explicit tool-first
system instructions reproduce all four context failures. Minimal independent
lookup and multiplication controls pass (`qwen36-tool-diagnostic-2/evidence.json`),
so parser/tool emission works in those controls. The source exposes/parses
`required` tools without hard decoder enforcement. No oracle or API behavior was
weakened to count direct answers as calls; no candidate quality claim is made.
Both diagnostic servers exited 143.

The platform model-owner service recreated VLLM after direct stop. The replacement
was stopped under the same authorization. Root-managed
`grove-beta-next-primary-model.service` remains in a restart cycle: `systemctl
--no-ask-password stop` reports interactive authentication required and `sudo -n`
cannot authenticate. A user-side stop of that exact unit was requested; no other
role, service setting or restart policy was changed. Retrieval processes were
preserved. No controlled performance window is claimed from the container stops.

To maintain disk headroom, completed-test derived prefix files were hashed and
removed (1,051,793,716 bytes), preserving case reports/raw logs and a checksum
inventory. An initial cleanup refused the quota metadata file and made no
deletion; the corrected allowlist includes the actual root quota metadata.
Another 167,451,510 bytes of inactive pip wheel-download cache bodies were
removed after ZIP-wheel/open-holder checks; installed environments remain.
Reports are `qwen36-exact-record-cleanup.json` and
`qwen36-wheel-download-cache-cleanup.json`. Restart record evidence remains on
disk; disk space is still limited. A read-only hint on the selected target's
four FTW shards is recorded in `qwen36-page-cache-advice.json`.

Compact reconciliation: `qwen36-validation.json` hashes source fixtures, wheel
and evidence, verifies all 32 state cases, 16 API cases, restart proof and failed
session summary, and lists unrun gates. Reproduction/eligibility/rollback is in
`FLARE_qwen36.md`. Ruff F and whitespace checks pass; all owned tests/servers are
terminal. Long 8K/32K recall/restart, completed tool sessions, protocol pressure,
matched performance, integrated soaks and original DFlash parity remain open.
No commit, publication, default change or certification promotion occurred.

## Authorized API constraint fix and long Qwen3.6 qualification

The user stopped the root-managed VLLM supervisor and requested continuation;
the initial audit confirmed inactive/dead, retrieval unchanged and over 600 GiB
disk free. Complete-model experiments retained the >=30 GiB pre-boot guard.

Eight same-partition 8K/32K memory/disk state/output cases passed in **224.24
seconds** (`qwen36-long-exact-state.json`, `qwen36-long-exact-state.log`). The
32K four-boot restart also passed: computed prefill **32769/32768/1/32769**, reuse
of 32768 tokens, strict output equality and all exits 143
(`qwen36-restart-32k-1/evidence.json`). These runs used the spawn-gap wheel;
later production changes are limited to API constraints and optional tokenization
boundary discovery. No broad quality/performance claim follows from these cases.

Prepared a concrete combined API preview: forced tool choice and unsupported
Responses text.format must reject before admission or opening SSE. Prior wheel:
**45 negative failures, 17 supported controls pass**. Preview: **62 focused
passes**, **162 including existing three-adapter checks**. The user explicitly
authorized issue publication/application. Existing-issue searches found no match;
published **https://github.com/sixteen-miles-labs/sparklab/issues/99**, then applied
the three-adapter patch and moved its tests into the repository. Source adapter
selection: **162 passed in 2.70 seconds** (`api-constraints-source.log`).
Auto/none remain supported. Actual constrained decoding is not implemented and
no direct answer is credited as a forced call. README documents compatibility.
No other external issue/comment/publication was made.

The API wheel hashes all 436 production modules; only the three adapters differ
from spawn-gap. SHA f9eaca916cd6d0af6033b09dc61f90935e83cca805f95b95c330fb2d0806acfd.
Its CPU selection passed **2284**, skipped **142**, 243.02 seconds; that command
omitted the separate MoE test directory, so it is not a like-for-like count
against the earlier combined native/MoE selection. The latest-wheel run below
restores that directory. Real HTTP validation rejects **18 stream/nonstream
requests** on Chat/Responses/Anthropic and unsupported Responses format. Two
ordinary generations bracket the errors and their IDs differ by exactly one,
proving the errors consumed no request IDs. Raw requests/responses remain in
`qwen36-recall-baseline-{api99,partialfix}-1/evidence.json`.

Added an independent factual-recall collector, distinct from failed tool sessions.
Its first baseline answered all six 8K/32K fact questions, but candidate request
construction exposed a production bug before model execution: optional checkpoint
discovery rendered a system-only prefix, and Qwen raises TemplateError without
a user query. Full requests were valid. Tokenization now omits unsupported
TemplateError/ValueError partial boundaries after full-request validation;
unexpected runtime and full-request failures remain authoritative. Prior wheel:
**two failures, two controls pass**. Focused source/verifier: **17 passes**;
the wider tokenizer/stage selection passed 45 (overlapping, not summed).
The failing candidate is preserved as failed in `qwen36-recall-candidate-api99-1/`.

The new wheel matches all 436 production modules, changing only tokenizer/tokenize.py
from the API wheel. SHA **5a70962d6d84554029da8982b634947cc812be5bbb16f349bce07504ea42e46e**,
`exact-partial-render-artifact.json`. Full native/product/MoE selection:
**2339 passed, 241 skipped, six deselected**, 246.33 seconds. Local CI product
selection with JIT disabled: **1558 passed, 95 skipped**, 233.16 seconds. A later
benchmark-only wire-order regression is separately covered below; counts are
not summed. Local aarch64/GPU-torch evidence is not hosted x86/CPU-torch/DCO proof.

The stronger paired factual-recall workload repeats every question: **12 cases
per arm**, seed/repeat × three typed/source-attributed facts × 8K/32K. All answers,
rendered prompt hashes and completed token counts match strictly. Candidate
repeats reuse **N-1**, 8191/32767 tokens, proving unaligned safe-suffix reuse;
page-aligned reuse alone cannot satisfy the verifier. Independent negative
controls reject bad facts, types, usage, truncated work, duplicate cases, changed
signatures, empty workloads, aggregate errors, mismatched identities/rendering
and aligned-only reuse. Final model counters: 21 publications, six exact hits,
zero readers/private restore pages/slots. Both servers exit 143; the candidate
also passes 16 ordinary HTTP branch controls. Reports:
`qwen36-recall-{baseline,candidate}-partialfix-1/recall.json`.

Latest-wheel real streaming/stage reconciliation passed in **26.60 seconds**
(`qwen36-protocol-stages-api99-1/observations.json`): completions, chat, Responses
and Anthropic all complete; five HTTP samples equal four admitted/generated/
completed requests plus one validation error; no invalid intervals or active
requests; exit 143. This does not measure tracing overhead, semantic quality,
long pressure or cancellation conservation.

A separate automatic arithmetic workflow was attempted without weakening actual
tool-call oracles. Its first request produced the correct parsed multiply call,
but the client's reference count was 322 versus the server's 323 tokens. The
HTTP helper canonicalizes JSON while the reference tokenizer had retained nested
schema insertion order. Corrected reference rendering to use the actual wire
order; preserved the pre-correction source and failure. **49 focused harness
checks pass in 16.65 seconds**, including an independent wire-order control;
that new control also passes in JIT-disabled CI mode (one pass, 5.08 seconds).
All 24 recorded factual-recall counts/hashes recheck unchanged under the
corrected provider (`recall-wire-order-reconciliation.json`).

The corrected automatic-tool baseline completes one actual multiply call and
consumes its real linked tool result. On the second arithmetic request it answers
directly with the correct result instead of making a call. Strict call observation
fails: two of three attempted model steps validate out of 40 planned steps;
candidate does not run. Raw failures are in `qwen36-auto-tools-baseline-{1,2}/`.
Auto permits a model to choose a direct answer; mandatory-call applications remain
unqualified, and actual forced decoding remains explicitly unsupported. This is
not attributed to a FLARE candidate regression or converted into a session pass.

`qwen36-validation-v2.json` reconciles artifact/source/evidence hashes and scope.
All owned test/model clients and servers are terminal; no forced kill is claimed
as successful cleanup. At the final service audit VLLM's supervisor was active
again (11:50:50 EDT), with its GPU worker visible; it was left untouched after
the tests. Retrieval remained present. Ruff F and whitespace checks pass. No
commit, push, merge, deployment, live default or certification change occurred.
Matched performance/overhead, full stop/cancel/restore concurrency and pressure,
60-minute/four-hour integrated soaks, original DFlash parity and hosted CI/DCO
remain open; the full original objective is not complete.


## Authorized paired Qwen3.6 performance and finite pressure

The user stopped the root primary owner. Same-wheel first baseline/exact pair
completed all 14 outputs per arm, but strict output equality fails for 32K
branching (13 match). Warmup and measured branch outputs agree within each
mode. Zero pairs qualify; four remaining pairs are unrun. Pure controls also
received different shared-prefix hits. No speed gate closes.

Independent baseline/memory/disk finite pressure collection passes 63 completed
requests and 21 client cancellations including warmups, six timed cycles and
six idle rebuilds each. Disk counters reach 13 completed restores and six
coalesced restores. The initial disk attempt failed at HTTP 503 busy rebuild;
collector-only handling is fixed, actual TCP busy/error regressions pass in the
37-test focused selection, and a fresh-store disk rerun passes. All six owned
servers exit 143, root owner remains inactive, retrieval is preserved. See
[full scoped evidence](FLARE_qwen36_performance.md) and
`$HOME/results/flare-runtime/qwen36-validation-v3.json`. Production modules and
latest wheel remain identical. No defaults/certification/deployment change.
