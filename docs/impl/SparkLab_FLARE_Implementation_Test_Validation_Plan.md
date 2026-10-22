# SparkLab: FLARE-inspired runtime improvement plan

Prepared: 2026-10-21
Target repository: https://github.com/sixteen-miles-labs/sparklab
Reference repository: https://github.com/NVIDIA/flare-local-inference

## 1. Objective and delivery contract

Improve responsiveness, reusable conversation state, and memory efficiency in SparkLab's native runtime without replacing its architecture. Implement and validate five workstreams:

1. Latency-aware prefill/decode scheduling.
2. Exact, complete conversation checkpoints.
3. Bounded NVMe retention of immutable prefix checkpoints.
4. Speculative depth selection based on measured acceptance and execution cost.
5. Targeted scratch-memory reuse, where profiling proves a benefit.

Deliver small, independently reviewable changes with executable tests, raw benchmark evidence, compatibility notes, and a working rollback. Keep existing behavior selectable until the relevant release gates pass. A feature implemented in source is not automatically qualified on every recipe.

This document is an implementation specification, not a benchmark report. All numerical gates below are proposed project acceptance criteria, not measured improvements or FLARE performance claims.

## 2. Source baseline and facts to recheck

The comparison used these main-branch commits:

| Repository | Reviewed commit |
| --- | --- |
| SparkLab | `715af2b3058f0b642bbb04c8d8a449e7816bba37` |
| FLARE | `5f37bd14062334f7687d168d58ea00567b9aa7fe` |

Before coding, inspect the actual checkout, record its HEAD and dirty state, read applicable AGENTS.md files and CONTRIBUTING.md, and reconcile any changes since this review. Do not reset user changes. These historical commits are comparison references, not instructions to downgrade the checkout.

Observed facts at the reviewed SparkLab commit:

- The supported product target is GB10/SM121 with 128 GB coherent unified memory.
- Native scheduling tries prefill before decode in `Scheduler._schedule_next_batch()`.
- Chunked prefill, CUDA graph capture, hybrid recurrent-state caching, tool-call anchors, speculative decoding, and NVMe expert loading already exist.
- Hybrid radix caching uses aligned checkpoints and skips donation of live final state when its endpoint cannot be represented safely at the aligned boundary.
- Multimodal requests carrying image embeddings are excluded from the shared prefix cache in the inspected native paths.
- Some portfolio recipes launch isolated containers. A native scheduler improvement does not improve those containers automatically.
- AOT kernel preparation already exists; do not propose introducing AOT as though it were absent.
- Existing benchmark evidence includes both confidence-adaptive DSpark and rejected cost-aware experiments. Review them before repeating those experiments.
- SparkLab already uses reference-model tests for prefix-cache lifecycles. Extend those rather than building a parallel test framework.

Relevant existing areas:

| Area | Existing files/directories |
| --- | --- |
| Scheduling | `python/sparklab/runtime/scheduler/{scheduler,prefill,decode,cache,status}.py` |
| Cache structures | `python/sparklab/runtime/kvcache/`, including `hybrid_radix_cache.py`, `swa_radix_cache.py`, `linear_state_pool.py` |
| Engine and graphs | `python/sparklab/runtime/engine/{engine,graph,config,cache_budget}.py` |
| Recipe/backend integration | `python/sparklab/{planner,deployment,catalog}.py`, `python/sparklab/backends/`, recipe JSON files |
| Expert loading | `python/sparklab/moe/{offload_cache,expert_banks,host_banks}.py` |
| APIs and metrics | `python/sparklab/serving/`, `python/sparklab/daemon/` |
| Tests | `tests/runtime/{scheduler,kvcache,engine}/`, `tests/serving/`, `tests/moe/`, `tests/e2e/` |
| Evidence | `benchmarks/gb10/`, `benchmarks/bench_openai_concurrency.py` |

Paths and new interfaces proposed later are suggestions. Adapt them to the current tree while preserving clear ownership.

## 3. Scope, invariants, and non-goals

### In scope

- Native runtime policy, state ownership, bounded background I/O, and related telemetry.
- Model-specific adapters only where the physical state or execution contract differs.
- Real GB10 validation using complete checkpoints.
- Existing OpenAI Chat Completions, Responses, and Anthropic compatibility.
- Application-shaped workloads representing Grove and tool-using agents, using synthetic or approved public data.

### Defer

- Rust runtime migration, adopting FLARE as the default backend, or exporting FLARE-style runtime packages.
- Replacing FTW, changing expert-weight precision, changing model semantics, or adding new model families.
- Multi-node serving, distributed cache servers, network prefix stores, and GPUDirect Storage.
- New embeddings, rerankers, document indexing, or application workflow changes.
- Shared multimodal prefix reuse in the first release; preserve the existing exclusion until content identity and complete vision state are implemented and tested.
- Broad performance refactors without a measured bottleneck.

### Mandatory invariants

1. A checkpoint represents one exact consumed prefix. Sampled-but-unconsumed tokens and presentation-filtered output do not define model state.
2. KV pages alone are insufficient for hybrid/recurrent models. Preserve every state component needed for continuation.
3. Shared published checkpoints are immutable. Branches write only private live state and private tail pages.
4. Physical storage cannot be reused until both logical owners and outstanding GPU/I/O users have released it.
5. Padding and inactive rows cannot read or mutate another request's state.
6. Admission accounts for weights, expert cache, KV, recurrent state, graph workspaces, I/O staging, CPU buffers, and OS reserve. Host and device allocations share physical capacity on GB10. Swap is not capacity.
7. Checkpoint-cache failures may fall back to recomputation only before unsafe state is published or executed. Unexpected invariant violations must remain visible.
8. Missing hardware checks are reported as unrun, not passed. A skipped test is not validation evidence.
9. Do not weaken numerical bounds, omit difficult workloads, or change checkpoints to make an optimization pass.
10. Changes to shared runtime code require regression checks across all affected cache schemes, even if the optimization is initially enabled for one qualified model.

## 4. Milestones and dependencies

| Milestone | Outcome | Depends on | Default status |
| --- | --- | --- | --- |
| M0 | Checkout audit, model eligibility map, independent baseline | None | No behavior change |
| M1 | Traceable latency/cost/resource measurements and workload harness | M0 | Low-cost counters; detailed tracing opt-in |
| M2 | Latency-aware scheduling | M1 | Opt-in until gates pass |
| M3 | Exact complete checkpoints for one native model, then affected schemes | M1 | Opt-in by validated state contract |
| M4 | Disk retention and restore of immutable checkpoints | M3 | Off by default |
| M5 | Cost-aware speculative depth for one existing implementation | M1; M3 interoperability tests | Fixed depth remains available |
| M6 | One measured scratch-memory optimization | M1; stable target execution paths | Enabled only on qualified paths |
| M7 | Integrated endurance, packaging, documentation, and rollout | Relevant earlier milestones | Promote each feature independently |

Implementation order: M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7. Reuse existing acceptance/graph timing infrastructure if adequate; do not require a generic scheduling framework.

If a phase fails its benefit gate, retain its useful measurements and tests, disable or remove the unsuccessful optimization, and continue independent phases. Correctness defects block dependent phases.

## 5. M0 — audit, eligibility, and baseline

### Tasks

- Record git HEAD, dependency versions, driver/CUDA, kernel, DGX OS, memory, filesystem, NVMe device, free space, power/thermal conditions, and other GPU processes.
- Read current contribution, test, release, and benchmark policies. Respect the requirement for design discussion before public API or checkpoint-format changes.
- Identify native versus container backend for every selected recipe; capture actual launch arguments, pinned checkpoint revision, artifact identity, attention backend, precision, speculative mode, and graph capacities.
- Inventory model continuation state and its existing snapshot, restore, commit, and rollback paths.
- Review historical DSpark adaptive/cost-aware evidence and explain why the proposed new experiment differs.
- Run existing focused tests before modifying code. Record pre-existing failures separately.
- Run an unchanged-baseline server and save workload results using the same harness planned for the candidate.

### Eligibility map

Choose locally available complete models rather than downloading a new portfolio unnecessarily.

| Role | Preferred starting point | Required distinction |
| --- | --- | --- |
| First hybrid checkpoint/scheduler target | Current native Qwen3.8-Flash-Next artifact | Use the actual current artifact; older Inferact evidence does not transfer |
| Dense attention control | Native Qwen3.8-27B, if available | Separate target-only from its DFlash profile |
| NVMe contention/control | Native DeepSeek V4 Flash, if available | Expert offload and prefix retention share storage bandwidth |
| Other cache-scheme coverage | Existing SWA/DSV4 and recurrent test fixtures; real models when available | CPU geometry tests do not certify real model continuation |
| Compatibility control | A container-backed recipe | Confirm no launch/API regression; make no native-performance claim |

If the selected native model cannot meet concurrency or context sizes below, record its actual supported envelope and select a compatible control model. Do not silently run a different backend.

### Exit gate

A reproducible baseline exists; selected recipes and state coverage are explicit; missing hardware or artifacts are listed. No performance claim comes from a reduced-layer or dummy-weight run.

## 6. M1 — observability and benchmark harness

### Measurement requirements

Use monotonic clocks. Separate:

- HTTP ingress, frontend/tokenization work, admission queue delay, first model dispatch, first generated token, first visible streamed content, and completion.
- GPU execution duration, host scheduling/submission overhead, and client-observed latency.
- Prefill tokens actually computed versus cached/reused tokens.
- Committed output tokens versus proposed/rejected draft tokens.
- Visible delivery events versus number of tokens delivered in one speculative burst.

Do not infer per-token latency by assigning separate arrival times to tokens delivered in the same burst. Report delivery gaps and accepted tokens per burst alongside GPU round time.

Add bounded counters/histograms for:

- Queue depth, prefill/decode work, dispatch reason, chosen chunk/graph shape, active output age.
- KV pages, recurrent/snapshot slots, pinned/retiring resources, scratch bytes.
- Speculative draft count, accepted count, rejection position, proposal/verification/commit time.
- Prefix hit length, exact/full hits, restore source, restore time, recompute time.
- Prefix-store bytes/records, queue occupancy, checksum failures, read/write failures, skipped publications, staging usage, and expert-cache hit/miss traffic.

Reuse existing metrics names where possible. Avoid request IDs, prompts, paths, or case names as metric labels. Use bounded opt-in traces for per-request details.

### Harness changes

Extend `benchmarks/bench_openai_concurrency.py` or add a focused companion harness; retain its existing behavior.

New workload definitions should support:

- Fixed tokenized input lengths, reproducible prompt content, output-token targets, seed/sampling, concurrency, arrival schedule, and backend identity.
- A burst of long prompts while other requests are decoding.
- Repeated messages, tool returns, branching continuations, cancellation, and prefix eviction.
- Client timestamps for visible delivery events, validated usage, error/timeout capture, and final output.
- Machine-readable raw observations and a compact summary, with baseline/candidate comparison.
- Trace-off production-like measurement plus a trace-on diagnostic run to quantify measurement overhead.

Use server tokenization or the exact serving tokenizer; record effective rendered length rather than estimating from words. Make cold-prefix and warm-prefix workloads separate. Control OS page-cache conditions separately from model/prefix-cache conditions.

### Exit gate

A known timestamp fixture verifies latency calculations; usage/event counts reconcile; harness failures remain failures. Raw observations can reproduce reported percentiles and throughput. Detailed tracing changes throughput by at most 2% on the measurement control or is disabled for release comparisons.

## 7. M2 — latency-aware scheduling

### Implementation

1. Preserve the current scheduler as an explicit baseline policy.
2. Track each decoder's last visible delivery and pending requests' queue age.
3. Measure duration distributions for actual prefill chunks and decode/verification shapes. Timing collection must not introduce a device synchronization per step.
4. Begin with a bounded policy: defer additional prefill work when it would excessively delay active decoders, and reduce chunk size under decoder pressure.
5. Include pending-prefill age/admission service so continuous decode traffic cannot starve new prompts. Conversely, continuous prompt arrivals cannot starve decoders.
6. Never preempt an in-flight GPU operation or split an atomic image span. Respect recurrent-prefill alignment where exact chunk boundaries are constrained.
7. Keep memory reservations, in-flight work, and cancellation semantics intact. Policy selects legal work; it does not bypass admission.
8. Make timing-cold-start behavior conservative and deterministic. Missing costs do not become zero-time operations.
9. Integrate speculative work only after its real cost is known; retain bounded graph shape selection.
10. Add internal configuration first. If a public option is needed, route it through the existing validated configuration/recipe system with a documented baseline fallback.

Suggested code placement: a small pure scheduling decision helper next to the scheduler; keep GPU execution in the engine. A decision may include work type, actual token budget, and reason. Do not build a general pluggable policy platform.

### Tests

- Use an independent discrete-event reference clock/trace to check decoder service and prefill admission under continuous arrivals.
- Long prefill arrives after decoding starts; verify decode progresses before the entire prompt finishes.
- Continuous decode load still admits queued prefill; prompt floods still serve active decode.
- Empty queues, one request, multiple decoders, odd batch sizes, zero work, max context, cancellation before/after dispatch.
- Prefill chunk and memory constraints remain respected; no duplicate or lost completion.
- Real-server overlapping requests verify visible timing; CPU scheduling tests alone cannot demonstrate GPU responsiveness.

### Benefit gate

On mixed 8K/32K-prompt workloads at supported concurrency:

- At least 25% improvement in p95 visible delivery gap versus the matched baseline.
- At most 5% aggregate output-throughput regression.
- At most 10% p95 TTFT regression for newly admitted prompts.
- No starvation in sustained overload tests; report queue wait distributions and the configured service bound.
- Pure-prefill and pure-decode controls regress no more than 5%.

Freeze a measurable service bound after M1 using observed nonpreemptible graph/chunk durations. A hard latency promise shorter than one legal GPU operation is invalid. If the three performance criteria conflict, keep the policy opt-in and document the measured tradeoff rather than changing thresholds after seeing results.

## 8. M3 — exact complete conversation checkpoints

### State inventory and contract

Create an explicit state inventory for every enabled model/cache scheme. The contract describes concrete state layout and continuation requirements, not a model-name dispatch inside the shared scheduler.

Include, when applicable:

- Immutable full KV/compressed/index pages and private partial tail pages.
- Recurrent matrices, convolution histories, sliding-window state, raw selector rings/tags, and compressor state.
- Position, logical-consumed-count, and multimodal continuation metadata.
- Target feedback, draft history/cursors, and state required by the selected speculative mode.
- Final hidden vector or equivalent safe resume data for complete-prompt hits.

Support operations equivalent to identify endpoint, snapshot, restore, and release with explicit completion/lifetime ownership. Reuse existing model snapshot and accepted-prefix commit mechanisms. Avoid copying irrelevant weights or whole oversized pools.

### Boundary semantics

- Obtain endpoints from the exact rendered/tokenized request, not raw string lengths or role assumptions.
- A message-prefix render is reusable only if it exactly matches a prefix of the complete rendered prompt.
- Cache eligible complete message boundaries, final prompt boundary, and terminal consumed boundary; deduplicate equal endpoints.
- Stop prefill at eligible endpoints when necessary and snapshot only after the state mutation is complete.
- Do not round the key backward while preserving a state advanced past that key.
- Store partial-page data privately where the endpoint falls inside a physical page.
- Share only immutable full pages; restore boundary pages into private storage.
- A full-prompt hit may use a saved hidden vector/resume graph when semantically valid. Otherwise explicitly recompute a safe suffix; do not pretend the full-hit optimization is supported.
- A new request starts from its own sampling settings and RNG state. Prefix reuse must not inherit the prior request's RNG stream.
- Initially preserve multimodal prefix-cache exclusion. Later content-aware caching is a separate change requiring conditioning hashes and vision-state qualification.
- If snapshot capacity is exhausted, continue inference and skip publication safely. Never grow an unbounded snapshot pool.

### Identity

In-memory and persistent compatibility identity must cover effective weights/artifact, relevant precision, effective model/tokenizer/template configuration, state-layout version, attention/cache geometry, continuation/speculative mode, and conditioning inputs where supported. Runtime settings belong in compatibility identity only if they change stored-state interpretation or continuation semantics.

A prefix key includes consumed token IDs and the effective conditioning identity. Add an isolation scope if the serving layer exposes tenants/workspaces; otherwise use an explicit single-instance scope and do not claim tenant isolation. Page IDs and live slot IDs are not portable record identity.

### Tests

Extend the existing radix reference model and lifecycle harness.

| Case | Independent oracle |
| --- | --- |
| Resume after each eligible endpoint | Uninterrupted continuation using the same effective model and execution geometry |
| Endpoints 1, P-1, P, P+1, 2P-1, plus recurrent chunk boundaries | Exact consumed count and independently reconstructed state; P is actual physical page size |
| Two branches from one checkpoint | Two independent cold-reference runs; parent bytes remain unchanged |
| Full prompt hit | Cold prompt's resume logits/state or justified repeatability bounds |
| Speculative accept/reject at every prefix length | Target-reference accepted-prefix state; no rejected-state contamination |
| Duplicate publication and ancestor eviction | Existing reference model plus resource conservation |
| Cancel/abort/rebuild during ownership transitions | Eventual release after GPU fences, no stale handles |
| Same token placeholders with different images | No shared hit while multimodal reuse is excluded |
| Changed template/artifact/state mode | Compatibility miss; never restore incompatible bytes |

Poison recycled storage and inactive rows. Vary page size and supported chunk geometry. Distinguish bit-exact transfer correctness from numerical differences caused by different compute partitions.

Do not set a single arbitrary tolerance for every tensor. Preserve existing qualified tolerances; derive new ones from dtype/operator semantics and measured repeatability. Cache copies of the same representation must preserve bytes exactly.

### Exit gate

All ownership and continuation tests pass on the enabled native model, with all applicable real model states included. No model is enabled merely because a similarly shaped model passed. With caching disabled, existing behavior and API results remain valid. Repeated-prefix TTFT benefit is measured without a quality regression.

## 9. M4 — bounded disk prefix retention

### Architecture

Treat this as a separate subsystem from expert-weight caching:

`published immutable checkpoint → bounded copy-out → versioned disk record → validated restore → ordinary in-memory checkpoint`

The existing GPU/in-memory cache remains authoritative. Offload only fully published immutable state. Never offload a mutable live request or an unresolved speculative round.

On GB10, prefer disk retention with bounded staging instead of a duplicate RAM retention tier. Include staging, record indexes, queued metadata, and page-cache pressure in memory admission and monitoring.

Suggested small components, adapted to the current tree:

- Record encoder/decoder and compatibility checks.
- Bounded disk-store index and eviction.
- Copy/restore ticket ownership.
- Background I/O worker and scheduler polling of completion events.
- Admission/cost decision for restore versus recomputation.

### Record and namespace requirements

- Versioned header, declared dimensions/lengths, compatibility fingerprint, consumed-prefix identity, and checksums.
- Length/size validation before allocation or read; reject overflow, truncation, unsupported versions, and oversized records.
- No pointers, executable payloads, Python pickle, or machine-specific physical page IDs.
- Atomic publication via temporary records and rename; define required flush/fsync ordering for the promised crash behavior.
- On restart, recover only complete records and clean incomplete temporary files safely.
- Use a bounded disk quota, bounded metadata index, bounded staging pool, and bounded queue. Evict only records not pinned by restore/read.
- Reconstruct shared full-page chains without double ownership; validate parent/dependency presence before publication.
- Serialize mutation within a namespace. Reject incompatible concurrent writers or use proven explicit locking.
- Use the user's existing deployment access controls; do not claim encryption-at-rest unless provided and verified by the deployment.
- A scope change invalidates cross-scope lookup. Do not expose prompt contents in record filenames or metrics.

### Runtime behavior

- The model worker does not block on disk reads/writes, queue capacity, fsync, or transfer completion.
- Pending restores may wait in a scheduler-owned state; unrelated ready requests continue.
- A cancelled request can detach, but outstanding I/O/GPU users keep ticket buffers alive until completion.
- Copy-out pressure skips publication rather than slowing active generation without bound.
- Coalesce simultaneous restores of the same compatible immutable prefix when ownership allows.
- Select restore only when estimated restore plus scheduling cost beats recomputing the matched prefix; calibrate using measured hit sizes and NVMe contention.
- If store overload, incompatibility, checksum failure, eviction, or bounded timeout makes restore unsuitable, recompute from an intact safe source.
- Restore into private allocations and publish only after validation/copies finish. An unexpected post-publication device-state error cannot simply be swallowed as a cache miss.
- Coordinate I/O with expert loading: bound outstanding reads and compare performance with both subsystems active.
- Start with staged I/O supported by the environment. Direct I/O requires explicit alignment and filesystem validation; do not add O_DIRECT or GDS just for appearance.

### Fault and lifecycle tests

- Disk hit after in-memory eviction; hit after clean restart; safe recovery after killed writer.
- Header/payload corruption, truncated record, wrong checksum, wrong identity/version, missing ancestor, and invalid length.
- ENOSPC, permission denial, short read/write, unavailable directory, queue saturation, and worker failure.
- Two simultaneous readers, duplicate publication, cancellation in every transfer stage, and shutdown while I/O is active.
- Disk quota and staging bounds enforced under sustained churn.
- Read/restore after cache geometry rebuild must either be compatible or miss safely.
- Namespace/isolation mismatch cannot hit; malicious or corrupt size fields cannot trigger unbounded allocations.
- Factual recall from an evicted/restored long context, with facts placed throughout it.

Fault injection may mock OS failure boundaries, but real-file round trips and real GPU restores must also pass.

### Exit gate

For eligible 8K/32K prefixes:

- Exact byte transfer and complete continuation correctness pass.
- Restart persistence is demonstrated with raw dispatch traces, not inferred from a hit counter alone.
- Median restore-path TTFT is at least 20% below matched recomputation on at least one declared target workload; slower sizes use recomputation or stay disabled.
- Active decode p95 delivery gap and throughput regress no more than 5% under copy-out/inbound pressure.
- Expert-offload workload throughput regresses no more than 5% with retention enabled.
- No unbounded queues, metadata growth, staging growth, leaked tickets, or runtime-attributable swap-out.
- Failures produce explicit counters and safe recomputation where allowed.

A failed interference gate leaves disk retention off by default.

## 10. M5 — cost-aware speculative depth

### Implementation

- Read existing confidence-adaptive DSpark and rejected cost-aware experiment records first.
- Choose one native, already-correct speculation implementation. Do not redesign MTP, DSpark, and DFlash simultaneously.
- Maintain bounded per-request conditional acceptance estimates by depth and proposal source.
- An accepted prefix supplies positive observations; the first observed rejection supplies one negative observation. Unverified deeper drafts are not rejection observations.
- Estimate complete round cost: proposal, target verification, state commit/rollback, host handoff, and relevant expert I/O.
- Select only from prepared/supported depths and batch shapes. Account for remaining context/output length and stop/grammar restrictions.
- Maximize expected committed target progress per real service time while respecting M2 delivery fairness. Include a target-only candidate where the implementation can switch safely.
- Preserve history/state through depth transitions; a target-only round must not leave later draft continuation stale.
- Use conservative startup estimates, bounded measurement refresh, hysteresis, and a fixed-depth override.
- Avoid a high-frequency host synchronization to decide the next depth.
- Keep greedy and probabilistic verification correctness separate. An optimization must not replace exact verification with approximate token acceptance.

### Tests

- Independent synthetic acceptance chains with known optimal depths and measured-cost fixtures.
- All-accept, first-reject, late-reject, depth-zero/target-only, and changing acceptance.
- Batch/context/output limits, cancellation, stop strings, tool boundaries, grammar constraints where supported.
- Fixed↔adaptive and depth↔depth transitions preserve complete target/draft state.
- No training update from unobserved positions.
- Real native runs across prose, coding, math, structured/tool output, and multilingual prompts.
- Greedy parity where the current execution path guarantees it; numerical-repeatability and quality gates where graph grouping can change arithmetic.
- For probabilistic mode, preserve the algorithm's acceptance correction and assess distributional behavior against the target sampler; fixed-seed equality alone is insufficient.

### Exit gate

- At least 5% sustained end-to-end throughput improvement over the best preselected fixed-depth control on the declared primary workload.
- No more than 5% regression on each required workload stratum versus the selected deployment control, including target-only where appropriate.
- Accepted-prefix state and output-quality gates pass; no gain comes from less completed work or a changed sampler.
- If no robust gain exists, retain diagnostics and fixed-depth mode; do not enable adaptive mode by default.

## 11. M6 — measured scratch-memory reuse

### Implementation

- Profile allocations and read/write lifetimes on the selected native target.
- Identify one material retained workspace or repeated intermediate allocation.
- Reuse storage across sequential layers/stages/graph variants only when no live consumer remains.
- Keep overlapping branches, residuals, retained target/draft probabilities, accepted-prefix state, and asynchronous transfer buffers disjoint.
- If concurrent operations need scratch, declare their combined live requirement and stream join explicitly.
- Build on the existing graph pool and AOT facilities; avoid a general allocator rewrite.
- Freeze prepared views before capture. Graph replay uses stable addresses.
- Do not change quantization, BF16/FP32 rounding, accumulation order, or checkpoint scales as part of this memory change.
- Measure fixed workspace with cache capacities held constant, then separately test whether released memory can increase KV/expert capacity.

### Tests and gate

- Poison reusable buffers between legal lifetimes; compare with unshared buffers using the same arithmetic.
- Alternate graph sizes, decode/prefill/speculative modes, active/inactive rows, and stream paths.
- Check graph replay allocation stability, cancellation, and cache rebuild.
- No illegal access, race, or state corruption; correctness remains within established bounds.
- Demonstrate a material benefit selected before implementation: at least 5% lower fixed non-weight/non-cache workspace, or another documented bottleneck reduction justified by profiling.
- Complete-graph and end-to-end latency regress no more than 2%.
- Do not count a smaller configured cache or unrelated allocator cleanup as workspace savings.

If profiling finds no material opportunity, close M6 with the measurements and do not add an abstraction.

## 12. Common correctness and application validation

### CPU tests

Use existing production logic with independent oracles:

- Extend the radix reference-model/state-machine harness for ownership and eviction.
- Deterministic event simulation for scheduling fairness and resource bounds.
- Record round trips and filesystem fault injection for disk retention.
- Independent expected-progress calculations for speculative policy.
- Randomized lifecycle sequences with recorded seeds and shrinking/reproducible minimal traces where available.

Mocks must not bypass the state transition whose correctness is being claimed. Avoid tests that inspect source text or merely reproduce the implementation's branches.

### GPU tests

- Complete state snapshot/restore/branch at page and recurrent boundaries.
- Partial acceptance at every supported depth.
- Batch-one and supported higher-concurrency shapes, odd active lane counts, inactive-row poisoning.
- Memory pressure, eviction, cache rebuild, cancellation, and repeated graph replay.
- Real-server API streaming and backpressure, not only direct engine calls.
- Focused compute-sanitizer/race tooling on changed low-level paths when available; report unavailable tools rather than claiming a clean run.

### Application-shaped cases

Use synthetic case files/prompts rather than private client data.

1. Repeated case session: stable context, successive questions, reused history.
2. Tool loop: assistant tool call → tool result → continuation, with parsed arguments and correct ordering.
3. Branching: two independent questions from one retained prefix.
4. Long prompt during another response: validate responsiveness and answer completion.
5. Long-context recall: multiple facts at beginning/middle/end, with answer and location checked against a fixed oracle.
6. Cancellation/retry: cancel during prefill, decode, restore, and output delivery; retry must not inherit state.
7. Legal-style summary and structured output: compare required facts, attribution, JSON/schema validity, and complete answers against the unchanged model baseline. This tests runtime preservation, not legal correctness.

Do not use model-as-judge alone to certify restored state. Check state/logits where possible plus deterministic facts and structured outputs.

## 13. Benchmark matrix and statistical protocol

### Workload matrix

Lengths refer to effective rendered input tokens. Reserve output space within the supported context limit.

| ID | Workload | Context/input | Concurrency | Main measurements |
| --- | --- | --- | --- | --- |
| B1 | Cold/warm single-stream control | Short, 8K, 32K | 1 | TTFT, decode rate, E2E |
| B2 | Steady decode | Short prompts, 256/512 outputs | 1, 2, 4 where supported | Delivery gaps, aggregate rate |
| B3 | Long-prompt interference | 8K/32K arrivals during decode | 2, 4 where supported | Active delivery gap, newcomer TTFT, fairness |
| B4 | Repeated exact prefixes | 8K/32K history + new suffix | 1, 2 | Reused tokens, compute avoided, TTFT |
| B5 | Branching/tool conversations | 20–50 turns and tool returns | 1, 2 | State correctness, hit rate, parser integrity |
| B6 | Disk restore | Evicted 8K/32K; 64K only if supported | 1, then mixed | Restore vs prefill, interference |
| B7 | NVMe contention | Disk experts + prefix traffic | Supported native envelope | Expert misses/I/O, decode rate, TTFT |
| B8 | Speculation content strata | Prose, code, math, tool/JSON, multilingual | 1; higher only if supported | Accepted tokens/time, quality |
| B9 | Pressure/churn | Bounded pools near admission limit | Up to supported limit | Resource accounting, rejection, recovery |
| B10 | Integrated soak | Mixed sessions, cancel/restore/rebuild | Supported envelope | Stability, memory, swap, latency drift |

Do not infer portfolio improvements from one model. Unsupported shapes are explicitly N/A with reason; they are not passing rows.

### Execution protocol

- Baseline and candidate must use identical checkpoint, artifact, precision, backend, sampling settings, effective prompt tokens, output work, hardware, memory/cache sizes, and expert storage.
- Separate scheduler policy from speculative depth initially. Do not compare target-only baseline against speculative candidate and attribute the whole gain to scheduling.
- Hold cache capacities fixed for isolated scheduling/scratch comparisons. Benchmark capacity changes separately.
- Run at least one full-path warmup and five paired baseline/candidate trials for release benefit claims. Use balanced A/B ordering to reduce thermal and cache-order bias.
- Prefer confidence intervals computed over independent trials or sessions. Token events within a request are correlated; do not treat them as independent samples.
- Quantify variance. A gain smaller than noise is inconclusive.
- Tail claims need adequate samples. Target at least 1,000 visible delivery gaps for delivery-gap p99, and 1,000 completed requests for TTFT p99; report counts. Smaller runs are smoke tests and should emphasize p50/p95 with sample-size caveats.
- Compare identical completed work. Record stop reasons, usage, cancellations, errors, and timeouts. Never drop failed requests from reported success rate or successful-work throughput accounting.
- Measure cold/warm prefix, cold/warm filesystem cache, and cold/warm expert cache as distinct labeled conditions.
- Do not globally drop OS caches or kill unrelated processes on a shared machine. Use isolated permitted runs and record uncontrolled conditions.
- Evaluate features individually, then interactions: scheduler+exact checkpoints; checkpoints+disk; scheduler+adaptive; disk+experts; and the final qualified combination.
- Report absolute values as well as percentage changes.

### Soak gate

Run a 60-minute integrated soak on each selected release profile, then one 4-hour run on the primary proposed default combination before promotion.

Required:

- No server crash, deadlock, OOM, stale-state hit, corrupted output, or unexpected request loss.
- No runtime-attributable sustained swap-out.
- Pool counts return to their expected retained-cache baseline after workload drain.
- Disk quota, index size, transfer queues, and staging remain bounded.
- No monotonically growing unexplained memory after warmup; investigate sustained drift rather than hiding it in allocator-reserved totals.
- Latency/throughput remain within the relevant gates after thermal steady state.
- Restart and disabling each feature recover the documented baseline behavior.

## 14. Commands, CI, and test placement

Follow the current repository setup. At the reviewed commit, the documented GB10 environment uses:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Reuse an existing valid environment; do not upgrade dependencies just to execute the plan.

Useful existing focused commands, from repository root:

```bash
.venv/bin/python -m pytest -q tests/runtime/scheduler -m "not slow and not needs_weights"
.venv/bin/python -m pytest -q tests/runtime/kvcache -m "not slow and not needs_weights"
.venv/bin/python -m pytest -q tests/runtime/engine -m "not slow and not needs_weights"
.venv/bin/python -m pytest -q tests/serving -m "not slow and not needs_weights"
.venv/bin/python -m pytest -q tests/sparklab -m "not slow and not needs_weights"
```

GPU-dependent cases may skip on a CPU host even without a `needs_weights` marker. Inspect skip summaries and do not mistake these commands for a complete GPU gate. For affected changes, expand to `tests/moe`, model-specific paths, and the real-server tests required by that phase.

Existing cache-rebuild smoke test, with a real small local model:

```bash
SPARKLAB_REBUILD_TEST_MODEL=/absolute/path/to/small-model \
  .venv/bin/python -m pytest -q tests/e2e/test_cache_rebuild.py
```

Existing concurrency smoke benchmark, after launching the exact recipe under test:

```bash
.venv/bin/python benchmarks/bench_openai_concurrency.py \
  --base-url http://127.0.0.1:1919 \
  --model ACTUAL_SERVED_MODEL_ID \
  --label unchanged-baseline \
  --concurrency 1,2,4 \
  --trials 3 \
  --output-tokens 256 \
  --output /absolute/path/to/results/baseline-concurrency.json
```

Use only supported concurrency values. This existing script is a smoke/control tool: it does not yet implement the mixed workloads, per-delivery gaps, or release statistics specified above.

Proposed new benchmark commands must be documented only after implemented; no invented CLI invocation should be presented as currently runnable.

### CI changes

- Preserve the existing hosted CPU/DCO checks.
- Explicitly add new CPU-safe scheduler/cache/record tests to hosted CI: at the reviewed commit, the hosted job does not include all `tests/runtime` directories.
- Select CPU-safe modules or mark GPU fixtures correctly; do not make hosted CI require CUDA.
- GPU/weights/endurance runs stay in trusted environments; do not expose a self-hosted runner to untrusted fork code.
- Do not require an external model download in ordinary PR tests.
- Run the full existing required checks before review, plus affected subsystem tests.
- Do not add a linter, dependency lockfile, or unrelated tooling migration as part of this work.

## 15. Evidence, certification, and rollout

### Per-phase evidence

Store compact, reviewable summaries using the existing benchmark/evidence conventions. Keep raw streams, large traces, checkpoint bytes, and generated binaries outside source control.

Every summary must include:

- Exact code commit and dirty status; effective model/checkpoint/artifact/recipe identity.
- Hardware/software/backend/configuration and enabled features.
- Workload definition, token counts, cache conditions, trial/sample counts.
- Absolute baseline/candidate values, variation/intervals, errors and timeouts.
- State/logit/quality results and affected cache-scheme coverage.
- Tests run, skipped, failed, and unrun; reason for every material gap.
- Gate outcome and rollback/default recommendation.

Extend existing schemas only where necessary. New metrics do not automatically qualify under the current certification schema. Keep evidence separate from automatic catalog promotion until the existing gate understands it.

### Promotion rules

1. First ship experimental opt-in behavior on exact qualified native profiles.
2. Preserve a reproducible baseline selector and a way to disable disk retention independently.
3. Validate the built wheel/container or actual release artifact, not only an editable checkout.
4. Run required API, parser, agent, context, memory, and endurance gates.
5. Update recipe eligibility and known limitations from measured evidence.
6. Promote a default only for profiles that pass all applicable gates. Container-backed and untested models keep existing behavior.
7. Never label the entire portfolio Certified because one native profile improved.

### Rollback

- Scheduling: select baseline policy without changing request API semantics.
- Checkpoints: stop new publication and drain outstanding GPU users before retiring storage; use existing safe prefix behavior.
- Disk retention: stop accepting work, detach/cancel waiters safely, drain tickets, and recompute as needed. Deleting retained cache is optional and must target only its owned namespace.
- Adaptive speculation: restore the fixed-depth recipe and compatible draft state; restart if a safe live transition is not implemented.
- Scratch reuse: use the prior allocation layout and rebuild captured graphs outside active execution.

Test rollback after actual feature activity, including in-flight requests. Do not assume changing a flag alone safely migrates live state.

## 16. Suggested reviewable change sequence

| Change | Scope | Required proof |
| --- | --- | --- |
| 1 | Baseline harness, eligibility map, minimal counters | Reproducible unchanged baseline; metric arithmetic checked |
| 2 | Bounded scheduling policy | Independent fairness tests and mixed-load GB10 results |
| 3 | Exact checkpoint contract for first native model | State/branch/full-hit/partial-page correctness |
| 4 | Shared cache-scheme regression and lifecycle integration | No stale ownership, safe cancellation/rebuild |
| 5 | Disk record format and bounded CPU store | Real-file round trips, corruption/crash/quota tests |
| 6 | Async GPU restore integration | Exact transfer, complete state, no blocked active streams |
| 7 | Disk/expert contention tuning | Benefit and interference gates |
| 8 | Adaptive depth for one path | State transitions, quality, sustained matched results |
| 9 | One profiled scratch reduction | Lifetime tests, memory and latency evidence |
| 10 | Integrated packaging, soak, docs, and eligibility updates | Final release-artifact gates and rollback |

Each change should explain trigger/problem, resulting behavior, evidence, and remaining limits. Follow repository DCO and PR workflow; never push directly to main. No commits, PRs, or deployments are required merely to prepare this plan.

## 17. Completion checklist

- [ ] Current checkout and source references reconciled.
- [ ] Existing native/container boundaries recorded.
- [ ] Baseline and measurement harness are reproducible.
- [ ] New CPU tests run in hosted CI, not just locally.
- [ ] M2 scheduler meets correctness, fairness, and mixed-load gates.
- [ ] M3 snapshots represent exact consumed prefixes and complete model state.
- [ ] M4 disk failures, restart, cancellation, quotas, and expert contention validated.
- [ ] M5 adaptive depth beats its matched fixed control without quality loss, or remains disabled.
- [ ] M6 produces a measured benefit, or is closed without speculative code.
- [ ] Shared-path regression coverage names every affected cache scheme.
- [ ] Built artifact, API/tool loops, long-context recall, and soaks pass.
- [ ] Missing GPU/model checks are explicit and no unrun path is promoted.
- [ ] Evidence, configuration, defaults, and rollback instructions agree.
- [ ] No unsupported model, capacity, performance, or certification claim was introduced.

## 18. Source references

SparkLab sources at the reviewed commit:

- [README and supported target](https://github.com/sixteen-miles-labs/sparklab/blob/715af2b3058f0b642bbb04c8d8a449e7816bba37/README.md)
- [Scheduler](https://github.com/sixteen-miles-labs/sparklab/blob/715af2b3058f0b642bbb04c8d8a449e7816bba37/python/sparklab/runtime/scheduler/scheduler.py)
- [Hybrid radix cache](https://github.com/sixteen-miles-labs/sparklab/blob/715af2b3058f0b642bbb04c8d8a449e7816bba37/python/sparklab/runtime/kvcache/hybrid_radix_cache.py)
- [Cache manager](https://github.com/sixteen-miles-labs/sparklab/blob/715af2b3058f0b642bbb04c8d8a449e7816bba37/python/sparklab/runtime/scheduler/cache.py)
- [Graph runner](https://github.com/sixteen-miles-labs/sparklab/blob/715af2b3058f0b642bbb04c8d8a449e7816bba37/python/sparklab/runtime/engine/graph.py)
- [Test instructions](https://github.com/sixteen-miles-labs/sparklab/blob/715af2b3058f0b642bbb04c8d8a449e7816bba37/tests/README.md)
- [Benchmark evidence and prior adaptive experiments](https://github.com/sixteen-miles-labs/sparklab/blob/715af2b3058f0b642bbb04c8d8a449e7816bba37/benchmarks/gb10/README.md)
- [Contribution policy](https://github.com/sixteen-miles-labs/sparklab/blob/715af2b3058f0b642bbb04c8d8a449e7816bba37/CONTRIBUTING.md)
- [Hosted PR checks](https://github.com/sixteen-miles-labs/sparklab/blob/715af2b3058f0b642bbb04c8d8a449e7816bba37/.github/workflows/pr-checks.yml)

FLARE references at the reviewed commit:

- [Architecture and exact checkpoints](https://github.com/NVIDIA/flare-local-inference/blob/5f37bd14062334f7687d168d58ea00567b9aa7fe/docs/core-design.md)
- [Scheduling](https://github.com/NVIDIA/flare-local-inference/blob/5f37bd14062334f7687d168d58ea00567b9aa7fe/docs/scheduling-runtime-design.md)
- [Disk prefix retention](https://github.com/NVIDIA/flare-local-inference/blob/5f37bd14062334f7687d168d58ea00567b9aa7fe/docs/prefix-cache-offload.md)
- [Acceptance estimator](https://github.com/NVIDIA/flare-local-inference/blob/5f37bd14062334f7687d168d58ea00567b9aa7fe/helios/src/scheduler/speculation.rs)
- [Scratch planner](https://github.com/NVIDIA/flare-local-inference/blob/5f37bd14062334f7687d168d58ea00567b9aa7fe/ignition/preparation/scratch.py)
- [Performance methodology](https://github.com/NVIDIA/flare-local-inference/blob/5f37bd14062334f7687d168d58ea00567b9aa7fe/docs/performance-work.md)

Use these as reference implementations and design evidence. If copying source, inspect the exact file's license/notice and dependency obligations and preserve required attribution. Do not import FLARE's entire runtime or its repository-specific maintainer workflow into SparkLab.
