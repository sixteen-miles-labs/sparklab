# Open implementation and validation gates

This is a working ledger against the supplied plan, not a completion or release
certificate. The original scope remains intact. A narrow copy test, CPU pass or
packaging pass cannot satisfy a broader continuation/API/performance/soak gate.
Raw evidence referenced here lives under `$HOME/results/flare-runtime/`.

Latest [Qwen3.6 performance/pressure results](FLARE_qwen36_performance.md): strict
32K branch output parity fails, stopping the five-pair series after its first
attempt. Finite baseline/memory/disk HTTP pressure passes 63 completed requests
and 21 cancellations including warmups, with real disk restores/coalescing.
This does not close pending-restore cancellation or long-duration gates.

| Requirement | Current evidence | Status / remaining proof |
| --- | --- | --- |
| M0 current source, native/container boundaries, complete artifacts, hardware and unchanged baseline | FLARE_progress.md; baseline 1175 CPU passes; complete FTW artifacts validated; raw baseline smoke | Partial: paired release baseline, power/thermal/NVMe conditions and controlled-host envelope remain |
| M1 monotonic event/usage/throughput arithmetic | Real HTTP harness, timestamp/error fixtures, real admission/selection fallback and selection-age controls | Implemented and CPU checked; worker ages are not client delivery measurements |
| M1 HTTP ingress, frontend/tokenization, admission, first dispatch, generated/visible output and completion separation | Fixed stage stamps, CPU ownership/ASGI checks and actual latest-wheel Qwen3.6 streaming on four routes | Four model requests plus one validation error reconcile stage/accounting counters with no invalid intervals or active requests; scoped plumbing proof only. <=2% overhead and sustained/contended workload gates remain |
| M1 bounded queues/costs/resource/restore/expert/speculation measurements | Rate-limited native_runtime export; bounded asynchronous live expert scalar sampling with existing collection/timing flags; six new CUDA controls plus CPU/real-file coverage | Implemented and locally checked; one pending sample and fixed buffer payloads, newer terminal precedence; separate live/reserved PyTorch allocator counters and diagnostic speculative phase intervals. Idle samples can be stale; complete-model ordering, contention and overhead qualification remain |
| M1 trace overhead at most 2% | Trace-off default | Unrun measured paired trace-on/off gate |
| M2 baseline selector and resource-safe bounded prefill/decode service | Independent event tests; baseline admission preserved, opt-in exact-page accounting checked, and real two-rank Gloo phase/chunk/service agreement and shared-failure controls | CPU and four small CUDA atomic-image token-budget admission/conservation checks; actual TP model/NCCL and communication overhead unqualified; no hard latency claim |
| M2 8K/32K mixed-load gap/throughput/TTFT and pure controls | Three-request confounded smoke only | Unrun balanced release trials; smoke is insufficient |
| M2 sustained overload/cancellation/fairness | CPU event and lifecycle tests | Real sustained overload and visible-delivery proof remain |
| M3 exact token endpoints, private tails, full-prompt safe suffix, immutable branches | Token-render oracle, radix harness, real model memory/disk state matrix | Initial MHA+GDN target checked at 1/15/16/17/31/63/64/65; long message/tool/session matrices remain |
| M3 complete relevant state and resource conservation | KV/conv/recurrent/aux exact-copy comparisons; real-model pre-reset pool checks; physical-page admission, all-active publication headroom and parallel fresh/chunked branch oracles; seeded shared reference-model ownership with 207 CPU passes, 11 skips and mutation controls | Initial target-only contract checked; randomized checks cover CPU KV/snapshot owners, not complete-model or CUDA/disk lifecycle qualification; other physical contracts remain explicitly excluded |
| M3 cross-partition repeatability and quality | Qwen3.6 first paired run: 13/14 matching outputs; 32K branch differs reproducibly between modes in warmup/measurement | **Strict quality failed**; zero qualified performance pairs. Diagnose state versus numerical partition effects without weakening equality |
| M3 images/conditioning identity | Shared multimodal exclusion preserved | No multimodal reuse enabled; different-image API regression remains |
| M3 target/draft checkpoint interoperability | Speculation rejected by exact-cache admission | Incomplete grouped target/draft portable-state contract; do not bypass rejection |
| M4 record bounds, identity/scope, checksums, atomic fsync/rename, namespace lock | 28 real-file record tests, recovery and fault injection | CPU checked; real model sparse-map writer bug fixed with restart regression |
| M4 bounded index/quota/staging/tickets/read pins | CPU pressure/cancellation/ticket tests and GPU staging | Root-wide byte/record/namespace bounds, whole-page snapshot plus private-branch admission, ownership-preserving reclamation, killed writer and startup/cleanup faults checked; sustained real pressure/host swap bounds remain |
| M4 scheduler-private restore/coalescing/unrelated progress | Production lifecycle CPU tests; CUDA adoption/eviction, six staged-read cancellation/retry cases and four completed-read cases with a real pending final DMA | Scoped ownership, pending-event retention, independent stream progress and byte checks pass; final DMA is explicitly injected into synthetic state. Real HTTP cancellation/backpressure/restore concurrency and full-model state remain |
| M4 real continuation after in-memory eviction | Complete-artifact same-partition memory/disk branches in source and M4 wheel | Short and 8K/32K byte/output branch matrices pass; factual recall and full-server restart dispatch traces remain |
| M4 clean restart, killed writer, corrupt/short/permission/ENOSPC/fsync errors | CPU recovery/fault injection, independent raw-dispatch verifier and Qwen3.6 short/32K four-boot passes | 65 and 32768 tokens reused; both restore modes compute one suffix token, strict outputs match, all exits 143. Original Qwen3.8 profile, natural recall after restart and active-GPU failure recovery remain unqualified |
| M4 measured restore versus recomputation and interference gates | Conservative measured-cost policy, unknown costs recompute | Unrun balanced 8K/32K TTFT and decode interference comparison |
| M4 expert/prefix NVMe contention | Separate bounded store; expert CPU regression | No enabled expert-offload prefix contract qualified; real eligible control/contention evidence remains unavailable |
| M5 bounded censored acceptance, complete cost and fixed override | Policy/preparation tests; 12 actual scheduler-drain CPU cases and one real CUDA-event case; prior-wheel negative control | Unreachable post-drain feedback repaired: telemetry on/off, EOS, abort and next-round ownership checked. Full-model API execution/quality and sustained benefit remain; no default change |
| M5 prepared graph shapes, zero/depth transitions and no stale draft context | Complete-artifact 897 KV immutability / 479 transaction checks | Latest source run: 1144 KV and 666 transaction checks, poisoned inactive lanes and all 24 accepted-prefix endpoints checked; strict quality gate still fails |
| M5 greedy/numerical/quality preservation | Prior five-stratum × six-mode matrix, 64 outputs each; fixture now adds repeated target-only controls and bounded first-divergence logit evidence; 30 CPU and three small CUDA diagnostic checks | Prior strict parity failed in 12 combinations. Expanded 35-case full-model matrix skips below 30 GiB free GPU memory; diagnostics/qualification remain required |
| M5 all accepted-prefix lengths and stop/cancel/tool/grammar restrictions | CPU transactions; natural GPU prefixes; byte-fallback stop controls; local Qwen tokenizer controls; issue 99 applied | All 24 forced-prefix GPU cases checked. Forced-tool/Responses constraints now reject before admission; 162 source adapter checks and 18 native HTTP errors with no UID consumption. Auto/none remain; full forced decoding and original tool-session qualification remain unavailable. Real speculative API stop/cancel restrictions still need qualification |
| M5 >=5% sustained gain over best preselected fixed control, <=5% per stratum regression | Instrumented/unbalanced correctness timings only | Unrun; those timings cannot qualify benefit |
| M6 profile before optimization; >=5% workspace benefit or measured negative decision | FLARE_memory.md; complete-artifact allocator/operator profile | Closed without allocation change for the selected 512-token-chunk control; no broad memory claim |
| M7 all shared native cache schemes and expert regression | Expanded built-wheel CPU selection includes all native cache/engine and MoE suites | Latest combined selection: 2250 passed, 241 skipped, six deselected; counts include product/API/harness tests. CUDA/vLLM skips explicit, scheme-specific hardware gates remain |
| M7 hosted CPU/DCO checks | Existing checks preserved; runtime directories added | Configuration only; hosted execution/DCO review still required |
| M7 native stop, partial launch and transport ownership | Actual CPU HTTP/supervisor/spawn/tokenizer/prefix-file matrix; CUDA paused-write shutdown ordering | Scoped ownership proof: finalizers, worker reap and semaphore release; constructed-worker exceptions, including the startup-report stop race, now preserve the fatal cause while draining local prefix owners before IPC closure, with nine CPU controls and a small actual GPU/file shutdown pair; active `/generate` drain under parent/group stop and disconnect/retry over real IPC. Foreground workers detach from terminal groups. Two actual Gloo rank-stop cases now close local owners and exit successfully without a new barrier; Two controlled pre-wrapper entry stops now reap/release queue ownership during partial/full-handle launch; Actual SIGINT/SIGTERM inside Process.start now preserve handle registration and reap; callable handlers/masks/error identity are checked. Nonreturning or failed OS spawn, default-fatal signals and NCCL/model/active-GPU interruption remain unqualified; forced stop does not prove a successful GPU/I/O drain |
| M7 actual built artifact | Spawn-gap wheel matches all 436 production modules; only launcher changed from startup-race wheel | 2250 CPU passes, 241 skips, six deselections, including real pre-registration signals, pre-wrapper stops, startup-report races and Gloo stops. No new CUDA run for parent-launch change. Earlier device/state evidence retains its artifact scope; full-model/GPU quality, performance, stage/export and arbitrary spawn failures remain open. Initial 4 KiB allocation OOM remains unexplained. |
| M7 real OpenAI streaming/tools/stop/accounting/cancel, Responses/Anthropic | Existing API CPU regressions, baseline real-server smoke, actual HTTP/IPC backend-failure and TCP backpressure matrices | Nine failure waiters terminate explicitly; ten finite backpressure cases across five protocols deliver exact resumed output or abort/retire disconnected work while another request completes. Synthetic model replies only; full experimental-path model/tool/GPU-cancel and sustained-pressure matrix remains |
| M7 synthetic application cases, factual recall, JSON/attribution oracles | Session runner executes 20–50 turns, actual tool returns, immutable-parent branches, structured summary, optional fictional legal-style summary/chronology and exact 8K/32K recall construction; latest 42 focused CPU controls, six recall and two case tokenizer prompts checked | Paired native fixture implements the case option, cache-geometry/activity checks and a 30 GiB floor; model execution skipped at that guard. Full-model answers/state, legal-summary preservation/quality, eviction/restore/restart and performance qualification remain |
| M7 paired warmup + five balanced trials; adequate tails/variance/completed work | Harness rejects incomplete or identity-mismatched comparisons | Release statistics not collected; no p99 from small samples |
| M7 60-minute profiles / four-hour primary proposed default | Executable bounded endurance collector: strict baseline outputs, cancellation/drain, resource sampling, optional idle rebuilds and preserved failure evidence; 18 short synthetic HTTP controls | Actual model integrated soaks unrun. Collection duration does not qualify state, GPU/I/O conservation, drift, thermal behavior or attribution; four-hour promotion gate not satisfied |
| M7 disable/restart/rebuild rollback after activity | Baseline modes retained; direct cache invalidation and CPU ownership tests | Exact-cache KV/recurrent public rebuild and oversize rejection pass on the declared ratio=0.85 profile; full disk/adaptive disable/restart and lifecycle/soak proof remain |
| Evidence, eligibility, defaults and no unsupported claims | Progress/design/memory ledger; all new execution opt-in | Maintain through final source/artifact reconciliation; no portfolio/default/certification promotion |

The earlier local audit completed the available CPU checks and the concrete
[resumption handoff](FLARE_handoff.md). The CI product selection with JIT disabled
passed 1469 and skipped 95; three complete-model preflights skipped below the
unchanged 30 GiB floor. Latest measured GPU free is 11.44 GiB. No owned validation
job remains live. The same hardware blocker has persisted across more than three
consecutive goal turns. Required quality diagnosis, full-model integration,
matched performance and endurance work need external resource availability;
the separate Responses preview needs publication permission. Blocked status
does not make any unrun gate N/A or qualify a smaller passing test as completion.

Subsequent authorized Qwen3.6 work supersedes that capacity observation: stopping
the VLLM container yielded 69.9 GiB free. The retained host Triton model passed
32 same-partition state/output cases and 16 HTTP branch/accounting/cold controls;
78 CPU fixture/CLI checks pass. Four-boot Qwen3.6 restart passes; the 20-turn
application fixture fails its baseline's first required tool call, leaving the
candidate unrun. The strict oracle is preserved. The systemd model
owner was then stopped by the user and now reports inactive. Eight long 8K/32K
state cases and a 32K four-boot restart pass (32768 reused tokens; one recomputed
suffix token; strict outputs). Issue 99 is published and the API rejection fix
is applied with permission. The latest wheel's full CPU/MoE selection passes
2339, skips 241 and deselects six. Paired independent factual recall passes 12
seed/repeat cases with strict outputs, N-1 cached tokens and six actual exact
hits. Four-protocol streaming/stage reconciliation passes. The separate
supported-auto arithmetic workflow's baseline completes one tool round, then
answers the next request directly; its strict tool-call observation fails and
candidate remains unrun. Wire-order fidelity in the reference tokenizer is
corrected and all 24 existing recall prompt hashes are rechecked unchanged.
These new scoped results do not qualify DFlash, long sessions, other state
contracts, performance or endurance. See [Qwen3.6 details](FLARE_qwen36.md).
