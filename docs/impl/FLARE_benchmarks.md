# Runtime workload measurements

Run from the isolated source worktree with the configured Python environment.
Use exact local artifact paths and an unoccupied loopback port. Never change an
existing live service or globally drop filesystem caches for these measurements.

Generate exact-token text using the actual serving tokenizer:

```bash
python benchmarks/make_runtime_workload.py \
  --tokenizer "$MODEL_PATH" --long-tokens 8192 --output-tokens 256 \
  --output "$HOME/results/flare-runtime/workload.json"
```

Keep input plus output within the selected server context. Supply an identity JSON
containing the effective artifact fingerprint/revision, backend, precision,
attention, sampling, cache capacities, graph shapes and expert storage settings.
The harness records the source commit, dirty state, client concurrency and timeout
separately, so candidate code/policy can differ without changing model identity.

```bash
python benchmarks/bench_runtime_mixed.py \
  --base-url http://127.0.0.1:19291 --model flare-control \
  --workload "$HOME/results/flare-runtime/workload.json" \
  --identity "$HOME/results/flare-runtime/identity.json" \
  --label unchanged-baseline --concurrency 2 \
  --output "$HOME/results/flare-runtime/baseline.json"
```

Use the same invocation with a candidate label/output and
`--baseline "$HOME/results/flare-runtime/baseline.json"` for matched comparison.
The comparison rejects changed workload or effective model identity and incomplete
work. It does not itself prove balanced trials, warmups, statistical significance,
quality, trace overhead or release acceptance. Inspect raw observations and run
the plan's paired balanced protocol before drawing benefit conclusions.

Check adaptive scheduler feedback through the actual output-drain path separately
from direct-engine accepted-state tests:

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q -rs tests/runtime/scheduler/test_adaptive_round_drain.py
```

Twelve CPU cases require exactly one cost/acceptance observation after the copy
fence and reply handoff, no training for aborted work, and a clear round before
the next preparation, with telemetry enabled and disabled. The CUDA-marked case
uses real round/copy events and a small tensor operation; run it on an available
GPU without hiding CUDA. It validates timing/drain integration, not complete-model
speculation, state parity, API behavior or a throughput gain. Cost mode remains
experimental and off by default because the separate strict quality gate fails.

The complete-model greedy gate now repeats the target-only control for each of
the five strata, then compares depths 3/7/11, transitions and adaptive execution
(35 natural cases, followed by the existing 24 forced state-commit cases). Run:

```bash
SPARKLAB_DFLASH_TEST_MODEL="$MODEL_PATH" \
SPARKLAB_DFLASH_TEST_DRAFT="$DRAFT_PATH" \
SPARKLAB_DFLASH_TEST_OUTPUT="$HOME/results/flare-runtime/dflash-quality.json" \
SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q -rs tests/e2e/test_adaptive_dflash.py
```

Before loading complete weights it requires 30 GiB free physical GPU memory;
`SPARKLAB_DFLASH_TEST_MIN_FREE_GIB` can raise that floor but cannot lower it.
Each natural case records at most 64 emitted-token observations with eight top
logits, exact maximum-tie count, top margin and a full FP32-vector digest. Real
greedy sampling and verification still choose every token. Discarded verifier
rows and intermediate prompt chunks do not become output observations, and
trace token IDs must match the completed output exactly. The first divergent
output position includes both token-aligned records after the same preceding
output tokens. This does not establish equality of the underlying model states.
The synchronous diagnostic copies invalidate performance use of these timings;
the unchanged strict parity assertion still fails on any differing output.

CPU controls run with `tests/test_runtime_quality.py`; its three CUDA cases use
the real verifier with small synthetic logits, testing emitted-prefix selection
and diagnostic transfers rather than full-model numerical repeatability. The
new complete-model diagnostic matrix remains unrun below the fixed memory floor.

A workload contains `cases`, each with a unique `id`, `at` (seconds after run start),
`request` (the API payload including `max_tokens`), optional `endpoint` (completions
or chat completions), `expected_prompt_tokens`, `require_output_budget` (defaults
to true) and `cancel_after_events`. Include pre-rendered message/tool histories in
chat payloads for repeated histories and branches. Dynamic tool execution and
state/logit correctness need separate API tests; this harness does not certify them.
Fixed text lengths are checked against server usage. Token-ID inputs remain
unsupported by SparkLab's existing completion API.

Each raw observation retains scheduled/start/end monotonic times, every visible
SSE delivery time and character count, final text, usage, cached tokens, stop reason,
errors and cancellation. Speculative bursts have one timestamp, without fabricated
per-token arrival times. Successful-work throughput divides tokens from complete
successful requests by the entire workload duration, including failures/timeouts.
Client queue delay is reported separately. p99 is withheld below 1000 relevant
samples. Large raw streams stay outside git.

Enable `SPARKLAB_RUNTIME_TIMING=1` on both sides of a diagnostic comparison to
collect matching GPU and host-stage measurements. Keep
`SPARKLAB_RUNTIME_TRACE_CAPACITY=0` for comparisons until the paired tracing
overhead gate passes; a positive value also retains a bounded frontend stage
ring in `/v1/stats` under `runtime_timing`. Server body sends, parsed semantic
readiness and client delivery timestamps measure different boundaries.

Eligible native expert-offload runs can also enable the existing
`--moe-collect-stats` flag. With runtime timing enabled, the worker then samples
aggregate device counters onto one reusable four-int64 pinned window, at most
once per second and with at most one copy pending. This does not enable counter
collection or add sampler allocations to ordinary launches with those options
off. `native_runtime.expert_decode` can refresh before a request terminates; a
newer terminal observation takes precedence. Its timestamp is the host submission
time of the sample, not a GPU execution timestamp. Idle exports may remain stale.
The buffers' payload bytes and pending flag are visible under
`native_runtime.retained_workspaces.expert_counter_sampling`; allocator and driver
overhead are not included in those byte counts.

Run the implemented counter oracles and real-stream checks without loading model
weights:

```bash
PYTHONPATH=python python -m pytest -q \
  tests/runtime/scheduler/test_expert_counter_sampling.py
```

Two CPU cases check observation precedence and disabled polling. Six CUDA cases
check independent aggregate/hybrid counts, zero activity, one-second submission
limits, immutable reports, actual scheduler export and a real pending copy. The
pending-copy case requires another stream to progress and rejects any reuse of
the in-flight window. Explicit synchronization/device scalar reads are forbidden
inside production polling by the test guards. Complete-model stream ordering,
host/device contention and the <=2% diagnostic-overhead gate still need matched
qualification; these synthetic-counter controls do not prove them.

An implemented real-server stage smoke drives streaming completions, chat,
Responses and Anthropic plus an invalid-budget request, then reconciles actual
stage histograms with frontend admission/output/HTTP totals. It records raw
client SSE events and the final stats snapshot outside git:

```bash
PYTHONPATH=python SPARKLAB_TIMING_TEST_MODEL="$MODEL_PATH" \
  SPARKLAB_TIMING_EVIDENCE_ROOT="$HOME/results/flare-runtime/timing-server" \
  SPARKLAB_TIMING_SERVER_ARGS='["--num-pages","512","--page-size","16","--max-seq-len-override","4096","--max-running-requests","2","--cuda-graph-max-bs","2","--max-prefill-length","512","--memory-ratio","0.85","--attention-backend","triton","--nvfp4-backend","triton","--nvfp4-prefill-backend","w4a16","--speculative-tokens","0"]' \
  python -m pytest -q -rs tests/e2e/test_request_timing.py
```

That declared profile requires at least 30 GiB free physical GPU memory before
launch. The current shared host failed that guard, so this real-model stage
check is **unrun**. The executable fixture is not a protocol/performance pass.
`SPARKLAB_TIMING_PACKAGE_PATH` can point the child server at an extracted wheel;
the caller's `PYTHONPATH` should use the same package for artifact validation.

The M0 shared-machine smoke uses 128/2048/512-token inputs and 64/32/32 outputs.
It is a correctness/control smoke, not the required 8K/32K release comparison.
The three-request run lacks balanced repetitions and full-path warmups; cache-hit
patterns differ causally with scheduling. Identical output strings were observed,
but no broad numerical or quality claim follows from that synthetic prompt.


## Complete-state checks

With the worktree source on `PYTHONPATH`, the available complete target-only
hybrid artifact can run this byte-transfer/continuation gate:

```bash
PYTHONPATH=python SPARKLAB_EXACT_TEST_MODEL="$MODEL_PATH" \
  python -m pytest -q tests/e2e/test_exact_checkpoints.py
```

The test loads one engine, then exercises memory and disk reuse across eight
page/recurrent endpoint lengths and two branches, comparing with cold runs using
identical compute partitions. The passing run (`m4-real-model-state-v4.log`)
completed in 66.24 seconds. It asserts actual disk-restore completion and prefix
admission before comparing every captured state byte and greedy output. This is
not the 8K/32K restore-performance or restart/endurance gate. A prior cross-partition
API probe found a tied-logit output difference, which remains unqualified.

CPU record/lifecycle checks use real temporary files and require no model download:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python \
  python -m pytest -q tests/runtime/checkpoint tests/runtime/scheduler \
  tests/runtime/kvcache/radix tests/runtime/engine
```

Run CUDA staging separately on an otherwise permitted GPU:

```bash
PYTHONPATH=python python -m pytest -q \
  tests/runtime/checkpoint/test_transfer.py \
  tests/runtime/checkpoint/test_transfer_cancel.py \
  tests/runtime/checkpoint/test_transfer_fence.py \
  --basetemp="$HOME/results/flare-runtime/transfer-owned-new-run"
```

Use a fresh, owned `--basetemp` path: pytest may replace an existing one. The
reported thirteen-test pass includes private restore adoption, shutdown ordering with
a paused GPU/file writer, and six cancellation/coalescing/retry cases. The latter
pause real staged reads before copying, after a copy and after checksum validation;
they check private ownership while the worker is paused, byte-exact state after
retry into poisoned storage, and eventual page/slot/file-pin conservation. The
transport, checksums and CUDA copies execute normally; only the worker's progress
is gated. Per-case `evidence.json` files retain observations and component hashes.
The state is synthetic, not model inference. Four additional pending-fence cases
add an idempotent final 24-byte auxiliary-state DMA to the actual restore stream.
A CUDA driver wait on a four-byte mapped host semaphore holds that write after
the file read finishes; event and stream queries must report pending. Another
stream must complete work during the hold. Cancellation, timeout and a short
shutdown attempt retain the completed ticket, staging, file pin, private pages and
state slot. A surviving coalesced waiter may publish only after the real DMA
drains, with byte-exact complete state. A ten-second host rescue bounds the hold;
rescue firing fails the test. Optional unsupported driver primitives skip before
the I/O worker starts. No event query or transfer-complete result is mocked.

These finite checks do not prove full-model HTTP cancellation, sustained resource
bounds or model quality. The final DMA is an explicit test injection, not evidence
that an ordinary record read naturally completes before its staged copies. CPU
transport tests do not substitute for this CUDA gate. Keep shared-host pressure
and pre-existing swap distinct from runtime-attributable memory behavior.

Native process shutdown checks run an actual HTTP server, supervisor, spawned
workers, named semaphores, local tokenizer and bounded prefix store. Model
execution is replaced, so these checks do not qualify model continuation or
active-generation cancellation:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python \
  python -m pytest -q tests/serving/test_worker_lifecycle.py
```

For persistent subprocess logs, set `SPARKLAB_LIFECYCLE_EVIDENCE_ROOT` to a new
directory outside the checkout. Each scenario creates its own directory and
rejects an existing one to prevent mixing runs. To validate the installed wheel,
point `PYTHONPATH` at its extracted package root instead of `python`; spawned
workers inherit the package location under test. The default-SIGTERM and foreground
worker process-group compatibility decisions and full-model qualification limits are described in
`FLARE_design.md` under native process shutdown.

Active HTTP drain and disconnect/retry checks use synthetic model replies over
the actual frontend ZMQ links and `/generate` SSE route:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python \
  python -m pytest -q tests/serving/test_http_worker_lifecycle.py
```

Set `SPARKLAB_HTTP_LIFECYCLE_EVIDENCE_ROOT` to a fresh external directory to keep
HTTP observations and process logs. Parent-only SIGTERM and terminal-group
SIGINT/SIGTERM must enter Uvicorn's active-connection drain before the fixture
releases its final reply. The disconnect case observes a worker abort and an
independent retry. These are frontend/process lifecycle proofs; model execution,
public-protocol semantics, GPU cancellation and full-model backpressure remain
separate gates.

Worker-failure checks admit nine concurrent requests across streaming/buffered
Completions, Chat, Responses and Anthropic, plus `/generate`, then raise a real
worker exception. They check terminal errors, request-map retirement, preserved
observed token totals and refusal to seal uncertain accounting:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python \
  python -m pytest -q tests/serving/test_backend_failure_lifecycle.py
```

`SPARKLAB_FAILURE_EVIDENCE_ROOT` retains HTTP results and subprocess logs in a
fresh external directory. Model replies and the tiny local chat template are
synthetic; frontend routes, ZMQ, supervisor, process failure and accounting are
actual production paths. Two additional loop/thread tests exercise failure
racing first send, duplicate notification, genuine queued terminal replies and
late accounting messages. This is not model continuation or GPU failure recovery.

Real TCP backpressure checks pause a streaming socket with a small receive buffer
until an actual ASGI body send is waiting. They verify a second request completes
through the same protocol, then resume or disconnect the slow client:

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=1 PYTHONPATH=python \
  python -m pytest -q tests/serving/test_http_backpressure.py
```

The ten cases cover `/generate`, Completions, Chat, Responses and Anthropic.
`SPARKLAB_PRESSURE_EVIDENCE_ROOT` retains compact counters, accounting, wire-body
hashes and subprocess logs under fresh `protocol/action` directories. Large
bodies are not saved. Resume must deliver all 8 MiB of synthetic text with the
protocol's terminal marker and reconciled usage; disconnect must deliver the
worker abort and its terminal accounting acknowledgement. Both retire frontend
request maps and queues. The synthetic producer emits 64 chunks; this is not an
8K/32K context test, model latency benchmark, full-model GPU cancellation gate or
sustained pressure/memory bound. The queue observation excludes the active
iterator, ASGI body and transport buffers; it is not a total resident-byte census.

## Application sessions and fixed answer oracles

`validate_runtime_sessions.py` drives buffered Chat Completions on an already
launched native server. It executes 20–50 turns, a real `lookup_fact` tool call
and return every fourth turn, two branches from one retained parent, and a
structured fact/source summary. A 20-turn run plans 28 requests; a 50-turn run
plans 65. Responses must satisfy strict JSON/value/source oracles and reconcile
usage with the local native tokenizer. Failed steps stay failed; dependent work
is recorded as not run. Partial evidence is replaced after each step, including
an in-flight record before the HTTP call. An interrupted report is not finalized.

Use the exact serving tokenizer and an identity JSON recording the actual pinned
artifact/backend/precision/cache profile. Launch each control separately with
matched capacities and sampling. The harness does not infer a remote server's
weight identity from its model alias:

```bash
PYTHONPATH=python python benchmarks/validate_runtime_sessions.py \
  --base-url http://127.0.0.1:19291 --model session-test \
  --tokenizer "$MODEL_PATH" --identity "$HOME/results/flare-runtime/session-identity.json" \
  --turns 20 --max-context 4096 --label baseline \
  --output "$HOME/results/flare-runtime/session-baseline.json"
```

For a candidate, change the output and label and add
`--baseline "$HOME/results/flare-runtime/session-baseline.json"`. Comparison
requires complete independent oracle passes, matching rendered token-ID hashes,
strict output signatures and equal completed output counts. Actual tool IDs are
preserved in assistant/tool history; the output signature excludes these generated
protocol IDs, while input hashes still detect any effect they have on rendering.
No timing benefit, state-byte proof, stochastic distribution or endurance claim
comes from this correctness runner.

Add `--legal-case` to include two independent requests over the same wholly
fictional service-dispute record: an attributed structured summary and a document
chronology. The summary must distinguish the original and corrective deadlines,
preserve each party's literal statement as unverified, cite the right document,
and retain an absent court finding as JSON null. Chronology sorts all four
document dates rather than a delivery date mentioned within a statement. Exact
JSON oracles reject omissions, extra fields, invented findings and changed types;
the matched baseline/candidate check still requires identical rendered inputs,
strict outputs and completed work. This tests runtime preservation, not legal
correctness or general legal-summary quality. The supplied record, queries and
oracles are versioned in `benchmarks/validate_runtime_sessions.py`; their hashes
enter workload identity when this option is selected.

This option adds two requests: 30 for a 20-turn run, 67 for a 50-turn run before
optional recall. It is off by default and leaves the original session workload
unchanged. Both new prompts fit the declared 4096-context/256-output profile with
the selected local Qwen tokenizer (344/309 prompt tokens; canonical oracle answers
132/84 tokens). This is a token-budget check, not a model-answer result. Other
artifacts still require their actual serving tokenizer and sufficient output space.

`--recall-lengths 8192 32768` adds three dispersed facts at each requested exact
rendered length, measured with the serving tokenizer rather than estimated from
words. These prompts need output space in a separately budgeted, supported context
profile. They cannot run in the default 4096-context server. Construction failure
remains explicit rather than substituting a shorter prompt. Duplicate JSON keys,
nonfinite values, wrong tool arguments/IDs, truncated answers and missing usage
are failures. The raw requests/responses stay in the external output report.

The implemented paired full-model fixture launches an unchanged target-only
baseline and an explicit candidate, captures actual cache geometry, rejects
changed capacities, checks feature activity, and stops each server before the
next boot:

```bash
PYTHONPATH=python SPARKLAB_SESSION_TEST_MODEL="$MODEL_PATH" \
  SPARKLAB_SESSION_EVIDENCE_ROOT="$HOME/results/flare-runtime/session-real" \
  python -m pytest -q -rs tests/e2e/test_runtime_sessions.py
```

Its default candidate is exact-prefix retention only; the existing baseline
scheduler and fixed target-only mode remain selected. `SPARKLAB_SESSION_FEATURES`
accepts explicit target-only checkpoint/scheduler/disk settings. A nonempty
`SPARKLAB_PREFIX_STORE` value there is redirected to this test's owned directory.
`SPARKLAB_SESSION_SERVER_ARGS` accepts only resource-profile overrides;
`SPARKLAB_SESSION_MAX_CONTEXT`, `SPARKLAB_SESSION_TURNS`,
`SPARKLAB_SESSION_OUTPUT_TOKENS` and `SPARKLAB_SESSION_RECALL_LENGTHS` declare the
workload envelope. `SPARKLAB_SESSION_PACKAGE_PATH` selects the extracted wheel.
Set `SPARKLAB_SESSION_LEGAL_CASE=1` to include the fictional case in both paired
boots; its only accepted values are `0` and `1`. For example:

```bash
PYTHONPATH=python SPARKLAB_SESSION_TEST_MODEL="$MODEL_PATH" \
  SPARKLAB_SESSION_LEGAL_CASE=1 \
  SPARKLAB_SESSION_EVIDENCE_ROOT="$HOME/results/flare-runtime/session-with-case" \
  python -m pytest -q -rs tests/e2e/test_runtime_sessions.py
```

The 30 GiB physical-memory floor applies before each boot and cannot be lowered;
longer/larger profiles require measured budgets and may raise it through
`SPARKLAB_SESSION_MIN_FREE_GIB`. This host currently fails the floor, so the paired
model session/recall gate remains **unrun**. CPU HTTP fixtures validate the runner,
and local tokenizer construction validates lengths, not model answers or state.

## Disk restart and disable rollback dispatch proof

The four-boot native fixture runs a cold query, publishes its exact prefix,
restarts with disk retention to run the extended query, then disables exact/disk
retention and repeats the query. Each boot uses the same target-only, non-overlap
profile and actual cache capacities. Raw bounded native dispatch logs must show
full recomputation on cold/disabled boots and only the uncached suffix after
restart. A restore counter alone cannot pass. Greedy output, usage, completed
output work, recovered startup records and drained transfer resources must agree;
the disabled boot must preserve the retained record inventory.

```bash
PYTHONPATH=python SPARKLAB_RESTART_TEST_MODEL="$MODEL_PATH" \
  SPARKLAB_RESTART_EVIDENCE_ROOT="$HOME/results/flare-runtime/prefix-restart" \
  python -m pytest -q -rs tests/e2e/test_prefix_restart.py
```

Use a fresh evidence directory. Defaults are a 65-token prefix with a private
partial tail, a one-token suffix, four output tokens, 4096 context tokens, 512
16-token KV pages and one running request. Before each boot the fixture requires
at least 30 GiB free physical GPU memory; `SPARKLAB_RESTART_MIN_FREE_GIB` may raise
that floor. The owned 2 GiB disk store requires at least 3 GiB free disk initially.
`SPARKLAB_RESTART_PACKAGE_PATH` selects an extracted wheel instead of `python`.
`SPARKLAB_RESTART_PREFIX_TOKENS`, `SPARKLAB_RESTART_MAX_CONTEXT` and
`SPARKLAB_RESTART_PAGES` declare larger supported profiles. An 8192/32768-token
prefix needs explicitly larger context/page capacities and a measured memory
budget; it does not fit the default profile.

Diagnostic timing and the 512-event dispatch trace stay enabled across all four
boots. Existing empty `/v1/cache/rebuild` control requests wake the idle scheduler
to observe asynchronous startup recovery. Empty rebuild requests perform no pool
resize or cache clear; unchanged geometry and an empty startup dispatch trace are
required. Restore policy `always` deliberately exercises recovery and does not
qualify measured restore-versus-recompute selection or a performance benefit.
Commands, requests, replies, actual geometry, raw logs and partial failure evidence
remain in the external directory. Ordinary shutdown is required; forced cleanup
cannot pass the verifier.

Recheck a completed evidence report independently:

```bash
python benchmarks/validate_prefix_restart.py \
  --evidence "$HOME/results/flare-runtime/prefix-restart/evidence.json"

CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=1 PYTHONPATH=python \
  python -m pytest -q tests/test_prefix_restart_evidence.py
```

The CPU controls use independent synthetic dispatch timelines, actual telemetry
serialization and failure injection. They validate the evidence checker, not
model restart correctness. The real fixture remains unrun below its memory guard.
Even a successful short restart run would leave the 8K/32K complete-state, recall,
interference, benefit and endurance gates open. Strict output mismatches fail;
the existing cross-partition quality findings are not waived.

## Exact-prefix capacity regression

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/runtime/checkpoint/test_admission_capacity.py

PYTHONPATH=python python -m pytest -q \
  tests/runtime/checkpoint/test_admission_capacity.py -k cuda
```

The four-token-page controls seat complete declared consumed work in tiny pools,
reject disk restores one page short, protect live readers, and require cold
fallback to progress after unread snapshot eviction. Further controls check
per-request rounding, whole-prompt reservations across chunks, all-active-request
publication headroom, allocator reclamation, and parallel fresh/chunked private
tails charged once. Sixteen CPU cases and ten actual CUDA state cases use
synthetic state functions and real production cache paths;
they do not run model inference. Logs and the rebuilt-wheel manifest are under
`$HOME/results/flare-runtime/m4-tail-capacity-*`.

## Endurance evidence collection

Use an explicitly owned test server and a fresh output directory. First obtain a
baseline with `bench_runtime_mixed.py` for the exact workload/identity below.
Cycles require 2..64 cases with explicit `temperature: 0`, fixed output budgets
up to 2048 tokens, arrivals within 60 seconds, and both completed and deliberately
cancelled requests. Include declared long arrivals and repeated/branching prefixes
in that workload; the collector does not invent model prompts or remove cases.

```bash
python benchmarks/validate_runtime_soak.py \
  --base-url http://127.0.0.1:1919 --model runtime-test \
  --workload "$HOME/results/flare-runtime/soak-workload.json" \
  --identity "$HOME/results/flare-runtime/identity.json" \
  --baseline "$HOME/results/flare-runtime/soak-baseline.json" \
  --duration-seconds 3600 --concurrency 4 --timeout 180 \
  --sample-interval 5 --rebuild-every 10 --max-evidence-mib 256 \
  --output "$HOME/results/flare-runtime/soak-60m"
```

For the four-hour gate use `--duration-seconds 14400` and a different fresh
directory. One full-path warmup precedes the timed interval. Cycles finish their
declared work before stopping, so actual duration can exceed the request. Counts
include warmup; cycle records label it separately. The collector reuses the
streaming client, checks exact completed text/token usage/finish reasons against
the baseline, requires the intended socket cancellations, samples `/v1/stats`
and host `/proc/vmstat` swap-out, and waits for frontend abort/accounting drain.
`--rebuild-every` explicitly opts into idle cache clears with unchanged page
capacity; bounded busy retries permit prefix I/O to drain. A changed instance ID,
output or geometry, failed request, missing drain or exhausted evidence budget
fails explicitly. There is no implicit server launch, stop or restart.

`cycles.jsonl` retains every completed cycle; `resources.jsonl` retains sampled
stats, including worker observation ages; `report.json` is atomically replaced
with progress and failure state. JSONL evidence is capped at the declared budget
(1..4096 MiB, default 256); prior files survive budget exhaustion. Existing output
directories are rejected. Request attempts whose observations were interrupted
remain explicit in `unobserved_requests`. Baseline, workload and source digests
are recorded. Concurrency is capped at 16, resource bodies at 2 MiB, and total
requested duration at eight hours. Only the current bounded cycle stays in RAM.

The result is **collection**, with `qualification: false`, even after the full
duration. Stale worker counters cannot establish drained GPU/I/O ownership, and
global swap-out cannot establish attribution on a shared host. Complete state,
memory/latency drift, temperatures, expert contention, dynamic tools, factual
answers, restart/disable and the remaining full integrated-profile gates still
need their separate evidence and review. Timed duration alone closes none of
those gates. The short HTTP controls use synthetic replies, not model inference;
the actual 60-minute and four-hour model runs remain unrun.


## Allocator observation control

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q -rs tests/runtime/scheduler/test_allocator_observation.py

PYTHONPATH=python python -m pytest -q \
  tests/runtime/scheduler/test_allocator_observation.py -k real_cuda
```

The CUDA control allocates one MiB in a real PyTorch CUDA tensor, observes the
live allocation, releases it, and requires the live-byte counter to return to
the preceding value. The reserved allocator may retain storage. Public device
wait and tensor readback methods are forbidden during observation. CPU controls
check disabled/CPU behavior, a single read per exported second, immutable prior
observations, coherent allocated/reserved pairs and missing statistics as null.
The prior tail-capacity wheel fails the controlled export because it has no
live-allocation field. This is allocator lifetime evidence, not full-model
resource conservation, physical memory accounting or a telemetry overhead result.

The experimental `native_runtime.resources` export now distinguishes current
`gpu_allocated_bytes` from `gpu_reserved_bytes`, from one host-side allocator
metadata read. Allocated is contained in reserved; do not sum them. Neither
includes all driver/non-PyTorch memory, and neither can simply be added to host
RSS on GB10. Named retained buffers/budgets overlap these totals. Allocator
backend semantics still apply. Snapshot ages and the complete-model/measurement
overhead gates remain explicit; defaults and the legacy vram gauge are unchanged.


## Adaptive phase boundary controls

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q -rs tests/runtime/engine/test_adaptive_phase_timing.py

PYTHONPATH=python python -m pytest -q \
  tests/runtime/engine/test_adaptive_phase_timing.py -k cuda
```

The CPU clock supplies independent proposal/snapshot/target/decision/commit costs
through actual Engine.forward_batch, Scheduler._forward and output drain. Four
cases cover target-only, first rejection, later rejection and all accept. CUDA
cases use small real tensor operations/events with synthetic target logits and
model/graph callbacks; they do not load model weights or validate actual graph
capture. The nonblocking output copy must follow the commit marker, phase sums
must reconcile to the GPU round, and all fixed histograms receive one observation.
Separate controls require optional bounded event ownership and completed markers.
A process-local wrong-boundary mutation fails all four CPU pipeline cases.

The complete DFlash matrix command above now enables telemetry, requires all phase
fields after actual drain and retains strict output/state assertions. Its memory
probe runs in a child to preserve the offline engine's fresh-CUDA precondition.
Use a fresh Python process for that fixture. The full-model run still skips below
30 GiB free physical GPU memory; tiny CPU/CUDA interval controls do not resolve
prior quality failures, prove multi-stream model ordering, or qualify overhead or
benefit. Interval definitions and limits are in FLARE_design.md.


## Dispatch outcome, age and rejection controls

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q -rs \
  tests/runtime/scheduler/test_dispatch_observations.py \
  tests/runtime/scheduler/test_runtime_telemetry.py \
  tests/runtime/scheduler/test_adaptive_round_drain.py
```

The dispatch cases use real small-pool PrefillAdder admission and DecodeManager
selection, with CPU-only preparation and a fixed independent clock. They require
truthful fallback reasons in baseline and latency modes, pre/post queue counts,
selection-time ages even when preparation advances time, and zero/unknown clock
handling. The prior phase wheel fails all eight cases. Additional scalar/drain
controls check first-rejection censoring, at most 16 position keys, invalid
observation atomicity and no aborted-round counter updates. Selection and age
semantics are documented in FLARE_design.md. These checks do not establish
model responsiveness, service deadlines, trace overhead or sustained overload.


## Tensor-parallel latency decision agreement

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/runtime/scheduler/test_tp_policy.py \
  tests/runtime/scheduler/test_latency_policy.py \
  tests/runtime/scheduler/test_dispatch_observations.py
```

The two spawned CPU workers use a real Gloo process group and deliberately
conflicting local timing inputs. They require the primary phase, chunk and
service budget on both ranks, prohibit peer selection, and check shared failure
for primary exceptions and incompatible runnable envelopes. Local baseline/TP1
controls prohibit added collectives. Actual Scheduler wiring is checked against
an authoritative received decision; the preceding wheel fails that control.
This is coordination evidence, not TP model correctness, NCCL, latency or
throughput qualification. Existing exact/adaptive TP1 eligibility is unchanged.


## Unexpected scheduler failure cleanup

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/serving/test_scheduler_failure_cleanup.py
```

Six worker controls cover inference/sync/SystemExit failures with successful or
failing cleanup, retaining the exact original fatal cause. Two actual Scheduler
shutdown controls require prefix-drain-before-IPC ordering and prohibit peer or
normal device teardown on the failure path. A real spawned CPU worker fails with
a paused prefix-file write active; cleanup releases the gate, completes the
write and releases its namespace. The test independently reopens and validates
the committed record, then requires worker reap and semaphore release. The
preceding TP-policy wheel fails all nine controls. This does not qualify a
broken CUDA context, constructor-partial owners, multi-rank failure teardown or
full-model failure during pending GPU transfers. Supplemental CUDA controls run
`tests/runtime/checkpoint/test_transfer.py -k scheduler`: both normal and failure
shutdown drain a real small GPU-backed record before IPC closes. The failure is
an explicit cleanup-mode injection with a usable device; the I/O gate pauses
before transfer, not inside a damaged CUDA operation.


## Byte-fallback Unicode stop strings

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/runtime/scheduler/test_unicode_stop.py \
  tests/runtime/scheduler/test_request_timing.py \
  tests/runtime/scheduler/test_adaptive_round_drain.py \
  tests/serving/test_generation_accounting.py
```

Five stop strings cover one- to four-byte characters, a combined Unicode stop
and ASCII. Real ByteLevel BPE encoding/decoding provides the independent oracle;
the scheduler matcher and burst drain, message codec, DetokenizeManager and
resource owners are production paths. The prior wheel fails six of ten new
controls and passes the ASCII/two-byte controls. The corrected source passes
all ten; the focused combined selection is 33 passes and one CUDA skip.
Full-model API stop/tool/grammar and existing strict quality gates remain open.


## Atomic image prompt deferral

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/runtime/scheduler/test_atomic_multimodal_admission.py

SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/runtime/scheduler/test_atomic_multimodal_admission.py -k cuda
```

The controls place a six-token image prompt after a four-token text/image request
under an eight-token prefill budget. Both prompts fit individually. Only the
first may enter the first batch; the second must retain its complete prompt and
conditioning for the next pass, with tentative ownership released between
passes. Page sizes one/four and both text/image predecessors are covered on CPU
and CUDA. An independent oversized-prompt control requires the existing error
before sizing/copy mutation. These test production admission and resource owners
with synthetic embeddings, not vision outputs or full-server latency.


## Multi-rank stop at different collective positions

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/serving/test_tp_worker_stop.py \
  tests/serving/test_scheduler_failure_cleanup.py
```

Two spawned workers use a real Gloo process group with a six-second timeout.
One peer has submitted an unmatched broadcast and waits in Work.wait; the
primary is between operations. Stop through the actual parent helper (SIGINT)
or direct worker SIGTERM must close local owners, exit both ranks successfully,
reap both processes and release owned named semaphores within the stop bound.
Production shutdown and worker wrappers execute; model scheduling and the CUDA
fence are replaced with CPU fixture callbacks. Both cases fail against the
preceding wheel because shutdown adds a mismatched barrier. These controls do
not qualify NCCL, model execution, active GPU cleanup or a forced-kill drain.


## Startup error reporting interrupted by stop

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/serving/test_startup_stop_race.py \
  tests/serving/test_scheduler_failure_cleanup.py \
  tests/serving/test_tp_worker_stop.py
```

Four unit controls distinguish constructed/unconstructed owners and successful
or failing cleanup while preserving the exact original exception object. Two
spawned CPU workers receive real SIGINT/SIGTERM after reporting an injected
synchronization failure, with a gated real prefix write still active. Only
cleanup unblocks the writer. Both must retain the original error/nonzero exit,
complete the write, release owned semaphores and reap; the parent independently
reopens and checksum-validates the record. The prior wheel fails four of the six
new cases, while both no-owner controls pass. This covers the specific reporting
boundary; partial construction, arbitrary interruption points and GPU/model
failure remain separate gates.


## Controlled stop before the worker wrapper

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/serving/test_pre_wrapper_stop.py
```

The real serve probe holds its child at first worker entry, before the explicit
Torch import and before stop handlers/group detach. It records the actual handler
and import state. Parent SIGTERM is tested while the launch callback still owns a
partial start and after the backend handle has returned with health loading.
Both cases require parent status 143, an expected child interrupt exit, no forced
kill or supervisor false alarm, process reap and all three owned queue semaphores
released. The optional gates exist only in the test probe; production is unchanged.
These controls do not cover arbitrary multiprocessing bootstrap/import bytecode
interruptions, model construction or active GPU ownership.


## Stop while the child exists but Process.start has not registered it

```bash
CUDA_VISIBLE_DEVICES="" SPARKLAB_DISABLE_JIT=0 PYTHONPATH=python \
  python -m pytest -q tests/serving/test_spawn_gap_stop.py
```

The probe injects actual SIGINT/SIGTERM after Popen launches a child but before
Process._popen assignment. Child import/handler state and missing handle are
observed directly. A separate controller is a Linux subreaper so a failing prior
artifact cannot strand the deliberately unregistered child. Positive assertions
require ordinary parent cleanup before controller recovery, complete worker reap
and queue-semaphore release. Unit controls check returning/raising callbacks,
coalescing, original startup-error identity, handler restoration, frame lifetime,
ignored/default signals and the non-main-thread fallback. The prior artifact's
actual early-stop status is measured separately; no generic Python signal exit
code is substituted for it. No model weights or GPU work are exercised.
