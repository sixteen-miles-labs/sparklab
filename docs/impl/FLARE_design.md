# Native runtime compatibility and ownership design

This implements the supplied plan in separate review units. Existing scheduler,
radix cache, CUDA graphs, accepted-prefix transactions, FTW and backend adapters
remain the runtime. Experimental features do not alter selected recipes.

## Measurements and scheduling

The explicit baseline is prefill-first. `SPARKLAB_SCHEDULER_POLICY=latency` selects
an internal experiment with a bounded prefill service interval and at most one
prefill dispatch between runnable decode dispatches. It selects legal work through
the existing PrefillAdder; allocations, cancellation and graph preparation remain
in their existing owners. Chunk budgets preserve page/recurrent alignment and
never split an image prompt. This cannot preempt a running operation.

`SPARKLAB_SCHEDULER_DELIVERY_MS` (default 100) is a desired output-ready budget,
not a hard client latency promise. `SPARKLAB_SCHEDULER_PREFILL_SERVICE_MS`
(default 1000) bounds opportunities for the oldest resource-admissible prefill;
it cannot promise bounded queue waits under unlimited overload or unavailable
memory. CPU tests exercise the selection with an independent execution clock.
Actual GPU operation durations and visible SSE gaps require real-server evidence.

For latency scheduling with TP greater than one, rank zero selects the phase,
prefill token budget and decode service budget from its observed ages/costs. The
existing CPU process group broadcasts five float64 scalars (40 bytes), then
all-reduces one int64 agreement vote before batch preparation. Peers do not run
the timing-based selector. Invalid budgets or incompatible runnable envelopes
fail on every rank; a primary selection exception is broadcast as failure before
raising. Collectives use the existing group timeout. Baseline and TP1 selection
add no collective. Queue/request/cache identity remains the existing TP invariant;
the vote checks the runnable envelope, not identity or recovery from corrupted
state. Rank-zero cost authority does not establish slowest-rank latency bounds.
Two-process Gloo CPU controls cover divergent ages/costs, phase/chunk agreement,
shared failure and subsequent group usability. Actual TP model/NCCL execution,
communication overhead and sustained benefit remain qualification gates.

Low-cost counters/histograms are bounded. `SPARKLAB_RUNTIME_TIMING=1` records GPU
events; latency mode also enables it. Timing polls use completed events without
adding synchronization. Pending event pairs are capped at 64 and cost shapes at
64, each retaining at most 128 timings. Cold timing estimates remain nonzero.
`SPARKLAB_RUNTIME_TRACE_CAPACITY=0` disables the diagnostic ring by default; its
maximum is 100000 records. `SPARKLAB_RUNTIME_LOG=1` emits a snapshot at idle.
No request strings, IDs, paths or case names become metric labels. Scheduler
output-ready measurements are explicitly distinct from HTTP/client delivery.

`SPARKLAB_RUNTIME_TIMING=1` also enables bounded frontend-stage histograms;
latency scheduling enables these measurements too. Internal messages carry a
fixed set of host-monotonic stamps from ASGI ingress, frontend submission,
tokenization, scheduler receipt/admission and first dispatch to first generated
output, detokenization and frontend receipt. Only the first generated token in
a speculative burst carries the first-output stamps. Chunked prefill preserves
the original admission/dispatch stamps. The generated timestamp is the host's
observation after the existing output fence, not an invented GPU token time;
GPU duration remains a separate measurement.

The existing `/v1/stats` response includes `runtime_timing` only when enabled.
The optional stats contract is recorded in
[design issue 97](https://github.com/sixteen-miles-labs/sparklab/issues/97).
It contains fixed-name histograms and counters plus an optional ring capped by
`SPARKLAB_RUNTIME_TRACE_CAPACITY` (default zero, maximum 100000). No new global
per-request map is retained. Stamps contain no request IDs, prompt text or paths.
The frontend and scheduler rings have separate bounded capacities. Missing stages
stay missing; errors and incomplete/disconnected HTTP requests remain counted.
Admission and first-output samples count generations, while HTTP samples count
responses; batched completions can contain multiple generations.

Parsed semantic-content readiness excludes empty/control-only deltas. ASGI first
body may be an earlier role/metadata frame. First body send start, send return
and final body send return are separate from semantic readiness and from client
arrival. A pure ASGI timing wrapper sits outside the buffering HTTP middleware,
so downstream send backpressure is included. These are still server observations;
only the client harness can measure visible delivery on the client. Measurement
overhead and complete-model protocol stage reconciliation remain qualification
gates, not consequences of passing timestamp fixtures.

The native worker also piggybacks a version-one `native_runtime` observation
on at most one ordinary output reply per second when runtime timing is enabled
(including disk-retention timing). This adds no metrics queue. The detokenizer
preserves the observation and the frontend retains only the latest one for
`/v1/stats`, with `observed_at` and `age_seconds`. Detailed trace rings never enter
this IPC payload. Idle/final snapshots can remain stale: this export alone is
not a drain-completion oracle.

The observation joins bounded scheduler histograms and GPU shape p90 costs,
KV/recurrent/window occupancy, separate GPU allocated/reserved bytes, exact-prefix ownership,
prefix tickets/staging/private restore resources and measured read/write service
costs. The production GPU shape table has 64 entries and 128 samples per entry;
prefix costs have 32 entries and 32 samples per entry. Named retained verification
buffers and the existing prefix/adaptive-graph reservations are reported from
tensor shape metadata and cached scalar values. These named buffers are not an
aggregate census of all graph/scratch storage; the full M6 inventory remains
separate. Missing components remain explicit rather than becoming zero bytes.

The allocator pair comes from one host-side PyTorch `memory_stats` observation
after the export rate limit, at most once per second. Live `gpu_allocated_bytes`
and `gpu_reserved_bytes` are current counters from the same allocator sample;
allocated bytes are contained in reserved bytes and must not be summed. Missing
statistics and non-CUDA devices report null. No tensor copy, allocator flush,
file read, device wait or new retained buffer is needed. The existing reply's
`gpu_mem_bytes`/frontend `vram_bytes` reserved-memory gauge is unchanged.
These are PyTorch allocator counters, with the selected allocator backend's
semantics. They exclude driver and non-PyTorch owners and do not measure total
physical memory. They cannot be added to host RSS on GB10 to establish an
independent physical-memory total. Named workspace/budget bytes also overlap the
allocator totals. Idle observations remain stale, and host/process accounting,
cross-subsystem reconciliation and measurement overhead remain qualification gates.

The prefix worker replaces a completed metadata observation after boot and each
I/O operation. It includes file counters and bounded quota/index totals; the actor
never scans files to answer telemetry. Root-quota counts carry the timestamp of
their last admission/publication observation, separately from the newer worker
observation after a read. Snapshotting does not trigger a root census. Unknown
files remain outside cache-owned accounting as described below.

Expert-disk counters are sampled from the existing host-only source statistics.
Their physical-byte field describes aligned source reads, not independent NVMe
device traffic; filesystem cache conditions still matter. With both runtime
timing and the existing `--moe-collect-stats` enabled, device-side expert counters
also receive a bounded live sample. One four-int64 device buffer, one equally
sized pinned host buffer and one event are reused; at most one copy is pending
and submissions are at least one second apart. Aggregation and copy follow
counter writes on the engine stream. The worker only queries completion; it
converts the pinned host values after the event passes. No cache/weight owner is
retained by the sampler. Existing rebuild/shutdown fences drain these operations.

`expert_decode.observed_at` is the sample's host submission time, a conservative
freshness timestamp rather than a GPU execution timestamp. Host prefill fields
are captured at submission. A newer existing terminal observation takes
precedence over an older asynchronously completed sample. Disabled collection
does not initialize the sampler. Its retained payload bytes and pending flag
appear in `retained_workspaces.expert_counter_sampling`; allocator/driver overhead
is outside those payload-byte counts. Pending means not yet consumed by a poll,
including a finished copy awaiting another output drain. The periodic exporter
does not call the synchronizing terminal counter readers. Sampling piggybacks on output drains, so
idle observations can remain stale; this is not a per-round or one-second delivery
guarantee. Full-model ordering and measurement overhead remain qualification gates.
This common export correlates observations with different freshness; it does
not make them an atomic cross-subsystem snapshot or qualify contention benefit.

## Exact checkpoint contract

The first complete-state target is the available dense-weight Qwen3.8-27B native
control, whose architecture is **hybrid GDN/full attention**, not pure MHA.
It fits on the shared machine; the preferred Flash-Next profile does not currently
fit alongside the active services. No qualification transfers between them.

The state contract includes consumed token IDs, MHA full-page chains,
a private partial boundary page, GDN recurrent and convolution state for all
layers, registered auxiliary state, exact position and a compatibility identity.
Target/draft continuation must be included before speculation is enabled for this
contract. Multimodal requests remain excluded. Existing state donation and copy
operations must materialize any deferred accepted-prefix transaction first.

Exact endpoints come from the tokenizer's effective rendering. A candidate message
prefix is eligible only when its token IDs are an exact prefix of the full prompt.
Equal endpoints are deduplicated. The terminal endpoint is consumed state, not
sampled output length. Overadvanced state cannot be published. Shared published
full pages remain immutable and pinned; partial pages restore to private pages.
A full prompt hit without saved resume data must explicitly recompute a safe suffix.

The implementation must extend existing radix ownership and reference-model tests.
It must not round a key backward while keeping a later recurrent state. Both
logical owner counts and outstanding GPU fences retain physical resources.
Pool exhaustion skips publication; it never expands an unbounded snapshot tier.

Exact-cache admission reserves whole private pages per request, including the
consumed rows of a copied partial tail. Disk restore counts both immutable source
pages and the rounded live branch budget. If a memory hit cannot fit, admission
may discard ready, unread exact records and retry once through the ordinary
lookup; active readers and pending GPU copies still prevent eviction.

The initial hit check counts the private tail before it is allocated. After
`restore_tail` debits the free list, later batch candidates are charged only for
the admitted request's future pages. A continuation similarly reserves only the
rounded pages beyond its existing `allocated_len`, through the full pending
prompt and output budget. Already-owned partial pages must not be charged twice.
This permits simultaneous branches that physically fit while retaining the
whole-page overcommit checks and immutable-parent ownership.

Each active request retains its whole pending prompt plus output reservation
across intermediate checkpoint chunks. The table-indexed future-page ledger is
bounded by the request table and cleared on finish/abort; rebuild requires it to
be empty. Publication skips a new partial snapshot if it would consume those
future pages, including reservations belonging to other active requests. Ordinary
allocation can reclaim ready, unread exact tails before evicting radix pages.
These rules preserve physical capacity; they do not qualify full-model quality
or a restore performance benefit.

## Persistent compatibility and ownership

Retention is a separate subsystem from expert-weight offload. Version-one records
have a bounded, versioned JSON header and nonexecutable typed tensor payloads,
compatibility fingerprint, consumed-prefix digest, dimensions, lengths and checksums.
Identity covers effective artifact, precision, tokenizer/template/config hashes,
state layout and cache geometry, continuation/speculation mode, plus explicit
single-instance scope. Live page IDs are never portable identity. A new scope misses.

The disk store must own a distinct namespace and lock it against concurrent writers.
It must bound bytes, record count, staging and queue metadata, publish with atomic
rename and fsync ordering, and recover only validated records. Cancellation detaches
a waiter without recycling a live transfer ticket. Shared immutable source state
remains pinned until copy-out completion. Restore remains private until validation
and transfers finish. Prepublication failures may recompute; postpublication
invariant failures must remain visible.

The stored-format design is recorded in
[issue 96](https://github.com/sixteen-miles-labs/sparklab/issues/96).
`SPARKLAB_PREFIX_STORE` enables the experiment only with `SPARKLAB_EXACT_PREFIX=1`.
It is unset by default. `SPARKLAB_PREFIX_SCOPE` defaults to `single-instance`;
this is not tenant isolation. Scope and portable effective-artifact identity select
the namespace. Source relocation or pool-capacity changes preserve identity;
changed precision, tokenizer/template, page geometry or continuation mode miss.

Default limits: 8 GiB disk (`SPARKLAB_PREFIX_DISK_GIB`, 1..1024), 128 records
(`SPARKLAB_PREFIX_DISK_RECORDS`, 1..4096), four outstanding tickets
(`SPARKLAB_PREFIX_IO_TICKETS`, 1..8), 8 MiB staging per ticket
(`SPARKLAB_PREFIX_STAGING_MIB`, 1..64), one inbound restore and at most
`min(128, max_running_req*2)` waiters. A ticket includes queued, running, completed
and unreleased ownership. Exact in-memory metadata is separately bounded by
`SPARKLAB_EXACT_PREFIX_METADATA_MIB` (default 16, 1..64). Physical admission
reserves staging and metadata from the unified memory pool at engine startup.

Disk byte and record limits apply across compatibility namespaces under the
configured root, including stale identities. A nonblocking `.quota.lock`
serializes admission, staging and publication on the I/O worker. Version-one
`.quota.json` persists the root's byte/record limits with fsync/atomic rename;
invalid or future metadata disables the cache rather than guessing a policy.
Changing those limits requires all other namespaces to be inactive. Concurrent
namespaces with identical limits may operate, but mutations contend without
waiting and may skip publication or restore through ordinary recomputation.
Writers predating this root protocol must not share the root concurrently.

Each namespace's lifetime lock protects its entire contents through outstanding
I/O and GPU ticket ownership, even after shutdown has been requested. Other
namespaces can reclaim only inactive records. The owning store may evict its
own unpinned records. Inactive incomplete files and empty owned namespaces are
removed without following symlinks or recursively deleting unknown files.
Discovery is capped at 32768 root entries and 32768 namespace entries in total;
overflow disables admission. A scan retains one oldest inactive candidate,
with no cross-identity prompt index. Repeated eviction can require repeated
scans; large-root startup performance is not qualified.

The quota counts serialized record/staging bytes, not filesystem block rounding
or filesystem metadata. Record and namespace counts separately bound owned
metadata: at most the configured record limit plus one namespace. Unknown
files are preserved and are outside cache-owned byte accounting. Root usage
metrics report the last I/O observation, not a synchronous actor-side scan.
Shutdown returns incomplete until the scheduler drains all ticket and GPU
owners; only the worker then closes files and releases the namespace lock.

`SPARKLAB_PREFIX_RESTORE_POLICY=measured` uses observed complete restore service
costs and measured prefill costs for otherwise computed tokens; unknown costs
choose recomputation. `calibrate` permits an idle first probe. `always` bypasses
only the cost decision for explicit correctness experiments. Resource, identity,
checksum and lifetime checks always apply. `SPARKLAB_PREFIX_RESTORE_TIMEOUT`
defaults to 30 seconds (maximum 300); timeout detaches the request and retains
private allocations until the I/O and GPU users finish.

The scheduler owns source leases, restore pages, state slots and waiter lists.
The I/O worker owns file calls and staged-copy waits. Polling publishes only a
fully validated image after its GPU fence completes. Duplicate or unwanted
restores return all private allocations. A corrupt prepublication image falls
back to ordinary recomputation; unexpected device/invariant failures remain
visible. Rebuild rejects undrained tickets before engine teardown.

Disable retention independently by unsetting `SPARKLAB_PREFIX_STORE` and
restarting after a successful shutdown drain. Namespace deletion is optional;
never remove live files or delete outside the owned fingerprint namespace.
There is no live configuration migration interface. Scheduling remains selectable
as `baseline`; disable exact reuse with `SPARKLAB_EXACT_PREFIX=0` on restart.
Scratch changes remain contingent on profiling. No release/default promotion is
supported by the current partial validation.

## Internal adaptive DFlash experiment

`SPARKLAB_DFLASH_POLICY=fixed` keeps existing behavior. `cost` prepares one
batch-one Qwen DFlash2 target graph plus a finite set of verification widths,
including a target-only transition that maintains draft context. It reserves
2 GiB from the unified physical budget and refuses a graph set whose retained
reservation exceeds that allowance. `SPARKLAB_DFLASH_DEPTHS` configures up to
four legal depths including zero and the configured maximum;
`SPARKLAB_DFLASH_FIXED_DEPTH` supplies the matched fixed control.

A request owns its bounded acceptance/cost history. Only accepted positions and
the first observed rejection train conditional estimates. Complete round costs
include proposal, target verification, accepted-prefix commit, scheduler writes
and output handoff. Proposal is deferred until its own selected decode round so
the next depth is not charged to the previous round. GPU timing uses the existing
drain fence after scheduler writes. The latency scheduler's remaining service
budget constrains depth selection. There is no new per-step device wait.

The scheduler retires each round at the end of its actual output drain, after
the existing copy fence and reply handoff. Continuing and EOS-completed rounds
train once; aborted rounds detach without training. This lifecycle runs with
telemetry both on and off. The next preparation must see no prior round. Native
runtime IPC observations remain rate-limited and last-observed; a snapshot
attached before handoff can precede that round's final feedback counters.

The real-model strict greedy-parity gate currently fails. State-transition and
immutable-KV tests do not replace it. This path is internal, off by default, and
has no model-support, quality or benefit qualification. Exact prefix checkpoints
continue to reject speculation. Roll back by unsetting the adaptive variables
and restarting through a drained shutdown; the original fixed-depth path and
recipe remain available. There is no supported live serving-mode migration API.

[The workspace investigation](FLARE_memory.md) did not justify changing allocation
layout for the selected control. Existing graph/scratch behavior remains.

## Native process shutdown

[Design issue 98](https://github.com/sixteen-miles-labs/sparklab/issues/98)
records the process-lifecycle compatibility decision. Worker stop handlers are
one-shot: SIGINT/SIGTERM/SIGHUP enter the existing scheduler shutdown path, and
repeated signals cannot interrupt its drain. The parent requests a Python
interrupt of owned POSIX workers so spawn-time imports can also unwind before
worker-specific handlers exist. Interrupted startup does not pretend to have a
fully constructed scheduler. Actual startup errors keep their original report.

Once a scheduler is fully constructed, an unexpected worker exception also
runs local failure cleanup before propagating the original error. Prefix tickets
receive the existing bounded shutdown drain, then owned IPC closes. This failure
path leaves final device teardown to process exit: it cannot require healthy TP
peers to enter a shutdown barrier or assume a usable CUDA context after an engine
error. A failed local drain is logged explicitly and does not replace the fatal
cause or imply successful GPU/I/O drain. TP1 signal-driven shutdown retains
its device fence, rank synchronization and engine teardown. Partial constructor
failure still has no fully constructed scheduler to drain.

A TP>1 signal stop also uses local cleanup: ranks can be interrupted at different
collective positions, so submitting a new shutdown barrier can wait behind
unmatched work. Each interrupted rank closes local owners and IPC, leaving final
device/communicator teardown to process exit. The worker log describes a local
rank stop rather than claiming a coordinated GPU drain. Real two-rank Gloo tests
hold one peer in an unmatched broadcast while the primary is between operations;
both parent SIGINT and direct SIGTERM stops complete without a new barrier.
They check owned ZMQ closure, exit codes, process reap and named-semaphore release
with CUDA uninitialized. This is CPU control-plane proof; NCCL/model/active-GPU
and earliest-spawn shutdown remain unqualified. Parent force/reap bounds remain.

A stop racing with startup-error reporting also drains a fully constructed
scheduler before re-raising the original startup error. The shared local failure
cleanup helper catches/logs cleanup errors without replacing that original
cause. When construction itself did not return an owner, cleanup remains a
no-op. Unit controls distinguish those cases; spawned CPU workers receive real
SIGINT/SIGTERM at the reporting boundary while a real prefix-file write is
paused. Cleanup alone opens the gate, and the parent independently reopens and
validates the complete record after the original nonzero exit. This proves the
specific reporting-boundary case with an injected sync failure, not arbitrary
constructor interleavings or GPU/model failure recovery.

The foreground server wraps a default-fatal SIGTERM so Uvicorn's replay unwinds
Python finalizers. Shell-visible status remains 143; `Popen.returncode` becomes
positive 143 rather than negative 15 on an orderly default SIGTERM. Explicit
embedding handlers are retained. Shell SIGINT still belongs to turn cancellation;
default shell stop/hangup unwinds its existing finally, while explicit ignored
stop signals remain ignored.

Foreground scheduler and tokenizer/detokenizer workers now leave the terminal's
process group, as shell workers already did. Uvicorn owns the frontend's terminal
signals and drains active HTTP work before lifespan/finally explicitly stops the
workers. Previously group Ctrl+C could stop the backend first, strand a stream
awaiting output, and trigger a false backend-crash report. This changes POSIX
worker process-group membership, not the existing HTTP drain timeout. Direct
worker signals still enter their one-shot cleanup. Two controlled first-entry
stops before the child imports Torch or installs worker handlers now pass on CPU:
parent stop during a partial launch, and parent stop after the full handle returns.
They require child reap, queue-semaphore release and no supervisor crash report.
A guard-removal fault control leaves both processes alive until explicit harness
recovery. Arbitrary spawn-bootstrap/import interruptions and GPU/model startup
remain unqualified.

Partial launches retain ownership until a complete backend handle is returned.
All frontend exits reap owned workers with one shared ten-second grace deadline,
followed by SIGKILL and a bounded join of survivors. A forced stop is logged and
does not prove successful transfer drain. Expected stops cannot reopen admission
or be reported by the supervisor as a backend crash. The frontend stop is
idempotent; async IPC closure runs on its owner loop. Owned transports use zero
linger at shutdown, while ordinary utility defaults stay unchanged. All owned
links receive a close attempt even if one fails.

For TP1 normal shutdown the native scheduler retains the existing order: prefix tickets/GPU users drain,
device work completes, engine teardown runs, then owned IPC closes. Real spawned
CPU workers, the actual tokenizer worker and real prefix files validate process
and semaphore ownership. A CUDA fixture validates this ordering with an actual
paused GPU/file transfer. An actual `/generate` stream over HTTP and ZMQ checks
finite output drain under parent-only SIGTERM and group SIGINT/SIGTERM. A client
disconnect checks abort delivery, retired frontend request maps and an independent
retry. These fixtures replace model execution and do not qualify complete-model,
multi-rank, all public protocols, restart or soak behavior. The ten-second worker
grace is not an overall HTTP shutdown bound; Uvicorn's active-request drain policy
has not changed.

Loss of a backend also wakes admitted generation waiters on the frontend loop.
The frontend appends a local terminal error only when a genuine terminal reply
is not already queued. Repeated notifications are idempotent. A failure racing
the first send takes the same path instead of queuing work toward a known dead
peer. Late real replies still reach token accounting, but cannot reopen delivery
to a failed HTTP request.

Admitted buffered generation returns HTTP 503 with `backend_unavailable`; committed OpenAI
streams emit `server_error` with that code, Responses emits `response.failed`,
and Anthropic emits `internal_error`. `/generate` also emits an explicit error.
Public messages omit worker exception details. A failed stream does not invent a
final usage snapshot. Existing client-input error classification is preserved.
The local error never passes through `StatsTracker.observe` as a scheduler
acknowledgement: requests missing real terminal acknowledgements remain active,
and the prepare-stop barrier refuses to seal their uncertain totals. These are
actual HTTP/IPC failure tests with synthetic model execution, not a qualification
of full-model protocol, GPU cancellation, restart or soak behavior.


## Adaptive phase interval diagnostics

Cost-mode rounds retain their existing complete host service cost for policy
training. With existing runtime telemetry enabled, two additional CUDA events
split the existing round into proposal, verification, engine commit and GPU
handoff intervals. There are five event owners per diagnostic round, three when
phase diagnostics are off, and at most one current round per eligible batch-one
request. Fixed-depth mode outside the cost experiment allocates none of these
adaptive events. No new public flag or measurement queue is introduced.

The markers bracket real engine/scheduler boundaries:

- `proposal_gpu_seconds`: round start through proposal-ready on the proposal stream.
- `verification_gpu_seconds`: proposal-ready through target sampling/acceptance
  decision, including ordered dispatch/preparation and rollback-state snapshots.
- `commit_gpu_seconds`: verification decision through engine state commit or
  rollback/replay and bookkeeping, before nonblocking output copy.
- `handoff_gpu_seconds`: engine commit through output copy and the scheduler's
  GPU token-pool write, ending at the existing round-end/output fence.

These are ordered GPU-timeline intervals, not exclusive kernel busy time; stream
waits and host submission/decision gaps can contribute. GPU handoff is distinct
from CPU reply delivery and client-visible output. Phase values are read only
at drain after the existing output fence. Missing, duplicate or out-of-order
markers are errors; disabled phases are not synthesized as zero measurements.
The fixed-name histograms `speculative_{proposal,verification,commit,handoff}_gpu`
and bounded diagnostic trace carry these values. The policy still uses whole
host round seconds, never a phase sum as a substitute for service cost.

The complete-model fixture enables these diagnostics and records all five span
fields after actual output drain. Its memory preflight now queries a short-lived
child because the offline engine requires CUDA uninitialized in the owning
process. This preserves the unchanged 30 GiB floor. Full-model phase ordering,
quality, contention and <=2% measurement overhead remain unqualified until that
fixture and paired model runs execute under adequate resources.


## Selection observations and first-rejection counters

Native dispatch observation keeps the policy inputs and actual dispatch outcome
separate. `selection_at` is the host-monotonic selection timestamp; the trace's
existing `time` is captured after batch preparation. `queued_before` counts
pending requests at selection, while `queued` retains its post-admission meaning.
The observed queue peak includes both values so successful admission cannot erase
that selection's backlog. This is a dispatch-observed peak, not an instantaneous
watermark for every frontend/tokenizer/I/O queue.

`active_output_age_seconds` is the maximum known age among runnable decoders,
using last worker-output-ready time or first dispatch if no output is ready yet.
`oldest_queue_age_seconds` is the maximum native pending-request age at selection.
An empty queue or wholly unknown output clocks yield null and no age histogram
observation, not a measured zero. A timestamp of zero is valid. The fixed
`active_output_age`/`oldest_queue_age` histograms sample these selection-level
maxima, not individual request TTFTs or client-visible delivery gaps.

`policy_reason` retains the initial choice; `reason=prefill_unavailable` records
actual decode/verification fallback when no prefill batch can be admitted. It
may reflect capacity, tables or an outstanding restore, and is not a specific
resource-failure classifier. `prefill_unavailable_dispatches` counts only such
actual fallbacks. `requests` remains padded request count; `active_requests`
records actual request count, `tokens` active query work, and `token_budget` the
chosen prefill budget (zero for a decode selection). These fields describe
logical dispatch shape, not a claim that a particular graph was replayed.
Detailed records remain in the existing bounded trace and are excluded from the
periodic IPC snapshot; the fixed histograms/counters follow existing export rules.

Completed non-aborted adaptive rounds retain existing proposal/acceptance totals.
`adaptive_rejection_position_N` records only the first observed rejection at
zero-based draft position N (0..15); deeper unverified drafts supply no negative
observations. `adaptive_rejected_rounds` and `adaptive_all_accepted_rounds` distinguish
rejected and fully accepted positive-depth rounds. Target-only rounds retain their
separate counter. Verification acceptance is not the number of user-delivered
tokens if a stop boundary truncates a burst. Invalid depth/acceptance or age inputs
are rejected before partial counter updates. All counter names are bounded by
the supported 16-position policy envelope. No learner, sampler, public flag,
trace queue, GPU synchronization or filesystem path was added.


## Unicode stop boundaries in speculative bursts

The common host drain evaluates stop strings after each emitted token, including
inside a verified multi-token result. Its decoded suffix window now uses the
longest stop's UTF-8 byte length plus one context token, clipped to generated
output. Character count is insufficient for byte-fallback tokenization: one
Unicode character can occupy three or four tokens. This retains bounded suffix
decoding instead of decoding the whole accumulated output. It preserves the
existing stop-list precedence and response semantics.

Controls use a real ByteLevel BPE tokenizer with one token per byte. They require
an exact match only when the UTF-8 sequence completes, bounded lookback after a
long generated prefix, truncation before a burst suffix, codec preservation,
stop-free detokenized output, and release of overadvanced recurrent/KV owners.
Separate controls load only the complete local Qwen artifact's tokenizer: two
characters split across three/four tokens miss on the prior wheel and match on
the corrected path. These are tokenizer/common-drain checks, not model inference,
complete API stop/tool/grammar qualification or a fix for existing M3/M5 quality
failures. Structured decoding eligibility remains unchanged.


## Atomic image admission under leftover token budget

PrefillAdder retains the initial token budget for each admission pass. An image
prompt that exceeds only the remaining budget after earlier admissions returns
no request from sizing, allowing the existing tentative-admission rollback to
release its table, cache pin and recurrent slots. The prompt remains pending and
can run whole in a later pass. An individually oversized image still raises the
existing single-chunk error, now before token copies or budget mutation.

This corrects mixed text/image and image/image token-budget pressure. It does
not enable multimodal prefix reuse or change cache conditioning identity. CPU
and small actual CUDA controls cover page sizes one/four, unchanged embeddings
and image positions, zero cached length, disjoint request tables, live-page
accounting and complete cleanup. They use synthetic embeddings with production
admission owners, not vision-tower inference. SWA-specific physical pressure,
full-model vision, HTTP error presentation for oversized images and concurrency
performance remain separate qualification requirements.


## Owned child registration during Process.start

Native launch wraps each Process.start in a main-thread POSIX guard. Callable
SIGINT/SIGTERM/SIGHUP handlers temporarily record one pending request per signal
while multiprocessing creates and registers the child. Original handlers are
restored, then pending signals are re-delivered through the OS. A stop can thus
unwind startup only after the Process handle is visible to the existing partial
ownership cleanup. Explicit default/ignored handlers remain untouched; non-main
thread or non-POSIX starts keep their existing path. If start itself fails, that
original error remains authoritative even if a replayed handler also raises.

The parent blocks signals only during handler restoration after spawn, so the
child inherits no new blocked-stop mask. Pending metadata contains at most three
signal numbers and retains no interrupted frames. Returning custom handlers keep
their authority; duplicate pending stop requests coalesce. The guard does not
promise a timeout for a nonreturning OS spawn or cover default-fatal signals that
have no Python handler.

Real SIGINT/SIGTERM controls interrupt Popen after an actual child reaches its
first-entry gate while Process._popen is still unset. A separate Linux subreaper
controller makes the negative control safe to recover and reap. The prior wheel
finalizes with a live unregistered child; the guarded path reaps it and releases
its queue semaphores. The unchanged frontend's measured early-stop exit statuses
(one for SIGINT, 143 for SIGTERM) remain the oracle. These are CPU registration and
signal-ownership checks; model/GPU construction and arbitrary OS spawn failures
remain unqualified.
