# Qwen3.6 checkpoint performance and finite pressure results

The user stopped the root-owned primary model service and authorized these
checks. All six owned native servers exited normally (143); the primary owner
remains inactive, retrieval services remain running. No defaults, weights,
sampling, precision or deployment changed. The plan is not complete.

All servers used the retained Qwen3.6-35B-A3B Triton NVFP4 artifact, the latest
`wheels-partial-render/` wheel (SHA-256
`5a70962d6d84554029da8982b634947cc812be5bbb16f349bce07504ea42e46e`),
8192 pages of 16 tokens, 65536 context, two running requests, all experts
resident, 512-token prefill chunks, baseline scheduling, no graphs/speculation,
runtime timing enabled and trace capacity zero. Each boot measured more than
80 GiB free, exceeding the fixed 30 GiB boot floor. All 436 production modules
still match the wheel; only the benchmark collector and its tests changed.

Raw evidence is under `$HOME/results/flare-runtime/`.
`qwen36-validation-v3.json` verifies artifact/source identity and hashes the
reports, collectors, relevant helpers and raw observations. Earlier evidence
keeps its original scope in `qwen36-validation-v2.json`.

## Strict quality failure stops performance qualification

`qwen36-performance-1/` retains the first complete baseline/candidate attempt,
including full-path warmups and ten workload reports per arm. All 14 declared
requests per arm completed their budgets without request errors. Thirteen
outputs match exactly, including both cold prompts, both repeated prompts,
8K branching, mixed traffic and the controls. **32K branching differs**: existing
cache reuse is 32704 tokens versus exact reuse of 32768 tokens; text first
differs at character 80. Each mode's warmup and measured branch output agree
with each other. This is reproducible within these two observations per mode;
it does not establish the cause or cross-boot repeatability. State copying and
cross-partition numerical behavior must be distinguished during diagnosis.

No strict equality was weakened. The series stopped before pairs 2–5, yielding
**zero qualified pairs**. `strict-output-reconciliation.json` retains every
comparison and the first text excerpts. Earlier byte-exact same-partition
state checks and factual-recall passes retain their narrower scope.

Single-pair diagnostic TTFT below is from one request per arm, without variance
or a release benefit claim. The 32K branch row fails quality.

| Workload | Baseline TTFT ms | Exact TTFT ms | Output parity |
| --- | ---: | ---: | --- |
| 8K cold | 4053.43 | 4070.62 | Equal |
| 8K repeat | 110.14 | 62.40 | Equal |
| 8K branch | 111.75 | 63.03 | Equal |
| 32K cold | 19097.07 | 19067.91 | Equal |
| 32K repeat | 172.34 | 96.93 | Equal |
| 32K branch | 177.03 | 100.41 | **Different** |

The purported pure controls also shared leading prefixes, causing different
intra-run cache hits. Their timings cannot isolate pure overhead and must not
be used to claim the 5% control gate. Before the next qualification attempt,
use distinct leading token sequences for those controls and validate their
cold-cache conditions. Do not remove or retune the failing 32K branch. The
latency-policy, tracing overhead, restore benefit and five-pair release gates
remain open; these target-only measurements establish none of them.

## Finite HTTP cancellation, overlapping requests and rebuilds

The original baseline and memory profiles collected successfully under
`qwen36-pressure-1/pressure-{baseline,exact}/cycles/`. Each performed one
warmup plus **six timed cycles**, **21 completed requests and seven deliberate
cancellations** including warmup, and six idle rebuilds. All completed text,
usage and finish reasons match the baseline reference. Requests drained after
client completion/disconnect; cache geometry and server instance stayed fixed.

The original disk attempt completed its four-request reference, then stopped
on HTTP 503 from rebuild. This exposed a collector defect: `get_json` raised on
the real API's non-2xx busy status before its bounded retry loop could inspect
`status: busy`. Prior TCP tests only returned HTTP 200 for busy. The failed
attempt is preserved in `qwen36-pressure-1/pressure-disk/`.

`benchmarks/validate_runtime_soak.py` now accepts only explicit rebuild-busy
responses with HTTP 409/503. Generation errors, other status codes, loading,
failed/rejected and malformed bodies still fail; the 2 MiB body bound remains.
The corresponding actual-HTTP regression/profile/client selection passes
**37 tests in 12.08 seconds** (`qwen36-pressure-busy-regressions-final.log`).
The previous collector source is saved with its hash independently checked
against the original baseline report. No production wheel change was needed.

The disk-only rerun uses a fresh store, the same wheel and original baseline
reference. `qwen36-pressure-busyfix-1/pressure-disk/cycles/` passes the same
six timed cycles, **21 completed requests, seven cancellations**, six rebuilds
and strict completed-output comparisons. Three HTTP 503 busy rebuild responses
were handled successfully. Sampled native counters reach **13 completed restores,
six coalesced restores and ten completed writes**. Sampled maxima are ten
records / 1837211162 bytes under the 2 GiB / 16-record limits, one unreleased
ticket under the two-ticket limit, and one 1 MiB staging window. Private restore
maxima are 512 pages and one slot. These are observed samples, not an exhaustive
memory/ownership proof; idle worker exports may be stale.

Across the three successful collectors: **63 completed requests and 21
cancellations**, including warmups and excluding separate reference requests.
All observed global swap-out deltas are zero; this is shared-host observation,
not attribution. These short collections do not qualify cancellation while a
restore is pending, allocator drift, expert/prefix NVMe offload contention,
60-minute profiles or the four-hour primary soak. Adaptive DFlash stays disabled.

Next: diagnose the 32K branch divergence without weakening its oracle, correct
the pure controls, then rerun full balanced trials. Keep experimental checkpoint
reuse opt-in and do not deploy it based on these results.
