# Spark Lab Beta 0.1 Release Plan

Status: proposed

Plan date: 2026-08-27

Target: public beta for one NVIDIA GB10
Related strategy: [Spark Lab rebrand and GB10 execution plan](spark-lab-rebrand-plan.md)

## 1. Release outcome

Beta 0.1 should prove one narrow promise:

> A GB10 owner can install Spark Lab, validate the machine, acquire a certified
> recipe, start a local server, and use its OpenAI- or Anthropic-compatible API
> without selecting low-level memory, kernel, or offload settings.

This is a product-validation release, not the completion of the 1.0 portfolio.
It should favor a small reproducible surface over additional experimental models.

### Success measures

- A new user reaches a streamed response in under 15 minutes, excluding checkpoint
  download and prepared-checkpoint conversion time.
- The normal install, first-run, CLI, server, documentation, and release experience
  presents only the Spark Lab product identity.
- Spark Lab reaches the native runtime only through a backend contract; product code no
  longer assumes its checkpoint format, CLI flags, process model, or health payloads.
- The release artifact installs on a clean supported ARM64/GB10 system without source
  edits or undocumented dependency workarounds.
- Every Beta FTW artifact preserves the source checkpoint's dtype and quantization; no
  catalog recipe performs conversion-time quantization.
- At least one precision-preserving recipe passes its correctness, context, performance,
  and 60-minute endurance gates on the exact release candidate. Historical evidence from
  a differently quantized artifact cannot satisfy this gate.
- Normal operation shows no swap growth, CUDA OOM, unbounded memory growth, service
  restart, or silent output corruption.
- Every public performance claim links to versioned, machine-readable GB10 evidence.
- The first seven days produce no open severity-0 or severity-1 defects.

## 2. Product identity and attribution boundary

Beta 0.1 is a **Spark Lab** release. FreeToken is the engine Spark Lab is built on,
not the product name presented to users.

### Public product surfaces

These surfaces must use only the Spark Lab name and `sparklab` terminology:

- Website, repository landing page, screenshots, release title, release notes, and
  downloadable artifact names
- Installation command and installed application/package display name
- CLI executable, help, prompts, errors, logs, diagnostics, daemon/service labels,
  metrics namespace, cache/state paths, and environment variables
- Model recipes, benchmark reports, support matrix, issue templates, and user guides
- API server identity where a product or implementation name is exposed

The supported path must use `sparklab`, `SPARKLAB_*`, and Spark Lab-owned state paths.
It must not require users to type `freetoken`, `ft`, or `FREETOKEN_*`, or install an
artifact visibly named `freetoken`.

### Engine and compatibility surfaces

FreeToken may remain visible only where technical or legal accuracy requires it:

- Apache license, copyright, source notices, software bill of materials, and security
  provenance
- Research-paper attribution and an acknowledgements/about page
- Internal source modules and developer-only architecture documentation
- A clearly secondary legacy migration page for existing `ft`, `freetoken.*`, and
  `FREETOKEN_*` users

Use factual wording such as **“Spark Lab is built on the FreeToken engine”** in those
locations. Do not lead the README, installation guide, quick start, release announcement,
or routine CLI output with the engine name. Attribution must remain complete; it is being
moved out of the product journey, not removed.

## 3. Version and release identity

Use **Spark Lab Beta 0.1** as the user-facing release name. Before code freeze, the
release owner must map that name to one immutable Spark Lab artifact version and Git tag.

The engine currently reports version `0.1.2`; that internal version must not determine
or appear as the public product version. Give the Spark Lab application/distribution its
own version source and record the public mapping as:

```text
Spark Lab Beta 0.1 -> Spark Lab <artifact-version> -> git tag <release-tag>
```

Use a valid prerelease version such as `0.1.0b1` if Beta 0.1 is distributed through a
PEP 440 package channel. The exact package/index name requires an availability and legal
check before it is placed in public instructions.

The tagged commit, Spark Lab artifact, internal engine and kernel-cache artifacts, recipe
version, checkpoint revision, and certification evidence must identify one another
exactly. Engine details belong in provenance/SBOM metadata rather than the primary release
title or user instructions.

## 4. Backend-neutral product architecture

Spark Lab is the product and orchestration layer. The current FreeToken engine becomes
the first built-in runtime backend, identified as `native` on public surfaces and by its
implementation identity only in developer diagnostics, certification provenance, notices,
and the SBOM. Future runtimes must be addable without changing the product CLI, catalog,
planner, daemon, or public API address.

### 4.1 Ownership boundary

```text
Spark Lab
├── CLI, configuration, version, paths, logs, and product identity
├── model catalog, tiers, recipes, and certification
├── GB10 platform detection and unified-memory admission
├── source acquisition and artifact manifests
├── stable API gateway and health/status schema
├── backend-neutral process supervisor
└── runtime backend contract
    ├── native       current FreeToken engine
    ├── gguf         potential future GGUF runtime
    ├── external     potential compatible local server
    └── future adapters
```

Spark Lab owns platform policy, artifact lifecycle, backend selection, user-facing APIs,
and certification. A backend owns its supported architectures and quantizations, prepared
artifact formats, preparation and validation, backend-specific memory requirements,
invocation, process health, metrics translation, and shutdown behavior.

### 4.2 Backend contract

The Beta contract should use structured data rather than raw command-line strings:

```python
class RuntimeBackend(Protocol):
    backend_id: str

    def probe(self, platform: PlatformSnapshot) -> BackendAvailability: ...
    def capabilities(self) -> BackendCapabilities: ...
    def prepare(self, source: Artifact, deployment: DeploymentRecipe) -> PreparedArtifact: ...
    def validate_artifact(self, artifact: Artifact) -> ValidationResult: ...
    def plan(self, request: RuntimeRequest) -> BackendPlan: ...
    def launch(self, plan: BackendPlan) -> BackendHandle: ...
    def health(self, handle: BackendHandle) -> BackendHealth: ...
    def metrics(self, handle: BackendHandle) -> BackendMetrics: ...
    def stop(self, handle: BackendHandle) -> None: ...
```

`BackendPlan` must contain a resolved executable, argument vector, environment, artifact
references, expected endpoint, capabilities, and declared resource requirements. Planning
must be side-effect free. Launching, monitoring, and stopping remain explicit lifecycle
operations.

Start with a built-in registry and explicit adapter allowlist. Do not load arbitrary
third-party entry points during Beta 0.1; external plugin discovery can follow after the
contract and security policy stabilize.

### 4.3 Deployment recipes and certification

Replace engine-specific `runtime_args` with a backend-qualified deployment in recipe
schema v2:

```yaml
model: Inferact/Qwen3.8-Flash-Next-NVFP4
deployment:
  backend: native
  backend_api: "1.0"
  source_format: safetensors-nvfp4
  runtime_format: ftw-nvfp4
  quantization: nvfp4
  execution_policy: nvme-moe
  backend_options:
    attention_backend: qsa
    moe_host_cache_gb: 3
    nvfp4_backend: triton
    max_running_req: 1
```

Backend options remain namespaced instead of forcing every runtime into a lowest-common-
denominator configuration. Provide a deterministic v1-to-v2 recipe migration and reject
unknown backend options fail-closed.

Certification belongs to the complete tuple:

```text
model + checkpoint revision + quantization + artifact format
+ backend ID/version + recipe version + platform + release artifact
```

Two backends serving the same model have separate status and evidence. Certification never
transfers automatically between backends, quantizations, or prepared artifact formats.

### 4.4 Artifact boundary

Spark Lab acquires immutable source artifacts and records manifests, sizes, fingerprints,
and provenance without assuming a runtime layout. The selected backend prepares and
validates its execution artifact:

- Safetensors can remain a shared source format.
- FTW v1 belongs to the `native` backend; FTW validation must leave product acquisition
  code and move behind that adapter.
- GGUF belongs to a backend that declares GGUF support.
- Quantization is recorded independently from the container format.
- Beta 0.1 FTW preparation preserves the source tensor dtypes and quantization. Alignment,
  sharding, fusion, and backend repacking are allowed only when they do not introduce a
  lower-precision representation.
- Precision-changing transforms, including BF16-to-NVFP4 expert conversion, are outside
  the supported Beta recipe path. If retained for research, they produce a separately
  named artifact, recipe version, status, and evidence record.
- Product manifests use generic roles such as `source` and `runtime`, plus backend-owned
  format/version metadata.

Existing FTW artifacts remain readable through the native adapter. New-user documentation
calls them Spark Lab prepared checkpoints; compatibility code may retain the legacy reader
and on-disk names until a versioned, migration-safe replacement exists.

### 4.5 NVMe and disk-backed MoE boundary

NVMe-backed execution is a first-class Spark Lab capability, not an implementation detail
to erase behind a generic backend interface. Spark Lab owns the storage policy and user
promise; the native backend owns its FTW layout and expert-streaming implementation.

The native backend already provides:

- 4096-byte-aligned, sharded FTW tensors with independent expert-row descriptors;
- `O_DIRECT` reads with an explicit `mmap` fallback when direct I/O is unavailable;
- bounded one- or two-layer pinned staging plus a byte-budgeted pageable host expert cache;
- global LRU and scan-resistant per-layer LRU admission policies;
- a persistent bounded reader pool, duplicate-route elimination, and coalesced reads for
  adjacent aligned expert rows;
- synchronous decode staging, hybrid CPU/GPU staging, and double-buffered asynchronous
  prefill lookahead;
- prefill-aware cache admission, cache-hit device copies, and route-first sparse prefill;
- counters for cache capacity/occupancy/hits/misses/evictions/bypasses, logical and physical
  bytes, operations, read time, reader count, staging buffers, and host allocations.

The implementation must remain backend-owned. Product code expresses a generic storage
request and consumes a normalized plan and metrics contract:

```yaml
storage:
  policy: nvme-moe
  runtime_artifact: ftw-v1
  memory_budget:
    host_cache_bytes: 2147483648
    staging_bytes: measured
  requirements:
    local_nvme: true
    direct_io: preferred
    swap_growth_bytes: 0
```

`runtime_artifact` and detailed cache/read options are namespaced to the backend. Public
recipes select the validated settings automatically; the normal user path does not expose
reader counts, cache policy, staging buffers, or kernel flags.

Spark Lab owns these backend-neutral storage responsibilities:

- discover local storage, free space, filesystem, block device, and direct-I/O suitability;
- plan one unified-memory budget covering resident weights, GPU expert slots, pinned
  staging, pageable host cache, KV/recurrent state, workspaces, OS reserve, and conversion
  transients;
- reject plans that depend on swap or unbounded operating-system page cache;
- record artifact sizes and fingerprints and present actionable capacity failures;
- normalize storage metrics across backends and attach them to certification evidence.

The native adapter owns FTW validation, expert addressing, direct reads, prefetch,
coalescing, cache policy, backend-specific allocations, and detailed diagnostics. Other
backends may implement `nvme-moe` differently; they earn support only by satisfying the
same product gates.

Before Beta 0.1, extend the normalized storage report with the current counters plus:

- selected read path (`O_DIRECT` or `mmap`) and fallback reason;
- read service time versus queue-wait/stall time;
- configured and observed queue depth;
- coalesced request/extent count and read amplification;
- prefetch requests, useful hits, late arrivals, cancellations, and unused bytes;
- logical and physical bytes per prompt token and generated token;
- pinned staging, pageable cache, GPU cache, and total unified-memory peaks;
- page-cache change, swap-in/out delta, NVMe temperature, and I/O errors during endurance.

Do not describe decode as fully asynchronous until cache entries have explicit loading and
ready states and concurrent requests can progress independently. The current persistent
parallel reader pool and prefill lookahead are real optimizations, while decode staging is
still synchronized at the layer boundary.

The current proof points are:

- Qwen3.6-35B-A3B NVFP4 ran its complete pinned checkpoint at 67.46 decode tok/s
  and 0.320 s warm TTFT on the selected fixed 64-token probe. The request caused
  no swap traffic, but this is performance evidence only: the 32K context and
  60-minute stability gates were not run, so the Fast recipe remains Experimental;
- Qwen3.8-Flash-Next 0.3.0 historically demonstrated converted NVFP4/NVMe execution at
  12.51 decode tok/s, 0.870 s warm TTFT, exact 65,536-token recall, and 60.50 minutes
  uninterrupted with no model-attributed swap growth. This evidence is retained for
  comparison but does not certify the publisher-quantized Inferact NVFP4 Beta recipe;
- Qwen3.8-Flash-Next 0.4.0's complete official FP8 artifact measured 4.99 decode tok/s
  and 0.580 s warm TTFT. The result proves preparation and end-to-end serving, but misses
  the 5 tok/s Frontier floor by 0.01 tok/s, does not include the remaining gates, and is
  historical evidence for a superseded checkpoint tuple;
- DeepSeek V4: Preview DS-FP4/NVMe execution at 9.217 decode tok/s and 14.045 s warm TTFT,
  with identical greedy output across matched controls and 142.63 GiB physical expert I/O
  on the fixed probe;
- GLM-5.2: Experimental Research NVFP4/NVMe execution at 0.802 decode tok/s and
  2.57 s TTFT on its selected 256-token trial. The trial remained bounded but swapped out
  680 KiB, so `GB10-GLM52-RESEARCH-001` records measured performance without granting
  Research admission.

### 4.6 API and supervision boundary

Spark Lab owns one stable loopback endpoint and normalizes lifecycle health, readiness,
errors, and metrics. Initially, the gateway may proxy protocol endpoints already implemented
by the native backend. Each deployment declares which OpenAI, Anthropic, reasoning, tool,
and streaming capabilities it provides, and certification tests them end to end.

Do not claim that a new backend supports an API merely because its route names resemble the
Spark Lab contract. Protocol translation can move into the gateway incrementally after the
native adapter reaches parity.

The Spark Lab supervisor launches the command supplied by an adapter, persists the backend
and deployment identity, polls normalized health, captures logs, and performs graceful stop
with a timed process fallback. It must not assume FreeToken arguments, FTW, or FreeToken-
specific health fields.

### 4.7 Beta architecture gate

Beta 0.1 requires the backend contract, built-in registry, recipe schema v2, native adapter,
backend-neutral acquisition/planning/supervision path, and a deterministic fake backend for
contract tests. The native adapter must preserve source precision during Beta preparation;
the resulting artifact must earn its own behavior and performance evidence.

A second production backend is deliberately **not** a Beta 0.1 blocker. After the beta, add
one real alternative backend as proof of portability before declaring the adapter API stable.

## 5. Scope

### Required in Beta 0.1

- NVIDIA GB10, ARM64 Linux/DGX OS, CUDA 13, SM121, and local NVMe only.
- A Spark Lab-branded installer, package, or bundle whose public filename, metadata,
  install command, and entry points do not expose the internal engine package name.
- A versioned runtime-backend contract, built-in registry, native adapter, and fake adapter
  used by backend contract tests.
- Recipe schema v2 with backend-qualified deployments and a deterministic migration from
  the existing recipes.
- Backend-neutral artifact manifests, preparation dispatch, runtime planning, process
  supervision, and normalized health/metrics.
- Native `nvme-moe` capability preserved behind the adapter, with bounded unified-memory
  planning, normalized storage telemetry, recipe-owned tuning, and failure-safe teardown.
- Qwen4-Exp loading for Qwen's official 128x128 block-FP8 tensor layout, including
  precision-preserving dense and expert FTW round trips with dtype, shape, scale, and
  payload checks.
- `sparklab doctor`, `models`, `plan`, `pull`, `run`, `status`, `shell`, and
  `launch` for the documented happy path.
- Inferact/Qwen3.8-Flash-Next-NVFP4 `0.5.0` as the primary precision-preserving
  publisher-quantized NVFP4 FTW candidate.
- At least one source-precision-preserving recipe must be Certified before release; Beta
  does not ship by reusing certification from a converted artifact.
- Fail-closed platform, storage, memory, checkpoint-revision, and recipe admission.
- Text generation at batch/concurrency one.
- OpenAI Chat Completions and Responses APIs, plus Anthropic Messages.
- Reasoning output, tool calls, streaming, prefix reuse, graceful stop, and restart.
- Quiet compatibility for the `ft` CLI, `freetoken.*` imports, and `FREETOKEN_*`
  fallbacks, documented only in a secondary legacy migration guide.
- Spark Lab-branded state/cache paths, daemon/service identity, logs, metrics, error
  messages, diagnostics, and API implementation metadata throughout the supported path.
- A reproducible release artifact, checksums, release notes, known limitations,
  upgrade instructions, and rollback instructions.

### Allowed but not release-blocking

- DeepSeek V4 remains the Preview Frontier recipe; GLM-5.3 Flash remains its primary
  Experimental Frontier peer.
- Qwen3.6 NVFP4 remains the primary Experimental Fast recipe. Its measured speed and
  TTFT pass the Fast performance thresholds, while context and endurance gates remain.
- NVIDIA Kimi-K3-NVFP4 `0.2.0` remains the Experimental Research recipe. Its
  precision-preserving FTW target retains the checkpoint's NVFP4 experts, block-FP8
  attention projections, and the source precision of remaining tensors.
- GLM-5.2 moves to the Research tier as an Experimental fallback outside the primary
  lineup. Its measured result remains visible, but its nonzero swap growth prevents
  Research admission.
- Experimental recipes may remain visible when their status and limitations are
  unmistakable and they never appear as the default recommendation.

### Explicitly deferred

- A Certified Fast recipe and certification of the remaining two Frontier-layer recipes.
- A second production runtime backend, third-party backend plugins, or a public backend SDK.
- Full protocol translation for backends that do not natively satisfy the certified API
  contract.
- Kimi K3 NVFP4 full-checkpoint enablement, including the ModelOpt mixed-format tensor
  mapping required by NVIDIA's checkpoint.
- Multimodal input, multi-GB10 execution, high concurrency, and non-GB10 support.
- Renaming the internal `freetoken.*` Python import package.
- A large marketing website or elaborate visual-identity system beyond the minimum
  coherent Spark Lab release assets.
- Any performance claim not reproduced by the release candidate.

Scope additions after freeze require removing work of comparable risk or moving the
target. A newly released model is not a reason to delay Beta 0.1.

## 6. Starting point

Snapshot from `main` at `ddb9c34` on 2026-08-27:

| Area | Current state | Beta gap |
|---|---|---|
| Product CLI | Core doctor/catalog/acquire/plan/run workflow exists | Exercise every documented command from the packaged artifact |
| Backend boundary | Spark Lab imports the engine directly for platform inspection, FTW validation/conversion, server launch, daemon, shell, and benchmarks | Route runtime behavior through structured backend and product-service contracts; confine direct engine imports to the native adapter and legacy compatibility layer |
| Recipe schema | Recipes contain engine checkpoint names and raw runtime argument vectors | Introduce backend-qualified schema v2 and preserve v1 migration coverage |
| NVMe execution | Aligned FTW expert rows, bounded caches/staging, persistent parallel reads, coalescing, prefill lookahead, sparse prefill, hybrid staging, and basic counters are implemented | Move the feature behind the native adapter; normalize missing queue/prefetch/stall/page-cache telemetry; prove cancellation, read-error, and memory-budget behavior |
| Certified model | The current Inferact Qwen3.8 NVFP4 checkpoint has loader coverage but no accepted preparation or GB10 measurement; prior converted-NVFP4 and official-FP8 results are historical | Prepare and validate the immutable 0.5.0 artifact, run every gate, and publish the exact artifact linkage |
| Other models | DeepSeek V4 Preview; four recipes Experimental | Keep clearly non-default; do not let them expand the critical path |
| Packaging | Tagged x86_64 CPython 3.10-3.13 workflow and rolling beta workflow publish engine-named wheels | Add a Spark Lab-branded ARM64/GB10 artifact and keep engine packages behind the product boundary |
| Automation | Nightly and tagged-release workflows exist | Add ordinary pull-request CI and a release rehearsal that cannot publish production artifacts |
| Tests | Non-slow local suite: 1,454 passed, 8 skipped, 11 deselected | Run slow, real-checkpoint, API, packaged-install, and GB10 hardware gates |
| Documentation | README, install, quick start, models, CLI, and migration docs exist | Verify them verbatim on a clean machine and add beta release/rollback notes |
| Brand/state migration | Product CLI exists, but README/install metadata still describe the FreeToken distribution and some legacy daemon/state surfaces remain | Remove the engine identity from the normal product journey while preserving legal attribution and quiet compatibility |

### 6.1 Refactor checkpoint — 2026-08-27

The first backend-boundary implementation slice is complete in the working tree:

- Spark Lab now owns its GB10 platform policy and public `0.1.0b1` version source;
- a versioned backend contract, explicit built-in registry, native adapter, native artifact
  validator, and deterministic fake-backend contract test are present;
- all packaged recipes use schema v2 with structured native options, while schema-v1
  recipes migrate deterministically and the legacy `ftw` format alias remains readable;
- the catalog records `primary` versus `fallback` portfolio roles: Qwen3.6 NVFP4 is
  Fast; Qwen3.8, GLM-5.3 Flash, and DeepSeek V4 Flash are Frontier; Kimi K3 NVFP4 is
  the primary Research target; and GLM-5.2 is retained as an Experimental Research
  fallback;
- acquisition emits generic `source` and `runtime` artifact roles, dispatches preparation
  and validation through the selected backend, and discovers schema-v1 manifests through
  an in-memory compatibility projection;
- runtime resolution, launch planning, product `run`, and advanced compatibility commands
  reach the engine only through the native adapter;
- a static architecture test rejects direct engine imports in Spark Lab modules outside
  `sparklab.backends.native*`;
- the non-slow regression suite passes: 1,465 passed, 8 skipped, and 11 deselected.

The superseded local Qwen3.8 NVFP4 prepared checkpoint also passes the adapter validator
with fingerprint
`dff6c5fd14658727`, 10 FTW shards, 1,175 tensors, 78,032,617,472 physical FTW bytes,
and one 102,400,491,520-byte external n-gram artifact. A synthetic zero-swap snapshot
reproduces the exact native QSA/NVFP4/disk launch plan with 4,929,474,560 bytes of admitted
headroom. These facts are historical implementation evidence, not Beta FP8 certification.
A later engineering run prepared, byte-validated, and served the complete official FP8
artifact despite the fail-closed product preflight remaining not-ready due unrelated swap.
`GB10-QWEN38-FP8-001` records 4.99 tok/s and 0.580 s warm TTFT without granting admission.
Recipe 0.5.0 now targets Inferact's native ModelOpt NVFP4 checkpoint, so neither historical
result transfers to the current tuple.

This checkpoint does **not** complete the Beta architecture or release gate. Remaining
work includes moving daemon supervision and stable gateway lifecycle handling into Spark
Lab, binding final certification evidence to the full backend/artifact/release tuple,
adding the missing normalized disk telemetry and failure tests, producing a Spark
Lab-branded ARM64 artifact, measuring the current NVFP4 checkpoint above the Frontier
floor, and running correctness, context, quality, and endurance after swap returns to zero.

## 7. Workstreams and exit gates

The `Owner` entries are roles to assign during kickoff; each needs one directly
responsible person.

| Workstream | Owner | Required work | Exit gate |
|---|---|---|---|
| Scope and release control | Release lead | Approve this scope; choose target date and Spark Lab artifact/tag mapping; create milestone and triage labels | Scope, owners, date, and go/no-go meeting are recorded |
| Brand boundary | Product/release | Inventory every user-visible name; define Spark Lab package/bundle, service, state, logs, metrics, errors, metadata, and release assets; move engine attribution to Notices/About/developer migration docs | A clean supported journey contains no engine-branded product surface, while license, SBOM, research attribution, and source notices remain complete |
| Backend architecture | Product/runtime | Define structured backend types and registry; implement native and fake adapters; move platform, preparation, planning, launch, health, metrics, and stop calls behind their proper product/backend boundaries | No Spark Lab product module imports runtime-engine modules except the native adapter/legacy facade; fake and native adapters pass the same lifecycle contract suite |
| Recipe and artifact schema | Product/model | Add schema v2 deployments, generic artifact roles, backend-owned format metadata, v1 migration, strict validation, and backend-qualified evidence identity | All catalog recipes load through v2; invalid backend/options fail closed; existing manifests and prepared checkpoints remain discoverable |
| Gateway and supervisor | API/runtime | Establish stable Spark Lab endpoint and normalized lifecycle schema; make supervisor consume adapter launch/health/stop contracts; proxy native APIs without changing semantics | CLI, daemon, APIs, restart, and accounting pass through the adapter path with native parity |
| NVMe execution | Storage/runtime | Expose native disk preparation/planning/launch behind the adapter; add normalized storage metrics; test direct-I/O fallback, bounded cache/staging, read errors, cancellation, page-cache pressure, and teardown | Qwen certification is reproduced; memory stays within plan; swap growth is zero; physical bytes are explainable; no reader, buffer, or corrupt cache survives failure |
| ARM64 packaging | Release engineering | Audit binary dependencies; build the Spark Lab product artifact plus internal runtime/kernel-cache artifacts for ARM64/CUDA 13/SM121; add checksums and provenance | Clean GB10 installs Spark Lab using a Spark Lab-branded command/artifact and reaches `sparklab doctor` with no local source tree |
| CI and quality | QA/runtime | Add hosted CPU-safe PR checks; run full suite on the trusted GPU runner; make wheel build/import/CLI smoke tests mandatory | Required checks are green on the release commit; no test result depends on an unrecorded local patch |
| GB10 product path | Product/runtime | Exercise doctor -> models -> plan -> pull -> run -> status -> API; improve errors and cleanup behavior found during rehearsal | A clean-machine scripted rehearsal passes twice, including one interrupted/resumed pull |
| Certified model | Model/runtime | Complete the Inferact Qwen3.8 NVFP4 FTW round trip; then run output, capability, exact 64K recall, latency, memory, disk, quality, and 60-minute endurance probes | New evidence passes the selected tier contract and names the release commit, recipe, source precision, checkpoint, OS, and artifacts |
| API and agent compatibility | API/runtime | Test OpenAI Chat Completions, Responses, Anthropic Messages, streaming, reasoning, tool calls, cancellation, restart, and one supported coding-agent task | Protocol matrix and fixed agent task pass with the packaged server |
| Safety and recovery | Runtime/QA | Test insufficient RAM/disk, active swap, wrong GPU/CUDA, corrupt or partial artifact, port conflict, process kill, restart, and disk-full behavior | Failures are early and actionable; no corrupt cache is treated as valid; restart/cleanup is documented |
| Documentation | Docs/product | Make Spark Lab the sole primary identity; follow install and quick start literally; add release notes, limitations, support matrix, artifact verification, diagnostics, upgrade, rollback, About/Notices, and a secondary legacy migration page | A reviewer unfamiliar with the implementation completes the clean-machine path using only published docs and encounters the engine name only in attribution or legacy material |
| Legal and public claims | Product/legal | Check Spark Lab naming/trademark risk, checkpoint licenses, attributions, and every benchmark statement | Written approval or a documented beta-safe naming fallback exists; every claim has evidence |
| Launch operations | Release/support | Prepare announcement, issue template, diagnostic collection, incident owner, rollback decision, and seven-day support rotation | Launch checklist has named approvers and an executable rollback path |

## 8. Delivery sequence

Use relative dates until the release lead chooses `T0`.

### T-15 to T-11 working days: freeze the contract

- Approve scope, non-goals, success measures, public brand boundary, Spark Lab artifact
  name/version, release tag, and date.
- Assign owners and create one tracking issue per workstream.
- Add PR CI and make its required status checks visible.
- Decide the Spark Lab-branded ARM64 artifact format and document its dependency support
  matrix and internal-engine provenance mechanism.
- Inventory and classify every current `FreeToken`, `freetoken`, `ft`, and
  `FREETOKEN_*` occurrence as internal attribution, legacy compatibility, or a public
  surface that must be changed.
- Approve backend protocol types, ownership boundary, built-in registry policy, recipe
  schema v2, artifact manifest v2, and certification identity tuple.
- Capture golden native-backend launch plans, health/status payloads, API responses,
  metrics, shutdown behavior, and Qwen evidence before refactoring.
- Freeze Qwen3.8 0.5.0's Inferact-NVFP4-preserving preparation policy, checkpoint revision,
  prompts, and evidence schema.
- Triage all open defects; mark release blockers explicitly.

Gate: no unresolved product/version decision can change packaging or certification work.

### T-10 to T-6: close product and packaging gaps

- Produce Spark Lab-branded ARM64 release-candidate artifacts from a clean build
  environment.
- Implement the native adapter and route acquisition preparation, artifact validation,
  planning, launch, health, metrics, and stop through it.
- Preserve FTW row addressing, bounded cache/staging, parallel/coalesced reads, prefill
  lookahead, sparse prefill, and hybrid staging inside the native adapter; do not duplicate
  the storage engine in Spark Lab product code.
- Keep Qwen3.8's per-expert ModelOpt NVFP4 loader and FTW writer path covered by exact
  dtype, shape, scale, and payload round-trip tests; optimize the measured serving path
  without dequantizing or requantizing the checkpoint.
- Add normalized disk path, queue, coalescing, prefetch, cache, bytes/token, allocation,
  page-cache, swap, temperature, and error telemetry.
- Move GB10 platform types, Spark Lab versioning, gateway identity, and supervisor policy
  into the product layer.
- Migrate catalog recipes and manifests to schema v2 while retaining deterministic,
  read-only discovery of v1 state.
- Add a fake subprocess backend and run the complete backend contract suite against it.
- Complete fresh-install and CLI happy-path rehearsals on a clean GB10 image.
- Verify public artifact filenames, package metadata, installation output, CLI/help,
  daemon/service identity, paths, logs, metrics, errors, and API metadata against the
  brand boundary.
- Fix only Beta 0.1 blockers and high-confidence hardening issues.
- Run API/agent compatibility, failure injection, upgrade, and rollback tests.
- Draft release notes, support matrix, limitations, and artifact verification steps.

Gate: the packaged product works end to end before model certification begins.

### T-5 to T-3: certify RC1

- Cut RC1 from a clean commit; prohibit feature merges into the release branch.
- Run the full repository suite, packaged-wheel smoke suite, exact 64K recall,
  capability/agent probes, performance suite, and 60-minute endurance test.
- Run native-adapter parity tests and confirm there are no direct runtime-engine imports
  outside the adapter and approved compatibility modules.
- Store raw logs externally and commit the compact evidence result.
- Measure the Inferact NVFP4 artifact on RC1 and clear the 5 tok/s Frontier threshold. Treat
  both existing Qwen3.8 result IDs only as historical systems comparisons, not as regression
  or quality baselines for the Beta artifact.
- Conduct an independent documentation rehearsal from a fresh OS/user state.

Gate: all mandatory checks pass on the same immutable RC1 artifacts.

### T-2 to T-1: release decision

- Close or explicitly defer every issue in the milestone.
- Verify artifact hashes, tag/version consistency, licenses, links, and release notes.
- Hold go/no-go review with release, runtime/model, QA, docs/product, and support owners.
- If fixes are required, cut RC2 and rerun every gate affected by the changes; never
  relabel modified RC1 bytes as the same candidate.

Gate: all approvers sign the go/no-go record and rollback remains tested.

### T0 through T+7: publish and stabilize

- Create the signed/annotated version tag and let the tagged workflow build artifacts.
- Compare produced hashes and metadata with the approved candidate before publishing.
- Publish release notes, known limitations, support matrix, and evidence links.
- Run an install-and-generate smoke test from the public download location.
- Monitor issues daily for seven days; publish workarounds and patch releases rather
  than silently replacing immutable artifacts.

Gate: no open severity-0/1 defect at T+7 and all launch incidents have owners.

## 9. Mandatory validation matrix

| Gate | Environment | Pass condition | Evidence |
|---|---|---|---|
| Unit/regression | Supported development environment | Full suite passes; skips are reviewed and explained | Test log and commit |
| Backend contract | CPU-safe fake backend plus native backend | Probe, prepare, validate, plan, launch, ready, metrics, failure, stop, and restart semantics pass the shared contract; unknown capabilities/options fail closed | Contract report |
| Architectural boundary | Static dependency check | Product modules have no direct runtime-engine imports outside the native adapter and approved legacy facade | Import-boundary report |
| Native parity | Release GB10 | Adapter path matches pre-refactor invocation, API semantics, output, memory safety, and accepted performance tolerance | Golden diff and GB10 evidence |
| Recipe migration | All packaged recipes plus v1 fixtures | v2 round-trips; v1 migrates deterministically; backend/artifact/evidence identity is preserved | Schema test report |
| Disk/RAM correctness | Native backend with a tractable reference checkpoint | Resident and NVMe modes produce identical deterministic token IDs across short, code, reasoning, prefill, and sustained-decode cases | Output tokens/hashes and configuration |
| Bounded storage memory | Release GB10, minimum and selected cache budgets | Pinned staging, pageable host cache, GPU cache, and total unified-memory use stay within the declared budget plus measured fixed overhead; no swap I/O occurs | Allocation timeline and swap counters |
| Direct-I/O behavior | Supported NVMe plus fallback fixture | Certified path records `O_DIRECT`; unsupported direct I/O either fails when required or reports a deliberate `mmap` fallback without presenting page-cache speed as disk speed | Storage report |
| Disk scheduler | Cold and warm routing traces | Logical/physical bytes, operations, queue depth, coalescing, amplification, prefetch usefulness, and stalls reconcile; bounds hold at every configured reader/cache setting | Normalized storage metrics |
| Disk failure and teardown | Injected short read, corrupt extent, disk-full/unmount where safe, request cancellation, and process stop | Request fails clearly; partial entries are never admitted; worker pools terminate; pinned buffers release; restart revalidates artifacts | Failure-injection report |
| Artifact integrity | Clean ARM64 build/install environments | Build succeeds reproducibly; Spark Lab metadata and entry points are correct; product, internal runtime, and kernel-cache provenance match | Artifact hashes, SBOM, and build log |
| Brand boundary | Clean install plus static inventory | Normal install, first run, help, logs, services, paths, API metadata, and docs present Spark Lab only; engine identity appears only in approved notices/legacy/developer surfaces | Surface inventory and install transcript |
| Fresh install | Clean supported GB10 image | Spark Lab-branded documented install works without repository checkout, engine-named user commands, or manual edits | Timed install transcript |
| Platform admission | GB10 plus negative fixtures/machines | Supported GB10 passes; unsupported architecture/CUDA/memory/storage fails with remediation | `doctor --json` reports |
| Artifact acquisition | Clean model cache and local NVMe | Immutable revision downloads, resumes after interruption, verifies, prepares FTW, and detects corruption/low disk | Pull/conversion log |
| Recipe launch | Certified Qwen recipe | Plan and run choose recipe-owned settings; health becomes ready; stop/restart is clean | Invocation, health, and status logs |
| Correctness | Release GB10 and pinned checkpoint | Deterministic output/capability probes match the accepted reference | GB10 evidence JSON |
| Context | Release GB10 | Exact 65,536-token recall gate passes | Context result JSON |
| API compatibility | Packaged server | Required OpenAI and Anthropic request/stream/error cases pass | Protocol report |
| Agent task | One documented supported client | Fixed repository task completes and its tests pass | Prompt, transcript, patch, test result |
| Performance | Release GB10 in recorded power/thermal state | Frontier floor holds: at least 5 decode tok/s and at most 20 s warm TTFT; publish observed values | Benchmark JSON |
| Endurance | Release GB10 | At least 60 minutes without OOM, swap growth, parser failure, restart, corrupt output, or unbounded memory | Soak report and metrics |
| Upgrade | Existing compatible legacy state | Spark Lab discovers compatible state without exposing legacy names in the new-user path; `ft` still works or gives exact migration guidance; legacy env precedence is correct | Upgrade transcript |
| Rollback | Previously published stable artifact | Service and cache return to the prior working version without checkpoint re-download or state loss | Rollback transcript |

Skipped hardware tests count as **not run**, not passed. All release-only evidence must be
produced on a physical GB10 using the exact candidate artifacts.

## 10. Defect policy and go/no-go rules

### Severity

- **S0 — stop ship:** data/checkpoint corruption, security issue, unsafe destructive
  behavior, or artifacts that cannot be rolled back.
- **S1 — stop ship:** clean install failure, certified recipe failure, wrong output,
  crash/OOM/swap growth in the supported path, broken required API, or false platform
  admission; engine branding exposed in the normal Spark Lab product journey; native
  runtime behavior bypassing the backend contract in a product module; unbounded disk
  cache/staging, unexplained physical reads, corrupt cache admission, or leaked I/O workers.
- **S2 — normally stop ship:** major documented workflow failure with a reliable but
  burdensome workaround, large unexplained performance regression, or misleading UX.
- **S3 — may defer:** cosmetic, documentation, or experimental-model issue that does not
  undermine the supported promise.

### Go requires all of the following

- Zero open S0 or S1 defects; every S2 has explicit release-lead and area-owner sign-off.
- All mandatory gates passed on immutable candidate artifacts.
- Certified evidence and public claims agree.
- Fresh install, upgrade, and rollback were exercised, not merely reviewed.
- Release notes clearly state GB10-only support and experimental-model limitations,
  using Spark Lab as the product identity.
- Brand-boundary validation passes and required engine attribution is present in the
  approved Notices/About/SBOM locations.
- Backend contract, import-boundary, schema migration, and native parity gates pass.
- NVMe correctness, bounded-memory, normalized-telemetry, failure, cancellation, and
  teardown gates pass on the native adapter.
- A named incident owner is available during the first 24 hours.

If any condition fails, delay or reduce scope. Do not weaken a Certified recipe gate to
preserve the date; demote the recipe or postpone the beta.

## 11. Publish and rollback checklist

### Before publish

- [ ] Milestone contains no unresolved blocker.
- [ ] Working tree is clean; Spark Lab version and tag mapping is correct.
- [ ] Spark Lab ARM64 artifact and its internal runtime/kernel-cache artifacts come from the approved commit.
- [ ] Artifact hashes, Spark Lab metadata, entry points, SBOM, and provenance match.
- [ ] Brand-boundary inventory passes; legal and research attribution remains complete.
- [ ] Backend contract, static import boundary, schema migration, and native parity pass.
- [ ] Native NVMe path passes disk/RAM correctness, bounded-memory, direct-I/O, scheduler-
  telemetry, failure-injection, cancellation, teardown, and endurance gates.
- [ ] Full tests, clean install, Qwen certification, API matrix, agent task, and soak pass.
- [ ] Release notes, limitations, support matrix, evidence, upgrade, and rollback docs are final.
- [ ] Legal/name/checkpoint-license review is recorded.
- [ ] Go/no-go approvers and launch support owner have signed off.

### Immediately after publish

- [ ] Install from the public artifact location on a clean GB10.
- [ ] Run `doctor`, `models`, `plan`, and one streamed API completion.
- [ ] Verify all links, hashes, recipe assets, and evidence downloads.
- [ ] Confirm issue intake and diagnostic instructions are visible.

### Roll back when

- A public artifact differs from the approved candidate.
- A supported clean install or certified recipe cannot complete.
- An S0/S1 issue appears without a safe immediate mitigation.
- Monitoring finds repeated OOM, swap growth, corruption, or API incompatibility.

Rollback means withdrawing the bad artifact from recommendations, restoring the last
known-good version in install docs/channels, publishing an incident note, and cutting a
new version for the fix. Never overwrite immutable versioned artifacts in place.

## 12. Definition of done

Beta 0.1 is complete when the public Spark Lab artifact—not a source checkout or a visibly
engine-branded package—delivers the scoped GB10 workflow, at least one precision-preserving
recipe is Certified on that artifact, a fresh user can follow the docs successfully, the engine is properly
credited outside the normal product journey, rollback is proven, and the seven-day
stabilization window closes without an open S0 or S1 defect.

The product architecture is done for Beta 0.1 when the current engine is reachable only
through the native adapter from Spark Lab product code, recipes and evidence are backend-
qualified, artifact preparation is backend-owned, supervision is backend-neutral, and the
fake adapter proves that no FTW, CLI-argument, process, or health-schema assumption leaked
into the product contract.

The storage work is done for Beta 0.1 when the certified native deployment retains its
measured NVMe performance and exact output, the complete routed expert pool stays off the
resident-memory path, all staging and cache allocations are bounded and planned in the
single GB10 memory pool, physical I/O is accounted for, swap is not used as capacity, and
read failures or cancellation leave no stale cache entry, worker, or pinned allocation.

The next milestone can then pursue the broader 1.0 requirements: Qwen3.6 earning Fast
certification, certification of the remaining Frontier recipes, calibrated unified-memory
planning, fuller state and daemon rebranding, and Kimi K3 NVFP4 as an independent Research
deliverable.
