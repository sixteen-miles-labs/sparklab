# Codex instructions: implement the SparkLab runtime improvement plan

Use this prompt with `SparkLab_FLARE_Implementation_Test_Validation_Plan.md` attached or copied into the checkout.

## Ready-to-paste instruction

You are working in https://github.com/sixteen-miles-labs/sparklab.

Implement the attached SparkLab_FLARE_Implementation_Test_Validation_Plan.md. Read the entire plan before editing. It defines phased implementation, independent correctness tests, real GB10 validation, acceptance gates, evidence, compatibility, and rollback. Treat numerical targets as proposed engineering gates, not established performance claims.

This is an implementation task: make the changes, execute the available checks, and leave working, reviewable results. Do not stop after producing another plan.

### Start by grounding the work

1. Inspect the checkout, git status, HEAD, applicable AGENTS.md, CONTRIBUTING.md, tests/README.md, and current benchmark/release conventions. Preserve existing user work.
2. Reconcile current source with the reviewed references:
   - SparkLab: 715af2b3058f0b642bbb04c8d8a449e7816bba37.
   - FLARE: 5f37bd14062334f7687d168d58ea00567b9aa7fe.
   Do not downgrade the current checkout to those commits.
3. Inspect the actual native scheduler, graph runner, cache managers, model continuation state, expert-offload implementation, and existing tests. Determine which installed recipes use the native runtime versus isolated containers.
4. Review prior DSpark confidence-adaptive and rejected cost-aware benchmark experiments before repeating them.
5. Record actual hardware and complete local checkpoint availability. Reuse the configured environment and artifacts; avoid unrelated dependency upgrades or downloading an entire model portfolio.
6. Establish an unchanged baseline and run existing affected tests before implementation. Separate pre-existing failures from regressions.

### Implement in this order

- M0: source/hardware audit, exact recipe eligibility, existing behavior, baseline.
- M1: minimal observability and a reproducible mixed-workload benchmark harness.
- M2: bounded latency-aware prefill/decode scheduling, preserving a baseline selector.
- M3: exact complete checkpoints at message/prompt/terminal consumed boundaries, beginning with one native model.
- M4: versioned, bounded, asynchronous disk prefix retention and validated restore.
- M5: measured cost-aware speculative depth on one existing correct speculation path.
- M6: one profiled scratch-memory optimization, only if the measurements show a material opportunity.
- M7: affected-scheme regressions, built-artifact checks, integrated soak, evidence, docs, and rollout.

Complete each phase's implementation and relevant available tests before dependent work. Split changes into small coherent review units. Continue independent work if a phase is blocked or fails its benefit gate; do not carry a correctness failure into dependent phases.

### Preserve these boundaries

- Evolve the Python native runtime; do not rewrite it in Rust or replace SparkLab with FLARE.
- Reuse existing CUDA graphs, AOT, hybrid radix tests, accepted-prefix commit logic, metrics, backend interfaces, and evidence schemas where they fit.
- Separate expert-weight offload from conversation-prefix retention.
- Do not change model weights, quantization, sampling semantics, public model-support claims, or certification to make an optimization look better.
- On GB10, account for host/device allocations in the same physical pool; swap is not capacity.
- A checkpoint must preserve all state at the exact consumed endpoint: relevant KV/tails, recurrent/convolution/window/compressor state, position metadata, and target/draft continuation state.
- Keep shared published state immutable and retain resources through GPU/I/O completion.
- Preserve current multimodal cache exclusion until content identity and complete vision state are separately implemented and qualified.
- Bound disk/staging/index/queue resources; the model worker must not block on prefix-cache disk I/O.
- Use measured restore-versus-prefill cost, including expert-I/O contention.
- Preserve fixed-depth speculation and the baseline scheduling path.
- Follow repository design-discussion requirements before public API or stored-format changes. Prepare the concrete design and compatibility decision early.
- Respect the repository PR/DCO workflow. Do not push directly to main, merge, publish a release, deploy, or change a live default without the appropriate explicit authorization.

### Testing expectations

Follow the detailed test matrices and gates in the plan.

- CPU: independent scheduling/event oracles; existing radix reference-model lifecycle tests; real-file record round trips; failure injection; resource conservation.
- GPU: complete state snapshot/restore, unaligned boundaries, full prompt hits, branching, all accepted-prefix lengths, cancellation/rebuild, padding/inactive-row poisoning, and replay stability.
- API: real-server streaming, tool loops, stop behavior, accounting, cancellation, OpenAI/Responses/Anthropic regressions.
- Performance: identical checkpoint/backend/precision/sampling/cache sizes and completed work; compare baseline/candidate in paired balanced trials.
- Workloads: long-prefill interference, repeated prefixes, tool conversations, branching, disk restore/restart, expert/prefix NVMe contention, diverse speculation content, and sustained pressure.
- Report delivery gaps rather than fabricating per-token times for speculative bursts.
- Correctness precedes speed claims. Keep skipped/unavailable checks distinct from passes.
- Add new CPU-safe runtime tests to hosted CI: the reviewed CPU workflow does not include all runtime test directories.
- Keep raw large traces outside git and compact summaries consistent with benchmark conventions.
- Validate a built artifact and rollback paths before default promotion.

If the environment has no GB10 or required full checkpoint, implement and run the CPU-safe work and available integration checks, then continue independent implementation where sound. Leave GPU-dependent features disabled/experimental and provide exact commands/artifacts needed for the missing checks. Do not claim GB10 validation from CPU tests, mocked models, or skipped tests.

If code or measurements establish that a proposed optimization offers no robust benefit, document the result and retain the safe existing behavior. Do not add speculative abstractions simply to check off a milestone.

### Work tracking and final handoff

Maintain a compact progress record using the repository's existing convention. Track each milestone as implemented, validated, blocked, failed-benefit-gate, or deferred, with a reason. Keep this separate from release certification.

At handoff provide:

1. Concrete behavior changed and affected recipes/backends/cache schemes.
2. Files/review units and configuration/default decisions.
3. Tests executed with pass/fail/skip counts and all unrun hardware checks.
4. Absolute matched baseline/candidate metrics and gate outcomes.
5. Complete-state/quality findings, resource bounds, and disk failure behavior.
6. Rollback steps, known limitations, and the next genuinely blocked action.
7. Paths to compact evidence and reproducible benchmark commands.

Start now with M0 and M1, then implement subsequent phases in order as evidence and dependencies allow. Continue until all feasible authorized work is done; do not ask for confirmation on routine implementation choices.
