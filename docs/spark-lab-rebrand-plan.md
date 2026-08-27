# Spark Lab Rebrand and GB10 Execution Plan

Status: implementation started
Date: 2026-08-27
Scope: NVIDIA GB10 only

## Implementation status — 2026-08-27

The first compatibility-preserving product slice has landed in the worktree:

- `sparklab` is a packaged CLI alongside the unchanged `ft` alias;
- `sparklab doctor` emits human and schema-versioned JSON GB10 reports covering
  Linux/ARM64, CUDA 13, SM121, physical and available memory, swap, storage, and
  installed kernel dependencies;
- `sparklab models` reads packaged, versioned recipes for the Fast, Frontier, and
  Research portfolio and keeps intended tier separate from admission status;
- `sparklab plan`, `sparklab pull`, and `sparklab run` provide fail-closed
  artifact planning, immutable resumable acquisition, optional FTW preparation,
  and recipe-owned launch arguments;
- Qwen3.8-Flash-Next now has a certified text runtime for Hyper-Connections, PLE
  with a disk-backed 95 GiB n-gram table, hybrid GDN/QSA attention, stacked BF16
  experts, and bounded-memory FTW conversion. The full checkpoint passed output,
  capability, exact 64K-context, performance, and 60-minute endurance gates in
  `GB10-QWEN38-FRONTIER-001`;
- `SPARKLAB_*` values take precedence in the central environment configuration,
  while corresponding `FREETOKEN_*` values remain fallbacks;
- new bandwidth profiles write under `~/.cache/sparklab`, with read-only discovery
  of an existing FreeToken profile;
- the measured DeepSeek V4 result is recorded as
  `benchmarks/gb10/results/GB10-BASELINE-001.json` under a versioned evidence schema;
- README, installation, quick-start, model, CLI, and migration documentation now
  lead with the GB10 product workflow and retain explicit FreeToken attribution.

This does not complete Stage 1 or the public rebrand gate. In particular, the
ARM64 release artifact, daemon state migration, visual identity, calibrated
runtime-memory measurements, and the remaining model certification gates remain
open.

### Active certification sequence

The current machine work is deliberately serial to preserve NVMe and unified
memory headroom:

1. Qwen3.8-Flash-Next 0.4.0 is the Fast-layer target. Its 0.3.0 text-only FTW/NVMe
   recipe earned historical Frontier certification in `GB10-QWEN38-FRONTIER-001`,
   but the new target remains Preview until it passes the stricter Fast gate.
2. Finish the pinned GLM-5.3-Flash acquisition, convert FTW, and validate the new
   text runtime against the complete checkpoint. The implementation now includes
   KDA, NoPE MLA/KPool, four-stream mHC, FP32-scaled block FP8, and NVMe experts;
   certification still requires measured correctness, performance, and endurance.
3. Resume the complete Kimi K3 experiment only after GLM-5.3 finishes.

No dummy, reduced-layer, or import-only result can promote a recipe beyond
Experimental. Only complete-checkpoint evidence can earn Preview or Certified.

## 1. Executive decision

Rebrand the user-facing project from **FreeToken** to **Spark Lab** and narrow the
product from a general consumer-GPU inference engine to a GB10-native frontier-model
runtime.

The product promise is:

> Spark Lab runs frontier open-weight models locally on NVIDIA GB10 with tested
> configurations, predictable memory use, and APIs that coding and agent tools can use
> directly.

Spark Lab should be opinionated about one device rather than broadly compatible with many
devices. The supported production platform is:

- NVIDIA GB10 Grace Blackwell Superchip
- 128 GB coherent unified memory
- ARM64 / DGX OS
- CUDA 13
- Compute capability 12.1 (SM121)
- Local NVMe for checkpoints and disk-backed expert storage

FreeToken remains the name of the existing research paper and, during migration, the
internal Python engine. Spark Lab becomes the product, CLI, documentation, benchmark
program, and model certification identity.

## 2. Positioning

### Category

**GB10-native frontier inference lab**

Spark Lab is not positioned as a generic inference framework, model chat application, or
replacement for every vLLM deployment. It is the shortest reliable path from a GB10 device
to a useful local frontier model.

### Primary message

> Turn one NVIDIA GB10 into a private frontier-AI workstation.

### Supporting messages

1. **Made for unified memory:** memory planning and execution policies are designed for
   GB10 rather than inherited from discrete GPUs.
2. **Models, not knobs:** certified model recipes choose kernels, cache geometry, context,
   and concurrency automatically.
3. **Frontier beyond memory:** NVMe-backed MoE execution allows selected models larger than
   the 128 GB physical memory pool to run locally.
4. **Agent ready:** OpenAI- and Anthropic-compatible APIs, reasoning output, tool calls, and
   prefix reuse work out of the box.
5. **Measured, not implied:** every certified recipe publishes TTFT, decode speed, memory,
   disk traffic, and correctness evidence from a real GB10.

### Flagship proof points

- **Practical flagship:** DeepSeek V4, already measured at 9.217 decode tok/s on GB10 with
  disk-backed experts, 14.045 s warm TTFT, no observed swap growth, and identical greedy
  output across matched configurations. This is the launch baseline, not an aspirational
  number; see `exp_dsv4_gb10.md`.
- **Daily-use tier:** a small set of 20B-120B dense and sparse models selected for
  interactive coding, reasoning, and agent work.
- **Research moonshot:** the complete official Kimi K3 checkpoint on one GB10 using
  NVMe-backed expert streaming. This must be described as an experimental capability until
  it meets the correctness and usability gates in Stage 5.

## 3. Product boundaries

### In scope

- Single-GB10 inference and optimization
- Text generation and agent/tool use
- Dense, MoE, hybrid recurrent, and hybrid-attention architectures
- NVFP4, MXFP4, FP8, and BF16 formats when appropriate for a certified model
- Unified-memory planning
- Explicit NVMe-backed expert storage
- OpenAI Chat Completions and Responses APIs
- Anthropic Messages API
- CLI, local service, model catalog, diagnostics, and benchmark reporting
- Multi-GB10 research after the single-device product is stable

### Not in scope for the initial product

- RTX 30/40/50, datacenter GPU, AMD, Apple Silicon, or CPU-only support claims
- Windows support
- Training or full fine-tuning
- General-purpose distributed serving
- High-concurrency production serving
- Supporting every Hugging Face architecture
- Claiming interactive speed merely because a checkpoint can produce a token
- A one-million-token Kimi K3 context at initial launch

Existing non-GB10 code may remain as an unsupported compatibility path while it does not
slow development. New product decisions and release gates are based only on GB10.

## 4. Brand and compatibility architecture

Use a staged rename to avoid coupling the brand launch to a risky package-wide refactor.

| Surface | Initial Spark Lab release | Later major release |
|---|---|---|
| Product name | Spark Lab | Spark Lab |
| Primary CLI | `sparklab` | `sparklab` |
| Legacy CLI | `ft`, supported alias | Deprecate only with migration data |
| Python distribution | `freetoken` | Decide after product validation |
| Python imports | `freetoken.*` | Optional `sparklab_engine.*` migration |
| Environment variables | `SPARKLAB_*`, with `FREETOKEN_*` fallback | `SPARKLAB_*` |
| User data | `~/.sparklab` with legacy discovery | `~/.sparklab` |
| API protocol | Unchanged | Unchanged |
| Research paper | FreeToken | FreeToken |

The command should be `sparklab`, not `spark`, to avoid collision with Apache Spark and
other executables. Before public launch, complete a name, domain, package-index, and NVIDIA
trademark review. If the name is not defensible, change it before visual assets or package
renames begin.

## 5. Product architecture

### 5.1 Platform profile

Replace scattered architecture checks with a single GB10 platform profile:

```text
GB10Platform
├── identity: arm64, SM121, CUDA 13
├── memory: physical, available, reclaimable, display-reserved, swap
├── compute: supported quantization and kernel capabilities
├── storage: NVMe path, direct-I/O support, bandwidth and queue-depth profile
└── health: driver, thermal, power, dependency, and kernel-cache status
```

The initial implementation belongs under `python/freetoken/platform/`, with the existing
`python/freetoken/utils/arch.py` retained as a compatibility facade.

### 5.2 Unified-memory planner

The current GPU-memory planner assumes weights, GPU cache, and KV cache consume a discrete
VRAM pool. Spark Lab needs a GB10 planner based on the physical unified-memory budget:

```text
physical unified memory
- OS and service reserve
- display reservation
- non-expert model weights
- expert cache or resident expert weights
- KV and recurrent-state cache
- CUDA graphs, activations, and workspaces
- transient load and conversion buffers
= safety margin
```

The planner must use Linux available/reclaimable memory in addition to CUDA allocator
information. It must never count swap as normal model capacity. Before allocating weights,
it should resolve a safe model recipe or reject it with a concrete explanation.

### 5.3 Execution policies

Expose internal capabilities as explicit policies:

```text
resident       all weights remain in unified memory
uma-moe        dense weights resident, experts share a unified-memory cache
nvme-moe       dense weights resident, routed experts stream from NVMe
multi-gb10     future tensor/expert parallel policy
```

The GB10 path should remove redundant host-to-device copies where shared allocations or
direct GPU access are faster. This is a measured decision per tensor class; it must not be
assumed universally.

### 5.4 Certified model recipes

Make a versioned recipe the deployable unit:

```yaml
model: deepseek-ai/DeepSeek-V4-Flash-0731
platform: gb10
status: certified
checkpoint_format: ftw
execution_policy: nvme-moe
profile: balanced
context_tokens: 2048
max_concurrency: 1
memory:
  host_cache_gib: 4
  expert_slots: 5900
storage:
  read_workers: 20
validation:
  output_hash: fbf178b2bde5
```

Recipe statuses are:

- **Certified:** correctness, stability, and benchmark gates pass on the release image.
- **Preview:** end-to-end generation works but one or more performance or feature gates are
  still open.
- **Experimental:** intended for engineering evaluation; no usability promise.

No model appears in the default catalog without a recipe and a recorded validation result.

### 5.5 User experience

The intended path is:

```bash
sparklab doctor
sparklab models
sparklab pull deepseek-v4
sparklab run deepseek-v4 --profile balanced
sparklab status
sparklab bench deepseek-v4
```

`sparklab doctor` is the first product feature, not an auxiliary debugging command. It
reports platform compatibility, usable unified memory, storage suitability, installed
kernels, model capacity, and recommended corrective actions.

## 6. Stages and milestones

Durations are planning ranges, not release promises. A stage exits only when its measurable
gate passes.

### Stage 0: Commit to the GB10 product

Target: Week 0-1

#### Milestones

- Approve the GB10-only scope and non-goals in this document.
- Complete naming, domain, package-index, and trademark checks for Spark Lab.
- Freeze new non-GB10 optimization work.
- Record the current DeepSeek V4 GB10 result as baseline `GB10-BASELINE-001`.
- Define a standard benchmark prompt set and artifact schema.
- Confirm the primary lineup: Qwen3.8-Flash-Next for Fast, GLM-5.3-Flash and
  DeepSeek-V4-Flash-0731 for Frontier, and Kimi K3 for Research. Retain
  Qwen3.6-35B-A3B-NVFP4 and GLM-5.2-NVFP4 only as Experimental fallbacks.

#### Exit gate

- One approved product brief, one approved name, one benchmark schema, and a signed-off
  model shortlist exist.
- Every active engineering item maps to GB10 detection, correctness, performance, model
  enablement, installation, or product UX.

### Stage 1: Establish the GB10 foundation

Target: Week 1-4

#### Milestones

- Add explicit GB10/SM121 detection and fail closed on unrecognized Blackwell kernels.
- Add `GB10Platform` and a machine-readable `sparklab doctor --json` report.
- Implement unified-memory capacity reporting that includes Linux reclaimable memory but
  excludes swap from safe capacity.
- Add an ARM64 build pipeline and produce reproducible CUDA 13 / SM121 wheels or an official
  container.
- Audit native extensions and binary dependencies for ARM64.
- Add a GB10 kernel capability registry and startup report.
- Preserve the current DeepSeek V4 result within a 5% decode-throughput tolerance.

#### Exit gate

On a clean, supported GB10 image:

- Installation completes without manual source edits.
- `sparklab doctor` identifies GB10, ARM64, CUDA, SM121, memory, NVMe, and kernel support.
- A certified smoke model reaches first token through the public API.
- DeepSeek V4 passes deterministic output validation without swap growth.

### Stage 2: Ship GB10-native memory and execution

Target: Week 3-7

#### Milestones

- Add a unified-memory planner alongside the current discrete-GPU planner.
- Separate resident, UMA-MoE, and NVMe-MoE policies.
- Measure zero-copy/shared allocations against explicit copies for each hot tensor class.
- Make page-cache pressure visible and add safe targeted checkpoint-page eviction.
- Replace hard-coded expert-cache tuning with recipe-driven allocation.
- Add preflight estimates for weights, KV/recurrent state, workspaces, and safety reserve.
- Make OOM handling deterministic: reduce context or concurrency before model load rather
  than failing during generation.

#### Exit gate

- The planner's predicted peak is within 5% of measured peak memory for each certified
  model.
- Repeated 30-minute runs show no swap growth, CUDA OOM, or unbounded page-cache growth.
- Startup explains every material allocation and the selected execution policy.

### Stage 3: Optimize the daily-use Fast and Frontier tiers

Target: Week 5-10

#### Milestones

- Promote DeepSeek V4 from experimental configuration to a versioned certified recipe.
- Profile SM121 attention, MoE, sampling, recurrent-state, and weight-read bottlenecks.
- Tune asynchronous reads, coalescing, queue depth, layer-ahead prefetch, and expert
  admission on the built-in NVMe.
- Tune MXFP4 and NVFP4 kernels specifically for SM121.
- Add long-running agent traces, repeated-prefix traces, and cold/warm prompt suites.
- Certify Qwen3.8-Flash-Next 0.4.0 for Fast and GLM-5.3-Flash plus DeepSeek V4
  for Frontier. Preserve Qwen3.8 0.3.0's evidence as historical Frontier evidence;
  do not transfer it to the Fast recipe.
- Validate reasoning and tool-call parsers with real coding-agent sessions.

#### Initial performance objectives

- DeepSeek V4: exceed the 9.217 tok/s baseline without changing the validated output.
- DeepSeek V4 stretch: at least 12 tok/s and warm TTFT below 10 seconds on the same AIME
  probe, or publish profiling evidence showing the next limiting component.
- Fast-tier model: at least 30 tok/s at batch size one.
- Every certified model: 60-minute agent session without OOM, swap growth, parser failure,
  or service restart.

These are engineering objectives. Public claims use only committed benchmark artifacts.

#### Exit gate

- Three model recipes are certified on the release image.
- Each has cold and warm TTFT, decode tok/s, peak unified memory, context capacity, power,
  disk bytes/token, and output-validation results.
- One supported coding agent completes a fixed repository task end to end with each recipe.

### Stage 4: Complete the product rebrand

Target: Week 7-11

#### Milestones

- Add the `sparklab` CLI while retaining `ft` as a compatibility alias.
- Add `SPARKLAB_*` environment variables with documented legacy fallback precedence.
- Move new user state to `~/.sparklab` and discover existing FreeToken installations
  without destructive migration.
- Replace README, installation, quick-start, model, CLI, and daemon documentation with the
  GB10 workflow.
- Create the visual identity, logo, screenshots, and GB10-focused website copy.
- Rename user-visible daemon, logs, metrics namespace, and desktop labels.
- Keep FreeToken paper attribution and a migration guide.
- Add recipe-backed `models`, `pull`, `run`, `status`, and `bench` commands.

#### Exit gate

- A new GB10 owner can go from the supported base image to a streamed API response in under
  15 minutes, excluding checkpoint download time.
- No primary documentation asks the user to select a low-level cache, kernel, or offload
  flag.
- `ft` workflows continue to work or produce an exact migration instruction.
- The public site contains only benchmark claims reproduced by the release artifacts.

### Stage 5: Kimi K3 on one GB10

Target: Week 9-18; independent experimental release gate

Kimi K3 is the flagship demonstration, not a dependency for the daily-use product launch.

#### Milestone 5.1: Checkpoint and format feasibility

- Record official checkpoint revision, license obligations, checksums, total disk size, and
  tensor inventory.
- Verify that source weights plus conversion workspace fit the target storage configuration.
- Implement resumable shard-by-shard conversion or direct checkpoint access so conversion
  never requires an unnecessary second full checkpoint copy.
- Extend FTW metadata for 896-expert, 93-layer K3 layouts.

Gate: the complete checkpoint is addressable, resumable, checksum-validated, and bounded in
RAM during preparation.

#### Milestone 5.2: Architecture correctness

- Add Kimi K3 configuration and model registration.
- Implement KDA, Gated MLA, Attention Residuals, Stable LatentMoE, SiTU-GLU, and the text
  tokenizer/input encoder.
- Reuse and extend the existing hybrid recurrent/KV state manager.
- Add K3 reasoning-effort and tool-call parsing.
- Defer MoonViT-V2 and video until text correctness passes.

Gate: a reduced synthetic configuration passes layer-by-layer reference tests and the full
checkpoint initializes far enough to validate all tensor mappings.

#### Milestone 5.3: First correct generation

- Keep non-expert weights and runtime state resident where feasible.
- Stream routed MXFP4 experts from NVMe into the unified-memory cache.
- Run a fixed 4K-or-smaller text prompt and generate at least 16 tokens.
- Compare token IDs or accepted numerical tolerances against a trusted K3 engine.
- Demonstrate with network disabled and no swap growth.

Gate: the full official model produces a validated completion on one physical GB10. This
earns **experimental / runs locally** status, not **interactive** status.

#### Milestone 5.4: Usability optimization

- Add multi-layer asynchronous prefetch and read coalescing.
- Measure expert locality by layer and workload.
- Evaluate cache pinning, router-informed admission, and repeated-agent-session locality.
- Evaluate DSpark speculative decoding by accepted tokens per physical NVMe byte, not only
  tokens per verification step.
- Publish cold and warm TTFT, tok/s, bytes/token, cache-hit rates, and power.

Gate: publish the observed performance without adjective inflation. K3 is called
**interactive** only if it sustains at least 1 tok/s on the agreed coding trace; otherwise it
remains an explicit research mode.

#### Milestone 5.5: Reproducible flagship demo

- Release a `sparklab run kimi-k3 --profile research` recipe.
- Show hardware identity, disabled networking, checkpoint fingerprint, live memory, NVMe
  traffic, and token latency.
- Complete a real coding task and run its tests.
- Publish the command, prompt, output tokens, logs, and benchmark JSON.

Gate: an independent GB10 owner can reproduce the result from the documented artifact set.

### Stage 6: Spark Lab 1.0

Target: after Stages 1-4 pass; Kimi K3 may remain experimental

#### Milestones

- Freeze the GB10 platform and recipe schema for the 1.x line.
- Publish signed ARM64 artifacts and checksums.
- Publish a support matrix containing only measured GB10 recipes.
- Add upgrade, rollback, cache cleanup, and diagnostic bundle workflows.
- Run fresh-install, upgrade-from-FreeToken, API compatibility, endurance, and model
  correctness release suites.
- Publish the DeepSeek V4 flagship benchmark and the Kimi K3 result at its earned status.

#### Exit gate

- All three daily-use model recipes pass correctness and endurance tests on the release
  artifact.
- Fresh install and legacy migration pass on separate GB10 systems or clean system images.
- Every public performance claim points to a machine-readable result containing environment,
  checkpoint, recipe, and source revision identifiers.
- There are no open severity-one correctness, data-integrity, OOM, or API compatibility
  defects.

## 7. Model portfolio policy

Do not optimize every newly released model. Score candidates against:

| Criterion | Weight |
|---|---:|
| Frontier quality for coding/reasoning/agents | 30% |
| Architectural reuse in the engine | 20% |
| Native GB10 quantization path | 20% |
| Useful single-user performance potential | 15% |
| License and checkpoint stability | 10% |
| Community demand | 5% |

Maintain three visible tiers. A tier belongs to a **checkpoint + Spark Lab recipe**, not to
a model family or parameter count. The same model may move tiers when a better
quantization, kernel, or storage policy is certified.

Use the standard `GB10-INTERACTIVE-001` workload to assign tiers: one GB10, batch size and
concurrency one, a 4K-token text prompt, 512 requested output tokens, the model's recommended
sampling settings, one warm-up request, and the median of three fresh server processes.

| Tier | User promise | Required GB10 gate |
|---|---|---|
| **Fast** | Default for routine chat, editing, and short agent loops | At least 20 decode tok/s, warm TTFT at most 5 s, at least 32K usable context, no normal-operation NVMe stalls after load, and a 60-minute agent trace without swap growth, OOM, or restart |
| **Frontier** | Quality-first model for hard coding, reasoning, and long agent work | At least 5 decode tok/s, warm TTFT at most 20 s, at least 64K usable context, bounded NVMe traffic, and the same 60-minute stability gate |
| **Research** | Complete or novel models beyond the current interactive envelope | Correct full-model output and bounded memory are required; no latency promise. Publish TTFT, tok/s, bytes/token, and limitations prominently |

Speed alone cannot promote a recipe. Fast and Frontier recipes must also pass output
correctness, reasoning/tool parsing, the fixed coding-agent task, and the benchmark evidence
contract in Section 8. Conversely, total parameter count does not force a model into
Research if sparse activation and a GB10-native format meet an interactive gate.

Promotion and demotion rules:

- Assign **Research** when a new model first produces validated full-model output.
- Promote to **Frontier** only after all Frontier performance, context, stability, and agent
  gates pass on a release artifact.
- Promote to **Fast** only after all Fast gates pass without a quality-degrading model
  modification outside the certified checkpoint recipe.
- Demote a recipe if a checkpoint revision, runtime release, or supported OS update causes
  any required gate to fail.

The catalog should remain small. A model is removed or demoted when its checkpoint changes,
its recipe cannot be reproduced, or a better model occupies the same product role.

### Recommended 1.0 shortlist

| Intended tier | Candidate | GB10 rationale | Admission status |
|---|---|---|---|
| Fast | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 125B language-model parameters with 6B active, plus a 51B n-gram embedding and 4B MTP; 0.3.0 established a 12.51 tok/s Frontier baseline | Version 0.4.0 is Preview until the Fast throughput and no-stall gates pass |
| Frontier | [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | Strong coding/reasoning story and an existing reproducible GB10 result | Baseline proven; Frontier recipe not yet productized |
| Frontier | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 320B total and 18B active parameters, native multimodality, FP8 weights, and hybrid sparse/linear attention | Text architecture implemented; Experimental until the complete checkpoint and GB10 gates pass; multimodality remains separate |
| Fast fallback | [Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | Native low precision, about 3B active parameters, existing model implementation | Experimental fallback outside the primary lineup |
| Frontier fallback | [GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | Existing implementation, NVIDIA-format checkpoint, and lower enablement risk | Certification candidate retained until GLM-5.3-Flash passes |
| Research | [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | 2.8T flagship that demonstrates inference beyond physical memory | Architecture and performance work required |

### Target Spark Lab 1.0 portfolio

The planned launch lineup is:

```text
Fast       Qwen3.8-Flash-Next
Frontier   GLM-5.3-Flash
Frontier   DeepSeek-V4-Flash-0731
Research   Kimi K3
```

These labels are target product roles, not certification claims. Each recipe keeps its
independent Certified, Preview, or Experimental admission status until its complete
checkpoint passes the required GB10 gate. In particular:

- **Qwen3.8-Flash-Next** is the primary Fast target. Its 0.3.0 text-only NVFP4/NVMe
  result records 12.51 tok/s, 0.870 s warm TTFT, exact 64K recall, and an uninterrupted
  60.50-minute endurance run. Version 0.4.0 remains Preview until it reaches the 20 tok/s
  floor with no normal-operation NVMe stalls.
- **GLM-5.3-Flash** and **DeepSeek V4 Flash** are the two Frontier targets. Each must
  independently pass the complete Frontier gate.
- **Kimi K3** is the flagship beyond-memory demonstration and does not block the daily-use
  1.0 release. It stays Research unless it independently satisfies every Frontier gate.

### Active promotion queue

1. **GLM-5.3-Flash text inference** is active. The `glm5_next` KDA/mHC/KPool runtime,
   FP8 loader, and bounded NVMe recipe are implemented; finish acquisition and run the
   complete checkpoint through parity, performance, context, and endurance gates.
2. **GLM-5.3-Flash multimodality** follows text-only Frontier certification. Vision support
   must pass separate image preprocessing, memory, correctness, and agent-tool tests and
   cannot inherit the text recipe's certification.
3. **Kimi K3** runs after Qwen and GLM-5.3 so its much larger artifacts cannot displace the
   daily-use certification work.
4. **GPT-OSS 120B** remains a comparison candidate. Add it only if it displaces an existing
   recipe on measured quality, speed, tool reliability, or maintenance cost.
5. **GPT-OSS 20B**, **Qwen3-30B-A3B**, and smaller Gemma variants are Fast-tier fallback
   candidates, not additional 1.0 commitments.

Qwen3.8-Flash-Next was tested in both official BF16 and GB10-oriented NVFP4 forms. The
certified NVFP4 recipe keeps routed experts independently addressable and uses a bounded
page-cached n-gram artifact rather than treating that embedding as ordinary MoE weights.
Its five-problem AIME sample scored 3/5, with two reasoning-budget caps recorded as an
explicit quality limitation.

GLM-5.3-Flash supersedes GLM-5.2 as the desired Frontier candidate, but it does not inherit
GLM-5.2's status. Its text-only `glm5_next` architecture, hybrid linear/sparse attention,
mHC, and FP8 format are implemented and now require complete-checkpoint validation. The
native multimodal path remains a separate later milestone. Keep GLM-5.2-NVFP4 as the
release fallback until 5.3 passes the complete Frontier gate.

The release still requires one Fast and two Frontier recipes to pass rather than merely
exist in the target lineup. Qwen3.8 is the Fast target; GLM-5.3 and DeepSeek V4 fill the
Frontier slots. Qwen3.6 and GLM-5.2 remain fallbacks and cannot replace a primary target
without benchmark evidence and an explicit portfolio decision.

### Stage dependency and release order

```text
Stage 0: product commitment
    |
    v
Stage 1: GB10 foundation
    |
    +-----------> Stage 5: Kimi K3 research (independent status)
    |
    v
Stage 2: UMA execution
    |
    v
Stage 3: certified daily-use models
    |
    v
Stage 4: public rebrand and product UX
    |
    v
Stage 6: Spark Lab 1.0
```

The repository may adopt internal Spark Lab terminology during Stages 1-3, but the public
launch should wait until installation, diagnostics, and at least three daily-use recipes
pass their gates. Kimi K3 can launch later without delaying Spark Lab 1.0.

## 8. Benchmark and evidence contract

Every published model result must include:

- Spark Lab and engine git revisions
- Recipe version
- Checkpoint repository, revision, format, size, and fingerprint
- DGX OS, kernel, driver, CUDA, PyTorch, Triton, and optional-kernel versions
- Physical and available unified memory before load
- NVMe model, filesystem, direct-I/O mode, and measured read bandwidth
- Context length, input/output tokens, sampling settings, batch, and concurrency
- Cold and warm startup/TTFT
- Decode tok/s plus latency p50 and p99
- Peak allocations, available memory floor, and swap delta
- Expert cache hits, misses, evictions, disk operations, bytes, and service time
- Output token IDs or stable hash for deterministic probes
- Thermal or power state for sustained tests

Store raw JSON outside the source repository and commit compact summaries plus schemas. A
result is comparable only when the workload, recipe, checkpoint, and output validation are
the same.

## 9. Workstreams and repository landing zones

| Workstream | Initial landing zone |
|---|---|
| GB10 detection and capability registry | `python/freetoken/platform/`, `utils/arch.py` |
| Unified-memory planner | `python/freetoken/engine/`, `python/freetoken/kvcache/` |
| NVMe expert pipeline | `python/freetoken/moe/`, `python/freetoken/checkpoint/` |
| SM121 kernels | `python/freetoken/kernel/` |
| Model recipes and catalog | `profiles/models/`, `python/sparklab/catalog.py` |
| Product CLI and doctor | `python/sparklab/` |
| API compatibility | `python/freetoken/server/` |
| Kimi K3 architecture | `python/freetoken/models/kimi_k3/` |
| Reproducible benchmarks | `benchmarks/gb10/` |
| Rebrand and migration docs | `README.md`, `docs/` |
| ARM64 release pipeline | `scripts/ci/`, packaging configuration |

## 10. Risks and controls

| Risk | Control |
|---|---|
| Spark Lab name conflicts with Apache Spark or NVIDIA branding | Complete naming and trademark gate before asset work |
| UMA is treated like large discrete VRAM | Separate GB10 planner and measure every allocation/copy policy |
| ARM64 dependencies lag x86 wheels | Official pinned container first; ARM64 wheel matrix second |
| Model-count ambition dilutes optimization | Three certified recipes maximum at 1.0 launch |
| Kimi K3 technically runs but is unusably slow | Separate correctness and interactivity gates; publish raw latency |
| NVMe endurance or heat limits long sessions | Record physical reads and temperature; minimize bytes/token |
| Linux page cache competes with CUDA allocations | Targeted eviction, capacity reporting, and endurance tests |
| Rebrand breaks existing users and paper citations | Preserve `ft`, `freetoken`, API behavior, and FreeToken attribution initially |
| Public benchmark cannot be reproduced | Require recipe, environment, checkpoint, and output identity in every result |

## 11. Immediate next actions

Execute these in order:

1. Approve or replace the Spark Lab name after availability and trademark review.
2. Convert the existing DeepSeek V4 GB10 experiment into `GB10-BASELINE-001` JSON plus a
   checked-in summary.
3. Implement explicit GB10/SM121 detection and `sparklab doctor`.
4. Add the unified-memory planner and tests before changing current cache behavior.
5. Create the recipe schema and encode the measured DeepSeek V4 configuration.
6. Produce a clean ARM64 installation artifact.
7. Select and certify the fast and quality daily-use models.
8. Begin Kimi K3 checkpoint inventory and reduced-configuration architecture tests in a
   separate experimental track.
9. Change user-facing branding only after the Stage 1 install and smoke-test gate passes.

## 12. Definition of success

The rebrand is complete when Spark Lab is recognized and experienced as a GB10 product,
not when every `freetoken` identifier has been renamed.

Success requires all of the following:

- A clean GB10 install reaches a working frontier-model API without manual tuning.
- The runtime plans coherent unified memory safely and does not rely on swap.
- At least three models have versioned, reproducible, GB10-certified recipes.
- DeepSeek V4 performance improves beyond the recorded 9.217 tok/s baseline or the limiting
  hardware bottleneck is proven with published evidence.
- Reasoning and tool use complete real agent tasks, not only synthetic token benchmarks.
- Kimi K3 produces a validated full-model completion on one GB10, with its actual speed
  disclosed and status assigned by the gates above.
- All public positioning, documentation, artifacts, commands, and diagnostics use Spark Lab,
  while legacy FreeToken users and academic citations have a documented compatibility path.
