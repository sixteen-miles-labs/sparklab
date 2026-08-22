# Disk-Backed MoE Inference Roadmap

## Objective

Add an explicit NVMe-backed expert-storage mode to FreeToken so that models whose routed
expert weights exceed system RAM can run with a bounded host-memory footprint. Performance
is secondary during the first implementation; correctness, bounded memory, and reliable
forward progress are the initial goals.

The first development and validation model is
`nvidia/Qwen3.6-35B-A3B-NVFP4`. It is a 35B-parameter MoE model with approximately 3B
parameters active per token, and FreeToken already supports its architecture and NVFP4
expert format.

## Target machine

- GPU: NVIDIA GeForce RTX 3090, 24 GiB VRAM, SM86
- Host memory: 64 GiB
- Driver: 580.173.02
- Primary experimental storage: `/mnt/ssd`
- Storage filesystem: local ext4 on NVMe
- Available experimental storage at planning time: approximately 3.4 TiB

The repository filesystem should contain source code only. Model checkpoints, FTW files,
logs, profiles, and benchmark results should live under `/mnt/ssd/freetoken`.

## Implementation status (2026-08-22)

- Phases 0-2: checkpoint downloaded and converted to FTW; RAM-backed baseline recorded.
- Phase 3: GPU expert-cache, host expert-cache, and disk byte/operation/time counters are
  available in the benchmark JSON. Asynchronous queue and prefetch counters remain.
- Phase 4: FTW expert-row descriptors and aligned independent row reads are implemented and
  covered by focused tests.
- Phase 5 correctness prototype: `--moe-storage disk` uses a byte-budgeted pinned-host LRU
  plus a shared expert-layer staging area instead of loading the complete 16.9 GiB host
  expert banks. `--moe-host-cache-gb` controls the LRU budget. Disk mode currently requires
  FTW, the offload backend, eager execution, and disabled prefill overlap.
- Matching 16-token RAM/disk smoke tests produced the same greedy output hash
  (`8400f78e0fc8`) and cache-miss counts. RAM decoded at 18.16 tok/s; synchronous disk
  decoded at 2.14 tok/s and read 22.02 GiB physical at 0.83 GiB/s.
- A 1 GiB host-cache smoke test held 604 experts, achieved a 1.65% hit rate, read 21.66 GiB,
  decoded at 2.11 tok/s, and preserved the same output hash. This validates bounded LRU
  operation but shows that prefill cache pollution must be addressed before sizing sweeps.
- Prefill-aware admission raised the host-cache hit rate to 6.28%, reduced disk reads to
  20.64 GiB, reduced warm TTFT by 9.3% to 22.89 s, decoded at 2.13 tok/s, and preserved the
  output hash. Prefill can consume cache hits but cannot insert, refresh, or evict entries.

The next implementation milestone is coalesced asynchronous reads followed by layer-aware
prefetching. The synchronous prototype is the correctness reference, not the final
performance design.

Suggested layout:

```text
/mnt/ssd/freetoken/
├── models/
│   └── Qwen3.6-35B-A3B-NVFP4/     # downloaded Hugging Face checkpoint
├── ftw/
│   └── Qwen3.6-35B-A3B-NVFP4/     # converted FTW checkpoint
├── results/
│   ├── baseline/
│   └── disk/
├── logs/
└── scratch/                        # disposable conversion and benchmark data
```

Do not use operating-system swap as the expert-storage mechanism. Disk access must be
explicit, measurable, and bounded by FreeToken.

## Design target

The current offload hierarchy is:

```text
checkpoint on disk -> complete pinned host expert banks -> GPU expert LRU -> compute
```

The target hierarchy is:

```text
FTW expert entries on NVMe
        |
        | aligned asynchronous reads
        v
bounded pinned-host expert cache
        |
        | host-to-device copy
        v
existing GPU expert LRU
        |
        v
MoE compute
```

Only routed experts are disk-backed initially. Attention weights, embeddings, routers,
shared experts, KV cache, activations, CUDA graphs, and kernel workspaces remain resident
in GPU or host memory as required by the current engine.

## Phase 0: Reproducible environment

### Work

- Record the git commit, Python, PyTorch, CUDA, driver, CPU, RAM, GPU, NVMe, and filesystem.
- Confirm the RTX 3090 selects the Marlin NVFP4 backend.
- Confirm `/mnt/ssd` supports the FTW `O_DIRECT` path; record if it falls back to `mmap`.
- Place all large artifacts under `/mnt/ssd/freetoken`.
- Ensure benchmarks run without unrelated GPU workloads or active memory pressure.
- Record swap counters before and after every benchmark. The new disk mode must not depend
  on swap activity.

### Measurements

```bash
ft bench bw --dtype nvfp4
```

Also record sequential and random direct-I/O performance for `/mnt/ssd`, including read
bandwidth, latency, and queue-depth scaling using a non-destructive read-only benchmark.

### Exit gate

- Hardware and software metadata are captured with the benchmark results.
- NVFP4 inference kernels run correctly on SM86.
- `/mnt/ssd` read behavior is known and reproducible.

## Phase 1: Acquire and validate the checkpoint

### Work

- Download `nvidia/Qwen3.6-35B-A3B-NVFP4` into:

  ```text
  /mnt/ssd/freetoken/models/Qwen3.6-35B-A3B-NVFP4
  ```

- Record repository revision and file checksums.
- Start FreeToken with the original checkpoint and complete a short deterministic request.
- Convert the checkpoint to FTW at:

  ```text
  /mnt/ssd/freetoken/ftw/Qwen3.6-35B-A3B-NVFP4
  ```

- Validate a deterministic request from the FTW checkpoint against the original checkpoint.

### Exit gate

- Original and FTW checkpoints both boot.
- Greedy token IDs match for a fixed prompt and generation length.
- No missing or incorrectly packed expert tensors are reported.

## Phase 2: RAM-backed performance baseline

Use the existing implementation as the control. Run batch size one, greedy decoding, an
8K sequence limit, and 256 generated tokens.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. \
python benchmarks/bench_decode_moe.py \
  --model /mnt/ssd/freetoken/ftw/Qwen3.6-35B-A3B-NVFP4 \
  --backend offload,cpu,hybrid \
  --decode 256 \
  --greedy \
  --json /mnt/ssd/freetoken/results/baseline/decode.json
```

Run each configuration at least three times in separate server processes and retain the
median. Capture startup time separately:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. \
python benchmarks/bench_load_weight_generic.py \
  --model /mnt/ssd/freetoken/models/Qwen3.6-35B-A3B-NVFP4 \
  --ftw-dir /mnt/ssd/freetoken/scratch/load-benchmark-ftw
```

### GPU-cache sweep

Measure fixed cache sizes of 256, 512, 1024, and 2048 slots, plus automatic sizing. If a
size cannot boot in 24 GiB, record it as an expected capacity result rather than silently
changing the configuration.

For each size, collect:

- Startup time
- Peak host RAM and VRAM
- Decode tokens/second and milliseconds/token
- Warm TTFT
- Token-latency p50 and p99
- GPU expert-cache hit and miss counts
- Output token hash
- Prompt and completion token counts

Use both repeated prompts and distinct prompts to separate warm expert locality from cold
routing behavior.

### Exit gate

- A reproducible RAM-backed reference result exists.
- The smallest stable GPU expert cache is known.
- Greedy outputs agree across valid backends, or any numerical divergence is documented.

## Phase 3: Instrumentation required for disk mode

Extend engine statistics and benchmark JSON with:

- Host expert-cache capacity, occupancy, hits, misses, and evictions
- GPU expert-cache hits and misses
- Disk bytes and operations read per request and per generated token
- Disk-read size distribution
- Disk-read service time and queue wait time
- Read queue depth
- Coalesced requests
- Prefetch requests, hits, late arrivals, and unused reads
- Peak pinned host memory, pageable host memory, and VRAM
- Time stalled on disk by layer and token
- Selected FTW read backend (`O_DIRECT` or `mmap`)

Metrics must be cheap enough to leave enabled during development. Expensive per-layer traces
may be placed behind a diagnostic flag.

### Exit gate

- The RAM-backed control still passes tests and shows no material performance regression.
- Cache activity and bytes moved can be reconciled from the reported statistics.

## Phase 4: FTW random-access expert index

Create a persistent descriptor for every routed expert row:

```text
(layer_id, expert_id, bank_name) -> (shard, aligned offset, logical bytes, stored bytes)
```

Requirements:

- Use FTW only for the first implementation.
- Preserve 4096-byte alignment for direct I/O.
- Validate all ranges and tensor metadata before serving.
- Keep file descriptors open for the engine lifetime.
- Avoid materializing complete expert banks during engine startup.
- Load dense and shared weights through the existing path.

Add unit tests for index construction, shard-boundary reads, short-tail alignment, corrupt
metadata, and exact expert-row round trips.

### Exit gate

- Any Qwen3.6 expert row can be fetched independently from FTW.
- Fetched bytes match the current fully resident expert-bank representation exactly.
- Engine startup no longer allocates the complete host expert bank in disk mode.

## Phase 5: Bounded pinned-host expert cache

Introduce an expert-row cache keyed by `(layer_id, expert_id)`. A cache entry owns all banks
needed to compute that expert.

Initial behavior:

- Fixed byte budget, not an unbounded entry count
- LRU eviction
- Complete expert rows as the allocation unit
- Pinned, aligned buffers suitable for existing GPU movement paths
- Request coalescing when multiple consumers request the same missing expert
- Explicit entry states: absent, loading, ready, failed, and evicting
- No eviction while an entry is referenced by an in-flight GPU copy
- Clean propagation of read errors without hanging the scheduler

Initial CLI surface:

```text
--moe-storage {ram,disk}
--moe-host-cache-gb <float>
--moe-disk-read-workers <int>
--moe-disk-prefetch-layers <int>
```

`--moe-storage ram` must preserve existing behavior. Disk mode should require an FTW
checkpoint until other formats gain a validated random-access implementation.

### Exit gate

- Peak expert-cache RAM remains within its configured budget plus documented fixed overhead.
- A forced one-entry cache can repeatedly load and evict experts without corruption.
- Concurrent requests for one entry issue one physical read.
- Read failures terminate the affected request cleanly.

## Phase 6: Synchronous disk-backed decode

Connect cache misses to the current expert-copy path using the simplest correct execution:

1. Resolve the experts selected for a layer.
2. Load missing experts synchronously into the bounded host cache.
3. Copy required experts into GPU slots.
4. Run the existing NVFP4 MoE kernel.
5. Release host-cache references.

Do not optimize prefetching yet. Decode may be very slow in this phase.

The current requirement that all host layers be permanently pinned must be relaxed only for
the new disk-cache source type. The existing RAM source path should retain its invariants.

### Correctness workload

- Batch size one
- Greedy decoding
- Fixed prompt corpus containing short chat, code, and reasoning prompts
- 16, 64, 256, and 1024 generated-token cases
- RAM and disk modes compared token-for-token
- Host-cache budgets of 0.5, 1, 2, 4, 8, and 16 GiB

### Exit gate

- Disk and RAM modes generate identical greedy token IDs for the validation corpus.
- A 4K-token soak test completes without memory growth or deadlock.
- Full routed expert banks are absent from resident host memory.
- The run makes forward progress with the smallest supported host-cache budget.
- No swap-in or swap-out activity is attributable to the benchmark.

## Phase 7: Asynchronous reads and prefetch

Once synchronous mode is correct:

- Add a bounded read queue and multiple I/O workers.
- Double-buffer staging memory.
- Overlap NVMe reads with computation and host-to-device copies.
- Prefetch one or more layers ahead when routing information is available.
- Coalesce adjacent FTW ranges where doing so reduces I/O without excessive read
  amplification.
- Cancel or deprioritize obsolete speculative reads.
- Preserve deterministic cache state transitions under shutdown and request cancellation.

Sweep:

| Variable | Values |
|---|---|
| Host cache | 0.5, 1, 2, 4, 8, 16 GiB |
| GPU slots | 256, 512, 1024, auto |
| Read workers | 1, 2, 4, 8 |
| Prefetch distance | 0, 1, 2 layers |
| Prompt pattern | repeated, alternating, unique |
| Process state | cold start, warm process |

Use `O_DIRECT` results as the primary disk measurement. Report `mmap` separately because a
warm operating-system page cache can otherwise make disk mode appear RAM-backed.

### Exit gate

- Async mode matches synchronous mode token-for-token under greedy decoding.
- p99 stalls and disk bytes/token are reported and explainable.
- Increasing queue depth or prefetch distance never violates memory budgets.
- Server shutdown and cancelled requests leave no I/O workers or pinned buffers behind.

## Phase 8: Prefill and long-context behavior

Decode correctness comes first. Then add bounded-memory prefill:

- Start with small prefill chunks and synchronous expert loading.
- Reuse the existing two-buffer overlap where compatible.
- Measure read amplification caused by many experts being selected within a chunk.
- Test 512, 2K, 4K, and 8K-token prompts before attempting longer context.
- Keep KV cache resident; disk-backed KV is out of scope.

### Exit gate

- Prefill obeys host and GPU memory limits.
- Disk mode matches RAM mode on prompt processing and generated tokens.
- TTFT and disk traffic scale predictably with prompt length.

## Phase 9: Scale to larger supported models

After Qwen3.6 is stable, validate in increasing capacity order:

1. MiniMax-M2.5 NVFP4, 229B
2. DeepSeek-V4-Flash, approximately 284B
3. GLM-5.2 NVFP4, 753B total / 40B active

Before downloading each model, estimate and then measure its non-pageable VRAM floor:

```text
resident dense/shared weights
+ embeddings and output head
+ minimum GPU expert slots
+ KV cache
+ activations and workspaces
+ CUDA allocator margin
```

Disk-backed experts cannot make a model run if this floor exceeds the RTX 3090's 24 GiB.
Use approximately 19--21 GiB as the initial safe engine budget rather than planning around
all 24 GiB.

If GLM-5.2's permanent weights exceed the budget, the next independent feature is dense
layer streaming or additional dense-weight quantization. That is not part of the initial
expert-disk implementation.

## Benchmark reporting

Every result should contain:

- Git commit and dirty-worktree status
- Model repository and exact revision
- Checkpoint format and path
- Backend and all cache settings
- Hardware/software metadata
- Workload description and random seed
- Startup time, TTFT, prefill throughput, and decode throughput
- Token-latency p50/p95/p99
- Peak VRAM and host RAM
- Host and GPU cache statistics
- Disk operations, bytes, latency, and bandwidth
- Output token hash and correctness status
- Whether the run used `O_DIRECT`, `mmap`, or resident RAM

Store machine-readable results below `/mnt/ssd/freetoken/results` and keep a small summarized
comparison table in the repository.

## Definition of the first milestone

The first milestone is complete when Qwen3.6-35B-A3B-NVFP4 generates the same greedy token
sequence in RAM-backed and NVMe-backed modes while:

- The complete expert bank is not resident in system RAM.
- Host expert memory stays within an explicit configured limit.
- VRAM remains within the RTX 3090's safe operating budget.
- A 4K-token decode soak test completes without deadlock, corruption, or memory growth.
- Reported metrics account for cache behavior and disk traffic.
- Existing RAM-backed tests and benchmarks continue to pass.

Performance optimization and larger-model validation begin only after this gate passes.
