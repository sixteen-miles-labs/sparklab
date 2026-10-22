# Native workspace investigation

The selected complete-artifact control is the native RadixArk Qwen3.8-27B artifact
`1ad54cc34ff10c49`, local DFlash2 draft revision `bd7a934213c4`, W4A16 prefill,
Triton attention/NVFP4, BF16 KV/compute and FP32 recurrent state. This inventory
uses 8192 cache tokens, 4096 context, batch one, page size 16 and 512-token prefill
chunks. Cache capacities and precision are unchanged between observations.
Other live services share the GB10. These are allocation measurements, not
end-to-end timing or portfolio qualification.

The implemented command is:

```bash
PYTHONPATH=python python benchmarks/profile_runtime_memory.py \
  --model "$MODEL_PATH" --draft "$DRAFT_PATH" --operator-profile \
  --output "$HOME/results/flare-runtime/m6-memory-operators.json"
```

Raw evidence is outside git: `m6-memory-profile.json`,
`m6-memory-operators.json` and their logs. Both processes completed. The operator
profile includes one actual cold 512-token prefill and warm execution of every
prepared depth. Profiler warnings about pre-existing allocations mean its
allocation totals are traffic counters, not live-storage totals. CUDA allocator
segments and the physical transaction tensors establish the retained inventory.

| Workspace | Measured bytes |
| --- | ---: |
| Per-prefix recurrent verification intermediates | 1,811,939,328 |
| Per-prefix convolution inputs | 11,796,480 |
| Verification index | 4 |
| Incremental graph reservation, decode plus three verify widths | 98,566,144 |
| Largest verify input/logit tensor set (12 rows) | 11,919,517 |
| Separate private pools: decode / 12 / 8 / 4 rows | 2,097,152 / 25,165,824 / 25,165,824 / 23,068,672 |

Graph private pools are counted using their allocator pool identities and reserved
segments, even when `allocated_size` is zero after capture. Counting only live
Python tensors would understate retained graph storage. Startup and all-depth
observations preserve the same pool sizes. After the 512-token prefill, allocated
storage rises from 25,865,263,616 to 25,890,917,888 bytes; peak allocated storage
is 26,089,945,088 bytes. Much larger cumulative allocation traffic (for example,
24,917,221,376 bytes attributed to `aten::empty`) is reused by the existing
stream-aware allocator; it is not simultaneously retained workspace.

The candidate investigated before changing allocation layout was sharing private
workspace across sequential graph widths. Those graphs run on one engine stream,
but graph inputs/logits, target/draft context, accepted-prefix transaction arrays
and asynchronous owners must retain their own valid lifetimes. Existing decode
batch variants already share a CUDA graph pool. No general allocator is needed.

The per-layer/per-prefix transaction states remain live until the full target
round determines the accepted endpoint. Aliasing their layers or prefix steps
would destroy rollback state. Their 1,823,735,812 bytes are round intermediates,
not model weights, KV capacity, or the live/snapshot GDN cache pool.

Even an unrealistically optimistic removal of every incremental graph byte except
the largest input/logit tensor set saves at most 86,646,627 bytes. That is **4.51%**
of the measured fixed-workspace lower bound (transactions plus graph reservation),
below the preselected 5% materiality gate. A realistic shared-pool saving is
smaller, and including other fixed workspace only lowers the percentage. No
scratch allocation layout was changed. M6 closes without an optimization for this
control profile; this is not a claim that larger chunks or other models lack
opportunities. No cache-size reduction, precision change, or allocator cleanup is
counted as a saving, and no scratch-reuse speed or latency claim is made.
