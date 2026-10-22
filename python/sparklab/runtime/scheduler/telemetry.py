"""Bounded native scheduling measurements. No device synchronization or I/O here."""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class Shape:
    phase: str
    requests: int
    tokens: int

    @classmethod
    def of(cls, phase, requests, tokens):
        # Only finite physical-shape buckets become metric labels.
        return cls(
            phase, max(1, requests), 1 << max(0, (max(tokens, 1) - 1).bit_length())
        )


class RuntimeTelemetry:
    """Fixed-size totals/histograms and optional bounded diagnostic ring.

    `output_ready` is the scheduler's delivery to detokenization, NOT HTTP ingress
    or client-visible delivery. Those are recorded by their actual owners.
    """

    BOUNDS = (
        0.0001,
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        5,
        30,
        math.inf,
    )
    METRICS = (
        "host_submit",
        "gpu",
        "queue_wait",
        "active_output_age",
        "oldest_queue_age",
        "output_ready_gap",
        "speculative_round_gpu",
        "speculative_proposal_gpu",
        "speculative_verification_gpu",
        "speculative_commit_gpu",
        "speculative_handoff_gpu",
    )

    def __init__(self, trace_capacity=0, shape_capacity=64, clock=time.monotonic):
        if not 0 <= trace_capacity <= 100000 or not 1 <= shape_capacity <= 256:
            raise ValueError("invalid telemetry bounds")
        self.clock = clock
        self.counters = Counter()
        self.histograms = {key: [0] * len(self.BOUNDS) for key in self.METRICS}
        self.costs = OrderedDict()
        self.shape_capacity = shape_capacity
        self.trace = deque(maxlen=trace_capacity)

    def observe(self, name, seconds):
        if name not in self.histograms or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid timing observation")
        for i, bound in enumerate(self.BOUNDS):
            if seconds <= bound:
                self.histograms[name][i] += 1
                break

    def dispatch(
        self,
        *,
        phase,
        requests,
        tokens,
        cached_tokens,
        queued,
        reason,
        queued_before=None,
        selection_at=None,
        output_age=None,
        queue_age=None,
        policy_reason=None,
        token_budget=None,
        active_requests=None,
    ):
        if phase not in ("prefill", "decode", "verify"):
            raise ValueError("invalid dispatch phase")
        if any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in (output_age, queue_age)
        ):
            raise ValueError("invalid dispatch age")
        if queued_before is None:
            queued_before = queued
        for name, value in (
            ("active_output_age", output_age),
            ("oldest_queue_age", queue_age),
        ):
            if value is not None:
                self.observe(name, value)
        self.counters[phase + "_dispatches"] += 1
        self.counters["prefill_computed_tokens"] += tokens if phase == "prefill" else 0
        self.counters["prefix_reused_tokens"] += cached_tokens
        self.counters["queue_depth_peak"] = max(
            self.counters["queue_depth_peak"], queued_before, queued
        )
        if reason == "prefill_unavailable":
            self.counters["prefill_unavailable_dispatches"] += 1
        if self.trace.maxlen:
            self.trace.append(
                {
                    "time": self.clock(),
                    "phase": phase,
                    "requests": requests,
                    "tokens": tokens,
                    "queued": queued,
                    "reason": reason,
                    "policy_reason": policy_reason
                    if policy_reason is not None
                    else reason,
                    "queued_before": queued_before,
                    "selection_at": selection_at,
                    "active_output_age_seconds": output_age,
                    "oldest_queue_age_seconds": queue_age,
                    "token_budget": token_budget,
                    "active_requests": active_requests,
                }
            )

    def speculative_round(self, proposed, accepted):
        """Only the first observed rejection has a position; deeper rows are censored."""
        if (
            type(proposed) is not int
            or type(accepted) is not int
            or not 0 <= accepted <= proposed <= 16
        ):
            raise ValueError("invalid speculative counter observation")
        self.counters["adaptive_rounds"] += 1
        self.counters["adaptive_proposed"] += proposed
        self.counters["adaptive_accepted"] += accepted
        self.counters["adaptive_target_only_rounds"] += int(proposed == 0)
        if accepted < proposed:
            self.counters["adaptive_rejected_rounds"] += 1
            self.counters["adaptive_rejection_position_" + str(accepted)] += 1
        elif proposed:
            self.counters["adaptive_all_accepted_rounds"] += 1

    def gpu_completed(self, shape, seconds):
        self.observe("gpu", seconds)
        if shape not in self.costs:
            if len(self.costs) == self.shape_capacity:
                self.costs.popitem(last=False)
            self.costs[shape] = deque(maxlen=128)
        self.costs.move_to_end(shape)
        self.costs[shape].append(seconds)

    def estimate(self, shape, fallback):
        """Conservative p90 of observed complete dispatch costs, or nonzero cold cost."""
        values = self.costs.get(shape)
        if not values:
            return fallback
        ordered = sorted(values)
        return ordered[math.ceil(0.9 * len(ordered)) - 1]

    def snapshot(self, *, include_trace=True):
        return {
            "counters": dict(self.counters),
            "histogram_bounds_seconds": [
                b if math.isfinite(b) else "+Inf" for b in self.BOUNDS
            ],
            "histograms": {k: list(v) for k, v in self.histograms.items()},
            "trace": list(self.trace) if include_trace else [],
            "shape_count": len(self.costs),
            "gpu_costs": [
                {
                    "phase": shape.phase,
                    "requests": shape.requests,
                    "tokens": shape.tokens,
                    "samples": len(values),
                    "p90_seconds": self.estimate(shape, None),
                }
                for shape, values in self.costs.items()
            ],
        }


# Only these host scalar fields enter the periodic IPC export. GPU counters are
# cached from the existing terminal observation, never read/synchronized here.
EXPERT_DECODE_FIELDS = (
    "active",
    "missing",
    "fetched",
    "calls",
    "layer_calls",
    "miss_rate",
    "prefill_hit_rows",
    "prefill_rows",
    "sparse_prefill_layers",
    "sparse_prefill_routes",
    "sparse_prefill_unique_rows",
    "sparse_prefill_fallback_layers",
    "shared_expert_overlap_calls",
)
EXPERT_DISK_FIELDS = (
    "cache_capacity_entries",
    "cache_capacity_bytes",
    "cache_allocated_bytes",
    "staging_allocated_bytes",
    "host_allocated_bytes",
    "cache_occupancy_entries",
    "cache_occupancy_bytes",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "cache_bypasses",
    "read_ops",
    "logical_bytes",
    "physical_bytes",
    "read_seconds",
    "read_workers",
    "staging_buffers",
)


def host_scalars(values, names):
    """No tensor conversion, nested traversal, dynamic labels or user strings."""
    return {
        name: value
        for name in names
        if type(value := values.get(name)) in (int, float) and math.isfinite(value)
    }


class ExpertCounterSampler:
    """One ordered, nonblocking device-to-host scalar sample at most per second.

    Only the existing offload cache's decode counters are read. No cache/weight
    owner is retained, so an idle cache rebuild may replace its source tensors.
    CUDA graph counter writes and this sample share the engine stream.
    """

    def __init__(self, cache):
        import torch
        from flashlib.kernels.slot_cache import N_STATS, Stat

        if torch.device(cache.device).type != "cuda":
            raise ValueError("live expert counter sampling requires CUDA")
        if N_STATS != 3 or (Stat.ACTIVE, Stat.MISS, Stat.CALLS) != (0, 1, 2):
            raise ValueError("unqualified expert counter layout")
        self.device = torch.empty(4, dtype=torch.int64, device=cache.device)
        self.parts = tuple(self.device[i] for i in range(4))
        self.host = torch.empty(4, dtype=torch.int64, pin_memory=True)
        self.event = torch.cuda.Event()
        self.event.record(torch.cuda.current_stream(cache.device))
        self.pending = None
        self.last_submit = None
        self.latest = None

    def poll(self, cache, stream, now):
        import torch

        if self.pending is not None and self.event.query():
            submitted_at, counters = self.pending
            active, missing, calls, fetched = self.host.tolist()
            counters.update(
                active=active,
                missing=missing,
                calls=calls,
                layer_calls=calls,
                fetched=fetched,
                miss_rate=missing / active if active else 0.0,
            )
            self.latest = {"observed_at": submitted_at, "counters": counters}
            self.pending = None
        if self.pending is None and (
            self.last_submit is None or now - self.last_submit >= 1.0
        ):
            # Host fields describe this submission, not a later poll. Reusing
            # the pinned window is legal only after the prior copy event passed.
            counters = host_scalars(vars(cache), EXPERT_DECODE_FIELDS)
            # The existing cache calls this field prefill_total_rows internally.
            counters.update(
                host_scalars(
                    {"prefill_rows": cache.prefill_total_rows},
                    EXPERT_DECODE_FIELDS,
                )
            )
            with torch.inference_mode(), torch.cuda.stream(stream):
                if cache.decode_target == "hybrid":
                    for destination, source in zip(
                        self.parts[:3],
                        (
                            cache.stat_active,
                            cache.stat_missing,
                            cache.stat_calls,
                        ),
                    ):
                        destination.copy_(source, non_blocking=True)
                else:
                    torch.sum(cache.lru_stats, dim=0, out=self.device[:3])
                self.parts[3].copy_(cache.stat_fetched, non_blocking=True)
                self.host.copy_(self.device, non_blocking=True)
                self.event.record(stream)
            self.last_submit = now
            self.pending = (now, counters)
        return self.latest


def allocator_snapshot(device):
    """One host allocator observation, excluding driver and non-PyTorch owners.

    Allocated bytes are contained in reserved bytes; they must not be summed.
    Missing allocator statistics remain unknown rather than implying zero use.
    """
    if device is None or device.type != "cuda":
        return {"gpu_allocated_bytes": None, "gpu_reserved_bytes": None}
    import torch

    stats = torch.cuda.memory_stats(device)
    return {
        "gpu_allocated_bytes": stats.get("allocated_bytes.all.current"),
        "gpu_reserved_bytes": stats.get("reserved_bytes.all.current"),
    }


def workspace_snapshot(engine, expert_sampler=None):
    """Named retained buffers from shape metadata, not a total allocator census."""
    pool = getattr(engine, "linear_state_pool", None)

    def nbytes(tensor):
        return tensor.numel() * tensor.element_size() if tensor is not None else 0

    return {
        "expert_counter_sampling": (
            {
                "device_bytes": nbytes(expert_sampler.device),
                "pinned_host_bytes": nbytes(expert_sampler.host),
                "pending": expert_sampler.pending is not None,
            }
            if expert_sampler is not None
            else None
        ),
        "prefix_reserved_bytes": getattr(engine, "_prefix_reserved_bytes", None),
        "adaptive_graph_reserved_bytes": getattr(engine, "adaptive_graph_bytes", None),
        "verification_transactions": (
            {
                name: nbytes(getattr(pool, name, None))
                for name in (
                    "verify_recurrent_states",
                    "verify_conv_inputs",
                    "verify_state_indices",
                )
            }
            if pool is not None
            else None
        ),
        "verification_auxiliary_bytes": (
            sum(nbytes(value) for value in pool._verify_aux_inputs.values())
            if pool is not None
            else None
        ),
    }
