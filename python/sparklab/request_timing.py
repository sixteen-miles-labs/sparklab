"""Bounded, opt-in native request stages; no tensor/runtime imports or I/O.

Monotonic timestamps cross processes on this single-host runtime. ASGI send
times describe server handoff, not client receipt or individual burst tokens.
"""

from collections import Counter, deque
from contextvars import ContextVar
import math
import os
import time


CURRENT_HTTP = ContextVar("sparklab_http_timing", default=None)
STAGES = frozenset(
    (
        "http_ingress",
        "frontend_submit",
        "tokenize_start",
        "tokenize_end",
        "scheduler_received",
        "admitted",
        "first_dispatch",
        "first_generated",
        "detokenize_start",
        "detokenize_end",
        "frontend_received",
        "semantic_ready",
        "first_body",
        "first_body_sent",
        "http_complete",
    )
)
PAIRS = {
    "admitted": {
        "frontend_work": ("http_ingress", "frontend_submit"),
        "tokenization_queue": ("frontend_submit", "tokenize_start"),
        "tokenization": ("tokenize_start", "tokenize_end"),
        "backend_transport": ("tokenize_end", "scheduler_received"),
        "admission_queue": ("scheduler_received", "admitted"),
    },
    "generated": {
        "dispatch_wait": ("admitted", "first_dispatch"),
        "first_generation": ("first_dispatch", "first_generated"),
        "ingress_first_generated": ("http_ingress", "first_generated"),
        "detokenization_queue": ("first_generated", "detokenize_start"),
        "detokenization": ("detokenize_start", "detokenize_end"),
        "frontend_transport": ("detokenize_end", "frontend_received"),
    },
    "http": {
        "semantic_ready": ("http_ingress", "semantic_ready"),
        "first_body": ("http_ingress", "first_body"),
        "first_body_sent": ("http_ingress", "first_body_sent"),
        "http_complete": ("http_ingress", "http_complete"),
    },
}


class RequestTiming:
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
        1.0,
        5.0,
        30.0,
        math.inf,
    )

    def __init__(self, enabled=False, trace_capacity=0, clock=time.monotonic):
        if not 0 <= trace_capacity <= 100000:
            raise ValueError("invalid request timing trace capacity")
        self.enabled = enabled
        self.clock = clock
        self.counters = Counter()
        self.histograms = {
            name: [0] * len(self.BOUNDS) for pairs in PAIRS.values() for name in pairs
        }
        self.trace = deque(maxlen=trace_capacity)

    @classmethod
    def from_env(cls):
        enabled = (
            os.getenv("SPARKLAB_RUNTIME_TIMING", "0") == "1"
            or os.getenv("SPARKLAB_SCHEDULER_POLICY", "baseline") == "latency"
        )
        return cls(enabled, int(os.getenv("SPARKLAB_RUNTIME_TRACE_CAPACITY", "0")))

    def record(self, phase, stamps):
        if not self.enabled:
            return
        if phase not in PAIRS:
            raise ValueError("unknown request timing phase")
        # Fixed names prevent accidental prompt/ID/path labels or unbounded
        # metadata. Missing stages remain missing; never fill them with zero.
        clean = {}
        for stage in STAGES:
            value = stamps.get(stage)
            if type(value) in (float, int) and math.isfinite(value) and value >= 0:
                clean[stage] = value
        self.counters[phase + "_samples"] += 1
        for name, (start, end) in PAIRS[phase].items():
            if start not in clean or end not in clean:
                continue
            seconds = clean[end] - clean[start]
            if seconds < 0:
                self.counters["invalid_intervals"] += 1
                continue
            for i, bound in enumerate(self.BOUNDS):
                if seconds <= bound:
                    self.histograms[name][i] += 1
                    break
        if self.trace.maxlen:
            self.trace.append({"phase": phase, "stages": clean})

    def snapshot(self):
        return {
            "clock": "host_monotonic_seconds",
            "counters": dict(self.counters),
            "histogram_bounds_seconds": [
                x if math.isfinite(x) else "+Inf" for x in self.BOUNDS
            ],
            "histograms": {k: list(v) for k, v in self.histograms.items()},
            "trace": list(self.trace),
        }


def semantic_ready():
    context = CURRENT_HTTP.get()
    if context is not None and "semantic_ready" not in context[1]:
        context[1]["semantic_ready"] = context[0].clock()


class RequestTimingMiddleware:
    """Pure ASGI wrapper: observes final streamed bodies and send backpressure."""

    PATHS = frozenset(
        (
            "/generate",
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
        )
    )

    def __init__(self, app, get_collector):
        self.app, self.get_collector = app, get_collector

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self.PATHS
        ):
            return await self.app(scope, receive, send)
        collector = self.get_collector()
        if collector is None or not collector.enabled:
            return await self.app(scope, receive, send)
        stamps = {"http_ingress": collector.clock()}
        token = CURRENT_HTTP.set((collector, stamps))
        status = 0

        async def timed_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            first = (
                message["type"] == "http.response.body"
                and message.get("body")
                and "first_body" not in stamps
            )
            if first:
                stamps["first_body"] = collector.clock()
            await send(message)
            if first:
                stamps["first_body_sent"] = collector.clock()
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                stamps["http_complete"] = collector.clock()

        try:
            await self.app(scope, receive, timed_send)
        finally:
            CURRENT_HTTP.reset(token)
            outcome = (
                "incomplete"
                if "http_complete" not in stamps
                else "error"
                if status >= 400
                else "complete"
            )
            collector.counters["http_" + outcome] += 1
            collector.record("http", stamps)
