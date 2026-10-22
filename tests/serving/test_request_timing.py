"""Independent clock/send fixtures for server stages and transport boundaries."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from sparklab.core import SamplingParams
from sparklab.message import BaseTokenizerMsg, TokenizeMsg, UserReply
from sparklab.request_timing import (
    CURRENT_HTTP,
    RequestTiming,
    RequestTimingMiddleware,
    semantic_ready,
)
from sparklab.serving.api_server import FrontendManager
from sparklab.serving.stats import StatsTracker
from sparklab.tokenizer.server import _tokenize_requests


class Clock:
    value = 10.0

    def __call__(self):
        return self.value

    def advance(self, value):
        self.value += value


def test_stage_intervals_use_actual_owners_without_inventing_missing_stages():
    collector = RequestTiming(True, trace_capacity=2)
    collector.record(
        "admitted",
        {
            "http_ingress": 10.0,
            "frontend_submit": 10.001,
            "tokenize_start": 10.02,
            "tokenize_end": 10.03,
            "scheduler_received": 10.04,
            "admitted": 11.0,
            "prompt": "must never be retained",
            "uid": 999,
        },
    )
    collector.record(
        "generated",
        {
            "admitted": 11.0,
            "first_dispatch": 11.05,
            "first_generated": 11.5,
            "detokenize_start": 11.7,
            "detokenize_end": 11.8,
            "frontend_received": 11.9,
        },
    )
    snapshot = collector.snapshot()
    assert sum(snapshot["histograms"]["tokenization"]) == 1
    assert snapshot["histograms"]["admission_queue"][9] == 1  # 960ms
    assert snapshot["histograms"]["first_generation"][8] == 1  # 450ms
    assert sum(snapshot["histograms"]["ingress_first_generated"]) == 0
    assert "prompt" not in snapshot["trace"][0]["stages"]
    assert "uid" not in snapshot["trace"][0]["stages"]


def test_histories_and_names_are_bounded_under_long_lived_traffic():
    collector = RequestTiming(True, trace_capacity=3)
    for i in range(10000):
        collector.record(
            "generated",
            {"first_dispatch": 1.0, "first_generated": 2.0, f"untrusted-{i}": i},
        )
    assert len(collector.trace) == 3
    assert len(collector.counters) == 1
    assert sum(collector.histograms["first_generation"]) == 10000
    collector.record("generated", {"first_dispatch": 2.0, "first_generated": 1.0})
    assert collector.counters["invalid_intervals"] == 1
    assert sum(collector.histograms["first_generation"]) == 10000


def test_asgi_stream_distinguishes_role_frame_content_and_backpressured_send():
    clock = Clock()
    collector = RequestTiming(True, 4, clock)
    sent = []

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body":
            clock.advance(0.25)  # send cannot finish until the transport is ready

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        clock.advance(0.1)
        await send({"type": "http.response.body", "body": b"role", "more_body": True})
        clock.advance(0.5)
        semantic_ready()
        clock.advance(0.1)
        semantic_ready()  # later deltas must not move the first-content timestamp
        await send(
            {"type": "http.response.body", "body": b"content", "more_body": False}
        )

    asyncio.run(
        RequestTimingMiddleware(app, lambda: collector)(
            {"type": "http", "method": "POST", "path": "/v1/responses"}, None, send
        )
    )
    stamps = collector.trace[0]["stages"]
    assert stamps["first_body"] == pytest.approx(10.1)
    assert stamps["first_body_sent"] == pytest.approx(10.35)
    assert stamps["semantic_ready"] == pytest.approx(10.85)
    assert stamps["http_complete"] == pytest.approx(11.2)
    assert [m.get("body") for m in sent] == [None, b"role", b"content"]
    assert collector.counters["http_complete"] == 1
    assert CURRENT_HTTP.get() is None


@pytest.mark.parametrize("status,cancel", [(400, False), (200, True)])
def test_http_errors_and_disconnects_remain_in_denominator(status, cancel):
    collector = RequestTiming(True, 3)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status})
        if cancel:
            raise asyncio.CancelledError()
        await send({"type": "http.response.body", "body": b"error"})

    async def send(message):
        pass

    async def run():
        try:
            await RequestTimingMiddleware(app, lambda: collector)(
                {"type": "http", "method": "POST", "path": "/v1/messages"}, None, send
            )
        except asyncio.CancelledError:
            assert cancel
        assert CURRENT_HTTP.get() is None

    asyncio.run(run())
    assert collector.counters["http_samples"] == 1
    assert collector.counters["http_incomplete" if cancel else "http_error"] == 1
    assert sum(collector.histograms["http_complete"]) == (0 if cancel else 1)
    assert sum(collector.histograms["semantic_ready"]) == 0


def test_disabled_measurement_adds_no_context_or_clock_calls():
    def clock():
        raise AssertionError("disabled tracing called the clock")

    collector = RequestTiming(False, clock=clock)

    async def app(scope, receive, send):
        assert CURRENT_HTTP.get() is None
        semantic_ready()

    asyncio.run(
        RequestTimingMiddleware(app, lambda: collector)(
            {"type": "http", "method": "POST", "path": "/generate"}, None, None
        )
    )
    assert not collector.counters and not collector.trace


def test_overlapping_http_tasks_keep_independent_request_contexts():
    collector = RequestTiming(True, 4)
    contexts = []

    async def app(scope, receive, send):
        context = CURRENT_HTTP.get()
        contexts.append(context)
        await asyncio.sleep(0)
        assert CURRENT_HTTP.get() is context
        semantic_ready()
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"content"})

    async def send(message):
        pass

    async def run():
        middleware = RequestTimingMiddleware(app, lambda: collector)
        await asyncio.gather(
            *(
                middleware(
                    {"type": "http", "method": "POST", "path": "/v1/responses"},
                    None,
                    send,
                )
                for _ in range(2)
            )
        )
        assert CURRENT_HTTP.get() is None

    asyncio.run(run())
    assert contexts[0][1] is not contexts[1][1]
    assert collector.counters["http_complete"] == 2
    assert sum(collector.histograms["semantic_ready"]) == 2


def test_installed_api_middleware_measures_downstream_send_backpressure(monkeypatch):
    """Exercise the real middleware stack, including its buffering HTTP wrapper."""
    from sparklab.serving import api_server

    clock = Clock()
    collector = RequestTiming(True, 4, clock)
    monkeypatch.setattr(api_server.request_ring, "record_request", lambda _: None)
    monkeypatch.setattr(
        api_server,
        "_GLOBAL_STATE",
        SimpleNamespace(
            maintenance_state="serving",
            stats=SimpleNamespace(timing=collector),
            config=SimpleNamespace(served_model_name="test"),
        ),
    )

    async def run():
        blocked, resume, never = asyncio.Event(), asyncio.Event(), asyncio.Event()
        received = False

        async def receive():
            nonlocal received
            if received:
                await never.wait()
            received = True
            return {
                "type": "http.request",
                "body": b'{"prompt":"hi","max_tokens":0}',
                "more_body": False,
            }

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                blocked.set()
                await resume.wait()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/generate",
            "raw_path": b"/generate",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 80),
        }
        task = asyncio.create_task(api_server.app(scope, receive, send))
        try:
            await asyncio.wait_for(blocked.wait(), 2.0)
            clock.advance(5.0)
            resume.set()
            await asyncio.wait_for(task, 2.0)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    stamps = collector.trace[0]["stages"]
    assert stamps["first_body"] == 10.0
    assert stamps["first_body_sent"] == stamps["http_complete"] == 15.0


def test_empty_parser_deltas_do_not_become_first_semantic_content(monkeypatch):
    from sparklab.serving import generation

    clock = Clock()
    stamps = {"http_ingress": clock()}
    collector = RequestTiming(True, clock=clock)

    async def events(*args):
        yield generation.ContentDelta("")
        clock.advance(0.5)
        yield generation.ReasoningDelta("visible reasoning channel")
        clock.advance(0.5)
        yield generation.ContentDelta("answer")

    monkeypatch.setattr(generation, "_generate_events_impl", events)

    async def run():
        token = CURRENT_HTTP.set((collector, stamps))
        try:
            observed = []
            async for event in generation.generate_events(1, None, None):
                observed.append(event)
                if len(observed) == 1:
                    assert "semantic_ready" not in stamps
            return observed
        finally:
            CURRENT_HTTP.reset(token)

    assert len(asyncio.run(run())) == 3
    assert stamps["semantic_ready"] == 10.5


def test_frontend_and_tokenizer_preserve_stages_through_message_codec(monkeypatch):
    clock = Clock()
    collector = RequestTiming(True, clock=clock)
    sent = []

    async def put(message):
        sent.append(BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(message)))

    state = FrontendManager.__new__(FrontendManager)
    state.initialized = True
    state.send_tokenizer = SimpleNamespace(put=put)

    async def submit():
        token = CURRENT_HTTP.set((collector, {"http_ingress": 10.0}))
        try:
            clock.advance(0.1)
            await state.send_one(
                TokenizeMsg(1, "synthetic", SamplingParams(max_tokens=2))
            )
        finally:
            CURRENT_HTTP.reset(token)

    asyncio.run(submit())
    clock.advance(0.1)

    def tokenize(messages):
        clock.advance(0.2)
        return [torch.tensor([1, 2, 3], dtype=torch.int32)]

    monkeypatch.setattr(
        "sparklab.tokenizer.server.time", SimpleNamespace(monotonic=clock)
    )
    ok, tensors, errors = _tokenize_requests(
        SimpleNamespace(tokenize=tokenize),
        sent,
        SimpleNamespace(warning=lambda _: None),
    )
    assert errors == [] and tensors[0].tolist() == [1, 2, 3]
    assert ok[0].timing == pytest.approx(
        {
            "http_ingress": 10.0,
            "frontend_submit": 10.1,
            "tokenize_start": 10.2,
            "tokenize_end": 10.4,
        }
    )


def test_first_output_and_admission_are_counted_once_without_changing_usage():
    tracker = StatsTracker()
    tracker.timing = RequestTiming(True)
    stamps = {
        "scheduler_received": 1.0,
        "admitted": 2.0,
        "first_dispatch": 3.0,
        "first_generated": 4.0,
        "detokenize_start": 5.0,
        "detokenize_end": 6.0,
    }
    tracker.on_new_user(1)
    tracker.observe(
        UserReply(
            1,
            "",
            False,
            prompt_tokens_delta=8,
            timing={k: v for k, v in stamps.items() if v <= 2.0},
        ),
        now=2.0,
    )
    for i in range(5):
        tracker.observe(
            UserReply(
                1,
                "output",
                i == 4,
                completion_tokens_delta=1,
                timing=stamps if i == 0 else None,
            ),
            now=7.0 + i,
        )
    assert tracker.timing.counters == {"admitted_samples": 1, "generated_samples": 1}
    assert sum(tracker.timing.histograms["frontend_transport"]) == 1
    assert tracker.prompt_tokens_total == 8 and tracker.completion_tokens_total == 5
    assert tracker.completed == 1 and tracker.active == 0
