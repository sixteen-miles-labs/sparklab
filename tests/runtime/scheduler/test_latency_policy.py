"""Independent event clocks verify finite service under two competing queues."""

from collections import deque
import pytest
from sparklab.runtime.scheduler.policy import SchedulingPolicy


def test_continuous_prompt_and_decode_load_have_bounded_dispatch_service():
    policy = SchedulingPolicy(
        "latency", delivery_budget=0.04, prefill_service_bound=0.2
    )
    now = 0.0
    last_output = 0.0
    queue = deque([0.0])
    next_arrival = 0.01
    prefill_times, delivery_times = [], []
    # Infinite decode and prompt arrivals: assert service opportunities, not an
    # impossible finite completion/queue bound when arrivals exceed capacity.
    for _ in range(2000):
        while next_arrival <= now:
            queue.append(next_arrival)
            next_arrival += 0.01
        decision = policy.choose(
            now=now,
            pending=bool(queue),
            decoders=True,
            oldest_queue_age=now - queue[0],
            output_age=now - last_output,
            prefill_budget=512,
            alignment=16,
            prefill_seconds=0.0512,
            decode_seconds=0.005,
        )
        assert decision.token_budget <= 512
        policy.dispatched(decision.phase, now)
        if decision.phase == "prefill":
            prefill_times.append(now)
            queue.popleft()
            # Independent execution model: .1ms per token, 5ms per decode.
            now += decision.token_budget / 10000
            assert decision.token_budget % 16 == 0 and decision.token_budget > 0
        else:
            now += 0.005
            last_output = now
            delivery_times.append(now)
    assert len(prefill_times) > 20 and len(delivery_times) > 20
    assert max(b - a for a, b in zip(prefill_times, prefill_times[1:])) <= 0.2051
    assert max(b - a for a, b in zip(delivery_times, delivery_times[1:])) <= 0.0401


def test_long_prefill_is_interleaved_with_decode_without_losing_work():
    policy = SchedulingPolicy("latency", delivery_budget=0.1, prefill_service_bound=0.1)
    now, last_output, remaining, outputs = 0.0, 0.0, 32768, 0
    while remaining:
        decision = policy.choose(
            now=now,
            pending=True,
            decoders=True,
            oldest_queue_age=now,
            output_age=now - last_output,
            prefill_budget=min(512, remaining),
            alignment=64,
            prefill_seconds=0.25,
            decode_seconds=0.02,
        )
        policy.dispatched(decision.phase, now)
        if decision.phase == "prefill":
            remaining -= decision.token_budget
            now += decision.token_budget / 2048
        else:
            now += 0.02
            last_output = now
            outputs += 1
        if remaining > 0 and now > 1:
            assert outputs > 0
    assert outputs > 100


@pytest.mark.parametrize("mode", ["baseline", "latency"])
def test_empty_single_queue_and_atomic_prompts_preserve_legal_budget(mode):
    policy = SchedulingPolicy(mode)
    common = dict(
        now=2, oldest_queue_age=2, output_age=0, prefill_budget=63, alignment=64
    )
    assert policy.choose(pending=False, decoders=False, **common).phase == "idle"
    assert policy.choose(pending=False, decoders=True, **common).phase == "decode"
    assert policy.choose(pending=True, decoders=False, **common).token_budget == 63
    assert (
        policy.choose(
            pending=True, decoders=True, atomic_prompt=True, **common
        ).token_budget
        == 63
    )


def test_baseline_is_unchanged_prefill_first_even_under_pressure():
    policy = SchedulingPolicy()
    for _ in range(10):
        decision = policy.choose(
            now=10,
            pending=True,
            decoders=True,
            oldest_queue_age=10,
            output_age=10,
            prefill_budget=8192,
        )
        assert decision.phase == "prefill" and decision.token_budget == 8192
        policy.dispatched(decision.phase, 10)
