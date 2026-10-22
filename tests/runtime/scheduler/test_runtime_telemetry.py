import math
import pytest
from sparklab.runtime.scheduler.telemetry import RuntimeTelemetry, Shape


def test_histogram_conservation_with_independent_known_intervals():
    metrics = RuntimeTelemetry()
    for seconds in [0.00001, 0.002, 0.015, 40]:
        metrics.observe("gpu", seconds)
    h = metrics.snapshot()["histograms"]["gpu"]
    assert sum(h) == 4
    assert h[0] == h[2] == h[4] == h[-1] == 1
    for value in [-1, math.inf, math.nan]:
        with pytest.raises(ValueError):
            metrics.observe("gpu", value)


def test_trace_and_shape_history_are_bounded_and_do_not_leak_requests():
    metrics = RuntimeTelemetry(trace_capacity=3, shape_capacity=2, clock=lambda: 1)
    for n in range(1000):
        metrics.dispatch(
            phase="prefill",
            requests=1,
            tokens=8,
            cached_tokens=2,
            queued=4,
            reason="baseline",
        )
        metrics.gpu_completed(Shape.of("prefill", 1, n + 1), 0.01)
    snapshot = metrics.snapshot()
    assert len(snapshot["trace"]) == 3 and snapshot["shape_count"] == 2
    assert snapshot["counters"]["prefill_computed_tokens"] == 8000
    assert snapshot["counters"]["prefix_reused_tokens"] == 2000
    assert snapshot["counters"]["queue_depth_peak"] == 4
    assert all(len(v) <= 128 for v in metrics.costs.values())
    assert RuntimeTelemetry().trace.maxlen == 0


def test_cost_estimate_is_not_zero_on_a_cold_shape():
    metrics = RuntimeTelemetry()
    shape = Shape.of("decode", 3, 3)
    assert metrics.estimate(shape, 0.1) == 0.1
    for ms in range(1, 11):
        metrics.gpu_completed(shape, ms / 1000)
    assert metrics.estimate(shape, 0.1) == 0.009


def test_first_rejection_positions_are_censored_and_have_finite_counter_keys():
    metrics = RuntimeTelemetry()
    for depth, accepted in [(0, 0), (3, 3), (3, 0), (7, 3)]:
        metrics.speculative_round(depth, accepted)
    counters = metrics.snapshot()["counters"]
    assert counters["adaptive_rounds"] == 4
    assert counters["adaptive_proposed"] == 13 and counters["adaptive_accepted"] == 6
    assert counters["adaptive_target_only_rounds"] == 1
    assert counters["adaptive_all_accepted_rounds"] == 1
    assert counters["adaptive_rejected_rounds"] == 2
    assert {
        k: v
        for k, v in counters.items()
        if k.startswith("adaptive_rejection_position_")
    } == {
        "adaptive_rejection_position_0": 1,
        "adaptive_rejection_position_3": 1,
    }
    for _ in range(10):
        for accepted in range(16):
            metrics.speculative_round(16, accepted)
    assert len(metrics.counters) == 22  # six totals plus 16 legal rejection positions


@pytest.mark.parametrize(
    "depth,accepted", [(-1, 0), (3, 4), (17, 1), (True, 0), (3, False)]
)
def test_invalid_rejection_observation_cannot_partially_update_counters(
    depth, accepted
):
    metrics = RuntimeTelemetry()
    with pytest.raises(ValueError):
        metrics.speculative_round(depth, accepted)
    assert not metrics.counters


def test_invalid_dispatch_ages_do_not_partially_record_a_dispatch():
    for output_age, queue_age in [(math.nan, 1.0), (1.0, -1.0)]:
        metrics = RuntimeTelemetry()
        with pytest.raises(ValueError, match="dispatch age"):
            metrics.dispatch(
                phase="decode",
                requests=1,
                tokens=1,
                cached_tokens=0,
                queued=0,
                reason="decode_only",
                output_age=output_age,
                queue_age=queue_age,
            )
        assert not metrics.counters and all(
            not any(h) for h in metrics.histograms.values()
        )
