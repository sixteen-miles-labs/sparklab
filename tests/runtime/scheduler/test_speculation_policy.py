"""Independent acceptance chains and cost/control constraints for depth selection."""

import math
import pytest

from sparklab.runtime.scheduler.speculation import SpeculationPolicy


def observe(policy, depth, accepted, seconds, n=1):
    for _ in range(n):
        policy.observe(
            proposed=depth, accepted=accepted, seconds=seconds, source="dflash2"
        )


def choose(policy, **kwargs):
    return policy.choose(output_remaining=100, context_remaining=100, **kwargs)


def test_first_rejection_censors_deeper_tokens_and_all_accept_censors_next():
    policy = SpeculationPolicy((0, 2, 4), 4)
    observe(policy, 4, 2, 0.4)
    assert [tuple(values) for values in policy.acceptance] == [
        (True,),
        (True,),
        (False,),
        (),
    ]
    observe(policy, 2, 2, 0.2)
    assert [len(values) for values in policy.acceptance] == [2, 2, 1, 0]
    # Independent conditional-chain expectation with Beta(1,1) priors.
    probabilities = [3 / 4, 3 / 4, 1 / 3, 1 / 2]
    reach, expected = 1.0, 1.0
    for probability in probabilities:
        reach *= probability
        expected += reach
    assert policy.expected_progress(4) == pytest.approx(expected)


@pytest.mark.parametrize(
    "accepted, costs, optimum",
    [
        (4, [1.0, 1.1, 1.2], 4),
        (0, [1.0, 2.0, 3.0], 0),
        (2, [1.0, 1.2, 4.0], 2),
    ],
)
def test_measured_optimum_for_all_accept_first_reject_and_late_reject(
    accepted, costs, optimum
):
    policy = SpeculationPolicy((0, 2, 4), 4, refresh_interval=128)
    for depth, cost in zip((0, 2, 4), costs):
        observe(policy, depth, min(accepted, depth), cost, n=20)
    assert choose(policy) == optimum
    assert choose(policy, fixed=True) == 4


def test_depth_limits_service_bound_and_stop_or_grammar_restriction():
    policy = SpeculationPolicy((0, 2, 4), 4)
    observe(policy, 0, 0, 0.1)
    observe(policy, 2, 2, 0.2)
    observe(policy, 4, 4, 0.4)
    assert policy.choose(output_remaining=3, context_remaining=100, fixed=True) == 2
    assert policy.choose(output_remaining=100, context_remaining=2) == 0
    assert choose(policy, allow_speculation=False) == 0
    assert choose(policy, service_budget=0.15) == 0
    assert policy.choose(output_remaining=0, context_remaining=10) == 0


def test_unknown_costs_are_not_free_and_refresh_is_bounded():
    policy = SpeculationPolicy((0, 2, 4), 4, refresh_interval=4)
    assert choose(policy) == 4
    assert choose(policy, service_budget=0.1) == 0
    observe(policy, 4, 4, 1.0, n=4)
    assert choose(policy) == 0  # explicit bounded calibration probe
    assert choose(policy) == 4  # no new observation => no repeated exploration
    assert policy.service_cost(0) is None
    observe(policy, 0, 0, 2.0)
    assert choose(policy) == 4


def test_refresh_visits_every_unknown_depth_before_repeating_a_measured_one():
    policy = SpeculationPolicy((0, 2, 4), 4, refresh_interval=4)
    observe(policy, 4, 4, 1.0, n=4)
    assert choose(policy) == 0
    observe(policy, 0, 0, 0.8)
    observe(policy, 4, 4, 1.0, n=3)
    assert choose(policy) == 2
    observe(policy, 2, 2, 0.9)
    assert all(policy.service_cost(depth) is not None for depth in (0, 2, 4))


def test_hysteresis_avoids_small_gain_and_history_is_bounded_and_changes():
    policy = SpeculationPolicy(
        (0, 2), 2, window=4, refresh_interval=128, hysteresis=0.1
    )
    observe(policy, 2, 0, 1.1, n=4)
    observe(policy, 0, 0, 1.0)
    # Conditional prior slightly offsets service cost; no noisy depth oscillation.
    assert choose(policy) == 2
    observe(policy, 2, 0, 4.0, n=8)
    assert choose(policy) == 0
    assert len(policy.acceptance[0]) == len(policy.costs[2]) == 4
    assert len(policy.acceptance[1]) == 0  # repeated unobserved is still censored
    observe(policy, 2, 2, 0.1, n=8)
    assert choose(policy) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"accepted": 5},
        {"proposed": 3},
        {"seconds": 0.0},
        {"seconds": math.nan},
        {"source": "unknown"},
        {"accepted": True},
    ],
)
def test_invalid_round_is_rejected_before_any_training(changes):
    policy = SpeculationPolicy((0, 2, 4), 4)
    observation = dict(proposed=4, accepted=2, seconds=0.1, source="dflash2")
    with pytest.raises(ValueError):
        policy.observe(**{**observation, **changes})
    assert policy.rounds == 0 and not any(policy.acceptance)
    assert not any(policy.costs.values())
