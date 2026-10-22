"""Request-local depth choice from censored acceptance and whole-round service.

This pure controller does not enable adaptive execution. Its caller must declare
prepared depths, prove target/draft transitions and supply complete round costs.
Fixed execution remains the default until that integration is qualified.
"""

from collections import deque
import math


class SpeculationPolicy:
    def __init__(
        self,
        prepared_depths,
        fixed_depth,
        *,
        source="dflash2",
        window=64,
        refresh_interval=16,
        hysteresis=0.05,
    ):
        depths = tuple(sorted(set(prepared_depths)))
        if (
            not depths
            or 0 not in depths
            or fixed_depth not in depths
            or any(type(d) is not int or not 0 <= d <= 16 for d in depths)
            or source != "dflash2"
            or not 4 <= window <= 128
            or not 4 <= refresh_interval <= 128
            or not math.isfinite(hysteresis)
            or not 0 <= hysteresis <= 0.5
        ):
            raise ValueError("invalid prepared speculation policy")
        self.depths = depths
        self.fixed_depth = fixed_depth
        self.source = source
        self.hysteresis = hysteresis
        self.refresh_interval = refresh_interval
        self.acceptance = [deque(maxlen=window) for _ in range(max(depths))]
        self.costs = {depth: deque(maxlen=window) for depth in depths}
        self.rounds = 0
        self.last_depth = fixed_depth
        self.probe_cursor = 0
        self._last_probe_round = -1

    def observe(self, *, proposed, accepted, seconds, source):
        # Validate atomically: an invalid measurement must not train acceptance.
        if (
            type(proposed) is not int
            or proposed not in self.costs
            or type(accepted) is not int
            or not 0 <= accepted <= proposed
            or source != self.source
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("invalid complete speculative round observation")
        for position in range(accepted):
            self.acceptance[position].append(True)
        if accepted < proposed:
            self.acceptance[accepted].append(False)
        # Positions after the first rejection were never observed. A shallower
        # all-accept round supplies no evidence about an unproposed next token.
        self.costs[proposed].append(seconds)
        self.rounds += 1

    def expected_progress(self, depth):
        if depth not in self.costs:
            raise ValueError("unprepared speculative depth")
        progress, reach = 1.0, 1.0
        for samples in self.acceptance[:depth]:
            reach *= (1.0 + sum(samples)) / (2.0 + len(samples))
            progress += reach
        return progress

    def service_cost(self, depth):
        values = self.costs[depth]
        if not values:
            return None
        ordered = sorted(values)
        return ordered[math.ceil(0.9 * len(ordered)) - 1]

    def choose(
        self,
        *,
        output_remaining,
        context_remaining,
        allow_speculation=True,
        service_budget=None,
        fixed=False,
    ):
        if (
            type(output_remaining) is not int
            or type(context_remaining) is not int
            or min(output_remaining, context_remaining) < 0
            or (
                service_budget is not None
                and (not math.isfinite(service_budget) or service_budget <= 0)
            )
        ):
            raise ValueError("invalid speculative execution envelope")
        maximum = min(output_remaining, context_remaining) - 1
        legal = [d for d in self.depths if d <= maximum]
        if not legal or not allow_speculation:
            return 0
        if fixed:
            return max(d for d in legal if d <= self.fixed_depth)
        known = {d: self.service_cost(d) for d in legal}
        measured = {d: cost for d, cost in known.items() if cost is not None}
        if service_budget is not None:
            measured = {
                d: cost for d, cost in measured.items() if cost <= service_budget
            }
        if not measured:
            # No measured cost is ever interpreted as zero. Use the existing
            # deployment control for startup, or target-only under a service cap.
            return (
                0
                if service_budget is not None
                else max(d for d in legal if d <= self.fixed_depth)
            )
        # One bounded refresh probe; repeated choose calls without an observation
        # cannot continually force exploration. Unknown costs use a conservative
        # upper estimate from observed whole rounds and never bypass a service cap.
        if (
            self.rounds >= 4
            and self.rounds % self.refresh_interval == 0
            and self._last_probe_round != self.rounds
        ):
            missing = [d for d in legal if not self.costs[d]]
            # Visit each unknown legal depth before refreshing an observed one.
            # Applying a rotating cursor to a changing combined list can keep
            # skipping a missing depth as other entries become measured.
            probe = missing[0] if missing else legal[self.probe_cursor % len(legal)]
            largest = max(measured)
            estimate = known[probe]
            if estimate is None:
                estimate = max(measured.values()) * max(
                    1.0, (probe + 1) / (largest + 1)
                )
            if not missing:
                self.probe_cursor += 1
            self._last_probe_round = self.rounds
            if service_budget is None or estimate <= service_budget:
                self.last_depth = probe
                return probe
        score = {d: self.expected_progress(d) / cost for d, cost in measured.items()}
        best = max(score, key=lambda d: (score[d], -d))
        current = self.last_depth
        if current in score and score[best] <= score[current] * (1.0 + self.hysteresis):
            best = current
        self.last_depth = best
        return best
