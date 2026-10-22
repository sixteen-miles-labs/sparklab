"""Small bounded native prefill/decode selector. Resource admission stays unchanged."""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Decision:
    phase: str
    token_budget: int
    reason: str


class SchedulingPolicy:
    def __init__(
        self, mode="baseline", *, delivery_budget=0.1, prefill_service_bound=1.0
    ):
        if mode not in ("baseline", "latency"):
            raise ValueError("scheduler policy must be baseline or latency")
        if not all(
            math.isfinite(v) and v > 0 for v in (delivery_budget, prefill_service_bound)
        ):
            raise ValueError("service bounds must be positive and finite")
        self.mode = mode
        self.delivery_budget = delivery_budget
        self.prefill_service_bound = prefill_service_bound
        self.last_phase = None
        self.last_prefill = None

    def choose(
        self,
        *,
        now,
        pending,
        decoders,
        oldest_queue_age,
        output_age,
        prefill_budget,
        alignment=1,
        prefill_seconds=0.25,
        decode_seconds=0.05,
        atomic_prompt=False,
    ):
        if not pending:
            return Decision(
                "decode" if decoders else "idle",
                0,
                "decode_only" if decoders else "idle",
            )
        if self.mode == "baseline" or not decoders:
            return Decision(
                "prefill",
                prefill_budget,
                "baseline" if self.mode == "baseline" else "prefill_only",
            )
        # Never chain prefills while decode is runnable. This bound applies even
        # when an old pending request remains a multi-chunk continuation.
        if self.last_phase == "prefill":
            return Decision("decode", 0, "decoder_service")
        age = (
            min(oldest_queue_age, now - self.last_prefill)
            if self.last_prefill is not None
            else oldest_queue_age
        )
        if age >= self.prefill_service_bound:
            phase, reason = "prefill", "prefill_service"
        elif output_age + prefill_seconds + decode_seconds > self.delivery_budget:
            phase, reason = "decode", "output_age"
        else:
            phase, reason = "prefill", "fits_delivery_budget"
        if phase == "decode":
            return Decision(phase, 0, reason)
        budget = prefill_budget
        if not atomic_prompt:
            # Conservative linear interpolation from an observed chunk cost. A
            # legal operation can exceed the desired latency; never promise a
            # deadline shorter than that operation or break cache alignment.
            slack = max(self.delivery_budget - output_age - decode_seconds, 0)
            budget = max(
                alignment,
                int(prefill_budget * min(1.0, slack / max(prefill_seconds, 1e-6))),
            )
            budget = min(
                prefill_budget, max(alignment, budget // alignment * alignment)
            )
        return Decision("prefill", budget, reason)

    def dispatched(self, phase, now):
        self.last_phase = "prefill" if phase == "prefill" else "decode"
        if phase == "prefill":
            self.last_prefill = now

    def choose_coordinated(self, *, tp_info=None, cpu_group=None, **inputs):
        """Make latency choices authoritative on rank zero before model dispatch.

        Queue/cache consistency remains the scheduler's existing TP invariant.
        The vote detects incompatible runnable envelopes, not request identity.
        """
        if self.mode != "latency" or tp_info is None or tp_info.size == 1:
            decision = self.choose(**inputs)
            return decision, (
                max(0.001, self.delivery_budget - inputs["output_age"])
                if self.mode == "latency"
                else None
            )
        if cpu_group is None:
            raise RuntimeError("TP latency scheduling requires the CPU process group")
        import torch

        phases = ("idle", "prefill", "decode")
        reasons = (
            "idle",
            "decode_only",
            "baseline",
            "prefill_only",
            "decoder_service",
            "prefill_service",
            "output_age",
            "fits_delivery_budget",
        )
        packet = torch.zeros(5, dtype=torch.float64, device="cpu")
        source_error = None
        if tp_info.is_primary():
            try:
                decision = self.choose(**inputs)
                packet[:] = torch.tensor(
                    [
                        1,
                        phases.index(decision.phase),
                        decision.token_budget,
                        reasons.index(decision.reason),
                        max(0.001, self.delivery_budget - inputs["output_age"]),
                    ],
                    dtype=torch.float64,
                    device="cpu",
                )
            except Exception as exc:
                source_error = exc
                packet[0] = -1
        cpu_group.broadcast(packet, root=0).wait()
        if packet[0].item() != 1:
            raise RuntimeError(
                "Primary TP scheduler could not select work"
            ) from source_error
        valid = False
        try:
            _, phase, budget, reason, service = packet.tolist()
            if not all(math.isfinite(v) for v in (phase, budget, reason, service)):
                raise ValueError("nonfinite scheduling packet")
            if not all(v == int(v) for v in (phase, budget, reason)):
                raise ValueError("nonintegral scheduling packet")
            if not (0 <= phase < len(phases) and 0 <= reason < len(reasons)):
                raise ValueError("unknown scheduling packet")
            decision = Decision(phases[int(phase)], int(budget), reasons[int(reason)])
            valid = service > 0
            if decision.phase == "prefill":
                valid &= (
                    bool(inputs["pending"]) and 0 < budget <= inputs["prefill_budget"]
                )
                if inputs.get("atomic_prompt", False):
                    valid &= budget == inputs["prefill_budget"]
                elif budget != inputs["prefill_budget"]:
                    valid &= budget % inputs.get("alignment", 1) == 0
            elif decision.phase == "decode":
                valid &= bool(inputs["decoders"]) and budget == 0
            else:
                valid &= (
                    not inputs["pending"] and not inputs["decoders"] and budget == 0
                )
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            valid = False
        vote = torch.tensor(int(valid), dtype=torch.int64, device="cpu")
        cpu_group.allreduce(vote).wait()
        if vote.item() != tp_info.size:
            raise RuntimeError(
                "TP scheduler decision disagrees with a rank's runnable envelope"
            )
        return decision, service
