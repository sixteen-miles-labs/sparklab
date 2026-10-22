"""Internal preparation envelope for the native Qwen DFlash2 experiment."""

from dataclasses import dataclass
import os
import time

from sparklab.runtime.scheduler.speculation import SpeculationPolicy

GRAPH_RESERVE_BYTES = 2 << 30


@dataclass(frozen=True)
class AdaptiveDFlash:
    depths: tuple[int, ...]
    fixed_depth: int
    override: int | None

    @classmethod
    def from_env(cls, config):
        mode = os.getenv("SPARKLAB_DFLASH_POLICY", "fixed")
        if mode == "fixed":
            return None
        if mode != "cost":
            raise ValueError("SPARKLAB_DFLASH_POLICY must be fixed or cost")
        maximum = int(config.speculative_tokens) - 1
        if (
            config.speculative_method != "dflash2"
            or maximum < 3
            or config.tp_info.size != 1
            or config.max_running_req != 1
            or config.attention_backend != "triton"
            or not getattr(config.model_config, "mtp_cuda_graph", False)
            or config.model_config.model_type not in {"qwen3_5", "qwen3_5_moe"}
        ):
            raise ValueError(
                "cost-aware DFlash requires graph-prepared Qwen DFlash2, TP=1, batch=1, Triton and block>=4"
            )
        raw = os.getenv("SPARKLAB_DFLASH_DEPTHS")
        depths = (
            tuple(sorted(set(map(int, raw.split(",")))))
            if raw
            else tuple(sorted({0, min(3, maximum), min(7, maximum), maximum}))
        )
        if (
            not 2 <= len(depths) <= 4
            or 0 not in depths
            or maximum not in depths
            or any(d != 0 and not 3 <= d <= maximum for d in depths)
        ):
            raise ValueError(
                "DFlash depths must include zero and the configured maximum, with at most four prepared depths"
            )
        forced = os.getenv("SPARKLAB_DFLASH_FIXED_DEPTH")
        override = int(forced) if forced is not None else None
        if override is not None and override not in depths:
            raise ValueError("fixed DFlash override must be a prepared depth")
        return cls(depths, maximum, override)

    def policy(self):
        return SpeculationPolicy(self.depths, self.fixed_depth)


@dataclass
class AdaptiveRound:
    policy: SpeculationPolicy
    depth: int
    started: float
    accepted: int = 0
    gpu_start: object | None = None
    gpu_proposal_end: object | None = None
    gpu_end: object | None = None
    gpu_verification_end: object | None = None
    gpu_commit_end: object | None = None
    _verification_recorded: bool = False
    _commit_recorded: bool = False

    def begin_gpu(self, device, *, phase_timing=False):
        import torch

        if device.type == "cuda":
            self.gpu_start = torch.cuda.Event(enable_timing=True)
            self.gpu_proposal_end = torch.cuda.Event(enable_timing=True)
            self.gpu_end = torch.cuda.Event(enable_timing=True)
            if phase_timing:
                self.gpu_verification_end = torch.cuda.Event(enable_timing=True)
                self.gpu_commit_end = torch.cuda.Event(enable_timing=True)
            self.gpu_start.record(torch.cuda.current_stream(device))

    def proposal_ready(self, device):
        if self.gpu_proposal_end is not None:
            import torch

            self.gpu_proposal_end.record(torch.cuda.current_stream(device))

    def verification_ready(self, device):
        if self.gpu_verification_end is not None:
            import torch

            if self._verification_recorded:
                raise AssertionError("adaptive verification marker recorded twice")
            self.gpu_verification_end.record(torch.cuda.current_stream(device))
            self._verification_recorded = True

    def commit_ready(self, device):
        if self.gpu_commit_end is not None:
            import torch

            if not self._verification_recorded or self._commit_recorded:
                raise AssertionError("adaptive commit marker out of order")
            self.gpu_commit_end.record(torch.cuda.current_stream(device))
            self._commit_recorded = True

    def gpu_timings(self):
        if self.gpu_end is None:
            return None
        if not self.gpu_end.query():
            raise AssertionError("adaptive timing observed before output fence")
        result = {
            "proposal_gpu_seconds": self.gpu_start.elapsed_time(self.gpu_proposal_end)
            / 1000.0,
            "round_gpu_seconds": self.gpu_start.elapsed_time(self.gpu_end) / 1000.0,
        }
        if self.gpu_verification_end is not None:
            if not self._verification_recorded or not self._commit_recorded:
                raise AssertionError("adaptive phase timing lacks completed markers")
            result.update(
                verification_gpu_seconds=self.gpu_proposal_end.elapsed_time(
                    self.gpu_verification_end
                )
                / 1000.0,
                commit_gpu_seconds=self.gpu_verification_end.elapsed_time(
                    self.gpu_commit_end
                )
                / 1000.0,
                handoff_gpu_seconds=self.gpu_commit_end.elapsed_time(self.gpu_end)
                / 1000.0,
            )
        return result

    def finish(self, *, clock=time.monotonic):
        seconds = clock() - self.started
        self.policy.observe(
            proposed=self.depth,
            accepted=self.accepted,
            seconds=seconds,
            source="dflash2",
        )
        return seconds
