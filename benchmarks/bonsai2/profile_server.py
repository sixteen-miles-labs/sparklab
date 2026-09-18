"""Diagnostic-only graph-replay profiler; touch /tmp/bonsai-profile.request."""

from pathlib import Path

import torch
from sparklab.runtime.engine.graph import GraphRunner

_original = GraphRunner.replay
_profiler = None
_steps = 0
_marker = Path("/tmp/bonsai-profile.request")


def replay(self, batch):
    global _profiler, _steps
    if _profiler is None and _marker.exists():
        _marker.unlink()
        _profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        _profiler.start()
        _steps = 0
    result = _original(self, batch)
    if _profiler is not None:
        _profiler.step()
        _steps += 1
        if _steps == 32:
            _profiler.stop()
            Path("/tmp/bonsai-profile.txt").write_text(
                _profiler.key_averages().table(
                    sort_by="self_device_time_total", row_limit=35
                )
            )
            _profiler.export_chrome_trace("/tmp/bonsai-profile.json")
            _profiler = None
    return result


GraphRunner.replay = replay
if __name__ == "__main__":
    from sparklab.serving import launch_server

    launch_server()
