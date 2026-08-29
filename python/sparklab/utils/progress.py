"""Loading progress bars, consistent across the framework's weight-load paths.

tqdm-backed (the framework's existing idiom) and byte-oriented -- load time tracks bytes
moved off disk, not tensor count, so a byte bar shows a meaningful GiB/s. Bars are disabled
off rank 0 (only the primary should draw) so multi-rank logs stay clean.
"""

from __future__ import annotations

from tqdm import tqdm

from sparklab.runtime.distributed import try_get_tp_info

import time
from typing import Callable, Optional

_PROGRESS_SINK: Optional[Callable[[str, int, int], None]] = None


def set_progress_sink(sink: Optional[Callable[[str, int, int], None]]) -> None:
    """Install (or clear, with None) a global progress callback invoked (throttled)
    by every ``byte_bar`` as ``sink(desc, done_bytes, total_bytes)``. The scheduler's
    rank-0 process installs one that forwards to its ack_queue; call sites are unchanged."""
    global _PROGRESS_SINK
    _PROGRESS_SINK = sink


def _on_primary() -> bool:
    info = try_get_tp_info()
    return info is None or info.is_primary()


def emit_progress(desc: str, done: int, total: int) -> None:
    """Push a one-off update to the installed sink for a phase that has no ``byte_bar`` — e.g.
    CUDA-graph capture / warmup, which moves no bytes. A ``total <= 0`` reads downstream as an
    indeterminate phase (no percentage). No-op off rank 0 or when no sink is installed."""
    sink = _PROGRESS_SINK
    if sink is not None and _on_primary():
        try:
            sink(desc, done, total)
        except Exception:  # noqa: BLE001 — progress reporting must never break load
            pass


class _SinkTqdm(tqdm):
    """A tqdm that also forwards its progress to the installed ``_PROGRESS_SINK``,
    throttled to <=1 emit / 0.5 s OR a >=1% delta (plus a guaranteed final emit)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sink_last_emit = 0.0
        self._sink_last_frac = -1.0

    def update(self, n: int = 1):  # type: ignore[override]
        ret = super().update(n)
        sink = _PROGRESS_SINK
        if sink is not None:
            total = int(self.total or 0)
            done = int(self.n or 0)
            frac = (done / total) if total else 0.0
            now = time.monotonic()
            if (
                now - self._sink_last_emit >= 0.5
                or frac - self._sink_last_frac >= 0.01
                or (total and done >= total)
            ):
                self._sink_last_emit = now
                self._sink_last_frac = frac
                try:
                    sink(self.desc or "", done, total)
                except Exception:  # noqa: BLE001 — progress reporting must never break load
                    pass
        return ret


def byte_bar(total: int, desc: str) -> tqdm:
    """A byte-scaled bar (shows e.g. ``12.8GiB [00:02, 6.1GiB/s]``); ``update(nbytes)`` it
    as each tensor/bank/shard finishes reading. Also drives the progress sink when installed.
    Thread-safe to update from a pool."""
    return _SinkTqdm(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024,
                     disable=not _on_primary(), leave=False, dynamic_ncols=True)


def count_bar(iterable, desc: str, total: int | None = None) -> tqdm:
    """A plain count bar over an iterable (use when total bytes aren't known up front)."""
    return tqdm(iterable, desc=desc, total=total, disable=not _on_primary(),
                leave=False, dynamic_ncols=True)


__all__ = ["byte_bar", "count_bar", "set_progress_sink"]
