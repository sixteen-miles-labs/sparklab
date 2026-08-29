"""Bounded in-memory log ring + SSE fan-out for the engine's captured stdout.

Modeled on ``server/request_ring.py`` (a monotonic all-time cursor over a bounded deque, pulled
incrementally with ``?since=``), but re-implemented here rather than imported — anything under
``sparklab.serving`` transitively pulls torch at import and would blow the daemon's §2 rule.

Two access paths, and they run on different threads, so everything is guarded by one internal
lock: the tailer thread ``append``s finalized log lines; SSE handlers on the event loop
``subscribe``/``unsubscribe`` push callbacks. ``append`` snapshots the subscriber set under the
lock and invokes callbacks OUTSIDE it (a callback must never call back in), so a mutating
subscriber can't corrupt the fan-out iteration."""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable

# A push callback receives one record dict. It must not raise and must not re-enter the ring.
PushFn = Callable[[dict], None]


class LogRing:
    def __init__(self, capacity: int = 4000) -> None:
        self._buf: deque[dict] = deque(maxlen=capacity)
        self._next = 0  # all-time monotonic seq, survives eviction
        self._subs: set[PushFn] = set()
        self._lock = threading.Lock()

    def append(self, text: str, *, kind: str = "line", ts: float = 0.0) -> dict:
        """Append one record and fan it out to subscribers. ``kind`` is ``"line"`` (finalized),
        ``"progress"`` (a throttled tqdm redraw), or ``"event"`` (a daemon-emitted lifecycle
        note). Returns the stored record (with its assigned ``seq``)."""
        with self._lock:
            rec = {"seq": self._next, "ts": ts, "kind": kind, "text": text}
            self._buf.append(rec)
            self._next += 1
            subs = tuple(self._subs)
        for push in subs:
            try:
                push(rec)
            except Exception:  # noqa: BLE001 — a broken subscriber must never kill the tailer
                pass
        return rec

    def since(self, cursor: int) -> tuple[list[dict], int]:
        """Records with ``seq >= cursor`` plus the next cursor to poll with (the all-time count).
        ``cursor`` is exclusive of already-seen records, matching request_ring's convention."""
        with self._lock:
            out = [r for r in self._buf if r["seq"] >= cursor]
            return out, self._next

    def cursor(self) -> int:
        with self._lock:
            return self._next

    def subscribe(self, push: PushFn) -> None:
        with self._lock:
            self._subs.add(push)

    def unsubscribe(self, push: PushFn) -> None:
        with self._lock:
            self._subs.discard(push)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)
