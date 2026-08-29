"""Follow the serve's logfile into the LogRing (the same file the serve writes stdout+stderr to).

One tailer thread per child. It follows a file rather than a pipe so the stream is
identical whether the serve is one we spawned or one we re-adopted after a daemon restart — the
re-adopted daemon just reopens the file where the last one left off.

It must terminate: a naive ``tail -f`` never sees EOF while the file exists. So it loops on a
``stop`` Event (set by ``ServeManager._reap`` when the child exits) with short non-blocking reads,
and after ``stop`` is set it drains whatever remains and exits — no thread/fd leak per
start/stop cycle.

ANSI is stripped and tqdm ``\\r`` redraws are collapsed via ``LineAssembler``; intermediate redraw
frames are emitted at most every ``throttle_s`` as ``kind="progress"`` so a multi-minute weight
load shows live movement in the Logs view without flooding the ring."""

from __future__ import annotations

import codecs
import threading
import time
from typing import Callable

from .logfmt import LineAssembler, strip_ansi


class LogTailer(threading.Thread):
    def __init__(
        self,
        log_path: str,
        ring,
        *,
        from_start: bool = True,
        idle_s: float = 0.1,
        throttle_s: float = 0.4,
        wall_now: Callable[[], float] = time.time,
        chunk: int = 65536,
    ) -> None:
        super().__init__(name=f"sparklab-daemon-tailer", daemon=True)
        self._path = log_path
        self._ring = ring
        self._from_start = from_start
        self._idle_s = idle_s
        self._throttle_s = throttle_s
        self._wall_now = wall_now
        self._chunk = chunk
        # NB: not `self._stop` — that name shadows threading.Thread's internal _stop() method,
        # which the runtime calls at thread teardown/join (→ 'Event not callable' crash).
        self._stopped = threading.Event()
        self._asm = LineAssembler()
        # Incremental decoder so a multibyte UTF-8 char split across two reads is stitched back
        # together instead of decoding each half to replacement chars.
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._last_progress = 0.0

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        fh = None
        try:
            # Wait briefly for the serve to create the file (spawn races the first write).
            deadline = self._wall_now() + 5.0
            while fh is None and not self._stopped.is_set():
                try:
                    fh = open(self._path, "rb")
                except FileNotFoundError:
                    if self._wall_now() > deadline:
                        break
                    time.sleep(self._idle_s)
            if fh is None:
                return
            if not self._from_start:
                fh.seek(0, 2)  # re-adopt: skip to EOF so we don't replay a huge stale file
            self._follow(fh)
        except Exception:  # noqa: BLE001 — the tailer must never take the daemon down
            pass
        finally:
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass

    def _follow(self, fh) -> None:
        while True:
            data = fh.read(self._chunk)
            if data:
                self._consume(data)
                continue
            # No data available right now.
            if self._stopped.is_set():
                # Child has exited: drain any final bytes (flushing the decoder so a dangling
                # partial codepoint becomes one replacement char), flush the partial line, stop.
                self._consume(fh.read(), final=True)
                for kind, text in self._asm.flush():
                    self._append(kind, text, force=True)
                return
            time.sleep(self._idle_s)

    def _consume(self, data: bytes, final: bool = False) -> None:
        text = self._decoder.decode(data, final)
        for kind, line in self._asm.feed(text):
            self._append(kind, line, force=(kind == "line"))

    def _append(self, kind: str, text: str, *, force: bool) -> None:
        clean = strip_ansi(text)
        if kind == "partial":
            # Throttle redraw frames; drop empties.
            if not clean:
                return
            now = self._wall_now()
            if not force and (now - self._last_progress) < self._throttle_s:
                return
            self._last_progress = now
            self._ring.append(clean, kind="progress", ts=now)
        else:
            self._ring.append(clean, kind="line", ts=self._wall_now())
