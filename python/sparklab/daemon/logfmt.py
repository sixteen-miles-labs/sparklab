"""Normalize a serve's raw stdout/stderr byte stream into display lines.

Two jobs, kept pure (no clock, no IO — so they unit-test trivially and the caller owns any
throttling): strip ANSI control sequences, and turn a terminal-style stream (``\\n`` newlines +
``\\r`` in-place redraws, as tqdm emits) into ("line" | "partial", text) events. A ``\\r`` marks
the current line as displayed-then-overwritten → emitted as a ``partial`` (the caller may throttle
these); a ``\\n`` finalizes a ``line``; ``\\r\\n`` is treated as a single newline.

Lives in the torch-free ``daemon`` package and imports only ``re`` — never anything from
``sparklab.serving`` / ``sparklab.utils`` (both pull torch/transformers at import).
"""

from __future__ import annotations

import re

# CSI/escape sequences: ESC [ ... final-byte, plus a few standalone two-char escapes. Enough to
# clean tqdm bars and coloured log lines without a full terminal emulator.
_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class LineAssembler:
    """Feed raw decoded chunks; get back finalized display lines. Stateful across chunks (a line
    may span reads) but deterministic and clock-free. ``feed`` returns a list of (kind, text);
    ``flush`` emits whatever trails after EOF."""

    __slots__ = ("_cur", "_cr")

    def __init__(self) -> None:
        self._cur: list[str] = []
        self._cr = False  # a bare '\r' was seen; the next char decides newline-vs-redraw

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for ch in text:
            if ch == "\r":
                # Coalesce a RUN of '\r'. A lone '\r' is a tqdm redraw (cursor→col 0); a run is
                # still just "return to col 0", so it collapses to one. This is what makes Windows
                # '\r\r\n' behave like '\r\n': the child's text-mode stdout adds a CR to output that
                # already ended in CRLF, and without coalescing the first '\r' was misread as a
                # redraw (emitting the real line as a throttled 'partial' plus a spurious empty
                # 'line' — the phantom blank rows in the Logs view).
                self._cr = True
                continue
            if self._cr:
                self._cr = False
                if ch == "\n":
                    # '\r'+…+'\n' — a single newline (covers '\r\n' and Windows '\r\r\n').
                    out.append(("line", "".join(self._cur)))
                    self._cur = []
                    continue
                # bare '\r' run then a real char: the line was drawn, then overwritten (redraw).
                out.append(("partial", "".join(self._cur)))
                self._cur = [ch]
                continue
            if ch == "\n":
                out.append(("line", "".join(self._cur)))
                self._cur = []
            else:
                self._cur.append(ch)
        return out

    def flush(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self._cr:
            self._cr = False
            out.append(("partial", "".join(self._cur)))
            self._cur = []
        elif self._cur:
            out.append(("line", "".join(self._cur)))
            self._cur = []
        return out
