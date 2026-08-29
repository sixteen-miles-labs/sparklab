"""Terminal rendering primitives for the SparkLab shell.

Pure presentation: cell-accurate wrapping, the dim ``Thinking...`` block, and the reverse-video
status footer. Nothing here talks to a server, and the reasoning/answer split arrives already
made -- the server's reasoning parser does it, and the renderer just gets two channels.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, List

from prompt_toolkit.utils import get_cwidth

THINK_BLOCK_START = "Thinking..."
THINK_BLOCK_END = "End Think"
THINK_MAX_LINES = 8
SHELL_FALLBACK_WIDTH = 100
SHELL_THINK_PREFIX = "| "
SHELL_OUTPUT_FLUSH_INTERVAL = 0.05
SHELL_STATUS_REFRESH_INTERVAL = 0.25
SHELL_INPUT_PROMPT = "$ "
ANSI_RESET = "\x1b[0m"
ANSI_DIM = "\x1b[2m"
ANSI_BOLD = "\x1b[1m"
ANSI_REVERSE = "\x1b[7m"
ANSI_CLEAR_LINE = "\x1b[2K"
ANSI_SAVE_CURSOR = "\x1b7"
ANSI_RESTORE_CURSOR = "\x1b8"
ANSI_RESET_SCROLL_REGION = "\x1b[r"


def wrap_visual_lines(text: str, display_width: int) -> List[str]:
    width = max(1, display_width)
    result: List[str] = []
    for line in text.splitlines():
        if line == "":
            result.append("")
            continue
        result.extend(wrap_line_by_cells(line, width))
    return result


def wrap_line_by_cells(line: str, width: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    current_width = 0
    for char in line:
        char_width = max(0, get_cwidth(char))
        if current and current_width + char_width > width:
            chunks.append(current)
            current = ""
            current_width = 0
        current += char
        current_width += char_width
        if current_width >= width:
            chunks.append(current)
            current = ""
            current_width = 0
    if current:
        chunks.append(current)
    return chunks


def pad_cells(text: str, width: int) -> str:
    clipped = clip_cells(text, width)
    return clipped + " " * max(0, width - get_cwidth(clipped))


def clip_cells(text: str, width: int) -> str:
    result = ""
    current_width = 0
    for char in text:
        char_width = max(0, get_cwidth(char))
        if current_width + char_width > width:
            break
        result += char
        current_width += char_width
    return result


def _ansi_dim(text: str) -> str:
    return f"{ANSI_DIM}{text}{ANSI_RESET}"


def _ansi_bold_dim(text: str) -> str:
    return f"{ANSI_BOLD}{ANSI_DIM}{text}{ANSI_RESET}"


def format_turn_header(cmd: str, display_width: int = SHELL_FALLBACK_WIDTH) -> str:
    width = max(1, display_width)
    formatted = []
    first = True
    content_width = max(1, width - 2)
    for line in cmd.splitlines() or [""]:
        chunks = wrap_line_by_cells(line, content_width) or [""]
        for chunk in chunks:
            prefix = "> " if first else "  "
            text = pad_cells(prefix + chunk, width)
            formatted.append(f"{ANSI_REVERSE}{text}{ANSI_RESET}")
            first = False
    return "\n" + "\n".join(formatted) + "\n\n"


def format_think_block(
    content: str,
    *,
    closed: bool,
    max_lines: int = THINK_MAX_LINES,
    display_width: int = SHELL_FALLBACK_WIDTH,
) -> str:
    body = content[1:] if content.startswith("\n") else content
    content_width = max(1, display_width - len(SHELL_THINK_PREFIX))
    lines = wrap_visual_lines(body, content_width)
    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    block = [THINK_BLOCK_START]
    block.extend(f"{SHELL_THINK_PREFIX}{line}" for line in lines)
    if closed:
        block.append(THINK_BLOCK_END)
    return "\n".join(block) + "\n"


def style_think_block(block: str) -> str:
    styled = []
    for line in block.splitlines():
        if line in (THINK_BLOCK_START, THINK_BLOCK_END):
            styled.append(_ansi_bold_dim(line))
        else:
            styled.append(_ansi_dim(line))
    suffix = "\n" if block.endswith("\n") else ""
    return "\n".join(styled) + suffix


def clear_rows(num_rows: int) -> str:
    if num_rows <= 0:
        return ""
    clear_lines = []
    clear_lines.append(f"\x1b[{num_rows}F")
    for idx in range(num_rows):
        clear_lines.append(f"\r{ANSI_CLEAR_LINE}")
        if idx + 1 < num_rows:
            clear_lines.append("\x1b[1E")
    if num_rows > 1:
        clear_lines.append(f"\x1b[{num_rows - 1}F")
    return "".join(clear_lines)


class ShellConsoleRenderer:
    """Streams a turn to the terminal, folding reasoning into a dim ``Thinking...`` block and
    printing the answer as plain text.

    The split is the server's: ``/v1/chat/completions`` emits ``delta.reasoning_content`` and
    ``delta.content`` on separate channels (one reasoning parser for every model family --
    gpt-oss Harmony, ``<think>`` for Qwen3/GLM/MiniMax, Gemma's thought channel), so a client
    only has to route them. A model with no reasoning parser configured simply never produces
    the reasoning channel and every delta prints as plain output.
    """

    def __init__(
        self,
        write: Callable[[str], None] | None = None,
        *,
        display_width: int = SHELL_FALLBACK_WIDTH,
        max_think_lines: int = THINK_MAX_LINES,
    ) -> None:
        self.write = write or self._write_stdout
        self.display_width = max(1, display_width)
        self.max_think_lines = max_think_lines
        self._think_closed = False
        self.think_content = ""
        self.current_think_block = ""
        self.live_think_rows = 0

    def begin_turn(self, cmd: str) -> None:
        self._think_closed = False
        self.think_content = ""
        self.current_think_block = ""
        self.live_think_rows = 0
        self.write(format_turn_header(cmd, self.display_width))

    def write_reasoning(self, text: str) -> None:
        if not text:
            return
        self.think_content += text
        self._render_think(closed=False)

    def write_content(self, text: str) -> None:
        """First answer delta closes the live think block; after that content is plain text."""
        if not text:
            return
        if self.think_content and not self._think_closed:
            self._render_think(closed=True)
            self._think_closed = True
        self.write(text)

    def finish_turn(self) -> None:
        # A turn that ended while still reasoning (truncated, no answer) still gets its block
        # closed so the display isn't left mid-"Thinking...".
        if self.think_content and not self._think_closed:
            self._render_think(closed=True)
            self._think_closed = True
        self.write("\n")

    def _render_think(self, *, closed: bool) -> None:
        if not self.think_content.strip():
            if closed and self.current_think_block:
                self.write(clear_rows(self.live_think_rows))
                self.current_think_block = ""
                self.live_think_rows = 0
            return
        block = format_think_block(
            self.think_content,
            closed=closed,
            max_lines=self.max_think_lines,
            display_width=self.display_width,
        )
        if block == self.current_think_block:
            return
        prefix = clear_rows(self.live_think_rows)
        self.current_think_block = block
        self.live_think_rows = len(block.splitlines())
        self.write(prefix + style_think_block(block))

    @staticmethod
    def _write_stdout(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()


class ShellOutputBuffer:
    """Coalesces per-token deltas into ~20 Hz repaints. Keeps the two channels in arrival
    order, so a model that goes back to reasoning after answering still renders in sequence."""

    def __init__(
        self,
        renderer: ShellConsoleRenderer,
        *,
        flush_interval: float = SHELL_OUTPUT_FLUSH_INTERVAL,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.renderer = renderer
        self.flush_interval = flush_interval
        self.now = now
        self.pending: List[List[str]] = []  # [kind, text] pairs, consecutive same-kind merged
        self.last_flush_at = now()

    def write_reasoning(self, text: str) -> None:
        self._append("reasoning", text)

    def write_content(self, text: str) -> None:
        self._append("content", text)

    def _append(self, kind: str, text: str) -> None:
        if not text:
            return
        if self.pending and self.pending[-1][0] == kind:
            self.pending[-1][1] += text
        else:
            self.pending.append([kind, text])
        if self.now() - self.last_flush_at >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        for kind, text in self.pending:
            if kind == "reasoning":
                self.renderer.write_reasoning(text)
            else:
                self.renderer.write_content(text)
        self.pending = []
        self.last_flush_at = self.now()


class ShellStatusLine:
    def __init__(
        self,
        format_status: Callable[[], str],
        write: Callable[[str], None] | None = None,
        *,
        display_width: int = SHELL_FALLBACK_WIDTH,
        display_height: int = 24,
        refresh_interval: float = SHELL_STATUS_REFRESH_INTERVAL,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.format_status = format_status
        self.write = write or ShellConsoleRenderer._write_stdout
        self.display_width = max(1, display_width)
        self.display_height = max(2, display_height)
        self.refresh_interval = refresh_interval
        self.now = now
        self.last_refresh_at: float | None = None
        self.active = False

    def activate(self) -> None:
        self.active = True
        self.write(self._render_footer())
        self.last_refresh_at = self.now()

    def deactivate(self) -> None:
        if not self.active:
            return
        input_row = self.display_height - 1
        self.write(
            f"{ANSI_SAVE_CURSOR}{ANSI_RESET_SCROLL_REGION}"
            f"\x1b[{input_row};1H{ANSI_CLEAR_LINE}"
            f"\x1b[{self.display_height};1H{ANSI_CLEAR_LINE}"
            f"{ANSI_RESTORE_CURSOR}"
        )
        self.active = False

    def force(self) -> None:
        self._draw()
        self.last_refresh_at = self.now()

    def maybe(self) -> None:
        current = self.now()
        if self.last_refresh_at is None or current - self.last_refresh_at >= self.refresh_interval:
            self.force()

    def write_output(self, text: str) -> None:
        if not text:
            return
        self.write(text)

    def _draw(self) -> None:
        self.write(self._render_status_line())

    def _render_status_line(self) -> str:
        status = pad_cells(self.format_status(), self.display_width)
        return (
            f"{ANSI_SAVE_CURSOR}"
            f"\x1b[{self.display_height};1H{ANSI_CLEAR_LINE}"
            f"{ANSI_REVERSE}{status}{ANSI_RESET}"
            f"{ANSI_RESTORE_CURSOR}"
        )

    def _render_footer(self) -> str:
        input_row = self.display_height - 1
        output_bottom = self.display_height - 2
        status = pad_cells(self.format_status(), self.display_width)
        return (
            f"\x1b[1;{output_bottom}r"
            f"\x1b[{input_row};1H{ANSI_CLEAR_LINE}{SHELL_INPUT_PROMPT}"
            f"\x1b[{self.display_height};1H{ANSI_CLEAR_LINE}{ANSI_REVERSE}{status}{ANSI_RESET}"
            f"\x1b[{output_bottom};1H"
        )
