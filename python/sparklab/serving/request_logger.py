"""Optional inbound-request logging.

When the ``SPARKLAB_API_LOG_DIR`` environment variable is set, every request that is
dispatched to one of the API handlers (``/v1/chat/completions``,
``/v1/completions``, ``/v1/messages``, ``/v1/responses``, ``/generate``) is
appended as one JSON record per line to a per-process
``requests-<start>-<pid>.jsonl`` file under that directory. (Requests rejected
by FastAPI's body validation never reach a handler, so they are not recorded.)

Records are handed to a dedicated background writer thread through a bounded
in-memory queue, so a slow / full / network-backed log filesystem can never
block the server's single async event loop; if the queue fills (pathologically
slow disk) records are dropped rather than applying backpressure to serving.
When the variable is unset the hooks are a no-op with effectively zero overhead,
so call sites can invoke :func:`log_request` unconditionally.

A logging failure must never break request handling, so the enqueue path
swallows every error (warning at most once).
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from typing import Any

from fastapi import Request
from pydantic import BaseModel

from sparklab.utils import init_logger

logger = init_logger(__name__, "ReqLog")

# Resolved once at import (with ~ and $VAR expansion); serving never reconfigures
# it at runtime.
_RAW_DIR = os.getenv("SPARKLAB_API_LOG_DIR")
_LOG_DIR = os.path.expanduser(os.path.expandvars(_RAW_DIR)) if _RAW_DIR else None

# Bounded so a stalled log disk drops records instead of growing memory without
# limit (and never blocks the event loop on a full queue).
_MAX_QUEUE = 10_000
_queue: "queue.Queue[str | None]" = queue.Queue(maxsize=_MAX_QUEUE)
_log_path: str | None = None
_fh = None
_worker: threading.Thread | None = None
_init_lock = threading.Lock()
_init_done = False
_init_failed = False
_warned = False


def enabled() -> bool:
    return bool(_LOG_DIR)


def _warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        logger.warning("%s", msg)
        _warned = True


def init() -> None:
    """Create the log directory and start the writer thread. No-op when
    disabled, idempotent. Called once at server startup so a misconfigured
    ``SPARKLAB_API_LOG_DIR`` surfaces immediately instead of on the first request."""
    if _LOG_DIR:
        _ensure_worker()


def _ensure_worker() -> None:
    """Open the per-process log file (owner-only) and spawn the writer thread,
    exactly once. Latches on failure so a bad path is not re-attempted on every
    request."""
    global _log_path, _fh, _worker, _init_done, _init_failed
    if _init_done:
        return
    with _init_lock:
        if _init_done:
            return
        _init_done = True  # try exactly once, success or fail
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            # One file per process: the PID keeps two servers sharing a dir (or a
            # restart within the same second) from colliding on one path, which
            # the per-process queue/thread could not otherwise serialize.
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            _log_path = os.path.join(_LOG_DIR, f"requests-{stamp}-{os.getpid()}.jsonl")
            # 0o600: request bodies can carry prompts/PII, so keep them owner-only.
            fd = os.open(_log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            _fh = os.fdopen(fd, "a", encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 — never fail a request over a log path
            _init_failed = True
            _warn_once(f"cannot open request log under {_LOG_DIR!r}: {exc}")
            return
        _worker = threading.Thread(target=_writer_loop, name="reqlog-writer", daemon=True)
        _worker.start()
        atexit.register(_shutdown)
        logger.info("Logging API requests to %s", _log_path)


def _writer_loop() -> None:
    while True:
        line = _queue.get()
        try:
            if line is None:  # shutdown sentinel
                return
            _fh.write(line)
            _fh.flush()
        except Exception as exc:  # noqa: BLE001 — a write failure must not kill the thread
            _warn_once(f"failed to write request log: {exc}")
        finally:
            _queue.task_done()


def _shutdown() -> None:
    """Best-effort flush of queued records at interpreter exit."""
    if _worker is None:
        return
    try:
        _queue.put(None, timeout=1.0)
        _worker.join(timeout=2.0)
    except Exception:  # noqa: BLE001
        pass


def flush(timeout: float = 5.0) -> bool:
    """Block until queued records have been written. Best-effort; returns False
    on timeout. Intended for tests and graceful shutdown, not the hot path."""
    if _worker is None:
        return True
    done = threading.Event()

    def _waiter() -> None:
        _queue.join()
        done.set()

    threading.Thread(target=_waiter, daemon=True).start()
    return done.wait(timeout)


def _to_payload(req: Any) -> Any:
    """Faithfully represent what the client sent: for a pydantic request model,
    dump only the fields it actually set (``exclude_unset``) in JSON mode so
    enums/nested models serialize cleanly. Anything else is passed through."""
    if isinstance(req, BaseModel):
        return req.model_dump(mode="json", exclude_unset=True)
    return req


def log_request(endpoint: str, req: Any, request: Request | None = None) -> None:
    """Enqueue one record for an inbound API request. No-op unless
    ``SPARKLAB_API_LOG_DIR`` is set. Non-blocking and swallows all errors (warns at
    most once)."""
    if not _LOG_DIR:
        return
    try:
        _ensure_worker()
        if _init_failed:
            return
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "endpoint": endpoint,
            "client": request.client.host if request and request.client else None,
            "request": _to_payload(req),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            _queue.put_nowait(line)
        except queue.Full:
            _warn_once("request log queue full; dropping records (slow log disk?)")
    except Exception as exc:  # noqa: BLE001 — logging must never break serving
        _warn_once(f"failed to enqueue request log: {exc}")
