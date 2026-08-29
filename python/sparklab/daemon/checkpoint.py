"""Supervise ``sparklab checkpoint`` conversions the same way the serve is supervised. GPU exclusivity
(a convert needs the GPU, so the caller stops the serve first) is enforced at the route layer
(app.py).

One job at a time, transient (no re-adoption): a conversion that outlives a daemon restart is not
worth re-attaching to. Output streams into the same LogRing tagged ``kind="line"``. Torch-free —
it only spawns ``sparklab checkpoint`` as a child."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

from . import osproc
from .serve_manager import Conflict
from .tailer import LogTailer


@dataclass
class Job:
    id: str
    pid: int
    started_at: float
    state: str = "running"  # running | done | failed | cancelled
    exit_code: int | None = None


class CheckpointManager:
    def __init__(
        self,
        ring,
        *,
        python: str = sys.executable,
        log_dir: str = ".",
        spawn_fn: Callable[[str, list[str]], object] | None = None,
        tailer_factory: Callable[[object], object] | None = None,
        now: Callable[[], float] = time.monotonic,
        wall_now: Callable[[], float] = time.time,
        grace_s: float = 10.0,
    ) -> None:
        self._ring = ring
        self._python = python
        self._log_dir = log_dir
        self._spawn_fn = spawn_fn or self._default_spawn
        self._tailer_factory = tailer_factory
        self._now = now
        self._wall_now = wall_now
        self._grace_s = grace_s
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._child = None

    def _default_spawn(self, job_id: str, args: list[str]):
        from .serve_manager import spawn_serve

        argv = [self._python, "-m", "sparklab.cli", "checkpoint", *args]
        log_path = os.path.join(self._log_dir, f"checkpoint-{job_id}.log")
        return spawn_serve(argv, log_path)

    def start(self, job_id: str, args: list[str]) -> dict:
        with self._lock:
            if self._job is not None and self._job.state == "running":
                if self._job.id == job_id:
                    # During the reserve→spawn window pid is still -1; don't hand a client a bogus pid.
                    pid = self._job.pid if self._job.pid > 0 else None
                    return {"jobId": job_id, "pid": pid, "idempotent": True}
                # Conflict (not a bare RuntimeError) so the route returns 409, not 500.
                raise Conflict(f"checkpoint {self._job.id!r} already running")
            # A cancelled/finished job whose child is still terminating hasn't freed the GPU yet —
            # block admission until it is fully reaped, else a cancel-then-start races two
            # conversions onto the GPU. The monitor sets child.reaped when gone.
            if self._child is not None and not self._child.reaped.is_set():
                raise Conflict("previous checkpoint is still terminating")
            # Reserve the slot inside THIS critical section (pid unknown until spawn), so a
            # concurrent different-id start is rejected before either spawns — no two GPU
            # conversions at once.
            self._job = Job(id=job_id, pid=-1, started_at=self._now())
            self._child = None
        try:
            child = self._spawn_fn(job_id, list(args))
        except Exception:
            with self._lock:
                if self._job is not None and self._job.id == job_id:
                    self._job = None  # release the reservation so the next start can proceed
            raise
        with self._lock:
            self._child = child
            self._job.pid = child.pid
            cancelled = self._job.state == "cancelled"  # a cancel arrived during the spawn window
        if cancelled:
            # Escalate SIGTERM→grace→SIGKILL off the request thread so a SIGTERM-ignoring child is
            # still force-killed, and the caller isn't blocked for grace_s.
            threading.Thread(target=self._terminate, args=(child,), daemon=True).start()
        if self._tailer_factory is not None:
            tailer = self._tailer_factory(child)
            child.tailer = tailer
            if tailer is not None:
                tailer.start()
        threading.Thread(
            target=self._monitor, args=(child, job_id), name=f"sparklab-daemon-ckpt-{child.pid}", daemon=True
        ).start()
        self._emit(f"checkpoint started (id={job_id} pid={child.pid})")
        return {"jobId": job_id, "pid": child.pid, "idempotent": False}

    def _monitor(self, child, job_id: str) -> None:
        try:
            info = child.wait()
            code = info.code
        except Exception:  # noqa: BLE001
            code = None
        child.reaped.set()  # wake a cancel() waiter
        with self._lock:
            job = self._job
            if job is not None and job.id == job_id:
                if job.state != "cancelled":
                    job.state = "done" if code == 0 else "failed"
                job.exit_code = code
        if getattr(child, "tailer", None) is not None:
            try:
                child.tailer.stop()
                child.tailer.join(timeout=self._grace_s)
            except Exception:  # noqa: BLE001
                pass
        child.close()
        self._emit(f"checkpoint {job_id} exited with code {code}")

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._job
            child = self._child
            if job is None or job.id != job_id or job.state != "running":
                return {"cancelled": False}
            job.state = "cancelled"
            # During the reserve→spawn window pid is -1 and child is None; mark cancelled and let
            # start() signal the child as soon as it exists.
            target = child if (child is not None and job.pid > 0) else None
        if target is None:
            return {"cancelled": True}
        self._terminate(target)
        return {"cancelled": True}

    def _terminate(self, child) -> None:
        """SIGTERM → grace → SIGKILL on the child's group (shared by cancel() and the
        cancel-during-spawn path)."""
        osproc.signal_group(child.pid, signal.SIGTERM)
        if not child.reaped.wait(timeout=self._grace_s):
            osproc.signal_group(child.pid, signal.SIGKILL)

    def status(self) -> dict:
        with self._lock:
            job = self._job
            if job is None:
                return {"job": None}
            return {
                "job": {
                    "id": job.id,
                    "pid": job.pid,
                    "state": job.state,
                    "uptimeS": int(self._now() - job.started_at),
                    "exitCode": job.exit_code,
                }
            }

    def _emit(self, text: str) -> None:
        try:
            self._ring.append(text, kind="event", ts=self._wall_now())
        except Exception:  # noqa: BLE001
            pass
