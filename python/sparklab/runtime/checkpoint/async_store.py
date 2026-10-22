"""Bounded single-worker prefix I/O; completion ownership stays with the scheduler."""

from __future__ import annotations
from dataclasses import dataclass, field
import queue
import threading
import time

from .store import PrefixStore


@dataclass
class IOTicket:
    kind: str
    payload: object
    key: str | None = None
    cancelled: bool = False
    result: object = None
    error: BaseException | None = None
    elapsed_seconds: float = 0.0
    finished: threading.Event = field(default_factory=threading.Event)
    reported: bool = False


class AsyncPrefixStore:
    """No submit/lookup/poll call does I/O or waits for filesystem/GPU completion.

    Admitted tickets include queued, running AND undrained completions, so a
    stalled scheduler cannot accumulate source leases or completion metadata.
    Release a ticket only after its GPU fence/source/private buffers are safe.
    """

    def __init__(self, root, compatibility, *, max_tickets=4, **store_options):
        if not 1 <= max_tickets <= 128:
            raise ValueError("invalid I/O queue bound")
        self.max_tickets = max_tickets
        self._jobs = queue.Queue(maxsize=max_tickets)
        self._done = queue.Queue(maxsize=max_tickets)
        self._mutex = threading.Lock()
        self._tickets = {}
        self._closing = threading.Event()
        self._io_stopped = threading.Event()
        self._released = threading.Event()
        self.ready = threading.Event()
        self.failure = None
        self.store = None
        self.observation = None
        self._thread = threading.Thread(
            target=self._run,
            args=(root, compatibility, store_options),
            name="sparklab-prefix-io",
            daemon=True,
        )
        self._thread.start()

    def _run(self, root, compatibility, options):
        try:
            self.store = PrefixStore(root, compatibility, **options)
            self._observe()
        except Exception as exc:
            if self.store is not None:
                self.store.close()
            self.failure = exc
            self.ready.set()
            self._io_stopped.set()
            return
        self.ready.set()
        try:
            while not self._closing.is_set() or not self._jobs.empty():
                try:
                    ticket = self._jobs.get(timeout=0.05)
                except queue.Empty:
                    continue
                started = time.monotonic()
                try:
                    if ticket.cancelled:
                        pass
                    elif ticket.kind == "write":
                        ticket.result = self.store.write(ticket.payload)
                    else:
                        ticket.result = self.store.read(ticket.key, ticket.payload)
                except BaseException as exc:
                    ticket.error = exc
                    if not isinstance(exc, Exception):
                        self.failure = RuntimeError(
                            "prefix I/O worker terminated unexpectedly"
                        )
                finally:
                    finished = getattr(ticket.payload, "io_finished", None)
                    if finished is not None:
                        try:
                            finished()
                        except BaseException as exc:
                            ticket.error = exc
                            self.failure = RuntimeError(
                                "prefix transfer finalization failed"
                            )
                    try:
                        self._observe()
                    except BaseException as exc:
                        ticket.error = exc
                        self.failure = RuntimeError("prefix observation failed")
                    ticket.elapsed_seconds = time.monotonic() - started
                    ticket.finished.set()
                    self._done.put_nowait(ticket)
                if self.failure is not None:
                    break
        finally:
            # Filesystem ownership outlives even a terminal I/O loop while a
            # completed/unreported ticket still pins the namespace. The actor
            # only wakes this worker; it never closes/flocks files itself.
            self._io_stopped.set()
            while True:
                with self._mutex:
                    if not self._tickets:
                        break
                    self._released.clear()
                self._released.wait()
            self.store.close()

    def _observe(self):
        # Construct and replace only on the worker. The actor reads one completed
        # observation, never a concurrently-mutating Counter or filesystem path.
        usage = self.store.root_quota.usage
        entries = self.store.descriptors()
        self.observation = {
            "observed_at": time.monotonic(),
            "counters": dict(self.store.stats),
            "records": len(entries),
            "disk_bytes": sum(entry.nbytes for entry in entries),
            "root_observed_bytes": usage.nbytes,
            "root_observed_records": usage.records,
            "root_observed_namespaces": usage.namespaces,
            "root_observed_at": usage.observed_at,
            "quota_bytes": self.store.quota_bytes,
            "max_records": self.store.max_records,
        }

    def descriptors(self):
        if (
            not self.ready.is_set()
            or self.failure
            or self.store is None
            or self.store.failed
        ):
            return ()
        return self.store.descriptors()

    def _submit(self, kind, payload, key=None):
        if not self.ready.is_set() or self.failure or self._closing.is_set():
            return None
        with self._mutex:
            if (
                self._closing.is_set()
                or self.store.failed
                or not self._thread.is_alive()
                or len(self._tickets) >= self.max_tickets
            ):
                return None
            ticket = IOTicket(kind, payload, key)
            if kind == "read" and not self.store.pin(key):
                return None
            self._tickets[id(ticket)] = ticket
            self._jobs.put_nowait(ticket)
            return ticket

    def submit_write(self, source):
        return self._submit("write", source)

    def submit_read(self, key, sink):
        return self._submit("read", sink, key)

    def cancel(self, ticket):
        # Detach only: buffers and record pins live through worker completion.
        ticket.cancelled = True

    def poll(self):
        completed = []
        for _ in range(self.max_tickets):
            try:
                ticket = self._done.get_nowait()
                ticket.reported = True
                completed.append(ticket)
            except queue.Empty:
                break
        if self._io_stopped.is_set():
            # A terminal worker cannot retain an invisible queued ticket forever.
            # Its GPU resources still belong to the ticket until explicit release.
            with self._mutex:
                for ticket in self._tickets.values():
                    if not ticket.finished.is_set():
                        ticket.error = RuntimeError(
                            "prefix I/O worker stopped before dispatch"
                        )
                        finished = getattr(ticket.payload, "io_finished", None)
                        if finished is not None:
                            finished()
                        ticket.finished.set()
                    if not ticket.reported:
                        ticket.reported = True
                        completed.append(ticket)
        return completed

    def release(self, ticket):
        if not ticket.finished.is_set():
            raise RuntimeError("I/O users still own ticket buffers")
        if not ticket.reported:
            raise RuntimeError("poll completion before releasing a ticket")
        ready = getattr(ticket.payload, "transfer_complete", None)
        if ready is not None and not ready():
            raise RuntimeError("GPU users still own ticket buffers")
        with self._mutex:
            if self._tickets.pop(id(ticket), None) is None:
                raise AssertionError("ticket released twice")
            if ticket.kind == "read":
                self.store.unpin(ticket.key)
            ticket.payload = None
            ticket.result = None
            self._released.set()

    @property
    def outstanding(self):
        with self._mutex:
            return len(self._tickets)

    def close(self, timeout=5.0):
        """Shutdown/idle-only drain. False means ownership MUST remain retained."""
        with self._mutex:
            self._closing.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()
