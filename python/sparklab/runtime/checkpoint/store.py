"""Worker-owned bounded prefix files with atomic publication and read pins."""

from __future__ import annotations
from collections import Counter, OrderedDict
from dataclasses import dataclass
import fcntl
import os
import re
import threading
import uuid

from .errors import StoreFull, StoreBusy, StoreUnavailable
from .quota import RootQuota

from .record import (
    HEX,
    PREFIX,
    RecordError,
    RecordLimits,
    _json,
    make_header,
    read_record,
    write_record,
)


@dataclass(frozen=True)
class DiskEntry:
    key: str
    consumed: int
    nbytes: int


class PrefixStore:
    """Synchronous I/O implementation; serving uses AsyncPrefixStore exclusively.

    The index stores only fixed-size descriptors, not prompts or whole headers.
    Each self-contained record owns its complete logical tensor chain. Namespace
    access controls are inherited from the deployment; no encryption is claimed.
    """

    def __init__(
        self, root, compatibility, *, quota_bytes=8 << 30, max_records=128, limits=None
    ):
        if (
            not HEX.fullmatch(compatibility.fingerprint)
            or not isinstance(compatibility.scope, str)
            or not compatibility.scope
        ):
            raise ValueError("invalid namespace compatibility")
        if (
            type(quota_bytes) is not int
            or type(max_records) is not int
            or not 1 <= quota_bytes <= 1 << 50
            or not 1 <= max_records <= 4096
        ):
            raise ValueError("invalid disk/index bounds")
        self.compatibility = compatibility
        self.limits = limits or RecordLimits(max_record_bytes=min(quota_bytes, 4 << 30))
        if self.limits.max_record_bytes > quota_bytes:
            raise ValueError("record bound exceeds disk quota")
        self.quota_bytes = quota_bytes
        self.max_records = max_records
        self.entries = OrderedDict()
        self.pins = Counter()
        self.stats = Counter()
        self.bytes = 0
        self.reserved_bytes = 0
        self._mutex = threading.Lock()
        self._writer = threading.Lock()
        self.failed = False
        self.closed = False
        self._lock_fd = None
        self.root_quota = RootQuota(root, quota_bytes, max_records)
        self.namespace = self.root_quota.root / compatibility.fingerprint
        try:
            with self.root_quota.mutation():
                self.root_quota.check_limits(self.namespace)
                created = not self.namespace.exists()
                if created:
                    # Account before mkdir too: repeated startup/cleanup failures
                    # must not accumulate a new directory for every identity.
                    self.root_quota.enforce(self, added_namespaces=1)
                try:
                    self.namespace.mkdir(mode=0o700, exist_ok=True)
                    if self.namespace.is_symlink():
                        raise ValueError("prefix namespace must not be a symlink")
                    fd = os.open(
                        self.namespace / ".lock",
                        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                        0o600,
                    )
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError as exc:
                        os.close(fd)
                        raise StoreBusy(
                            "prefix namespace already has a writer"
                        ) from exc
                    self._lock_fd = fd
                    self._recover()
                    self.root_quota.enforce(self)
                    self.root_quota.commit_limits()
                except BaseException:
                    self._close_namespace()
                    if (
                        created
                        and self.namespace.is_dir()
                        and not self.namespace.is_symlink()
                    ):
                        self.root_quota.remove_empty(self.namespace)
                    raise
        except BaseException:
            self.close()
            raise

    def _path(self, key):
        if not isinstance(key, str) or not HEX.fullmatch(key):
            raise ValueError("invalid record key")
        return self.namespace / (key + ".pxc")

    def _check(self):
        if self.closed or self.failed:
            raise StoreUnavailable("prefix store is unavailable")

    def _sync_directory(self):
        fd = os.open(self.namespace, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _release_page_cache(fd):
        # Best effort for this owned file only. Staged retention must not become
        # a second long-lived RAM tier on coherent-memory GB10. Never globally
        # drop caches, and do not turn an unsupported advisory into a miss.
        advise = getattr(os, "posix_fadvise", None)
        if advise is not None:
            try:
                advise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass

    def _recover(self):
        temp = re.compile(r"^\.tmp-[0-9a-f]{64}-[0-9a-f]{32}\.pxc$")
        with os.scandir(self.namespace) as files:
            for item in files:
                if not item.is_file(follow_symlinks=False):
                    continue
                if temp.fullmatch(item.name):
                    os.unlink(item.path)
                    self.stats["incomplete_temps_removed"] += 1
                    continue
                if not item.name.endswith(".pxc") or not HEX.fullmatch(item.name[:-4]):
                    continue
                key = item.name[:-4]
                fd = os.open(item.path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    header, _ = read_record(fd, self.compatibility, self.limits)
                    if header["key"] != key:
                        raise RecordError("filename/key mismatch")
                    size = os.fstat(fd).st_size
                    self._release_page_cache(fd)
                except (OSError, RecordError):
                    os.close(fd)
                    os.unlink(item.path)
                    self.stats["recovery_rejected"] += 1
                    continue
                else:
                    os.close(fd)
                self._make_room(size)
                self.entries[key] = DiskEntry(key, len(header["tokens"]), size)
                self.bytes += size
                self.stats["recovered"] += 1

    def _evict_one(self):
        with self._mutex:
            victim = next(
                (e for key, e in self.entries.items() if not self.pins[key]), None
            )
            if victim is None:
                return False
            del self.entries[victim.key]
            self.bytes -= victim.nbytes
        try:
            os.unlink(self._path(victim.key))
        except BaseException:
            with self._mutex:
                self.entries[victim.key] = victim
                self.bytes += victim.nbytes
            raise
        self.stats["evictions"] += 1
        return True

    def _make_room(self, size):
        if size > self.quota_bytes:
            raise StoreFull("record exceeds disk quota")
        while (
            self.bytes + self.reserved_bytes + size > self.quota_bytes
            or len(self.entries) >= self.max_records
        ):
            if not self._evict_one():
                raise StoreFull("disk quota/index is pinned or full")

    def descriptors(self):
        """Small immutable index snapshot. Never performs filesystem I/O."""
        with self._mutex:
            return tuple(self.entries.values())

    def pin(self, key):
        with self._mutex:
            if key not in self.entries or self.closed or self.failed:
                return False
            self.pins[key] += 1
            return True

    def unpin(self, key):
        with self._mutex:
            if self.pins[key] <= 0:
                raise AssertionError("record read pin underflow")
            self.pins[key] -= 1
            if not self.pins[key]:
                del self.pins[key]

    def write(self, source):
        if not self._writer.acquire(blocking=False):
            raise StoreBusy("prefix namespace already has a mutation in progress")
        try:
            with self.root_quota.mutation():
                self.root_quota.check_limits(self.namespace, allow_update=False)
                return self._write_owned(source)
        finally:
            self._writer.release()

    def _write_owned(self, source):
        self._check()
        if source.compatibility != self.compatibility:
            raise RecordError("incompatible publication")
        header = make_header(source)
        key = header["key"]
        with self._mutex:
            if key in self.entries:
                self.entries.move_to_end(key)
                self.stats["duplicate_publications"] += 1
                return self.entries[key]
        predicted = PREFIX.size + len(_json(header)) + header["payload_bytes"]
        if predicted > self.limits.max_record_bytes:
            raise StoreFull("record exceeds configured size")
        self._make_room(predicted)
        self.root_quota.enforce(self, predicted, 1)
        self.reserved_bytes = predicted
        temp = self.namespace / f".tmp-{key}-{uuid.uuid4().hex}.pxc"
        final = self._path(key)
        renamed = False
        try:
            fd = os.open(
                temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            try:
                _, actual = write_record(fd, source, self.limits)
                if actual != predicted:
                    raise AssertionError("publication reservation mismatch")
                os.fsync(fd)
                self._release_page_cache(fd)
            finally:
                os.close(fd)
            os.replace(temp, final)
            renamed = True
            self._sync_directory()
            entry = DiskEntry(key, len(source.tokens), actual)
            with self._mutex:
                self.entries[key] = entry
                self.bytes += actual
            self.root_quota.published(actual)
            self.stats["writes"] += 1
            return entry
        except BaseException:
            self.stats["write_failures"] += 1
            try:
                if renamed:
                    final.unlink(missing_ok=True)
                temp.unlink(missing_ok=True)
            except OSError:
                # Unknown orphan bytes cannot silently disappear from admission.
                # Fail closed until a new worker recovers the namespace.
                self.failed = True
            raise
        finally:
            self.reserved_bytes = 0

    def read(self, key, sink=None):
        self._check()
        if not self.pin(key):
            raise KeyError("prefix record was evicted")
        try:
            fd = os.open(self._path(key), os.O_RDONLY | os.O_NOFOLLOW)
            try:
                header, layout = read_record(fd, self.compatibility, self.limits, sink)
                if header["key"] != key:
                    raise RecordError("record filename identity mismatch")
                validated = getattr(sink, "validated_record", None)
                if validated is not None:
                    validated(header, layout)
                self._release_page_cache(fd)
            finally:
                os.close(fd)
            os.utime(self._path(key), None, follow_symlinks=False)
            with self._mutex:
                self.entries.move_to_end(key)
            self.stats["reads"] += 1
            return header, layout
        except (OSError, RecordError):
            self.stats["read_failures"] += 1
            raise
        finally:
            self.unpin(key)

    def _close_namespace(self):
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def close(self):
        if not self.closed:
            self.closed = True
            self._close_namespace()
            self.root_quota.close()
