"""Worker-only root accounting across compatibility namespaces.

A short-lived root mutation lock serializes admission/publication. Each store's
lifetime namespace lock protects all its records, including read/restore pins.
Inactive namespaces can be reclaimed without loading their prompt/header data.
Scans retain one eviction candidate, never an unbounded cross-identity index.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import time
import uuid

from .errors import StoreBusy, StoreFull
from .record import HEX, RecordError

TEMP = re.compile(r"^\.tmp-[0-9a-f]{64}-[0-9a-f]{32}\.pxc$")
MAX_SCAN_ENTRIES = 32768


@dataclass(frozen=True)
class RootUsage:
    nbytes: int = 0
    records: int = 0
    namespaces: int = 0
    observed_at: float | None = None


class RootQuota:
    def __init__(self, root, quota_bytes, max_records):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.quota_bytes = quota_bytes
        self.max_records = max_records
        self.fd = os.open(
            self.root / ".quota.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        self.usage = RootUsage()
        self._limits = None

    @contextmanager
    def mutation(self):
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StoreBusy("prefix root has a mutation in progress") from exc
        try:
            yield
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def _namespaces(self):
        with os.scandir(self.root) as entries:
            for count, entry in enumerate(entries):
                if count >= MAX_SCAN_ENTRIES:
                    raise StoreFull("prefix root discovery bound exceeded")
                if HEX.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False):
                    yield Path(entry.path)

    @contextmanager
    def _lease(self, namespace):
        try:
            fd = os.open(namespace / ".lock", os.O_RDWR | os.O_NOFOLLOW)
        except FileNotFoundError:
            yield False
            return
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
            else:
                try:
                    yield True
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def check_limits(self, own_namespace, *, allow_update=True):
        path = self.root / ".quota.json"
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            limits = None
        else:
            try:
                with os.fdopen(fd, "rb") as source:
                    data = source.read(4097)
                if len(data) > 4096:
                    raise RecordError("oversized root quota metadata")
                limits = json.loads(data)
            except (ValueError, TypeError, RecursionError) as exc:
                raise RecordError("invalid root quota metadata") from exc
            if (
                not isinstance(limits, dict)
                or set(limits) != {"version", "bytes", "records"}
                or type(limits["version"]) is not int
                or limits["version"] != 1
                or type(limits["bytes"]) is not int
                or not 1 <= limits["bytes"] <= 1 << 50
                or type(limits["records"]) is not int
                or not 1 <= limits["records"] <= 4096
            ):
                raise RecordError("unsupported root quota metadata")
        with os.scandir(self.root) as files:
            for count, item in enumerate(files):
                if count >= MAX_SCAN_ENTRIES:
                    raise StoreFull("prefix root discovery bound exceeded")
                if re.fullmatch(
                    r"\.quota-[0-9a-f]{32}\.tmp", item.name
                ) and item.is_file(follow_symlinks=False):
                    os.unlink(item.path)
        target = {"version": 1, "bytes": self.quota_bytes, "records": self.max_records}
        if limits != target:
            if not allow_update:
                raise StoreBusy("prefix root limits changed during serving")
            for namespace in self._namespaces():
                if namespace == own_namespace:
                    continue
                with self._lease(namespace) as inactive:
                    if not inactive:
                        raise StoreBusy(
                            "cannot change prefix root limits while another namespace is active"
                        )
        self._limits = limits

    def commit_limits(self):
        target = {"version": 1, "bytes": self.quota_bytes, "records": self.max_records}
        if self._limits == target:
            return
        temp = self.root / f".quota-{uuid.uuid4().hex}.tmp"
        try:
            fd = os.open(
                temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "wb") as output:
                output.write(json.dumps(target, sort_keys=True).encode())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, self.root / ".quota.json")
            fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            self._limits = target
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def remove_empty(namespace):
        # Called under root mutation + inactive namespace ownership. Unknown
        # files and symlinks are preserved. Never recursively remove a directory.
        with os.scandir(namespace) as entries:
            if any(entry.name != ".lock" for entry in entries):
                return False
        (namespace / ".lock").unlink()
        namespace.rmdir()
        return True

    def _scan(self, store):
        nbytes = records = namespaces = 0
        victim = None
        inspected = 0
        for namespace in self._namespaces():
            if namespace == store.namespace:
                inactive = False
                manager = _active_namespace()
            else:
                manager = self._lease(namespace)
            with manager as inactive:
                if inactive and self.remove_empty(namespace):
                    store.stats["root_namespaces_removed"] += 1
                    continue
                namespaces += 1
                with os.scandir(namespace) as entries:
                    for entry in entries:
                        inspected += 1
                        if inspected > MAX_SCAN_ENTRIES:
                            raise StoreFull(
                                "prefix root record discovery bound exceeded"
                            )
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        temporary = bool(TEMP.fullmatch(entry.name))
                        committed = entry.name.endswith(".pxc") and bool(
                            HEX.fullmatch(entry.name[:-4])
                        )
                        if not temporary and not committed:
                            continue
                        if temporary and inactive:
                            os.unlink(entry.path)
                            store.stats["root_temps_removed"] += 1
                            continue
                        try:
                            metadata = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        nbytes += metadata.st_size
                        records += 1
                        if inactive and committed:
                            candidate = (
                                metadata.st_mtime_ns,
                                entry.path,
                                metadata.st_ino,
                            )
                            if victim is None or candidate < victim:
                                victim = candidate
        return RootUsage(nbytes, records, namespaces, time.monotonic()), victim

    def enforce(self, store, added_bytes=0, added_records=0, added_namespaces=0):
        """Under root mutation lock; protected namespaces are never evicted."""
        if added_bytes > self.quota_bytes or added_records > self.max_records:
            raise StoreFull("record exceeds prefix root quota")
        while True:
            usage, victim = self._scan(store)
            self.usage = usage
            if (
                usage.nbytes + added_bytes <= self.quota_bytes
                and usage.records + added_records <= self.max_records
                and usage.namespaces + added_namespaces <= self.max_records + 1
            ):
                return
            if victim is not None:
                _, filename, inode = victim
                path = Path(filename)
                with self._lease(path.parent) as inactive:
                    if not inactive:
                        raise StoreBusy("prefix eviction victim changed owners")
                    if path.stat(follow_symlinks=False).st_ino != inode:
                        raise StoreBusy("prefix eviction victim changed identity")
                    path.unlink()
                    store.stats["root_evictions"] += 1
            elif not store._evict_one():
                raise StoreFull("prefix root quota is protected or full")

    def published(self, size):
        self.usage = RootUsage(
            self.usage.nbytes + size,
            self.usage.records + 1,
            self.usage.namespaces,
            time.monotonic(),
        )

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@contextmanager
def _active_namespace():
    yield False
