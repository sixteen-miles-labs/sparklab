"""Root-wide filesystem oracle across identities, owners, crash debris and pins."""

import fcntl
import json
import threading

import pytest

from sparklab.runtime.checkpoint.async_store import AsyncPrefixStore
from sparklab.runtime.checkpoint.record import TensorSource, RecordLimits, RecordError
from sparklab.runtime.checkpoint.state import StateImage
from sparklab.runtime.checkpoint.store import PrefixStore, StoreBusy, StoreFull
from tests.runtime.checkpoint.test_records import identity, image, Sink


def source(scope, tokens=(1, 2, 3, 4)):
    state = image(tokens)
    return TensorSource(
        StateImage(identity(scope), state.tokens, state.layout, state.inspect())
    )


def open_store(root, scope, *, quota=5000, records=3):
    return PrefixStore(
        root,
        identity(scope),
        quota_bytes=quota,
        max_records=records,
        limits=RecordLimits(min(4096, quota), 4096),
    )


def disk_usage(root):
    # Independent on-disk observation, never the store's counters/index.
    files = [
        file
        for directory in root.iterdir()
        if directory.is_dir()
        for file in directory.iterdir()
        if file.name.endswith(".pxc")
    ]
    return sum(file.stat().st_size for file in files), len(files)


def test_repeated_identity_changes_share_byte_record_and_namespace_bounds(tmp_path):
    for i in range(40):
        store = open_store(tmp_path, f"scope-{i}")
        try:
            entry = store.write(source(f"scope-{i}"))
            assert store.read(entry.key)[0]["scope"] == f"scope-{i}"
            nbytes, records = disk_usage(tmp_path)
            assert nbytes <= 5000 and records <= 3
            assert len([d for d in tmp_path.iterdir() if d.is_dir()]) <= 4
        finally:
            store.close()
    limits = json.loads((tmp_path / ".quota.json").read_text())
    assert limits == {"version": 1, "bytes": 5000, "records": 3}


def test_empty_inactive_namespaces_do_not_accumulate(tmp_path):
    for i in range(40):
        store = open_store(tmp_path, str(i), records=1)
        store.close()
        assert len([d for d in tmp_path.iterdir() if d.is_dir()]) == 1
    assert disk_usage(tmp_path) == (0, 0)


def test_other_active_namespace_is_protected_even_if_it_has_no_reader(tmp_path):
    a = open_store(tmp_path, "A", records=1)
    first = a.write(source("A"))
    b = open_store(tmp_path, "B", records=1)
    try:
        with pytest.raises(StoreFull, match="protected"):
            b.write(source("B"))
        assert a.read(first.key)[0]["scope"] == "A"
        assert disk_usage(tmp_path) == (first.nbytes, 1)
        a.close()
        last = b.write(source("B"))
        assert b.read(last.key)[0]["scope"] == "B"
        assert not a.namespace.exists()
    finally:
        a.close()
        b.close()


def test_quota_change_requires_other_owners_to_drain_then_reclaims(tmp_path):
    a = open_store(tmp_path, "A", quota=5000, records=4)
    for i in range(3):
        a.write(source("A", (i, i + 1, i + 2, i + 3)))
    original_limits = (tmp_path / ".quota.json").read_bytes()
    with pytest.raises(StoreBusy, match="limits"):
        open_store(tmp_path, "B", quota=2000, records=2)
    assert (tmp_path / ".quota.json").read_bytes() == original_limits
    a.close()
    b = open_store(tmp_path, "B", quota=2000, records=2)
    try:
        b.write(source("B"))
        nbytes, records = disk_usage(tmp_path)
        assert nbytes <= 2000 and records <= 2
        assert json.loads((tmp_path / ".quota.json").read_text())["bytes"] == 2000
    finally:
        b.close()


def test_simultaneous_writers_skip_pressure_without_waiting_or_staging(tmp_path):
    a, b = open_store(tmp_path, "A"), open_store(tmp_path, "B")
    entered, proceed = threading.Event(), threading.Event()
    plain = source("A")
    errors = []

    class Paused:
        compatibility, tokens, layout = plain.compatibility, plain.tokens, plain.layout

        def chunks(self, name, size):
            entered.set()
            assert proceed.wait(5)
            yield from plain.chunks(name, size)

    def write():
        try:
            a.write(Paused())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=write)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(StoreBusy, match="root"):
            b.write(source("B"))
        proceed.set()
        worker.join(5)
        assert not worker.is_alive() and not errors
        b.write(source("B"))
        assert disk_usage(tmp_path)[1] == 2
    finally:
        proceed.set()
        worker.join(5)
        a.close()
        b.close()


def test_shutdown_keeps_namespace_until_undrained_gpu_read_pin_is_released(
    tmp_path, monkeypatch
):
    seed = open_store(tmp_path, "A", records=1)
    first = seed.write(source("A"))
    seed.close()

    class FencedSink(Sink):
        fence = False

        def transfer_complete(self):
            return self.fence

    sink = FencedSink()
    a = AsyncPrefixStore(
        tmp_path,
        identity("A"),
        max_tickets=1,
        quota_bytes=5000,
        max_records=1,
        limits=RecordLimits(4096, 4096),
    )
    assert a.ready.wait(5) and a.failure is None
    read = a.submit_read(first.key, sink)
    assert read.finished.wait(5) and read.error is None
    assert a.poll() == [read]
    assert not a.close(timeout=0)
    assert a._io_stopped.wait(5)
    b = open_store(tmp_path, "B", records=1)
    try:
        with pytest.raises(StoreFull):
            b.write(source("B"))
        with pytest.raises(RuntimeError, match="GPU"):
            a.release(read)
        assert a.store.pins[first.key] == 1
        sink.fence = True
        actor = threading.get_ident()
        original_close, original_flock = __import__("os").close, fcntl.flock

        def close(fd):
            assert threading.get_ident() != actor, "actor performed filesystem cleanup"
            return original_close(fd)

        def flock(fd, operation):
            assert threading.get_ident() != actor, "actor performed filesystem locking"
            return original_flock(fd, operation)

        with monkeypatch.context() as guard:
            guard.setattr("os.close", close)
            guard.setattr(fcntl, "flock", flock)
            a.release(read)
        assert a.close(timeout=5)
        b.write(source("B"))
        assert disk_usage(tmp_path)[1] == 1
    finally:
        assert a.close(timeout=5)
        b.close()


def test_stale_temp_cleanup_and_foreign_files_or_symlinks_are_preserved(tmp_path):
    old = open_store(tmp_path, "A")
    namespace = old.namespace
    old.close()
    temp = namespace / (".tmp-" + "1" * 64 + "-" + "2" * 32 + ".pxc")
    temp.write_bytes(b"incomplete owned cache")
    foreign = namespace / "keep.txt"
    foreign.write_text("not owned by the prefix store")
    target = tmp_path / "outside"
    target.write_text("must remain")
    (namespace / ("3" * 64 + ".pxc")).symlink_to(target)
    (tmp_path / (".quota-" + "4" * 32 + ".tmp")).write_text("incomplete metadata")
    fresh = open_store(tmp_path, "B")
    try:
        assert not temp.exists()
        assert foreign.read_text() == "not owned by the prefix store"
        assert target.read_text() == "must remain"
        assert (namespace / ("3" * 64 + ".pxc")).is_symlink()
        assert not list(tmp_path.glob(".quota-*.tmp"))
    finally:
        fresh.close()


@pytest.mark.parametrize(
    "data", [b"{", b"x" * 4097, b'{"version":2,"bytes":5000,"records":3}']
)
def test_corrupt_or_future_root_metadata_fails_closed_without_deleting_records(
    tmp_path, data
):
    store = open_store(tmp_path, "A")
    entry = store.write(source("A"))
    record = store.namespace / (entry.key + ".pxc")
    before = record.read_bytes()
    store.close()
    (tmp_path / ".quota.json").write_bytes(data)
    with pytest.raises(RecordError):
        open_store(tmp_path, "B")
    assert record.read_bytes() == before


def test_global_byte_quota_binds_before_record_limit(tmp_path):
    for i in range(10):
        store = open_store(tmp_path, str(i), quota=2500, records=8)
        try:
            store.write(source(str(i)))
            actual_bytes, count = disk_usage(tmp_path)
            assert actual_bytes <= 2500 and count <= 2
        finally:
            store.close()


def test_killed_writer_releases_root_lock_and_new_identity_recovers_debris(tmp_path):
    import subprocess
    import sys

    old = open_store(tmp_path, "A", records=2)
    original = old.write(source("A"))
    path = old.namespace / (original.key + ".pxc")
    before = path.read_bytes()
    old.close()
    program = """
import os, sys
from tests.runtime.checkpoint.test_root_quota import open_store, source
s = source('A', (5,6,7,8))
class Fatal:
    compatibility, tokens, layout = s.compatibility, s.tokens, s.layout
    def chunks(self, name, budget):
        os._exit(17)
        yield b''
store = open_store(sys.argv[1], 'A', records=2)
store.write(Fatal())
"""
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path)], timeout=10)
    assert result.returncode == 17
    assert list(old.namespace.glob(".tmp-*"))
    new = open_store(tmp_path, "B", records=2)
    try:
        new.write(source("B"))
        assert not list(old.namespace.glob(".tmp-*"))
        assert path.read_bytes() == before
        assert disk_usage(tmp_path)[0] <= 5000
    finally:
        new.close()


def test_discovery_is_bounded_before_loading_an_unbounded_index(tmp_path, monkeypatch):
    store = open_store(tmp_path, "A")
    store.close()
    for i in range(8):
        (tmp_path / f"foreign-{i}").touch()
    monkeypatch.setattr("sparklab.runtime.checkpoint.quota.MAX_SCAN_ENTRIES", 4)
    with pytest.raises(StoreFull, match="discovery bound"):
        open_store(tmp_path, "B")
    assert len(list(tmp_path.glob("foreign-*"))) == 8


def test_root_metadata_fsync_failure_releases_locks_and_restart_cleans_up(
    tmp_path, monkeypatch
):
    import os

    with monkeypatch.context() as fault:

        def fail(fd):
            raise OSError("injected root fsync failure")

        fault.setattr(os, "fsync", fail)
        with pytest.raises(OSError):
            open_store(tmp_path, "A")
    assert not list(tmp_path.glob(".quota-*.tmp"))
    store = open_store(tmp_path, "A")
    try:
        store.write(source("A"))
        assert disk_usage(tmp_path)[1] == 1
    finally:
        store.close()


def test_failed_orphan_cleanup_disables_admission_and_retains_root_accounting(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from sparklab.runtime.checkpoint.errors import StoreUnavailable

    a = AsyncPrefixStore(
        tmp_path,
        identity("A"),
        max_tickets=1,
        quota_bytes=5000,
        max_records=2,
        limits=RecordLimits(4096, 4096),
    )
    assert a.ready.wait(5) and a.failure is None
    original_unlink = Path.unlink

    def fail_sync():
        raise OSError("injected directory fsync failure")

    def deny_cleanup(path, *args, **kwargs):
        if path.suffix == ".pxc" and not path.name.startswith(".tmp-"):
            raise PermissionError("injected orphan cleanup failure")
        return original_unlink(path, *args, **kwargs)

    try:
        with monkeypatch.context() as fault:
            fault.setattr(a.store, "_sync_directory", fail_sync)
            fault.setattr(Path, "unlink", deny_cleanup)
            ticket = a.submit_write(source("A"))
            assert ticket.finished.wait(5)
            assert isinstance(ticket.error, OSError) and a.store.failed
        assert a.poll() == [ticket]
        a.release(ticket)
        assert not a.descriptors() and a.submit_write(source("A")) is None
        with pytest.raises(StoreUnavailable):
            a.store.read(identity("A").prefix_key((1, 2, 3, 4)))
        assert disk_usage(tmp_path)[1] == 1
        assert a.close()
        # Recovery validates the complete orphan and adopts its real byte count.
        recovered = open_store(tmp_path, "A", records=2)
        try:
            assert len(recovered.descriptors()) == 1
            assert recovered.bytes == disk_usage(tmp_path)[0]
        finally:
            recovered.close()
    finally:
        assert a.close()


def test_repeated_startup_and_cleanup_failures_cannot_grow_namespaces(
    tmp_path, monkeypatch
):
    from sparklab.runtime.checkpoint.quota import RootQuota

    def fail(*args):
        raise PermissionError("injected metadata/cleanup failure")

    with monkeypatch.context() as fault:
        fault.setattr(RootQuota, "commit_limits", fail)
        fault.setattr(RootQuota, "remove_empty", fail)
        for i in range(10):
            with pytest.raises(PermissionError):
                open_store(tmp_path, str(i), records=1)
            assert len([d for d in tmp_path.iterdir() if d.is_dir()]) <= 1
    recovered = open_store(tmp_path, "recovered", records=1)
    recovered.close()
    assert len([d for d in tmp_path.iterdir() if d.is_dir()]) == 1
