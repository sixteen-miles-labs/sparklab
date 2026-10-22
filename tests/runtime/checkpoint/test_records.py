"""Real filesystem records and independent byte/length/failure oracles."""

import errno
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from sparklab.runtime.checkpoint.state import Compatibility, StateImage, StateLayout
from sparklab.runtime.checkpoint.record import (
    MAGIC,
    PREFIX,
    MAX_HEADER,
    RecordError,
    RecordLimits,
    TensorSource,
    read_record,
    write_record,
)
from sparklab.runtime.checkpoint.store import PrefixStore, StoreBusy, StoreFull
from sparklab.runtime.checkpoint.async_store import AsyncPrefixStore


def identity(scope="single-instance", artifact="weights-a"):
    return Compatibility.from_config(
        dict(
            artifact=artifact,
            precision="bf16",
            model_config="c",
            tokenizer="t",
            template="j",
            state_layout="mha-gdn-v1",
            cache_geometry={"page_size": 4},
            continuation="target-only",
            scope=scope,
        )
    )


def image(tokens=(1, 2, 3, 4)):
    tensors = {
        "kv": torch.arange(2 * len(tokens) * 2, dtype=torch.float32).reshape(
            2, 1, len(tokens), 1, 2
        ),
        "conv": torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16),
        "recurrent": torch.tensor([[[7.0, 11.0], [13.0, 17.0]]]),
        "aux.ple": torch.tensor([1, 2, 3], dtype=torch.int64),
    }
    layout = StateLayout(
        tuple(sorted((k, tuple(t.shape), str(t.dtype)) for k, t in tensors.items())),
        4,
        None,
    )
    return StateImage(identity(), tokens, layout, tensors)


class Sink:
    def __init__(self):
        self.data = {}
        self.header = None
        self.published = False

    def begin(self, header, layout):
        self.header = header
        self.layout = layout
        self.data = {p["name"]: bytearray(p["nbytes"]) for p in header["components"]}

    def chunk(self, name, offset, data):
        self.data[name][offset : offset + len(data)] = data


def test_record_roundtrip_has_exact_raw_bytes_and_no_physical_ids(tmp_path):
    source = TensorSource(image())
    store = PrefixStore(
        tmp_path,
        identity(),
        quota_bytes=12000,
        max_records=3,
        limits=RecordLimits(4096, 4096),
    )
    try:
        entry = store.write(source)
        sink = Sink()
        header, layout = store.read(entry.key, sink)
        assert layout == source.layout
        for name, tensor in source.image._tensors.items():
            assert bytes(sink.data[name]) == tensor.view(torch.uint8).numpy().tobytes()
        assert header["tokens"] == [1, 2, 3, 4]
        assert "page" not in "".join(p["name"] for p in header["components"])
        assert store.reserved_bytes == 0 and store.bytes == entry.nbytes
        assert list(store.namespace.glob("*.pxc")) == [
            store.namespace / (entry.key + ".pxc")
        ]
    finally:
        store.close()
    recovered = PrefixStore(
        tmp_path,
        identity(),
        quota_bytes=12000,
        max_records=3,
        limits=RecordLimits(4096, 4096),
    )
    try:
        assert recovered.stats["recovered"] == 1
        assert recovered.read(entry.key)[0]["key"] == entry.key
    finally:
        recovered.close()


@pytest.mark.parametrize(
    "scope,artifact", [("other", "weights-a"), ("single-instance", "weights-b")]
)
def test_identity_changes_cannot_read_a_record(tmp_path, scope, artifact):
    fd = os.open(tmp_path / "record", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        write_record(fd, TensorSource(image()), RecordLimits(4096, 4096))
        with pytest.raises(RecordError, match="identity"):
            read_record(fd, identity(scope, artifact), RecordLimits(4096, 4096))
    finally:
        os.close(fd)


def test_header_length_and_payload_corruption_reject_without_shared_publication(
    tmp_path,
):
    store = PrefixStore(
        tmp_path, identity(), quota_bytes=12000, limits=RecordLimits(4096, 4096)
    )
    try:
        entry = store.write(TensorSource(image()))
        path = store.namespace / (entry.key + ".pxc")
        original = path.read_bytes()
        path.write_bytes(
            PREFIX.pack(MAGIC, MAX_HEADER + 1, bytes(32)) + original[PREFIX.size :]
        )
        sink = Sink()
        with pytest.raises(RecordError, match="framing"):
            store.read(entry.key, sink)
        assert sink.header is None
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        sink = Sink()
        with pytest.raises(RecordError, match="checksum"):
            store.read(entry.key, sink)
        assert not sink.published
        path.write_bytes(original[:-1])
        with pytest.raises(RecordError, match="length"):
            store.read(entry.key)
        assert store.stats["read_failures"] == 3
    finally:
        store.close()
    recovered = PrefixStore(
        tmp_path, identity(), quota_bytes=12000, limits=RecordLimits(4096, 4096)
    )
    try:
        assert not recovered.descriptors() and recovered.stats["recovery_rejected"] == 1
    finally:
        recovered.close()


def test_quota_eviction_respects_read_pins_and_duplicate_publications(tmp_path):
    store = PrefixStore(
        tmp_path,
        identity(),
        quota_bytes=5000,
        max_records=2,
        limits=RecordLimits(4096, 4096),
    )
    try:
        a = store.write(TensorSource(image((1, 2, 3, 4))))
        b = store.write(TensorSource(image((5, 6, 7, 8))))
        assert store.pin(a.key) and store.pin(b.key)
        assert store.write(TensorSource(image((1, 2, 3, 4)))) == a
        with pytest.raises(StoreFull):
            store.write(TensorSource(image((9, 10, 11, 12))))
        store.unpin(b.key)
        c = store.write(TensorSource(image((9, 10, 11, 12))))
        assert {e.key for e in store.descriptors()} == {a.key, c.key}
        assert store.bytes <= store.quota_bytes
        store.unpin(a.key)
    finally:
        store.close()


def test_namespace_lock_and_scope_isolation(tmp_path):
    store = PrefixStore(
        tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    try:
        with pytest.raises(StoreBusy):
            PrefixStore(
                tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
            )
        separate = PrefixStore(
            tmp_path,
            identity("other"),
            quota_bytes=5000,
            limits=RecordLimits(4096, 4096),
        )
        assert separate.namespace != store.namespace
        separate.close()
    finally:
        store.close()


def test_short_io_loops_are_complete_and_enospc_leaves_no_record(tmp_path, monkeypatch):
    store = PrefixStore(
        tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    write, read = os.write, os.read
    monkeypatch.setattr(
        os, "write", lambda fd, data: write(fd, data[: min(7, len(data))])
    )
    monkeypatch.setattr(os, "read", lambda fd, n: read(fd, min(5, n)))
    try:
        entry = store.write(TensorSource(image()))
        assert store.read(entry.key)[0]["tokens"] == [1, 2, 3, 4]

        def full(*_):
            raise OSError(errno.ENOSPC, "injected full disk")

        monkeypatch.setattr(os, "write", full)
        with pytest.raises(OSError):
            store.write(TensorSource(image((2, 3, 4, 5))))
        assert store.bytes == entry.nbytes and store.reserved_bytes == 0
        assert len(list(store.namespace.glob("*.pxc"))) == 1
        assert not list(store.namespace.glob(".tmp-*"))
    finally:
        store.close()


def test_killed_writer_recovers_only_complete_records(tmp_path):
    store = PrefixStore(
        tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    entry = store.write(TensorSource(image()))
    namespace = store.namespace
    store.close()
    script = """
import os,sys
sys.path.insert(0,sys.argv[2])
from test_records import image,identity
from sparklab.runtime.checkpoint.record import TensorSource,RecordLimits
from sparklab.runtime.checkpoint.store import PrefixStore
s=PrefixStore(sys.argv[1],identity(),quota_bytes=5000,limits=RecordLimits(4096,4096))
os.replace=lambda *_: os._exit(19)
s.write(TensorSource(image((5,6,7,8))))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(Path(__file__).parent)],
        timeout=30,
    )
    assert process.returncode == 19 and list(namespace.glob(".tmp-*"))
    recovered = PrefixStore(
        tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    try:
        assert {e.key for e in recovered.descriptors()} == {entry.key}
        assert not list(namespace.glob(".tmp-*"))
        assert recovered.stats["incomplete_temps_removed"] == 1
    finally:
        recovered.close()


def test_async_queue_cancellation_and_shutdown_keep_buffers_owned(tmp_path):
    source = TensorSource(image())
    entered = threading.Event()
    proceed = threading.Event()
    thread_ids = []

    class DelayedSource:
        compatibility = source.compatibility
        tokens = source.tokens
        layout = source.layout

        def chunks(self, name, budget):
            thread_ids.append(threading.get_ident())
            entered.set()
            assert proceed.wait(5)
            yield from source.chunks(name, budget)

    store = AsyncPrefixStore(
        tmp_path,
        identity(),
        max_tickets=1,
        quota_bytes=5000,
        limits=RecordLimits(4096, 4096),
    )
    assert store.ready.wait(5) and store.failure is None
    ticket = store.submit_write(DelayedSource())
    try:
        assert ticket is not None and entered.wait(5)
        assert store.submit_write(source) is None
        store.cancel(ticket)
        with pytest.raises(RuntimeError, match="users"):
            store.release(ticket)
        assert not store.close(timeout=0)
        assert ticket.payload is not None and store.outstanding == 1
        proceed.set()
        assert ticket.finished.wait(5)
        assert not store.close(timeout=0)
        assert store.poll() == [ticket]
        assert store.outstanding == 1  # polling alone must not drop transfer ownership
        store.release(ticket)
        assert store.close(timeout=5)
        assert store.outstanding == 0 and ticket.payload is None
        assert thread_ids and all(t != threading.get_ident() for t in thread_ids)
    finally:
        proceed.set()
        store.close(timeout=5)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda h: h.update(version=2),
        lambda h: h.update(version=True),
        lambda h: h.update(tokens=[-1]),
        lambda h: h.update(layer_map=[-2]),
        lambda h: h["components"][0].update(offset=-1),
        lambda h: h["components"][0].update(dtype=["torch.float32"]),
        lambda h: h["components"][0].update(shape=[2**63]),
        lambda h: h.update(extra="unknown"),
        lambda h: h["components"].append(dict(h["components"][0])),
    ],
)
def test_valid_header_checksum_does_not_bypass_structural_bounds(tmp_path, mutation):
    path = tmp_path / "record"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        write_record(fd, TensorSource(image()), RecordLimits(4096, 4096))
    finally:
        os.close(fd)
    raw = path.read_bytes()
    _, length, _ = PREFIX.unpack(raw[: PREFIX.size])
    header = json.loads(raw[PREFIX.size : PREFIX.size + length])
    body = raw[PREFIX.size + length :]
    mutation(header)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(
        PREFIX.pack(MAGIC, len(encoded), hashlib.sha256(encoded).digest())
        + encoded
        + body
    )
    fd = os.open(path, os.O_RDONLY)
    sink = Sink()
    try:
        with pytest.raises(RecordError):
            read_record(fd, identity(), RecordLimits(4096, 4096), sink)
        assert sink.header is None
    finally:
        os.close(fd)


def test_async_gpu_fence_retains_ticket_after_io_completion(tmp_path):
    source = TensorSource(image())

    class Fenced:
        compatibility = source.compatibility
        tokens = source.tokens
        layout = source.layout
        safe = False

        def chunks(self, name, budget):
            yield from source.chunks(name, budget)

        def transfer_complete(self):
            return self.safe

    payload = Fenced()
    store = AsyncPrefixStore(
        tmp_path,
        identity(),
        max_tickets=1,
        quota_bytes=5000,
        limits=RecordLimits(4096, 4096),
    )
    try:
        assert store.ready.wait(5)
        ticket = store.submit_write(payload)
        assert ticket.finished.wait(5)
        store.poll()
        with pytest.raises(RuntimeError, match="GPU"):
            store.release(ticket)
        assert store.submit_write(source) is None
        payload.safe = True
        store.release(ticket)
        assert store.outstanding == 0
    finally:
        assert store.close()


def test_two_readers_keep_a_quota_victim_pinned(tmp_path):
    store = PrefixStore(
        tmp_path,
        identity(),
        quota_bytes=5000,
        max_records=1,
        limits=RecordLimits(4096, 4096),
    )
    entry = store.write(TensorSource(image()))
    gate = threading.Barrier(3)
    release = threading.Event()
    errors = []

    class DelayedSink(Sink):
        first = True

        def chunk(self, name, offset, data):
            if self.first:
                self.first = False
                gate.wait(timeout=5)
                assert release.wait(5)
            super().chunk(name, offset, data)

    def reader():
        try:
            store.read(entry.key, DelayedSink())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        gate.wait(timeout=5)
        with pytest.raises(StoreFull):
            store.write(TensorSource(image((2, 3, 4, 5))))
        release.set()
        for thread in threads:
            thread.join(5)
        assert not errors and not any(t.is_alive() for t in threads)
        assert not store.pins
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
        store.close()


def test_unavailable_directory_disables_worker_without_blocking_submit(tmp_path):
    root = tmp_path / "file"
    root.write_bytes(b"not a directory")
    store = AsyncPrefixStore(
        root, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    assert store.ready.wait(5)
    assert (
        store.failure is not None and store.submit_write(TensorSource(image())) is None
    )
    assert store.close()


def test_worker_death_completes_queued_tickets_and_retains_sources(tmp_path):
    source = TensorSource(image())
    entered = threading.Event()
    proceed = threading.Event()

    class Fatal:
        compatibility = source.compatibility
        tokens = source.tokens
        layout = source.layout

        def chunks(self, name, budget):
            entered.set()
            assert proceed.wait(5)
            raise SystemExit("injected worker termination")
            yield b""

    store = AsyncPrefixStore(
        tmp_path,
        identity(),
        max_tickets=2,
        quota_bytes=5000,
        limits=RecordLimits(4096, 4096),
    )
    assert store.ready.wait(5)
    first = store.submit_write(Fatal())
    assert entered.wait(5)
    second = store.submit_write(source)
    proceed.set()
    assert first.finished.wait(5)
    assert store._io_stopped.wait(5)
    assert not store.close(timeout=0)
    done = store.poll()
    assert {id(t) for t in done} == {id(first), id(second)}
    assert first.error is not None and second.error is not None
    assert first.payload is not None and second.payload is not None
    assert store.submit_write(source) is None
    for ticket in done:
        store.release(ticket)
    assert store.close(timeout=5)
    assert store.outstanding == 0


@pytest.mark.parametrize(
    "fault", ["permission", "zero-write", "file-fsync", "directory-fsync"]
)
def test_publication_failure_never_exposes_partial_record(tmp_path, monkeypatch, fault):
    store = PrefixStore(
        tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    try:
        original = store.write(TensorSource(image()))

        def fail(*args, **kwargs):
            raise OSError(
                errno.EACCES if fault == "permission" else errno.EIO, "injected"
            )

        if fault == "permission":
            original_open = os.open

            def denied(path, flags, *args, **kwargs):
                if str(path).split("/")[-1].startswith(".tmp-"):
                    fail()
                return original_open(path, flags, *args, **kwargs)

            monkeypatch.setattr(os, "open", denied)
        elif fault == "zero-write":
            monkeypatch.setattr(os, "write", lambda *args: 0)
        elif fault == "file-fsync":
            monkeypatch.setattr(os, "fsync", fail)
        else:
            monkeypatch.setattr(store, "_sync_directory", fail)
        with pytest.raises(OSError):
            store.write(TensorSource(image((9, 10, 11, 12))))
        assert store.bytes == original.nbytes and store.reserved_bytes == 0
        assert store.descriptors() == (original,)
        assert len(list(store.namespace.glob("*.pxc"))) == 1
        assert not list(store.namespace.glob(".tmp-*"))
    finally:
        store.close()


def test_finished_ticket_cannot_release_before_completion_is_polled(tmp_path):
    store = AsyncPrefixStore(
        tmp_path,
        identity(),
        max_tickets=1,
        quota_bytes=5000,
        limits=RecordLimits(4096, 4096),
    )
    assert store.ready.wait(5)
    try:
        ticket = store.submit_write(TensorSource(image()))
        assert ticket.finished.wait(5)
        with pytest.raises(RuntimeError, match="poll"):
            store.release(ticket)
        assert store.submit_write(TensorSource(image())) is None
        assert store.poll() == [ticket]
        store.release(ticket)
        assert store.outstanding == 0
    finally:
        assert store.close()


def test_hybrid_sparse_layer_map_survives_writer_validation_and_restart(tmp_path):
    from dataclasses import replace

    state = image()
    state.layout = replace(state.layout, layer_map=(-1, 0, -1))
    with_store = PrefixStore(
        tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    entry = with_store.write(TensorSource(state))
    with_store.close()
    recovered = PrefixStore(
        tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    try:
        header, layout = recovered.read(entry.key)
        assert header["layer_map"] == [-1, 0, -1]
        assert layout == state.layout
    finally:
        recovered.close()
