"""Worker metadata snapshots retain real-file counters without actor I/O."""

from sparklab.runtime.checkpoint.async_store import AsyncPrefixStore
from sparklab.runtime.checkpoint.record import RecordLimits, TensorSource
from sparklab.runtime.checkpoint.store import PrefixStore
from tests.runtime.checkpoint.test_records import identity, image
from tests.runtime.checkpoint.test_service import seeded


def open_async(root):
    store = AsyncPrefixStore(
        root, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    assert store.ready.wait(5) and store.failure is None
    return store


def test_completed_worker_observations_match_files_and_preserve_old_snapshots(tmp_path):
    store = open_async(tmp_path)
    initial = store.observation
    try:
        first = store.submit_write(TensorSource(image()))
        assert first.finished.wait(5) and first.error is None
        assert store.poll() == [first]
        first_key = first.result.key
        saved = store.observation
        assert (
            initial["root_observed_at"]
            <= saved["root_observed_at"]
            <= saved["observed_at"]
        )
        files = list(store.store.namespace.glob("*.pxc"))
        assert saved["records"] == len(files) == 1
        assert (
            saved["disk_bytes"]
            == saved["root_observed_bytes"]
            == files[0].stat().st_size
        )
        assert saved["counters"]["writes"] == 1
        assert initial["counters"].get("writes", 0) == initial["records"] == 0
        store.release(first)
        second = store.submit_write(TensorSource(image((5, 6, 7, 8))))
        assert second.finished.wait(5) and second.error is None
        assert store.poll() == [second]
        assert store.observation["counters"]["writes"] == 2
        assert saved["counters"]["writes"] == 1
        root_time = saved["root_observed_at"]
        store.release(second)
        # A read observes the worker again without inventing a new root census.
        latest_root_time = store.observation["root_observed_at"]
        assert latest_root_time >= root_time
        read = store.submit_read(first_key, None)
        assert read.finished.wait(5) and read.error is None
        assert store.poll() == [read]
        assert store.observation["root_observed_at"] == latest_root_time
        store.release(read)
    finally:
        assert store.close()


def test_retention_snapshot_uses_cached_worker_metadata_without_filesystem_calls(
    tmp_path, monkeypatch
):
    service, *_ = seeded(tmp_path, monkeypatch)
    try:

        def forbidden(*args, **kwargs):
            raise AssertionError("actor snapshot performed filesystem I/O")

        with monkeypatch.context() as guards:
            guards.setattr("os.open", forbidden)
            guards.setattr("os.scandir", forbidden)
            snapshot = service.snapshot()
        assert snapshot["worker_observation"]["records"] == 1
        assert snapshot["worker_observation"]["counters"]["recovered"] == 1
        assert (
            snapshot["worker_observation"]["root_observed_bytes"]
            == snapshot["disk_bytes"]
        )
    finally:
        assert service.close()


def test_observation_failure_during_boot_releases_namespace_ownership(
    tmp_path, monkeypatch
):
    with monkeypatch.context() as fault:

        def fail(self):
            raise ValueError("injected observation failure")

        fault.setattr(AsyncPrefixStore, "_observe", fail)
        store = AsyncPrefixStore(
            tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
        )
        assert store.ready.wait(5)
        assert isinstance(store.failure, ValueError)
        assert store.close()
    recovered = PrefixStore(
        tmp_path, identity(), quota_bytes=5000, limits=RecordLimits(4096, 4096)
    )
    recovered.close()


def test_retention_exports_bounded_measured_costs_without_prompt_metadata(
    tmp_path, monkeypatch
):
    service, *_ = seeded(tmp_path, monkeypatch)
    try:
        for seconds in [0.1, 0.2, 0.3]:
            service._record_cost("read", 9, seconds)
        costs = service.snapshot()["service_costs"]
        assert costs == [
            {"kind": "read", "tokens": 16, "samples": 3, "p90_seconds": 0.3}
        ]
        for i in range(1000):
            service._record_cost("read" if i % 2 else "write", 1 << (i % 17), 0.1)
        costs = service.snapshot()["service_costs"]
        assert len(costs) <= 32
        assert all(item["samples"] <= 32 for item in costs)
        assert all(
            set(item) == {"kind", "tokens", "samples", "p90_seconds"} for item in costs
        )
    finally:
        assert service.close()


def test_observation_failure_completes_ticket_and_drains_before_worker_closes(
    tmp_path, monkeypatch
):
    store = open_async(tmp_path)

    def fail():
        raise ValueError("injected observation failure")

    monkeypatch.setattr(store, "_observe", fail)
    ticket = store.submit_write(TensorSource(image()))
    assert ticket.finished.wait(5)
    assert isinstance(ticket.error, ValueError)
    assert store._io_stopped.wait(5)
    assert not store.close(timeout=0)
    assert store.poll() == [ticket]
    assert store.failure is not None
    store.release(ticket)
    assert store.close()
