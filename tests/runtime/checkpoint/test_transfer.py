"""Real-file GPU streaming uses bounded windows and complete private state."""

from types import SimpleNamespace

import pytest
import torch
from sparklab.models.config import LinearGatedDeltaGroupConfig
from sparklab.runtime.kvcache.mha_pool import MHAKVCache
from sparklab.runtime.kvcache.linear_state_pool import LinearStatePool
from sparklab.runtime.checkpoint.exact import Endpoint, KVAnchor
from sparklab.runtime.checkpoint.state import Compatibility, MHARecurrentAdapter
from sparklab.runtime.checkpoint.transfer import GPUStateSource, GPUStateSink
from sparklab.runtime.checkpoint.async_store import AsyncPrefixStore
from sparklab.runtime.checkpoint.record import RecordLimits

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA staging gate"
)


@torch.inference_mode()
def test_real_file_staging_roundtrip_private_tail_and_recurrent_state(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "sparklab.runtime.kvcache.mha_pool.get_tp_info", lambda: SimpleNamespace(size=1)
    )
    device = torch.device("cuda")
    kv = MHAKVCache(1, 2, 8, 16, 4, torch.bfloat16, device)
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 1),
        num_key_heads=1,
        num_value_heads=1,
        key_head_dim=4,
        value_head_dim=4,
        conv_kernel_dim=3,
        output_gate=True,
    )
    linear = LinearStatePool(group, 8, torch.bfloat16, device, tp_size=1)
    linear.ensure_aux_state("ple", (7,), torch.int64, fill=-1)
    src, dst = linear.alloc(2)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(73)
        kv._kv_buffer.normal_()
        linear.conv_states[:, src].normal_()
        linear.recurrent_states[:, src].normal_()
        linear._aux_states["ple"][src].copy_(torch.arange(7, device=device))
    adapter = MHARecurrentAdapter(kv, linear)
    tokens = tuple(range(9))
    producer = torch.cuda.Event()
    producer.record()
    endpoint = Endpoint(
        tokens,
        KVAnchor(8, None, torch.arange(8, device=device, dtype=torch.int32)),
        src,
        torch.arange(8, 12, device=device, dtype=torch.int32),
        producer,
    )

    def release(entry):
        assert entry.readers > 0
        entry.readers -= 1

    cache = SimpleNamespace(
        adapter=adapter, manager=SimpleNamespace(page_size=4), release=release
    )
    compatibility = Compatibility.from_config(
        dict(
            artifact="test-weights",
            precision="bf16-fp32",
            model_config="c",
            tokenizer="t",
            template="j",
            state_layout="mha-gdn-v1",
            cache_geometry={"page_size": 4},
            continuation="target-only",
            scope="transfer-test",
        )
    )
    expected = adapter.snapshot(
        compatibility, tokens, torch.arange(9), src, max_bytes=1 << 20
    )
    source = GPUStateSource(cache, endpoint, compatibility, chunk_bytes=4096)
    store = AsyncPrefixStore(
        tmp_path,
        compatibility,
        max_tickets=2,
        quota_bytes=1 << 20,
        limits=RecordLimits(1 << 20, 4096),
    )
    assert store.ready.wait(5)
    try:
        write = store.submit_write(source)
        assert write is not None and write.finished.wait(10)
        assert store.poll() == [write]
        assert write.error is None
        key = write.result.key
        assert source.transfer_complete() and endpoint.readers == 1
        source.release_source()
        store.release(write)
        assert endpoint.readers == 0
        # Restore to a separate private physical span, leaving source bytes intact.
        mapping = torch.arange(32, 41, device=device, dtype=torch.int32)
        ready = torch.cuda.Event()
        ready.record()
        sink = GPUStateSink(
            adapter,
            compatibility,
            tokens,
            mapping,
            dst,
            chunk_bytes=4096,
            allocation_ready=ready,
        )
        read = store.submit_read(key, sink)
        assert read is not None and read.finished.wait(10)
        assert store.poll() == [read] and read.error is None
        assert sink.validated and sink.transfer_complete()
        restored = adapter.snapshot(
            compatibility, tokens, mapping, dst, max_bytes=1 << 20
        )
        torch.cuda.synchronize()
        for name, tensor in expected._tensors.items():
            assert torch.equal(
                tensor.view(torch.uint8), restored._tensors[name].view(torch.uint8)
            ), name
        assert source._buffer.numel() == sink._buffer.numel() == 4096
        assert source.transferred_bytes == sink.transferred_bytes == expected.nbytes
        store.release(read)
        assert store.outstanding == 0
    finally:
        assert store.close(timeout=10)


@torch.inference_mode()
def test_gpu_disk_service_adopts_private_pages_after_eviction(tmp_path, monkeypatch):
    import time
    from sparklab.core import SamplingParams
    from sparklab.message import UserMsg
    from sparklab.runtime.scheduler.utils import PendingReq
    from sparklab.runtime.scheduler.prefill import ChunkedReq
    from sparklab.runtime.checkpoint.config import DiskSettings
    from sparklab.runtime.checkpoint.service import PrefixRetention
    from tests.runtime.checkpoint.test_exact import setup, consume, finish
    from tests.runtime.checkpoint.test_state import identity

    cm, tm, pm, linear, kv = setup(monkeypatch, device="cuda")
    settings = DiskSettings(
        str(tmp_path), 1 << 20, 16, 2, 4096, "always", 10.0, "single-instance-test"
    )
    service = PrefixRetention(
        cm.exact_endpoints, identity(), settings, prefill_cost=lambda n: None
    )
    assert service.store.ready.wait(5)

    def drain():
        deadline = time.monotonic() + 10
        while service.outstanding and time.monotonic() < deadline:
            service.poll()
            time.sleep(0.001)
        assert not service.outstanding

    try:
        tokens = tuple(range(1, 10))
        pm.add_one_req(
            UserMsg(
                1, torch.tensor(tokens, dtype=torch.int32), SamplingParams(max_tokens=4)
            )
        )
        while pm.runnable:
            batch = pm.schedule_next_batch(512)
            req = batch.reqs[0]
            cm.allocate_paged(batch.reqs)
            consume(cm, linear, kv, req)
            if isinstance(req, ChunkedReq):
                cm.exact_endpoints.publish(req)
            else:
                cm.cache_req(req, finished=False)
        finish(cm, tm, req)
        torch.cuda.synchronize()
        endpoint = cm.exact_endpoints.entries[tokens]
        mapping = torch.cat([endpoint.anchor.kv_indices, endpoint.tail[:1]])
        expected = cm.exact_endpoints.adapter.snapshot(
            identity(), tokens, mapping, endpoint.state_slot, max_bytes=1 << 20
        )
        # Publication can safely skip an unfinished producer. Offer the completed
        # immutable endpoint explicitly, as the next duplicate publication would.
        service.publish(endpoint)
        drain()
        assert any(e.consumed == 9 for e in service.store.descriptors())
        cm.check_integrity()
        cm.rebuild(cm.num_pages, cm.page_table)
        cm.check_integrity()
        pending = PendingReq(
            2,
            torch.tensor(tokens + (91,), dtype=torch.int32),
            SamplingParams(max_tokens=4),
        )
        assert cm.match_req(pending).cuda_handle.cached_len == 0
        assert service.maybe_restore(pending)
        drain()
        endpoint = cm.exact_endpoints.entries[tokens]
        mapping = torch.cat([endpoint.anchor.kv_indices, endpoint.tail[:1]])
        restored = cm.exact_endpoints.adapter.snapshot(
            identity(), tokens, mapping, endpoint.state_slot, max_bytes=1 << 20
        )
        torch.cuda.synchronize()
        for name, tensor in expected.inspect().items():
            assert torch.equal(
                tensor.view(torch.uint8), restored.inspect()[name].view(torch.uint8)
            ), name
        assert cm.match_req(pending).cuda_handle.cached_len == 9
        cm.check_integrity()
    finally:
        assert service.close(timeout=10)


@torch.inference_mode()
def test_scheduler_shutdown_drains_real_gpu_file_users_before_engine_and_ipc(tmp_path, monkeypatch, failed=False):
    import threading
    from collections import deque
    from sparklab.core import SamplingParams
    from sparklab.message import UserMsg
    from sparklab.runtime.scheduler.prefill import ChunkedReq
    from sparklab.runtime.scheduler.scheduler import Scheduler
    from sparklab.runtime.checkpoint.config import DiskSettings
    from sparklab.runtime.checkpoint.service import PrefixRetention
    from tests.runtime.checkpoint.test_exact import setup, consume, finish
    from tests.runtime.checkpoint.test_state import identity

    cm, tm, pm, linear, kv = setup(monkeypatch, device="cuda")
    entered, proceed = threading.Event(), threading.Event()

    class PausedSource(GPUStateSource):
        def chunks(self, name, size):
            entered.set()
            assert proceed.wait(5.)
            yield from super().chunks(name, size)

    service = PrefixRetention(cm.exact_endpoints, identity(),
        DiskSettings(str(tmp_path), 1 << 20, 16, 2, 4096, "always", 10., "single-instance-test"),
        prefill_cost=lambda n: None, source_factory=PausedSource)
    assert service.store.ready.wait(5.)
    timer = None
    try:
        tokens = tuple(range(1, 10))
        pm.add_one_req(UserMsg(1, torch.tensor(tokens, dtype=torch.int32), SamplingParams(max_tokens=4)))
        while pm.runnable:
            batch = pm.schedule_next_batch(512)
            req = batch.reqs[0]
            cm.allocate_paged(batch.reqs)
            consume(cm, linear, kv, req)
            if isinstance(req, ChunkedReq):
                cm.exact_endpoints.publish(req)
            else:
                cm.cache_req(req, finished=False)
        finish(cm, tm, req)
        endpoint = cm.exact_endpoints.entries[tokens]
        service.publish(endpoint)
        assert entered.wait(5.) and service.outstanding and endpoint.readers > 0
        events = []

        def teardown():
            assert proceed.is_set() and service.outstanding == 0
            assert endpoint.readers == 0 and not service.store._thread.is_alive()
            cm.check_integrity()
            events.append("engine")

        def stop(*, linger):
            assert linger == 0 and events == ([] if failed else ["engine"])
            assert proceed.is_set() and service.outstanding == 0
            assert endpoint.readers == 0 and not service.store._thread.is_alive()
            cm.check_integrity()
            events.append("ipc")

        owner = Scheduler.__new__(Scheduler)
        owner.prefix_retention = service
        owner.device = torch.device("cuda")
        owner._runtime_events = deque()
        def sync():
            assert not failed, "failed cleanup must not wait for healthy ranks"
        owner.sync_all_ranks = sync
        owner.engine = SimpleNamespace(shutdown=teardown)
        owner._send_into_tokenizer = SimpleNamespace(stop=stop)
        timer = threading.Timer(.05, proceed.set)
        timer.start()
        owner.shutdown(failed=failed)
        assert events == (["ipc"] if failed else ["engine", "ipc"])
        assert any(entry.consumed == 9 for entry in service.store.store.entries.values())
    finally:
        proceed.set()
        if timer is not None:
            timer.join()
        assert service.close(timeout=10.)


@torch.inference_mode()
def test_scheduler_failure_drains_real_gpu_file_users_before_process_exit(tmp_path, monkeypatch):
    test_scheduler_shutdown_drains_real_gpu_file_users_before_engine_and_ipc(
        tmp_path, monkeypatch, failed=True
    )
