"""Real Gloo workers stopped while peers are at different control phases."""

from datetime import timedelta
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest


def _worker(rank, root, queue):
    import torch
    import torch.distributed as dist
    import sparklab.runtime.scheduler as module
    from sparklab.runtime.distributed.info import DistributedInfo, set_tp_info
    from sparklab.runtime.scheduler.scheduler import Scheduler as RealScheduler
    from sparklab.serving.launch import _run_scheduler
    from sparklab.utils import ZmqPushQueue

    directory = Path(root)
    torch.set_num_threads(1)
    set_tp_info(rank, 2)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/group",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=6),
    )

    class Worker:
        def __init__(self, args):
            self.config = args
            self.device = torch.device("cpu")
            self.prefix_retention = None
            self._poll_runtime_timings = lambda: None
            self.locks = [multiprocessing.Lock()]
            (directory / f"rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "cuda_initialized": torch.cuda.is_initialized(),
                        "semaphores": [lock._semlock.name for lock in self.locks],
                    }
                )
            )
            writer = ZmqPushQueue("tcp://127.0.0.1:*", True, lambda value: value)

            def stop(*, linger):
                writer.stop(linger=linger)
                (directory / f"rank-{rank}.io-closed").touch()

            self._send_into_tokenizer = SimpleNamespace(stop=stop)
            self.engine = SimpleNamespace(shutdown=lambda: dist.destroy_process_group())

        def sync_all_ranks(self):
            dist.group.WORLD.barrier().wait()

        def run_forever(self):
            if rank:
                work = dist.group.WORLD.broadcast(torch.zeros(1), root=0)
                assert not work.is_completed()
                (directory / "peer-waiting").touch()
                work.wait()
                raise AssertionError("unmatched broadcast unexpectedly completed")
            (directory / "primary-waiting").touch()
            while True:
                time.sleep(0.01)

        def shutdown(self, *, failed=False):
            (directory / f"rank-{rank}.cleanup-{failed}").touch()
            RealScheduler.shutdown(self, failed=failed)
            (directory / f"rank-{rank}.drained").touch()

    module.Scheduler = Worker
    torch.cuda.synchronize = lambda *args, **kwargs: None  # CPU lifecycle fixture only
    args = SimpleNamespace(tp_info=DistributedInfo(rank, 2), silent_output=False)
    _run_scheduler(args, queue)


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX signals and Gloo ownership")
@pytest.mark.parametrize("stop_kind", ["parent", "direct"])
def test_tp_stop_closes_local_owners_without_joining_peer_work(tmp_path, stop_kind):
    from sparklab.serving.api_server import (
        _terminate_backend_workers,
        _reap_backend_workers,
    )

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    semaphores = [
        queue._rlock._semlock.name,
        queue._wlock._semlock.name,
        queue._sem._semlock.name,
    ]
    children = [
        ctx.Process(target=_worker, args=(rank, str(tmp_path), queue))
        for rank in range(2)
    ]
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + 20
        while not all(
            (tmp_path / marker).exists()
            for marker in ("peer-waiting", "primary-waiting")
        ):
            assert all(child.is_alive() for child in children)
            assert time.monotonic() < deadline
            time.sleep(0.01)
        for rank in range(2):
            report = json.loads((tmp_path / f"rank-{rank}.json").read_text())
            assert report["cuda_initialized"] is False
            semaphores.extend(report["semaphores"])
        started = time.monotonic()
        if stop_kind == "parent":
            _terminate_backend_workers(children)
        else:
            for child in children:
                os.kill(child.pid, signal.SIGTERM)
        _reap_backend_workers(children, timeout=8.0)
        assert [child.exitcode for child in children] == [0, 0]
        assert time.monotonic() - started < 4.0
        for rank in range(2):
            assert (tmp_path / f"rank-{rank}.drained").exists()
            assert (tmp_path / f"rank-{rank}.io-closed").exists()
            assert (tmp_path / f"rank-{rank}.cleanup-True").exists()
        (tmp_path / "stop-report.json").write_text(
            json.dumps(
                {
                    "stop_kind": stop_kind,
                    "seconds": time.monotonic() - started,
                    "exitcodes": [child.exitcode for child in children],
                    "cuda_initialized": False,
                    "group_timeout_seconds": 6,
                    "scope": "Real two-process Gloo and owned ZMQ/locks; synthetic scheduler, no GPU",
                },
                indent=2,
            )
        )
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
            child.join(timeout=3.0)
        queue.close()
        queue.join_thread()
        queue = None  # release the parent's own queue semaphore references
        import gc

        gc.collect()
    assert not any(Path(f"/proc/{child.pid}").exists() for child in children)
    assert all(
        not (Path("/dev/shm") / ("sem." + name.lstrip("/"))).exists()
        for name in semaphores
    )
