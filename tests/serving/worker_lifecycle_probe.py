"""Real serve/signal/worker probe without weights or CUDA initialization.

Only model execution is replaced. The API, supervisor, scheduler wrapper,
spawned processes, named semaphores and operating-system signals are real.
"""

import argparse
from contextlib import nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace


def tokenizer_worker(ack_queue, root, index):
    import torch
    from sparklab.serving.launch import _run_tokenize_worker

    directory = Path(root)
    (directory / f"worker-{index}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "cuda_initialized": torch.cuda.is_initialized(),
                "semaphores": [],
            }
        )
    )
    (directory / f"worker-{index}.running").touch()
    _run_tokenize_worker(
        True,
        tokenizer_path=str(Path(__file__).resolve().parents[1] / "dsv41/fixtures/tiny"),
        addr=f"ipc://{root}/tokenizer",
        create=False,
        backend_addr=f"ipc://{root}/backend",
        frontend_addr=f"ipc://{root}/frontend",
        local_bs=1,
        tokenizer_id=0,
        ack_queue=ack_queue,
    )
    (directory / f"worker-{index}.drained").touch()


def worker(
    ack_queue,
    root,
    index,
    locks,
    ignore_stop,
    load_delay,
    fail_startup,
    prefix_io,
    http_stream=False,
    hold_entry=False,
):
    if hold_entry:
        import sys
        directory = Path(root)
        marker = directory / f"worker-{index}.entry.pending"
        marker.write_text(json.dumps({
            "pid": os.getpid(), "torch_imported": "torch" in sys.modules,
            "default_sigterm": signal.getsignal(signal.SIGTERM) == signal.SIG_DFL,
        }))
        marker.replace(directory / f"worker-{index}.entry.json")
        while not (directory / "release-entry").exists():
            time.sleep(.01)
    import torch
    from sparklab.serving.launch import _run_scheduler
    import sparklab.runtime.scheduler as scheduler_module

    directory = Path(root)

    class Scheduler:
        def __init__(self, args):
            self.locks = [mp.Lock() for _ in range(locks)]
            self.store = None
            self.incoming = self.outgoing = None
            self.pending = set()
            self.pressure = {}
            (directory / f"worker-{index}.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "cuda_initialized": torch.cuda.is_initialized(),
                        "semaphores": [lock._semlock.name for lock in self.locks],
                    }
                )
            )
            (directory / f"worker-{index}.initializing").touch()
            if load_delay:
                time.sleep(load_delay)
            if fail_startup:
                raise ValueError("probe startup failure")
            if http_stream:
                from sparklab.message import BaseTokenizerMsg, BaseFrontendMsg
                from sparklab.utils import ZmqPullQueue, ZmqPushQueue

                self.incoming = ZmqPullQueue(
                    f"ipc://{root}/tokenizer", False, BaseTokenizerMsg.decoder
                )
                self.outgoing = ZmqPushQueue(
                    f"ipc://{root}/frontend", False, BaseFrontendMsg.encoder
                )
            if prefix_io:
                from sparklab.runtime.checkpoint.async_store import AsyncPrefixStore
                from sparklab.runtime.checkpoint.record import (
                    TensorSource,
                    RecordLimits,
                )
                from tests.runtime.checkpoint.test_records import identity, image

                self.gate = threading.Event()
                original = TensorSource(image())

                class Paused:
                    compatibility, tokens, layout = (
                        original.compatibility,
                        original.tokens,
                        original.layout,
                    )

                    def chunks(inner, name, size):
                        (directory / f"worker-{index}.io-started").touch()
                        if not self.gate.wait(10.0):
                            raise TimeoutError("probe I/O gate not released")
                        yield from original.chunks(name, size)

                self.store = AsyncPrefixStore(
                    directory / f"prefixes-{index}",
                    identity(),
                    quota_bytes=5000,
                    limits=RecordLimits(4096, 4096),
                )
                assert self.store.ready.wait(5.0) and self.store.failure is None
                self.ticket = self.store.submit_write(Paused())
            ack_queue.put("worker ready")

        def sync_all_ranks(self):
            pass

        def run_forever(self):
            if ignore_stop:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            (directory / f"worker-{index}.running").touch()
            while True:
                if (directory / "fail-runtime").exists():
                    raise RuntimeError("probe inference failure with prefix I/O")
                if self.incoming is not None:
                    from sparklab.message import AbortMsg, TokenizeMsg, UserReply

                    if not self.incoming.empty():
                        msg = self.incoming.get()
                        if isinstance(msg, TokenizeMsg):
                            self.pending.add(msg.uid)
                            (directory / f"request-{msg.uid}.admitted").touch()
                            # Protocol adapters normalize text into different
                            # message/block shapes; only fixture markers matter.
                            text = str(msg.text)
                            if "pressure" in text:
                                self.pressure[msg.uid] = 64
                                continue
                            self.outgoing.put(
                                UserReply(
                                    msg.uid,
                                    "first",
                                    False,
                                    prompt_tokens_delta=1,
                                    completion_tokens_delta=1,
                                )
                            )
                            if "fast" in text:
                                self.pending.discard(msg.uid)
                                self.outgoing.put(
                                    UserReply(
                                        msg.uid, "last", True, completion_tokens_delta=1
                                    )
                                )
                        elif isinstance(msg, AbortMsg):
                            self.pending.discard(msg.uid)
                            self.pressure.pop(msg.uid, None)
                            (directory / f"request-{msg.uid}.aborted").touch()
                            self.outgoing.put(UserReply(msg.uid, "", True))
                    for uid, remaining in list(self.pressure.items()):
                        self.outgoing.put(
                            UserReply(
                                uid,
                                "x" * (128 << 10),
                                remaining == 1,
                                prompt_tokens_delta=int(remaining == 64),
                                completion_tokens_delta=1,
                            )
                        )
                        if remaining == 1:
                            self.pressure.pop(uid)
                            self.pending.discard(uid)
                        else:
                            self.pressure[uid] = remaining - 1
                    if (directory / "release-http").exists():
                        for uid in self.pending:
                            self.outgoing.put(
                                UserReply(uid, "last", True, completion_tokens_delta=1)
                            )
                        self.pending.clear()
                    if (directory / "fail-http").exists():
                        raise RuntimeError(
                            "probe backend failure at /private/model/path"
                        )
                time.sleep(0.05)

        def shutdown(self, *, failed=False):
            if failed:
                (directory / f"worker-{index}.failure-cleanup").touch()
            (directory / f"worker-{index}.draining").touch()
            if self.store is not None:
                timer = threading.Timer(0.3, self.gate.set)
                timer.start()
                self.store.close(timeout=0)
                deadline = time.monotonic() + 5.0
                while self.store.outstanding and time.monotonic() < deadline:
                    for ticket in self.store.poll():
                        assert ticket.error is None
                        self.store.release(ticket)
                    time.sleep(0.005)
                assert self.store.close()
                timer.join()
            else:
                time.sleep(0.3)
            self.locks.clear()
            for link in (self.incoming, self.outgoing):
                if link is not None:
                    link.stop(linger=0)
            (directory / f"worker-{index}.drained").touch()

    scheduler_module.Scheduler = Scheduler
    args = SimpleNamespace(
        shell_mode=False,
        silent_output=True,
        tp_info=SimpleNamespace(is_primary=lambda: False),
    )
    _run_scheduler(args, ack_queue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--locks", type=int, default=4)
    parser.add_argument("--ignore-stop", action="store_true")
    parser.add_argument("--load-delay", type=float, default=0.0)
    parser.add_argument("--fail-startup", action="store_true")
    parser.add_argument("--fail-launch", action="store_true")
    parser.add_argument("--prefix-io", action="store_true")
    parser.add_argument("--tokenizer", action="store_true")
    parser.add_argument("--http-stream", action="store_true")
    parser.add_argument("--hold-entry", action="store_true")
    parser.add_argument("--pause-launch", action="store_true")
    parser.add_argument("--interrupt-spawn", type=int, default=0,
                        choices=[0, int(signal.SIGINT), int(signal.SIGTERM)])
    args = parser.parse_args()
    if args.interrupt_spawn and not args.hold_entry:
        parser.error("--interrupt-spawn requires --hold-entry")
    args.root.mkdir(parents=True, exist_ok=True)
    mp.set_start_method("spawn", force=True)
    from sparklab.serving.api_server import run_api_server
    from sparklab.serving.supervisor import BackendHandle
    from sparklab.serving import launch

    handles = []

    def start():
        queue = mp.Queue()
        processes = []
        handle = BackendHandle(queue, processes, args.workers)
        handles.append(handle)
        # Allow the same ready/stop probe against the prior artifact, which has
        # no partial-start ownership helper. Current-source failure cases use it.
        ownership = getattr(launch, "_startup_ownership", lambda *args: nullcontext())
        with ownership(processes, queue):
            for index in range(args.workers):
                worker_args = (
                    queue,
                    str(args.root),
                    index,
                    args.locks,
                    args.ignore_stop,
                    args.load_delay,
                    args.fail_startup,
                    args.prefix_io,
                    args.http_stream,
                    args.hold_entry,
                )
                process = mp.Process(
                    target=tokenizer_worker if args.tokenizer else worker,
                    args=(queue, str(args.root), index)
                    if args.tokenizer
                    else worker_args,
                    name=f"sparklab-probe-{index}",
                )
                processes.append(process)
                starter = getattr(launch, "_start_owned_process", lambda child: child.start())
                if args.interrupt_spawn and index == 0:
                    import multiprocessing.popen_spawn_posix as spawn_popen
                    original_launch = spawn_popen.Popen._launch

                    def interrupt_gap(popen, process_obj):
                        original_launch(popen, process_obj)
                        (args.root / "spawn-gap.json").write_text(json.dumps({
                            "worker": popen.pid, "handle_assigned": process_obj._popen is not None,
                            "semaphores": [queue._rlock._semlock.name,
                                           queue._wlock._semlock.name, queue._sem._semlock.name],
                        }))
                        deadline = time.monotonic() + 15.
                        while not (args.root / "worker-0.entry.json").exists():
                            assert time.monotonic() < deadline
                            time.sleep(.01)
                        os.kill(os.getpid(), args.interrupt_spawn)

                    spawn_popen.Popen._launch = interrupt_gap
                    try:
                        starter(process)
                    finally:
                        spawn_popen.Popen._launch = original_launch
                else:
                    starter(process)
                (args.root / "parent.json").write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "workers": [p.pid for p in processes],
                            "semaphores": [
                                queue._rlock._semlock.name,
                                queue._wlock._semlock.name,
                                queue._sem._semlock.name,
                            ],
                        }
                    )
                )
                if args.pause_launch and index == 0:
                    (args.root / "launch-paused").touch()
                    while not (args.root / "release-launch").exists():
                        time.sleep(.01)
                if args.fail_launch:
                    raise RuntimeError("probe partial launch failure")
            return handle

    config = SimpleNamespace(
        sampling_defaults="framework",
        use_dummy_weight=True,
        model_path="unused-probe",
        served_model_name="lifecycle-probe",
        server_host="127.0.0.1",
        server_port=args.port,
        cors_origins="",
        zmq_frontend_addr=f"ipc://{args.root}/frontend",
        zmq_tokenizer_addr=f"ipc://{args.root}/tokenizer",
        frontend_create_tokenizer_link=True,
    )
    if args.http_stream:
        from sparklab.serving.api_server import app, get_global_state, FrontendManager

        config.model_path = str(
            Path(__file__).resolve().parents[1] / "dsv41/fixtures/tiny"
        )
        original_tokenizer = FrontendManager.frontend_tokenizer

        def fixture_tokenizer(self):
            manager = original_tokenizer(self)
            manager.tokenizer.chat_template = (
                "{% for m in messages %}{{ m['content'] }}{% endfor %}"
            )
            return manager

        FrontendManager.frontend_tokenizer = fixture_tokenizer
        pressure = {"entered": 0, "completed": 0, "cancelled": 0}

        class PressureProbe:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http" or (
                    b"x-probe-pressure",
                    b"1",
                ) not in scope.get("headers", []):
                    return await self.app(scope, receive, send)

                async def observed(message):
                    if message["type"] != "http.response.body":
                        return await send(message)
                    pressure["entered"] += 1
                    try:
                        await send(message)
                    except BaseException:
                        pressure["cancelled"] += 1
                        raise
                    else:
                        pressure["completed"] += 1

                await self.app(scope, receive, observed)

        app.add_middleware(PressureProbe)

        @app.get("/probe-state")
        async def probe_state():
            state = get_global_state()
            return {"acks": len(state.ack_map), "events": len(state.event_map)}

        @app.get("/probe-accounting")
        async def probe_accounting():
            stats = get_global_state().stats
            return {
                "active": stats.active,
                "completed": stats.completed,
                "prompt": stats.prompt_tokens_total,
                "completion": stats.completion_tokens_total,
            }

        @app.get("/probe-pressure")
        async def probe_pressure():
            state = get_global_state()
            return {
                **pressure,
                "queued_replies": sum(len(q) for q in state.ack_map.values()),
                "queued_characters": sum(
                    len(reply.incremental_output)
                    for q in state.ack_map.values()
                    for reply in q
                ),
            }

    try:
        run_api_server(config, start, run_shell=False)
    finally:
        (args.root / "parent.finalized").write_text(
            json.dumps(
                {
                    "exitcodes": [
                        p.exitcode for handle in handles for p in handle.processes
                    ],
                }
            )
        )


if __name__ == "__main__":
    main()
