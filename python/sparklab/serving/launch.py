from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

from sparklab.runtime.distributed import DistributedInfo
from sparklab.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs
    from .supervisor import BackendHandle


def _install_worker_stop_handlers():
    """Worker-only one-shot stop: repeated signals cannot interrupt the drain."""
    stopping = False

    def begin_shutdown():
        nonlocal stopping
        stopping = True

    def request_stop(signum, frame):
        if not stopping:
            begin_shutdown()
            raise KeyboardInterrupt

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if (sig := getattr(signal, name, None)) is not None:
            signal.signal(sig, request_stop)
    return begin_shutdown


def _close_ack_writer(ack_queue):
    if ack_queue is not None:
        try:
            ack_queue.close()
            ack_queue.join_thread()
        except Exception:  # reporting must not replace an original worker error
            pass


@contextmanager
def _startup_ownership(processes, ack_queue):
    """The launcher owns partial starts until the complete handle is returned."""
    try:
        yield
    except BaseException:
        from .api_server import _terminate_backend_workers, _reap_backend_workers

        _terminate_backend_workers(processes)
        _reap_backend_workers(processes)
        _close_ack_writer(ack_queue)
        raise


def _start_owned_process(process):
    """Defer callable parent stops until start() installs the child handle."""
    if (threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "pthread_sigmask")):
        process.start()
        return
    previous, pending = {}, {}
    completed = False

    def defer(signum, frame):
        pending.setdefault(signum, None)  # bounded signals, no retained frames

    try:
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None and callable(handler := signal.getsignal(sig)):
                previous[sig] = handler
                signal.signal(sig, defer)
        process.start()
        completed = True
    finally:
        try:
            # Mask only during restoration, after spawn: children must not
            # inherit blocked stops. Restore all handlers before replay.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(previous))
            try:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            finally:
                try:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                except BaseException:
                    if completed:
                        raise
            for sig in tuple(pending):
                pending.pop(sig)
                if completed:
                    signal.raise_signal(sig)
                else:
                    try:
                        signal.raise_signal(sig)
                    except BaseException:
                        pass  # the original start error remains authoritative
        finally:
            pending.clear()  # do not retain interrupted startup frames


def _report_startup_error(ack_queue: mp.Queue, exc: BaseException) -> None:
    """Tell the parent WHY this worker is dying — push an ("error", reason) ack before it exits,
    so the supervisor reports the real cause (e.g. a config ValueError) instead of the generic
    "backend worker … exited during load". Best-effort and flushed (close + join_thread), since
    the process is about to terminate; a failure to report must never mask the original error."""
    try:
        ack_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        ack_queue.close()
        ack_queue.join_thread()
    except Exception:  # noqa: BLE001 -- reporting is a nicety; never shadow the real exception
        pass


def _detach_process_group() -> None:
    """Move owned workers out of the terminal's foreground process group.

    The shell binds ^C to "cancel this turn", but a terminal delivers SIGINT to the whole
    foreground group — which, with the engine running in this same process, includes the
    workers. They would take the same ^C and exit (``_run_scheduler`` below stops gracefully on
    KeyboardInterrupt), leaving the shell chatting with a dead engine. Nothing depends on the
    signal reaching them: the parent tears the workers down explicitly on every stop path
    (uvicorn's lifespan, plus the SIGTERM/SIGHUP handler and the reap backstop in api_server).

    In foreground serving, Uvicorn must finish its active HTTP drain before the
    workers exit. Group-wide delivery would stop them first and strand streams
    awaiting output. The parent explicitly stops and reaps them after its drain.
    """
    try:
        os.setpgrp()
    except OSError:  # no job control (already a group leader / unusual environment)
        pass


def _run_tokenize_worker(detach: bool, **kwargs) -> None:
    """Module-level so it survives the spawn pickle; exists only to detach the group first."""
    if detach:
        _detach_process_group()
    begin_shutdown = _install_worker_stop_handlers()
    try:
        from sparklab.tokenizer import tokenize_worker

        tokenize_worker(**kwargs)
    except KeyboardInterrupt:
        begin_shutdown()
    finally:
        begin_shutdown()
        _close_ack_writer(kwargs.get("ack_queue"))


def _cleanup_failed_scheduler(scheduler) -> None:
    """Retain a constructed owner's drain even while preserving another error."""
    if scheduler is None:
        return
    try:
        import torch

        with torch.inference_mode():
            scheduler.shutdown(failed=True)
    except BaseException:
        # Cleanup cannot replace the original fatal startup/inference cause.
        init_logger(__name__).exception("Scheduler failure cleanup did not complete")


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    _detach_process_group()
    begin_shutdown = _install_worker_stop_handlers()
    scheduler = None
    progress_sink = None
    startup_error = None
    try:
        import torch
        from sparklab.runtime.scheduler import Scheduler

        if args.tp_info.is_primary():
            from sparklab.utils.progress import set_progress_sink

            progress_sink = set_progress_sink
            progress_sink(lambda desc, done, total: ack_queue.put(("progress", desc, done, total)))

        with torch.inference_mode():
            try:
                scheduler = Scheduler(args)
                scheduler.sync_all_ranks()
            except Exception as exc:
                startup_error = exc
                _report_startup_error(ack_queue, exc)
                raise
            if args.tp_info.is_primary():
                try:
                    from sparklab.runtime.kvcache.cache_status import compute_cache_status_meta

                    ack_queue.put(("meta", compute_cache_status_meta(scheduler.engine)))
                except Exception:  # metadata failure must not prevent readiness
                    pass
                ack_queue.put("Scheduler is ready")
                progress_sink(None)
            if args.silent_output:
                logging.disable(logging.INFO)
            scheduler.run_forever()
    except KeyboardInterrupt:
        begin_shutdown()
        if startup_error is not None:
            _cleanup_failed_scheduler(scheduler)
            raise startup_error  # a racing stop must not turn a real failure into success
        if scheduler is not None:
            logger = init_logger(__name__)
            local_only = getattr(args.tp_info, "size", 1) > 1
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler stopping local rank..." if local_only
                            else "Scheduler exiting gracefully...")
            with torch.inference_mode():
                if local_only:
                    # Signals can interrupt ranks at different collective
                    # positions. A new barrier can then wait behind unmatched
                    # work. Drain local owners and let process exit tear down
                    # the interrupted device/communicator state.
                    scheduler.shutdown(failed=True)
                else:
                    scheduler.shutdown()
    except BaseException:
        begin_shutdown()
        _cleanup_failed_scheduler(scheduler)
        raise
    finally:
        begin_shutdown()
        if progress_sink is not None:
            progress_sink(None)
        _close_ack_writer(ack_queue)


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(
        sys.argv[1:] if argv is None else argv,
        run_shell,
        prog=prog,
    )
    init_logger(__name__, "initializer")

    def start_subprocess() -> "BackendHandle":
        import multiprocessing as mp

        from .supervisor import BackendHandle

        mp.set_start_method("spawn", force=True)
        detach = True  # the frontend owns terminal signals; see _detach_process_group

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue = mp.Queue()
        processes: list[mp.Process] = []

        with _startup_ownership(processes, ack_queue):
            for i in range(world_size):
                new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
                p = mp.Process(
                    target=_run_scheduler,
                    args=(new_args, ack_queue),
                    daemon=False,
                    name=f"sparklab-TP{i}-scheduler",
                )
                processes.append(p)
                _start_owned_process(p)

            num_tokenizers = server_args.num_tokenizer
            p = mp.Process(
                target=_run_tokenize_worker,
                kwargs={
                    "detach": detach,
                    "tokenizer_path": server_args.model_path,
                    "vision_model": server_args.vision_model,
                    "addr": server_args.zmq_detokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": num_tokenizers,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name="sparklab-detokenizer-0",
            )
            processes.append(p)
            _start_owned_process(p)
            for i in range(num_tokenizers):
                p = mp.Process(
                    target=_run_tokenize_worker,
                    kwargs={
                        "detach": detach,
                        "tokenizer_path": server_args.model_path,
                        "vision_model": server_args.vision_model,
                        "addr": server_args.zmq_tokenizer_addr,
                        "backend_addr": server_args.zmq_backend_addr,
                        "frontend_addr": server_args.zmq_frontend_addr,
                        "local_bs": 1,
                        "create": server_args.tokenizer_create_addr,
                        "tokenizer_id": i,
                        "ack_queue": ack_queue,
                    },
                    daemon=False,
                    name=f"sparklab-tokenizer-{i}",
                )
                processes.append(p)
                _start_owned_process(p)

            # Expected ready acks: 1 primary scheduler + num_tokenizers + 1 detokenizer.
            return BackendHandle(
                ack_queue=ack_queue,
                processes=processes,
                expected_acks=num_tokenizers + 2,
            )

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
