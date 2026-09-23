"""Start the pinned engine and gateway, terminate both if either exits."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def stop_processes(children):
    for child in reversed(children):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    for child in reversed(children):
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def wait_ready(child, url, *, timeout=3600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup ({child.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError("vLLM startup timed out")


def commands(args):
    engine = ["vllm", "serve", args.model, "--served-model-name", args.served_model_name,
              "--host", "127.0.0.1", "--port", "8010", "--attention-backend", "TRITON_ATTN",
              "--diffusion-config", '{"canvas_length": %d}' % args.canvas_length,
              "--max-logprobs", "128", "--enable-prefix-caching", "--async-scheduling",
              "--max-num-seqs", str(args.max_num_seqs), "--max-model-len", str(args.max_model_len),
              "--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes)]
    gateway = [sys.executable, "-m", "sparklab.external.djev.gateway",
               "--upstream", "http://127.0.0.1:8010", "--model", args.served_model_name,
               "--tokenizer", args.model, "--canvas", str(args.canvas_length)]
    return engine, gateway


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--canvas-length", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 << 30)
    args = parser.parse_args()
    children = []

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        engine, gateway = commands(args)
        children.append(subprocess.Popen(engine, start_new_session=True))
        wait_ready(children[0], "http://127.0.0.1:8010/health")
        children.append(subprocess.Popen(gateway, start_new_session=True))
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        raise RuntimeError("a djev service exited; stopping the paired service")
    finally:
        stop_processes(children)


if __name__ == "__main__":
    main()
