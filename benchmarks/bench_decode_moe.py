"""Single-stream (bs=1) decode benchmark for any MoE model on any offload backend.

Measures through the real serving path: for each backend the bench spawns ``ft serve``,
sends a warmed chat request over /v1/chat/completions with ``stream=true``, and
timestamps every SSE event as it arrives. Numbers therefore include the scheduler,
detokenizer, and HTTP/SSE hop -- what a client actually sees -- not bare engine forwards.

Method -- at bs=1 the server emits one delta event per decode step, and the final chunk
(``stream_options.include_usage``) reports exact token counts, so

    decode_tok_s = (completion_tokens - 1) / (t_last_event - t_first_event)

which stays correct even when the detokenizer coalesces a few tokens into one event
(multibyte characters): the window is still anchored on the first and last token's
arrival. ``ignore_eos`` keeps the step count at exactly ``D`` regardless of sampling.
TTFT is the measured run's warm first-token latency (template rendering + prefill
included). Expert-cache, hybrid-fetch, and disk-I/O diagnostics come from the measured
delta of the server's /v1/stats counters; VRAM is the same endpoint's live figure.

Prompt: an AIME-25 problem sent as a chat message with thinking enabled -- a real
reasoning workload, so expert routing is representative. The server renders the chat
template (including checkpoint-shipped encoders like DSV4's ``encoding_dsv4.py``). The
problems come from the ``math-ai/aime25`` dataset on the Hub, downloaded into the usual
HF cache on first run; ``--aime`` points at a local jsonl instead.

Sampling: the checkpoint's recommended params (``generation_config.json``), falling back
to temperature 1.0 / top_p 0.95 / top_k 64 for fields the checkpoint does not specify --
resolved here and sent explicitly, because the server's own unspecified-field defaults
are greedy and would silently degrade the routing workload for checkpoints without a
full sampling recommendation. The generated text is per-server-process deterministic
(fresh server, fixed request sequence), so one text sha1 per backend is a real
cross-backend check; token ids are not visible over the API, so this is a weaker
invariant than the old in-process id hash. ``--greedy`` sends temperature 0 for the
stricter comparison.

Run (one backend):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_decode_moe.py \
        --model /path/to/model

Run (all three backends, one server per backend):
    ... --model /path/to/model --backend offload,cpu,hybrid --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import fmean

# Applied for every field the checkpoint's generation_config.json does not specify.
FALLBACK_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

# AIME-25 problems, pulled from the Hub into the usual HF cache on first run.
AIME_REPO = "math-ai/aime25"
AIME_FILE = "test.jsonl"
# Reasoning models need the answer format spelled out; the boxed answer is also what makes
# a run spot-checkable by eye.
BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


class GpuTelemetry:
    """Low-rate GPU and host telemetry scoped to one request (best effort)."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[tuple[float, float, float, float]] = []
        self.system_samples: list[tuple[float, float, float | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._vm_start: dict[str, int] = {}

    @staticmethod
    def _vmstat() -> dict[str, int]:
        try:
            return {
                key: int(value)
                for key, value in (
                    line.split() for line in Path("/proc/vmstat").read_text().splitlines()
                )
                if key in {"pswpin", "pswpout", "pgfault", "pgmajfault"}
            }
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _system_sample() -> tuple[float, float | None]:
        available_gib = 0.0
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available_gib = int(line.split()[1]) / 2**20
                    break
        except (OSError, ValueError):
            pass
        nvme_temp = None
        for name in Path("/sys/class/hwmon").glob("hwmon*/name"):
            try:
                if name.read_text().strip() != "nvme":
                    continue
                value = name.parent / "temp1_input"
                nvme_temp = int(value.read_text().strip()) / 1000
                break
            except (OSError, ValueError):
                continue
        return available_gib, nvme_temp

    def start(self) -> None:
        self._started = time.perf_counter()
        self._vm_start = self._vmstat()
        have_nvidia_smi = shutil.which("nvidia-smi") is not None

        def sample() -> None:
            while not self._stop.is_set():
                if have_nvidia_smi:
                    try:
                        raw = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=5,
                        ).splitlines()[0]
                        power, temp, util = (float(value.strip()) for value in raw.split(","))
                        self.samples.append((time.perf_counter(), power, temp, util))
                    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
                        pass
                available_gib, nvme_temp = self._system_sample()
                self.system_samples.append((time.perf_counter(), available_gib, nvme_temp))
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, name="bench-gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        elapsed = time.perf_counter() - self._started if self._started else 0.0
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
        result = {
            "samples": len(self.system_samples),
            "duration_s": elapsed,
        }
        if self.samples:
            powers = [sample[1] for sample in self.samples]
            temperatures = [sample[2] for sample in self.samples]
            utilization = [sample[3] for sample in self.samples]
            average_power = fmean(powers)
            result.update({
                "power_w_avg": average_power,
                "power_w_peak": max(powers),
                "temperature_c_peak": max(temperatures),
                "utilization_pct_avg": fmean(utilization),
                "request_energy_j_est": average_power * elapsed,
            })
        if self.system_samples:
            result["mem_available_gib_min"] = min(sample[1] for sample in self.system_samples)
            nvme = [sample[2] for sample in self.system_samples if sample[2] is not None]
            if nvme:
                result["nvme_temperature_c_peak"] = max(nvme)
        vm_end = self._vmstat()
        for key, label in (
            ("pswpin", "swap_in_pages_delta"),
            ("pswpout", "swap_out_pages_delta"),
            ("pgfault", "page_faults_delta"),
            ("pgmajfault", "major_faults_delta"),
        ):
            if key in self._vm_start and key in vm_end:
                result[label] = vm_end[key] - self._vm_start[key]
        return result


def runtime_provenance() -> dict:
    """Versions and tracked-worktree identity needed to reproduce a result row."""
    repo = Path(__file__).resolve().parents[1]

    def command(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=repo, text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    packages = {}
    for name in ("torch", "triton", "flashinfer-python", "sglang-kernel", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    gpu = command(
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader,nounits",
    )
    return {
        "git_revision": command("git", "rev-parse", "HEAD"),
        "git_tracked_dirty": bool(command("git", "status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "gpu_driver": gpu,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint dir (or .ftw)")
    p.add_argument(
        "--backend",
        default="offload",
        help="comma list of offload|cpu|hybrid; one server per backend",
    )
    p.add_argument(
        "--storage", choices=("ram", "disk"), default="ram",
        help="routed-expert backing store",
    )
    p.add_argument(
        "--nvfp4-backend",
        choices=("auto", "marlin", "flashinfer", "triton"),
        default="triton",
        help="NVFP4 expert kernel/layout; must match an FTW checkpoint's converted layout",
    )
    p.add_argument(
        "--host-cache-gb", type=float, default=1.0,
        help="disk mode pageable host expert-LRU budget in GiB",
    )
    p.add_argument(
        "--cpu-threads", type=int, default=0,
        help="CPU MoE worker threads; 0 keeps server auto-selection",
    )
    p.add_argument(
        "--aime",
        default=os.environ.get("FREETOKEN_AIME25_JSONL"),
        help=f"local jsonl instead of downloading {AIME_REPO}; default $FREETOKEN_AIME25_JSONL",
    )
    p.add_argument("--problem", type=int, default=0, help="0-based AIME problem index")
    p.add_argument("--decode", type=int, default=256, help="decode tokens to measure (D)")
    p.add_argument(
        "--cache",
        type=int,
        default=0,
        help="GPU expert cache slots; 0 = auto-size from free VRAM",
    )
    p.add_argument("--cache-rate", type=float, default=None, help="cache slots as a fraction of L*E")
    p.add_argument("--cache-policy", choices=("lru", "layer_lru"), default="lru")
    p.add_argument(
        "--hybrid-fetch",
        type=int,
        default=-1,
        help="hybrid: max PCIe fetches/layer; -1 = auto (benched pcie/cpu bandwidth fraction)",
    )
    p.add_argument("--mem-ratio", type=float, default=0.9, help="target VRAM utilization")
    p.add_argument(
        "--num-tokens", type=int, default=0,
        help="explicit KV capacity; 0 keeps server auto-sizing",
    )
    p.add_argument(
        "--disable-prefill-overlap", action="store_true",
        help="disable the two-layer prefill staging reservation (permits a one-layer cache)",
    )
    p.add_argument(
        "--prefill-hit-d2d", action="store_true",
        help="copy prefill cache hits device-side and stream only misses (CUDA >= 13)",
    )
    p.add_argument(
        "--prefill-sparse-max-tokens", type=int, default=0,
        help="route-first sparse prefill threshold; 0 keeps full-layer staging",
    )
    p.add_argument(
        "--shared-expert-overlap", action="store_true",
        help="overlap supported resident shared experts with disk staging",
    )
    p.add_argument(
        "--collect-moe-stats", action="store_true",
        help="collect expert-cache counters and report the measured-request delta",
    )
    p.add_argument(
        "--include-output", action="store_true",
        help="include generated text in JSON output for first-divergence analysis",
    )
    p.add_argument("--no-graph", action="store_true", help="eager decode instead of CUDA graph")
    p.add_argument(
        "--greedy",
        action="store_true",
        help="force temperature 0 (ignore the checkpoint's sampling) so ids are comparable",
    )
    p.add_argument(
        "--server-timeout",
        type=float,
        default=1800,
        help="seconds to wait for the spawned server to become ready",
    )
    p.add_argument("--json", dest="json_out", default=None, help="append the result rows here")
    return p.parse_args(argv)


def load_problem(path: str | None, index: int) -> tuple[str, str]:
    """One AIME-25 (problem, answer). Downloads the dataset unless ``path`` overrides it.

    Accepts both the Hub schema (``problem``) and the pre-formatted jsonl some local copies
    use (``prompt``, answer instruction already appended)."""
    if not path:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(AIME_REPO, AIME_FILE, repo_type="dataset")
        except Exception as e:  # offline, rate-limited, repo moved
            sys.exit(f"could not fetch {AIME_REPO}/{AIME_FILE} ({e}); pass --aime <local jsonl>")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not 0 <= index < len(rows):
        sys.exit(f"--problem {index} out of range ({len(rows)} problems available)")
    row = rows[index]
    text = row.get("problem") or row["prompt"]
    if "boxed" not in text:
        text = f"{text}\n{BOXED_INSTRUCTION}"
    return text, str(row.get("answer", ""))


def resolve_sampling(model_path: str, greedy: bool) -> tuple[dict, str]:
    """Checkpoint-recommended sampling with per-field fallback; returns (params, source).

    Resolved client-side and sent explicitly: the server fills unspecified fields with
    its framework defaults (temperature 0 / no filtering), not with these fallbacks."""
    if greedy:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "greedy (--greedy)"
    recommended: dict = {}
    cfg = Path(model_path) / "generation_config.json"
    if cfg.is_file():
        raw = json.loads(cfg.read_text())
        recommended = {k: raw[k] for k in FALLBACK_SAMPLING if raw.get(k) is not None}
        if raw.get("do_sample") is False or recommended.get("temperature") == 0.0:
            return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "checkpoint (greedy)"
    params = {**FALLBACK_SAMPLING, **recommended}
    if params["top_k"] == 0:
        params["top_k"] = -1  # HF spells "no top-k filtering" as 0; the API as -1
    taken = sorted(recommended)
    source = f"checkpoint{taken} + fallback" if taken else "fallback (no generation_config)"
    return params, source


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_cmd(args: argparse.Namespace, backend: str, port: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port),
        "--moe-backend", backend,
        "--nvfp4-backend", args.nvfp4_backend,
        "--moe-storage", args.storage,
        "--moe-host-cache-gb", str(args.host_cache_gb),
        "--max-running-requests", "1",
        "--max-seq-len-override", str(8192 + args.decode),
        "--memory-ratio", str(args.mem_ratio),
        "--cuda-graph-max-bs", "0" if args.no_graph else "1",
        "--moe-hybrid-max-fetch", str(args.hybrid_fetch),
        "--moe-cache-policy", args.cache_policy,
    ]
    if args.num_tokens > 0:
        cmd += ["--num-tokens", str(args.num_tokens)]
    if args.cpu_threads > 0:
        cmd += ["--moe-cpu-threads", str(args.cpu_threads)]
    if args.disable_prefill_overlap:
        cmd.append("--disable-moe-prefill-overlap")
    if args.prefill_hit_d2d:
        cmd.append("--moe-prefill-hit-d2d")
    if args.prefill_sparse_max_tokens > 0:
        cmd += ["--moe-prefill-sparse-max-tokens", str(args.prefill_sparse_max_tokens)]
    if args.shared_expert_overlap:
        cmd.append("--moe-shared-expert-overlap")
    if args.collect_moe_stats:
        cmd.append("--moe-collect-stats")
    if args.cache > 0:
        cmd += ["--moe-cache-size", str(args.cache)]
    elif args.cache_rate is not None:
        cmd += ["--moe-cache-rate", str(args.cache_rate)]
    else:
        cmd.append("--moe-cache-auto")
    return cmd


def die_with_log(msg: str, log_path: str) -> None:
    tail = "".join(Path(log_path).read_text().splitlines(keepends=True)[-30:])
    sys.exit(f"[bench] {msg}\n[bench] server log tail ({log_path}):\n{tail}")


def wait_ready(origin: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(f"server exited with code {proc.returncode} during startup", log_path)
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except (OSError, ValueError):  # not bound yet / reset / partial response
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            die_with_log(f"server reported startup error: {health}", log_path)
        if health.get("maintenance") == "serving":
            return
        time.sleep(1.0)
    die_with_log(f"server not ready after {timeout:.0f}s", log_path)


def pump_output(src, log_f) -> None:
    """Mirror the server's output to our terminal while keeping the log file complete.

    Raw byte chunks (read1, not line-buffered) so \\r progress bars render live."""
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)
        log_f.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()


def stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM the whole session (frontend + scheduler/tokenizer workers), escalate.

    Best-effort by design: it runs in ``finally`` and must not mask the real error.
    killpg runs even when the frontend already exited -- a crashed frontend leaves live
    non-daemon workers in the group, and they hold the GPU."""
    for sig, wait_s in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:  # whole group already gone
            pass
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    time.sleep(3)  # let the driver reclaim VRAM before the next backend's server


def stream_generate(origin: str, model_id: str, problem: str, sampling: dict,
                    args: argparse.Namespace) -> dict:
    """One streamed chat completion; returns per-token arrival stamps, text, and usage."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": args.decode,
        "ignore_eos": not getattr(args, "normal_eos", False),
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling,
    }
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict | None = None
    finish_reason: str | None = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=max(1800, args.decode * 3))
    except urllib.error.HTTPError as e:
        sys.exit(f"[bench] request failed: HTTP {e.code}: {e.read()[:500]!r}")
    # Iterate the SSE stream line by line as bytes; json.loads decodes UTF-8 itself.
    # (A text-mode reader keyed off the content-type would decode latin-1: the server
    # sends ensure_ascii=False JSON with no charset on text/event-stream.)
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue  # blank separators between events
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    stamps.append(now)
                    pieces.append(text)
    if usage is None:
        sys.exit("[bench] stream ended without a usage chunk; is this a FreeToken server?")
    return {
        "t0": t0,
        "stamps": stamps,
        "text": "".join(pieces),
        "usage": usage,
        "finish_reason": finish_reason,
    }


def run_one(args: argparse.Namespace, backend: str) -> dict:
    problem, answer = load_problem(args.aime, args.problem)
    sampling, sampling_src = resolve_sampling(args.model, args.greedy)
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix=f"bench-serve-{backend}-", suffix=".log")
    cmd = serve_cmd(args, backend, port)

    print(
        f"[bench] model={args.model}\n"
        f"[bench] backend={backend} cache={args.cache or args.cache_rate or 'auto'} "
        f"mem_ratio={args.mem_ratio} decode={args.decode} graph={not args.no_graph}\n"
        f"[bench] sampling={sampling} <- {sampling_src}\n"
        f"[bench] server log: {log_path}",
        flush=True,
    )

    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            cache_status = get_json(f"{origin}/v1/cache/status")
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            print(f"[bench] model_id={model_id}", flush=True)
            print(f"[bench] AIME25 #{args.problem} (answer {answer})", flush=True)

            # Warm the expert cache to a steady-state decode working set.
            stream_generate(origin, model_id, problem, sampling, args)
            stats_before = get_json(f"{origin}/v1/stats")
            telemetry = GpuTelemetry()
            telemetry.start()
            try:
                r = stream_generate(origin, model_id, problem, sampling, args)
            finally:
                gpu_telemetry = telemetry.stop()
            stats = get_json(f"{origin}/v1/stats")
        finally:
            stop_server(proc)
            pump.join(timeout=10)

    stamps, usage = r["stamps"], r["usage"]
    if len(stamps) < 2:
        sys.exit(f"[bench] need >=2 token events to measure decode, got {len(stamps)}")
    completion = usage["completion_tokens"]
    if completion != args.decode:
        print(f"[bench] WARNING: completion_tokens={completion} != --decode {args.decode}", flush=True)
    steps = completion - 1
    decode_time = stamps[-1] - stamps[0]
    gaps = sorted((b - a) * 1e3 for a, b in zip(stamps, stamps[1:]))
    row = {
        "model": args.model,
        "backend": backend,
        "configuration": {
            "storage": args.storage,
            "nvfp4_backend": args.nvfp4_backend,
            "host_cache_gb": args.host_cache_gb,
            "requested_cache_size": args.cache,
            "requested_cache_rate": args.cache_rate,
            "memory_ratio": args.mem_ratio,
            "requested_num_tokens": args.num_tokens,
            "cpu_threads": args.cpu_threads,
            "hybrid_fetch": args.hybrid_fetch,
            "disk_read_workers": int(os.getenv("FREETOKEN_DISK_READ_WORKERS", "16")),
            "cache_policy": args.cache_policy,
            "prefill_overlap": not args.disable_prefill_overlap,
            "prefill_hit_d2d": args.prefill_hit_d2d,
            "prefill_sparse_max_tokens": args.prefill_sparse_max_tokens,
            "shared_expert_overlap": args.shared_expert_overlap,
            "cuda_graph": not args.no_graph,
        },
        "cache_geometry": cache_status.get("geometry", {}),
        "problem": args.problem,
        "prompt_tokens": usage["prompt_tokens"],
        "decode_steps": steps,
        "decode_tok_s": steps / decode_time if decode_time > 0 else 0.0,
        "ms_per_token": decode_time / steps * 1e3 if steps > 0 else 0.0,
        "event_ms_p50": gaps[len(gaps) // 2],
        "event_ms_p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))],
        "ttft_ms": (stamps[0] - r["t0"]) * 1e3,
        "events": len(stamps),
        "completion_tokens": completion,
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "sampling": sampling,
        "output_sha1": hashlib.sha1(r["text"].encode()).hexdigest()[:12],
        "server_log": log_path,
        "gpu_telemetry": gpu_telemetry,
        "provenance": runtime_provenance(),
    }
    if args.include_output:
        row["output_text"] = r["text"]
    before_moe = stats_before.get("moe") or {}
    after_moe = stats.get("moe") or {}
    if after_moe:
        delta = {
            key: int(after_moe.get(key, 0)) - int(before_moe.get(key, 0))
            for key in (
                "active", "missing", "fetched", "calls", "sparse_prefill_layers",
                "sparse_prefill_routes", "sparse_prefill_unique_rows",
                "sparse_prefill_fallback_layers", "shared_expert_overlap_calls",
            )
        }
        active, missing, fetched, calls = (
            delta["active"], delta["missing"], delta["fetched"], delta["calls"]
        )
        row["moe"] = {
            **delta,
            "active_per_layer": active / calls if calls else 0.0,
            "missing_per_layer": missing / calls if calls else 0.0,
            "miss_rate": missing / active if active else 0.0,
            "fetched_per_layer": fetched / calls if calls else 0.0,
            "fetch_rate": fetched / missing if missing else 0.0,
        }
        before_disk = before_moe.get("disk") or {}
        after_disk = after_moe.get("disk") or {}
        if after_disk:
            disk_delta = {
                key: after_disk.get(key, 0) - before_disk.get(key, 0)
                for key in (
                    "cache_hits", "cache_misses", "cache_evictions", "cache_bypasses", "read_ops",
                    "logical_bytes", "physical_bytes", "read_seconds",
                )
            }
            for key in (
                "cache_capacity_entries", "cache_capacity_bytes",
                "cache_allocated_bytes", "cache_occupancy_entries", "cache_occupancy_bytes",
                "staging_allocated_bytes", "host_allocated_bytes",
                "read_workers", "cache_policy", "staging_buffers",
            ):
                disk_delta[key] = after_disk.get(key, 0)
            row["moe"]["disk"] = disk_delta

    print(f"\n==== decode bs=1 [{backend}] via /v1/chat/completions ====", flush=True)
    print(f"  decode throughput : {row['decode_tok_s']:8.2f} tok/s  ({row['ms_per_token']:.3f} ms/token)")
    print(f"  TTFT (warm)       : {row['ttft_ms']:8.1f} ms  (prompt {row['prompt_tokens']} tok)")
    print(f"  decode measured   : {steps} steps in {decode_time:.3f} s  "
          f"(event p50 {row['event_ms_p50']:.3f} / p99 {row['event_ms_p99']:.3f} ms, "
          f"{len(stamps)} events)")
    print(f"  vram (server)     : {row['vram_gib']:8.2f} GiB")
    if "power_w_avg" in row["gpu_telemetry"]:
        gpu = row["gpu_telemetry"]
        print(
            f"  board power/temp  : {gpu['power_w_avg']:.1f} W avg / "
            f"{gpu['power_w_peak']:.1f} W peak, {gpu['temperature_c_peak']:.0f} C peak"
        )
    sha_note = "greedy" if args.greedy else "sampled, per-server deterministic"
    print(f"  output sha1       : {row['output_sha1']}  ({sha_note}; compare across backends)")
    print(f"  output sample     : {r['text'][:240]!r}")
    if row.get("moe"):
        moe = row["moe"]
        print(
            f"  expert cache      : miss={moe['miss_rate']:.2%} "
            f"({moe['missing_per_layer']:.2f}/{moe['active_per_layer']:.2f} per layer)"
        )
        if moe.get("disk"):
            disk = moe["disk"]
            seconds = disk["read_seconds"]
            gib = disk["physical_bytes"] / 2**30
            print(
                f"  expert disk I/O   : {disk['read_ops']} reads, {gib:.2f} GiB physical, "
                f"{gib / seconds if seconds else 0.0:.2f} GiB/s"
            )
            requests = disk["cache_hits"] + disk["cache_misses"]
            print(
                f"  host expert LRU   : {disk['cache_hits'] / requests if requests else 0.0:.2%} hit, "
                f"{disk['cache_occupancy_entries']}/{disk['cache_capacity_entries']} entries, "
                f"{disk['cache_evictions']} evictions, {disk['cache_bypasses']} bypasses"
            )
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    unknown = [b for b in backends if b not in ("offload", "cpu", "hybrid")]
    if unknown:
        sys.exit(f"unknown backend(s): {unknown}")

    failed = []
    for backend in backends:
        try:
            row = run_one(args, backend)
        # SystemExit inherits BaseException, not Exception, so name both: a mid-decode
        # connection drop (server crash) must not abort the remaining backends either.
        except (SystemExit, Exception) as e:
            if len(backends) == 1:
                raise
            print(f"\n[bench] backend {backend} failed: {e!r}", flush=True)
            failed.append(backend)
            continue
        if args.json_out:
            with open(args.json_out, "a") as f:
                f.write(json.dumps(row) + "\n")
    if failed:
        print(f"\n[bench] backends that failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
