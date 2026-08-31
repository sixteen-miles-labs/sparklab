#!/usr/bin/env python3
"""Launch one framework, benchmark its OpenAI endpoint, and preserve failure evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
USER_HOME = Path.home()
DEFAULT_MODEL = USER_HOME / "models/frameworks/qwen3.6-35b-a3b/hf"
DEFAULT_GGUF = USER_HOME / "models/frameworks/qwen3.6-35b-a3b/gguf/qwen3.6-35b-a3b.gguf"
DEFAULT_FTW = USER_HOME / ".sparklab/models/qwen3.6-35b-a3b/prepared/0.2.0"
INSTALL_ROOT = Path(os.environ.get(
    "SPARKLAB_FRAMEWORK_ROOT", USER_HOME / ".local/share/sparklab-frameworks"
))
SERVED_MODEL = "qwen3.6-35b-a3b"
DEFAULT_OLLAMA_MODEL = "qwen3.6:35b-a3b"
KIB_PER_GIB = 1 << 20
NVFP4_FRAMEWORKS = {"vllm", "sglang"}


class MemorySafetyAbort(RuntimeError):
    pass


def command_for(name: str, model: Path, gguf: Path, port: int, ftw: Path | None = None,
                model_family: str = "qwen36") -> tuple[list[str], str]:
    ftw = ftw or DEFAULT_FTW
    served_model = "qwen3.8-flash-next" if model_family == "qwen38" else SERVED_MODEL
    if name == "vllm":
        exe = INSTALL_ROOT / "vllm/bin/vllm"
        return ([str(exe), "serve", str(model), "--host", "127.0.0.1", "--port", str(port),
                 "--served-model-name", SERVED_MODEL, "--max-model-len", "32768",
                 "--gpu-memory-utilization", "0.50", "--trust-remote-code"], str(exe))
    if name == "sglang":
        exe = INSTALL_ROOT / "sglang/bin/python"
        return ([str(exe), "-m", "sglang.launch_server", "--model-path", str(model),
                 "--host", "127.0.0.1", "--port", str(port), "--served-model-name", SERVED_MODEL,
                 "--context-length", "32768", "--mem-fraction-static", "0.50",
                 "--max-running-requests", "1", "--max-total-tokens", "32768",
                 "--max-mamba-cache-size", "5", "--watchdog-timeout", "1200",
                 "--moe-runner-backend", "flashinfer_cutlass", "--disable-flashinfer-autotune",
                 "--disable-cuda-graph", "--trust-remote-code"], str(exe))
    if name == "llama.cpp":
        exe = INSTALL_ROOT / "llama.cpp/build/bin/llama-server"
        return ([str(exe), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
                 "--alias", SERVED_MODEL, "-ngl", "99", "-c", "32768", "--parallel", "1",
                 "-fa", "on", "--no-webui"], str(exe))
    if name == "ollama":
        exe = INSTALL_ROOT / "ollama/bin/ollama"
        return ([str(exe), "serve"], str(exe))
    if name == "ollama-no-mtp":
        exe = INSTALL_ROOT / "ollama/lib/ollama/llama-server"
        return ([str(exe), "--model", str(gguf), "--host", "127.0.0.1", "--port", str(port),
                 "--alias", SERVED_MODEL, "--no-webui", "--offline", "-c", "32768", "-np", "1",
                 "-ngl", "99",
                 "--no-jinja", "--chat-template", "chatml", "--load-mode", "dio",
                 "--flash-attn", "auto", "-b", "1024", "-ub", "1024", "--keep", "4"], str(exe))
    if name == "ktransformers":
        exe = INSTALL_ROOT / "ktransformers/bin/python"
        return ([str(exe), "-m", "ktransformers.server.main", "--model_path", str(model),
                 "--gguf_path", str(gguf), "--host", "127.0.0.1", "--port", str(port)], str(exe))
    if name in {"freetoken", "sparklab"}:
        exe = INSTALL_ROOT / "freetoken/bin/ft" if name == "freetoken" else ROOT.parents[1] / ".venv/bin/sparklab"
        common = [str(exe), "serve", "--model", str(ftw), "--host", "127.0.0.1",
                 "--port", str(port), "--served-model-name", served_model,
                 "--max-running-requests", "1", "--max-seq-len-override", "32768",
                 "--num-tokens", "32832", "--cuda-graph-max-bs", "1",
                 "--moe-backend", "offload", "--moe-cache-rate", "1.0",
                 "--nvfp4-backend", "triton", "--moe-prefill-hit-d2d"]
        if model_family == "qwen38":
            common.extend(["--attention-backend", "qsa" if name == "sparklab" else "qsa_sparse",
                           "--cache-type", "naive",
                           "--page-size", "16"])
            if name == "sparklab":
                common.extend(["--moe-storage", "disk", "--moe-host-cache-gb", "0",
                               "--moe-preload-all"])
        return (common, str(exe))
    raise ValueError(name)


def validate_nvfp4_checkpoint(model: Path) -> None:
    config_path = model / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read checkpoint quantization metadata: {error}") from error
    quantization = config.get("quantization_config") or {}
    layers = quantization.get("quantized_layers") or {}
    algorithms = {
        layer.get("quant_algo")
        for layer in layers.values()
        if isinstance(layer, dict)
    }
    if "W4A16_NVFP4" not in algorithms:
        raise ValueError(f"checkpoint is not the required ModelOpt NVFP4 format: {model}")


def environment_for(name: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if name == "sglang":
        # These are separate from --disable-flashinfer-autotune. On GB10, the FP8
        # configuration sweep previously launched 22 cicc processes that used 60 GiB
        # of host RAM while the model already held unified memory.
        env.setdefault("SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE", "0")
        # Four compiler jobs keep peak compiler RSS below the host safety margin
        # while avoiding the impractical startup time of fully serial compilation.
        env.setdefault("MAX_JOBS", "4")
        env.setdefault("FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS", "1")
    if name == "ollama":
        env.setdefault("OLLAMA_MODELS", str(USER_HOME / "models/frameworks/ollama"))
        env.setdefault("OLLAMA_NUM_PARALLEL", "1")
        env.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
        env.setdefault("OLLAMA_CONTEXT_LENGTH", "32768")
    if name == "ollama-no-mtp":
        lib = INSTALL_ROOT / "ollama/lib/ollama"
        cuda = lib / "cuda_v13"
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib}:{cuda}" + (f":{current}" if current else "")
        env["GGML_BACKEND_PATH"] = str(cuda / "libggml-cuda.so")
    if name in {"freetoken", "sparklab"}:
        env.setdefault("MAX_JOBS", "4")
    return env


def memory_safety_reason(
    *, min_available_gib: float, min_swap_free_gib: float, meminfo: Path = Path("/proc/meminfo")
) -> str | None:
    values: dict[str, int] = {}
    for line in meminfo.read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    available = values["MemAvailable"] / KIB_PER_GIB
    swap_free = values.get("SwapFree", 0) / KIB_PER_GIB
    if available < min_available_gib:
        return f"MemAvailable {available:.1f} GiB fell below {min_available_gib:.1f} GiB"
    if values.get("SwapTotal", 0) and swap_free < min_swap_free_gib:
        return f"SwapFree {swap_free:.1f} GiB fell below {min_swap_free_gib:.1f} GiB"
    return None


def terminate_group(process: subprocess.Popen, timeout: int = 30) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def version_for(name: str, exe: str) -> str:
    commands = {
        "vllm": [exe, "--version"],
        "sglang": [exe, "-c", "import sglang; print(sglang.__version__)"],
        "llama.cpp": [exe, "--version"],
        "ollama": [exe, "--version"],
        "ollama-no-mtp": [str(INSTALL_ROOT / "ollama/bin/ollama"), "--version"],
        "ktransformers": [exe, "-c", "import importlib.metadata as m; print(m.version('ktransformers'))"],
        "freetoken": [exe, "--version"],
        "sparklab": [exe, "--version"],
    }
    try:
        return subprocess.check_output(commands[name], text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=3) as response:
            body = response.read()
            if body:
                try:
                    health = json.loads(body)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    health = None
                if isinstance(health, dict) and "status" in health:
                    return health["status"] in {"ok", "ready", "healthy"}
            return response.status < 400
    except urllib.error.HTTPError as error:
        # A live 503 means the framework is still loading or warming up. Do not
        # mistake an early /v1/models response for readiness, as the old harness did.
        if error.code != 404:
            return False
    except Exception:
        return False
    try:
        with urllib.request.urlopen(base_url + "/v1/models", timeout=3) as response:
            return response.status < 400
    except Exception:
        return False


def classify(log: str, returncode: int | None) -> tuple[str, str]:
    low = log.lower()
    if re.search(r"out of memory|cuda.*oom|cannot allocate memory|oom-kill", low):
        return "oom", "server log reports an out-of-memory allocation failure"
    if re.search(r"unsupported|not supported|unknown model|model architecture.*not", low):
        match = next((line.strip() for line in log.splitlines() if "support" in line.lower()), "")
        return "unsupported", match[-500:] or "framework reports the model or quantization is unsupported"
    if "no such file or directory" in low or returncode == 127:
        return "unavailable", "framework executable or required artifact is missing"
    tail = "\n".join(log.splitlines()[-8:])[-1200:]
    return "failed", tail or f"server exited with status {returncode}"


def write_failure(path: Path, framework: str, version: str, status: str, reason: str, log: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "1.0", "framework": framework,
        "framework_version": version, "status": status, "reason": reason,
        "server_log": str(log)}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=["vllm", "sglang", "llama.cpp", "ktransformers", "ollama", "ollama-no-mtp", "freetoken", "sparklab"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--ftw", type=Path, default=DEFAULT_FTW)
    parser.add_argument("--model-family", choices=["qwen36", "qwen38"], default="qwen36")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--min-available-gib", type=float, default=12.0,
        help="terminate the framework before host available memory drops below this value")
    parser.add_argument("--min-swap-free-gib", type=float, default=2.0,
        help="terminate the framework before free swap drops below this value")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/qwen3.6-35b-a3b")
    args = parser.parse_args()
    output = args.output_dir / f"{args.framework.replace('.', '-')}.json"
    log_path = args.output_dir / f"{args.framework.replace('.', '-')}.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.framework in NVFP4_FRAMEWORKS:
        try:
            validate_nvfp4_checkpoint(args.model)
        except ValueError as error:
            write_failure(output, args.framework, "not-run", "unsupported", str(error), log_path)
            raise SystemExit(2)
    command, exe = command_for(args.framework, args.model, args.gguf, args.port, args.ftw,
        args.model_family)
    version = version_for(args.framework, exe)
    if not Path(exe).exists():
        write_failure(output, args.framework, version, "unavailable", f"executable not found: {exe}", log_path)
        raise SystemExit(2)
    if args.framework == "ktransformers":
        probe = subprocess.run([exe, "-c", "import ktransformers, kt_kernel"],
            text=True, capture_output=True)
        if probe.returncode:
            reason = (
                "KTransformers runtime is unavailable on this platform: its pinned kt-kernel "
                "release has no aarch64 wheel"
            )
            write_failure(output, args.framework, version, "unavailable", reason, log_path)
            raise SystemExit(2)
    if args.framework in {"llama.cpp", "ktransformers", "ollama-no-mtp"} and not args.gguf.exists():
        write_failure(output, args.framework, version, "unsupported", f"GGUF artifact not found: {args.gguf}", log_path)
        raise SystemExit(2)
    if args.framework in {"freetoken", "sparklab"} and not args.ftw.exists():
        write_failure(output, args.framework, version, "unavailable", f"FTW artifact not found: {args.ftw}", log_path)
        raise SystemExit(2)

    env = environment_for(args.framework)
    if args.framework == "ollama":
        env["OLLAMA_HOST"] = f"127.0.0.1:{args.port}"
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
            env=env, start_new_session=True, text=True)
        benchmark_process: subprocess.Popen | None = None
        try:
            deadline = time.monotonic() + args.startup_timeout
            base_url = f"http://127.0.0.1:{args.port}"
            while time.monotonic() < deadline and process.poll() is None:
                unsafe = memory_safety_reason(
                    min_available_gib=args.min_available_gib,
                    min_swap_free_gib=args.min_swap_free_gib,
                )
                if unsafe:
                    raise MemorySafetyAbort(unsafe)
                if ready(base_url):
                    break
                time.sleep(2)
            if process.poll() is not None or not ready(base_url):
                log_file.flush()
                text = log_path.read_text(errors="replace")
                status, reason = classify(text, process.poll())
                write_failure(output, args.framework, version, status, reason, log_path)
                raise SystemExit(3)
            benchmark_process = subprocess.Popen(
                [sys.executable, str(ROOT / "bench_openai.py"), "--base-url", base_url,
                "--model", args.ollama_model if args.framework == "ollama" else (
                    "qwen3.8-flash-next" if args.model_family == "qwen38" else SERVED_MODEL),
                "--framework", args.framework, "--version", version,
                "--weight-source", (
                    "oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW@cbbcf69f52b9815b8a987fe839003fae12aa8050" if args.framework in {"freetoken", "sparklab"} and args.model_family == "qwen38"
                    else "oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW@fecab7acfd0590d2b268d8fb9ea1c88431471111" if args.framework in {"freetoken", "sparklab"}
                    else "ollama.com/library/qwen3.6:35b-a3b" if args.framework in {"ollama", "ollama-no-mtp"}
                    else "ollama.com/library/qwen3.6:35b-a3b (GGUF layer sha256:d372de8e9348...)" if args.framework in {"llama.cpp", "ktransformers"}
                    else "nvidia/Qwen3.6-35B-A3B-NVFP4"
                ),
                "--weight-format", (
                    "FTW NVFP4" if args.framework in {"freetoken", "sparklab"}
                    else "Q4_K_M" if args.framework in {"llama.cpp", "ktransformers", "ollama", "ollama-no-mtp"}
                    else "ModelOpt NVFP4"
                ),
                "--trials", str(args.trials), "--output", str(output)],
                start_new_session=True,
            )
            while benchmark_process.poll() is None:
                unsafe = memory_safety_reason(
                    min_available_gib=args.min_available_gib,
                    min_swap_free_gib=args.min_swap_free_gib,
                )
                if unsafe:
                    raise MemorySafetyAbort(unsafe)
                time.sleep(1)
            if benchmark_process.returncode:
                write_failure(output, args.framework, version, "failed",
                    f"benchmark client exited {benchmark_process.returncode}", log_path)
                raise SystemExit(benchmark_process.returncode)
        except MemorySafetyAbort as error:
            message = f"memory safety watchdog stopped the run: {error}"
            log_file.write(f"\n[benchmark] {message}\n")
            log_file.flush()
            write_failure(output, args.framework, version, "oom_prevented", message, log_path)
            raise SystemExit(4)
        finally:
            if benchmark_process is not None:
                terminate_group(benchmark_process)
            terminate_group(process)


if __name__ == "__main__":
    main()
