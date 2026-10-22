"""Four actual native boots; raw dispatch proof of disk recovery and disable rollback."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import torch

from tests.e2e.runtime_profiles import profile_arguments, target_only_server_profile

from tests.e2e.test_cache_rebuild import (
    _free_port,
    _free_gpu_gib,
    _wait_until_serving,
    _get,
    _post,
)

pytestmark = pytest.mark.needs_weights


def module(name):
    path = Path(__file__).resolve().parents[2] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_real_disk_prefix_restart_dispatch_and_disable_rollback(tmp_path):
    model = os.getenv("SPARKLAB_RESTART_TEST_MODEL")
    if not model or not Path(model).is_dir() or not torch.cuda.is_available():
        pytest.skip(
            "requires CUDA and SPARKLAB_RESTART_TEST_MODEL complete FTW hybrid artifact"
        )
    guard = max(30.0, float(os.getenv("SPARKLAB_RESTART_MIN_FREE_GIB", "30")))
    if _free_gpu_gib() < guard:
        pytest.skip(
            f"complete-server restart profile requires {guard:g} GiB free physical GPU memory"
        )
    root = Path(os.getenv("SPARKLAB_RESTART_EVIDENCE_ROOT", str(tmp_path)))
    if root != tmp_path:
        root.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(root).free < 3 << 30:
        pytest.skip("restart fixture needs 3 GiB free disk for its owned 2 GiB store")
    size = int(os.getenv("SPARKLAB_RESTART_PREFIX_TOKENS", "65"))
    context = int(os.getenv("SPARKLAB_RESTART_MAX_CONTEXT", "4096"))
    pages = int(os.getenv("SPARKLAB_RESTART_PAGES", "512"))
    if (
        not 1 <= size <= 32768
        or not size + 16 <= context <= 65536
        or not context + 16 <= pages * 16 <= 262144
    ):
        raise ValueError(
            "declared prefix/context/page profile cannot hold the test work"
        )
    package = os.getenv("SPARKLAB_RESTART_PACKAGE_PATH", "python")
    proof = module("validate_prefix_restart")
    make = module("make_runtime_workload")
    from sparklab.utils import load_tokenizer
    from sparklab.tokenizer.tokenize import TokenizeManager
    from sparklab.message import TokenizeMsg
    from sparklab.core import SamplingParams

    tokenizer = load_tokenizer(model)
    manager = TokenizeManager(tokenizer)
    prefix = make.exact_text(tokenizer, size)
    query = prefix + " compass"

    def encode(text):
        return manager.tokenize(
            [TokenizeMsg(uid=0, text=text, sampling_params=SamplingParams())]
        )[0].tolist()

    pids, qids = encode(prefix), encode(query)
    assert len(pids) == size and qids[:size] == pids and len(qids) > size

    def digest(ids):
        return hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()

    model_profile = os.getenv("SPARKLAB_RESTART_MODEL_PROFILE", "default")
    profile = target_only_server_profile(
        model, model_profile, context=context, pages=pages, max_running=1, graphs=1,
    )
    identity = {
        "backend": "native",
        "model_profile": model_profile,
        "model_path": str(Path(model).resolve()),
        "launch_profile": profile,
        "overlap": False,
        "recurrent_state_dtype": os.getenv("SPARKLAB_MAMBA_SSM_DTYPE", "float32"),
        "metadata_sha256": {
            name: hashlib.sha256((Path(model) / name).read_bytes()).hexdigest()
            for name in (
                "config.json",
                "freetoken_weight.json",
                "conversion-manifest.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
            if (Path(model) / name).is_file()
        },
    }
    report = {
        "identity": identity,
        "prefix_tokens": size,
        "query_tokens": len(qids),
        "output_tokens": 4,
        "prefix_ids_sha256": digest(pids),
        "query_ids_sha256": digest(qids),
        "query_prefix_ids_sha256": digest(qids[:size]),
        "runs": {},
        "passed": False,
    }
    evidence = root / "evidence.json"
    store = root / "prefixes"
    geometry = None

    def save():
        evidence.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for mode in proof.MODES:
            if _free_gpu_gib() < guard:
                pytest.skip(
                    "free physical GPU memory fell below the restart profile guard"
                )
            if mode == "rollback":
                report["records_before_rollback"] = proof.inventory(store)
            enabled = mode in ("publish", "restore")
            port = _free_port()
            base = f"http://127.0.0.1:{port}"
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(
                    (
                        "SPARKLAB_EXACT_",
                        "SPARKLAB_PREFIX_",
                        "SPARKLAB_SCHEDULER_",
                        "SPARKLAB_DFLASH_",
                    )
                )
            }
            env.update(
                PYTHONPATH=package,
                SPARKLAB_DISABLE_OVERLAP_SCHEDULING="1",
                SPARKLAB_SCHEDULER_POLICY="baseline",
                SPARKLAB_DFLASH_POLICY="fixed",
                SPARKLAB_EXACT_PREFIX="1" if enabled else "0",
                SPARKLAB_PREFIX_STORE=str(store) if enabled else "",
                SPARKLAB_PREFIX_RESTORE_POLICY="always",
                SPARKLAB_PREFIX_SCOPE="owned-restart-fixture",
                SPARKLAB_PREFIX_DISK_GIB="2",
                SPARKLAB_PREFIX_DISK_RECORDS="16",
                SPARKLAB_PREFIX_IO_TICKETS="2",
                SPARKLAB_PREFIX_STAGING_MIB="1",
                SPARKLAB_RUNTIME_TIMING="1",
                SPARKLAB_RUNTIME_TRACE_CAPACITY="512",
                SPARKLAB_RUNTIME_LOG="1",
            )
            command = [
                sys.executable,
                "-m",
                "sparklab",
                "serve",
                "--model-path",
                model,
                "--served-model-name",
                "restart-test",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                *profile_arguments(profile),
                "--enable-cache-report",
            ]
            log_path = root / f"{mode}.server.log"
            run = {
                "mode": mode,
                "identity": identity,
                "command": command,
                "exact_enabled": enabled,
                "disk_enabled": enabled,
                "prefix_ids_sha256": report["prefix_ids_sha256"],
                "query_ids_sha256": report["query_ids_sha256"],
            }
            report["runs"][mode] = run
            save()
            with log_path.open("w") as log:
                proc = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env
                )
                try:
                    _wait_until_serving(base, proc, time.monotonic() + 300.0)
                    measured = _get(base, "/v1/cache/status")["geometry"]
                    run["geometry"] = {
                        key: measured[key]
                        for key in (
                            "num_pages",
                            "page_size",
                            "num_mamba_slots",
                            "moe_cache_size",
                            "num_swa_pages",
                            "swa_page_size",
                        )
                    }
                    if geometry is None:
                        geometry = run["geometry"]
                    assert run["geometry"] == geometry, (
                        "cache capacities changed across boots"
                    )
                    # A no-op control message wakes the idle actor without
                    # dispatching model tokens or rebuilding any pool. Observe
                    # recovered worker metadata before the first restore query.
                    deadline = time.monotonic() + 30.0
                    run["control_wakeups"] = []
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(
                                "server exited while awaiting prefix startup observation"
                            )
                        code, reply = _post(base, "/v1/cache/rebuild", {}, timeout=5.0)
                        assert code == 200 and reply["status"] == "ok", reply
                        run["control_wakeups"].append(
                            {"status": reply["status"], "observed_at": time.monotonic()}
                        )
                        observed = proof.read_measurements(log_path)
                        ready = not enabled or (
                            observed["prefix"]
                            and observed["prefix"].get("worker_observation") is not None
                        )
                        if mode == "restore":
                            ready = ready and observed["prefix"].get("records", 0) > 0
                        if observed["metrics"] is not None and ready:
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "prefix startup observation did not become ready"
                            )
                        time.sleep(0.05)
                    run["startup_metrics"] = observed["metrics"]
                    run["startup_prefix"] = observed["prefix"]
                    after = _get(base, "/v1/cache/status")["geometry"]
                    assert {key: after[key] for key in run["geometry"]} == run[
                        "geometry"
                    ], "control wakeup changed cache geometry"
                    request = {
                        "model": "restart-test",
                        "prompt": prefix if mode == "publish" else query,
                        "max_tokens": 1 if mode == "publish" else 4,
                        "temperature": 0.0,
                        "ignore_eos": True,
                    }
                    run["request"] = request
                    save()
                    code, reply = _post(
                        base,
                        "/v1/completions",
                        request,
                        timeout=300.0,
                    )
                    run["reply"] = reply
                    assert code == 200, reply
                    expected_outputs = 1 if mode == "publish" else 4
                    deadline = time.monotonic() + 60.0
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(
                                "server exited while awaiting post-request dispatch evidence"
                            )
                        observed = proof.read_measurements(log_path)
                        metric = observed["metrics"]
                        retention = observed["prefix"]
                        finished = (
                            metric
                            and metric.get("counters", {}).get(
                                "committed_output_tokens", 0
                            )
                            >= expected_outputs
                        )
                        if enabled:
                            field = (
                                "writes_completed"
                                if mode == "publish"
                                else "restores_completed"
                            )
                            finished = (
                                finished
                                and retention
                                and retention.get("unreleased_tickets") == 0
                                and retention.get("counters", {}).get(field, 0) > 0
                            )
                        if finished:
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "post-request trace/prefix drain evidence did not arrive"
                            )
                        time.sleep(0.05)
                    run.update(metrics=observed["metrics"], prefix=observed["prefix"])
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=60.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=30.0)
                    run["exit_code"] = proc.returncode
                    text = log_path.read_text()
                    run["forced_stop"] = "exceeded shutdown grace" in text
                    run["semaphore_warning"] = "resource_tracker:" in text
                    save()
            if mode == "publish":
                assert proof.inventory(store), "clean stop retained no checkpoint"
        report["records_after_rollback"] = proof.inventory(store)
        report["verification"] = proof.verify(report)
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
