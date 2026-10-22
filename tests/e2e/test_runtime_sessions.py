"""Real-model paired application gate; CPU fixtures are not model qualification."""

import hashlib
import json
import os
from pathlib import Path
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
)

pytestmark = pytest.mark.needs_weights


def test_paired_native_application_sessions(tmp_path):
    model = os.getenv("SPARKLAB_SESSION_TEST_MODEL")
    if not model or not Path(model).is_dir() or not torch.cuda.is_available():
        pytest.skip("requires CUDA and SPARKLAB_SESSION_TEST_MODEL complete checkpoint")
    guard = max(30.0, float(os.getenv("SPARKLAB_SESSION_MIN_FREE_GIB", "30")))
    if _free_gpu_gib() < guard:
        pytest.skip(
            f"complete-server profile needs at least {guard:g} GiB free physical GPU memory"
        )
    root = Path(os.getenv("SPARKLAB_SESSION_EVIDENCE_ROOT", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=False) if root != tmp_path else None
    package = os.getenv("SPARKLAB_SESSION_PACKAGE_PATH", "python")
    extra = json.loads(os.getenv("SPARKLAB_SESSION_SERVER_ARGS", "[]"))
    features = json.loads(
        os.getenv("SPARKLAB_SESSION_FEATURES", '{"SPARKLAB_EXACT_PREFIX":"1"}')
    )
    allowed = {
        "SPARKLAB_EXACT_PREFIX",
        "SPARKLAB_SCHEDULER_POLICY",
        "SPARKLAB_PREFIX_STORE",
        "SPARKLAB_PREFIX_RESTORE_POLICY",
        "SPARKLAB_PREFIX_DISK_GIB",
        "SPARKLAB_PREFIX_DISK_RECORDS",
        "SPARKLAB_PREFIX_IO_TICKETS",
        "SPARKLAB_PREFIX_STAGING_MIB",
    }
    if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
        raise ValueError("server args must be a JSON string array")
    resource_options = {
        "--num-pages",
        "--page-size",
        "--max-running-requests",
        "--cuda-graph-max-bs",
        "--max-prefill-length",
        "--memory-ratio",
    }
    if len(extra) % 2 or any(flag not in resource_options for flag in extra[::2]):
        raise ValueError(
            "extra arguments may override only the declared resource profile"
        )
    if (
        not isinstance(features, dict)
        or not set(features) <= allowed
        or not all(isinstance(v, str) for v in features.values())
    ):
        raise ValueError("unsupported target-only session feature settings")
    if features.get("SPARKLAB_PREFIX_STORE"):
        features["SPARKLAB_PREFIX_STORE"] = str(root / "prefixes")
    context = int(os.getenv("SPARKLAB_SESSION_MAX_CONTEXT", "4096"))
    turns = int(os.getenv("SPARKLAB_SESSION_TURNS", "20"))
    lengths = [
        int(n) for n in os.getenv("SPARKLAB_SESSION_RECALL_LENGTHS", "").split(",") if n
    ]
    budget = int(os.getenv("SPARKLAB_SESSION_OUTPUT_TOKENS", "256"))
    legal_case = os.getenv("SPARKLAB_SESSION_LEGAL_CASE", "0")
    if legal_case not in ("0", "1"):
        raise ValueError("SPARKLAB_SESSION_LEGAL_CASE must be 0 or 1")
    if (
        not 20 <= turns <= 50
        or not 1 <= budget <= 1024
        or not budget < context <= 131072
    ):
        raise ValueError("invalid session/context envelope")
    if len(set(lengths)) != len(lengths) or any(
        n not in (8192, 32768) or n + budget > context for n in lengths
    ):
        raise ValueError("recall plus output must fit the declared context")
    model_profile = os.getenv("SPARKLAB_SESSION_MODEL_PROFILE", "default")
    profile = target_only_server_profile(model, model_profile, context=context)
    profile["--tool-call-parser"] = "auto"
    profile.update(dict(zip(extra[::2], extra[1::2])))
    identity = {
        "backend": "native",
        "model_profile": model_profile,
        "model_path": str(Path(model).resolve()),
        "launch_profile": profile,
        "cache_report": True,
        "metadata_sha256": {
            name: hashlib.sha256((Path(model) / name).read_bytes()).hexdigest()
            for name in (
                "config.json",
                "conversion-manifest.json",
                "freetoken_weight.json",
            )
            if (Path(model) / name).is_file()
        },
    }
    (root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    script = (
        Path(__file__).resolve().parents[2] / "benchmarks/validate_runtime_sessions.py"
    )
    for label in ("baseline", "candidate"):
        if _free_gpu_gib() < guard:
            pytest.skip("free physical GPU memory fell below the complete-server guard")
        # Clear inherited experimental knobs; profile changes are explicit and
        # all prefix writes remain inside this test's owned evidence directory.
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
            SPARKLAB_SCHEDULER_POLICY="baseline",
            SPARKLAB_EXACT_PREFIX="0",
            SPARKLAB_PREFIX_STORE="",
            SPARKLAB_DFLASH_POLICY="fixed",
            SPARKLAB_RUNTIME_TIMING="1",
            SPARKLAB_RUNTIME_TRACE_CAPACITY="0",
        )
        if label == "candidate":
            env.update(features)
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "sparklab",
            "serve",
            "--model-path",
            model,
            "--served-model-name",
            "session-test",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *profile_arguments(profile),
            "--enable-cache-report",
        ]
        with (root / f"{label}.server.log").open("w") as log:
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env
            )
            try:
                _wait_until_serving(base, proc, time.monotonic() + 300.0)
                geometry = _get(base, "/v1/cache/status")["geometry"]
                observed = {
                    key: geometry[key]
                    for key in (
                        "num_pages",
                        "page_size",
                        "num_mamba_slots",
                        "moe_cache_size",
                        "num_swa_pages",
                        "swa_page_size",
                    )
                }
                if label == "baseline":
                    identity["observed_cache_geometry"] = observed
                    (root / "identity.json").write_text(
                        json.dumps(identity, indent=2) + "\n"
                    )
                else:
                    assert observed == identity["observed_cache_geometry"], (
                        "candidate cache capacities differ from the baseline"
                    )
                client = [
                    sys.executable,
                    str(script),
                    "--base-url",
                    base,
                    "--model",
                    "session-test",
                    "--tokenizer",
                    model,
                    "--identity",
                    str(root / "identity.json"),
                    "--output",
                    str(root / f"{label}.json"),
                    "--label",
                    label,
                    "--turns",
                    str(turns),
                    "--output-tokens",
                    str(budget),
                    "--max-context",
                    str(context),
                ]
                if lengths:
                    client.extend(["--recall-lengths", *map(str, lengths)])
                if legal_case == "1":
                    client.append("--legal-case")
                if label == "candidate":
                    client.extend(["--baseline", str(root / "baseline.json")])
                result = subprocess.run(
                    client,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3600.0,
                )
                (root / f"{label}.client.log").write_text(result.stdout)
                assert result.returncode == 0, result.stdout
                stats = _get(base, "/v1/stats")
                (root / f"{label}.stats.json").write_text(
                    json.dumps(stats, indent=2) + "\n"
                )
                assert stats["requests"]["active"] == 0
                if (
                    label == "candidate"
                    and features.get("SPARKLAB_EXACT_PREFIX") == "1"
                ):
                    exact = stats["native_runtime"]["exact_prefix"]
                    assert exact["published"] > 0 and exact["hits"] > 0, (
                        "exact path was not exercised"
                    )
                if label == "candidate" and features.get("SPARKLAB_PREFIX_STORE"):
                    assert (
                        stats["native_runtime"]["prefix_store"]["counters"][
                            "writes_completed"
                        ]
                        > 0
                    )
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=60.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30.0)
                (root / f"{label}.launch.json").write_text(
                    json.dumps(
                        {
                            "command": command,
                            "features": {k: v for k, v in env.items() if k in allowed},
                            "exit_code": proc.returncode,
                        },
                        indent=2,
                    )
                    + "\n"
                )
        assert proc.returncode == 143, (
            "server did not finish its ordinary SIGTERM cleanup"
        )
        stop_log = (root / f"{label}.server.log").read_text()
        assert (
            "exceeded shutdown grace" not in stop_log
            and "resource_tracker:" not in stop_log
        )
