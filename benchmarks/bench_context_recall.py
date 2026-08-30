"""Exact-token long-context recall gate for a complete SparkLab checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from bench_decode_moe import (
    GpuTelemetry,
    checkpoint_provenance,
    free_port,
    gb10_evidence,
    get_json,
    pump_output,
    runtime_provenance,
    serve_cmd,
    stop_server,
    wait_ready,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--json", dest="json_out", required=True)
    p.add_argument("--context-tokens", type=int, default=65_536)
    p.add_argument("--max-output-tokens", type=int, default=64)
    p.add_argument("--max-extend-tokens", type=int, default=8192)
    p.add_argument("--backend", default="offload")
    p.add_argument("--storage", default="disk")
    p.add_argument("--attention-backend", default="qsa")
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--cache-type", default="naive")
    p.add_argument("--nvfp4-backend", default="triton")
    p.add_argument("--host-cache-gb", type=float, default=2.0)
    p.add_argument("--cache", type=int, default=0)
    p.add_argument("--cache-rate", type=float, default=None)
    p.add_argument("--cache-policy", default="lru")
    p.add_argument("--hybrid-fetch", type=int, default=-1)
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--mem-ratio", type=float, default=0.9)
    p.add_argument("--disable-prefill-overlap", action="store_true")
    p.add_argument("--preload-all", action="store_true")
    p.add_argument("--prefill-hit-d2d", action="store_true")
    p.add_argument("--prefill-sparse-max-tokens", type=int, default=0)
    p.add_argument("--shared-expert-overlap", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--server-timeout", type=float, default=1800)
    p.set_defaults(collect_moe_stats=True, normal_eos=True, decode=64)
    args = p.parse_args()
    args.max_seq_len = args.context_tokens + args.max_output_tokens
    args.num_tokens = args.max_seq_len
    return args


def templated_token_count(tokenizer, content: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    return len(encoded.input_ids)


def exact_context_prompt(tokenizer, target: int, secret: str) -> str:
    prefix = f"Remember this access code exactly: {secret}.\n"
    suffix = "\nWhat is the access code? Reply with only the code."

    def candidate(n: int) -> str:
        return prefix + (" x" * n) + suffix

    lo, hi = 0, target
    while lo <= hi:
        mid = (lo + hi) // 2
        count = templated_token_count(tokenizer, candidate(mid))
        if count == target:
            return candidate(mid)
        if count < target:
            lo = mid + 1
        else:
            hi = mid - 1
    raise ValueError(f"could not construct exactly {target} templated tokens")


def post_json(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def counter_delta(before: dict, after: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def main() -> int:
    args = parse_args()
    output = Path(args.json_out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    from sparklab.catalog import get_recipe
    from transformers import AutoTokenizer

    recipe = get_recipe(args.recipe)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    secret = "SPARK-7319"

    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix="bench-context-recall-", suffix=".log")
    cmd = serve_cmd(args, args.backend, port)
    cmd += ["--max-prefill-length", str(args.max_extend_tokens)]
    started = time.monotonic()
    telemetry = GpuTelemetry()
    telemetry_running = False
    with os.fdopen(fd, "wb") as log_file:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_file), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            calibration_content = "Reply with OK."
            calibration_local = templated_token_count(tokenizer, calibration_content)
            calibration_response = post_json(
                f"{origin}/v1/chat/completions",
                {
                    "model": model_id,
                    "messages": [{"role": "user", "content": calibration_content}],
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=1800,
            )
            calibration_observed = int(
                (calibration_response.get("usage") or {}).get("prompt_tokens", 0)
            )
            template_offset = calibration_local - calibration_observed
            # The serving tokenizer may add or remove a small, stable set of
            # control tokens relative to the local Transformers template.  A
            # signed offset is valid; the measured request below remains the
            # authority for the exact-token gate.
            if calibration_observed <= 0 or abs(template_offset) > 256:
                raise ValueError(
                    "unexpected local/server chat-template offset: "
                    f"local={calibration_local}, observed={calibration_observed}"
                )
            prompt = exact_context_prompt(
                tokenizer, args.context_tokens + template_offset, secret
            )
            before = get_json(f"{origin}/v1/stats")
            telemetry.start()
            telemetry_running = True
            request_started = time.monotonic()
            response = post_json(
                f"{origin}/v1/chat/completions",
                {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": args.max_output_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=7200,
            )
            request_seconds = time.monotonic() - request_started
            gpu = telemetry.stop()
            telemetry_running = False
            after = get_json(f"{origin}/v1/stats")
        finally:
            if telemetry_running:
                telemetry.stop()
            stop_server(proc)
            pump.join(timeout=10)

    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    usage = response.get("usage") or {}
    log = Path(log_path).read_text(encoding="utf-8", errors="replace")
    before_disk = (before.get("moe") or {}).get("disk") or {}
    after_disk = (after.get("moe") or {}).get("disk") or {}
    observed_prompt = int(usage.get("prompt_tokens", 0))
    recall_ok = secret in content
    oom_count = len(re.findall(r"CUDA out of memory|torch\.OutOfMemoryError", log, re.I))
    vram_bytes = int(after.get("vram_bytes", 0))
    result = {
        "schema_version": "1.0",
        "kind": "context_recall",
        "recipe": {
            "slug": recipe.slug,
            "recipe_version": recipe.recipe_version,
            "revision": recipe.revision,
        },
        "engine": runtime_provenance(),
        "checkpoint": checkpoint_provenance(args.model),
        "platform": gb10_evidence(args.model),
        "validation": {
            "requested_context_tokens": args.context_tokens,
            "observed_prompt_tokens": observed_prompt,
            "exact_context": observed_prompt == args.context_tokens,
            "recall": recall_ok,
            "memory_bounded": (
                recipe.runtime_memory is not None
                and vram_bytes <= recipe.runtime_memory["total_bytes"]
            ),
            "oom_count": oom_count,
        },
        "response": {
            "content": content,
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
        },
        "metrics": {
            "request_seconds": request_seconds,
            "vram_bytes": vram_bytes,
            "physical_disk_bytes": counter_delta(
                before_disk, after_disk, "physical_bytes"
            ),
            "gpu_telemetry": gpu,
        },
        "configuration": {
            "max_seq_len": args.max_seq_len,
            "max_extend_tokens": args.max_extend_tokens,
            "preload_all": args.preload_all,
            "prefill_hit_d2d": args.prefill_hit_d2d,
            "prefill_sparse_max_tokens": args.prefill_sparse_max_tokens,
            "template_token_offset": template_offset,
            "calibration_local_tokens": calibration_local,
            "calibration_observed_tokens": calibration_observed,
        },
        "duration_seconds": time.monotonic() - started,
        "server_log": log_path,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)
    print(json.dumps(result, indent=2), flush=True)
    passed = all(
        result["validation"][key]
        for key in ("exact_context", "recall", "memory_bounded")
    ) and oom_count == 0
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
