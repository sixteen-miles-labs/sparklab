"""Figure-3-style AIME-25 serving evaluation for one long-lived FreeToken server.

Unlike ``bench_decode_moe.py`` (a deterministic fixed-length regression probe), this
runner evaluates every selected problem once with checkpoint-recommended sampling,
thinking enabled, and normal EOS. It appends each request immediately to JSONL so a long
run remains auditable/recoverable, then writes a suite summary with accuracy and latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import tempfile
import threading
from pathlib import Path

from bench_decode_moe import (
    AIME_FILE,
    AIME_REPO,
    BOXED_INSTRUCTION,
    GpuTelemetry,
    free_port,
    get_json,
    pump_output,
    resolve_sampling,
    runtime_provenance,
    serve_cmd,
    stop_server,
    stream_generate,
    wait_ready,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--aime", default=os.environ.get("FREETOKEN_AIME25_JSONL"))
    p.add_argument("--problems", default="all", help="all, comma list, or inclusive range (e.g. 0-4)")
    p.add_argument("--decode", type=int, default=1024, help="maximum output tokens per problem")
    p.add_argument("--backend", default="hybrid", choices=("offload", "cpu", "hybrid"))
    p.add_argument("--storage", default="disk", choices=("ram", "disk"))
    p.add_argument(
        "--nvfp4-backend",
        choices=("auto", "marlin", "flashinfer", "triton"),
        default="triton",
    )
    p.add_argument("--host-cache-gb", type=float, default=40.0)
    p.add_argument("--cache", type=int, default=0)
    p.add_argument("--cache-rate", type=float, default=None)
    p.add_argument("--cache-policy", choices=("lru", "layer_lru"), default="lru")
    p.add_argument("--hybrid-fetch", type=int, default=3)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--mem-ratio", type=float, default=0.9)
    p.add_argument("--num-tokens", type=int, default=2048)
    p.add_argument("--disable-prefill-overlap", action="store_true")
    p.add_argument("--prefill-hit-d2d", action="store_true")
    p.add_argument("--prefill-sparse-max-tokens", type=int, default=0)
    p.add_argument("--shared-expert-overlap", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--server-timeout", type=float, default=1800)
    p.add_argument("--json", dest="json_out", required=True)
    p.add_argument("--include-output", action="store_true")
    p.set_defaults(collect_moe_stats=True, normal_eos=True)
    return p.parse_args()


def load_rows(path: str | None) -> list[dict]:
    if not path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(AIME_REPO, AIME_FILE, repo_type="dataset")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        text = row.get("problem") or row["prompt"]
        if "boxed" not in text:
            text = f"{text}\n{BOXED_INSTRUCTION}"
        row["_eval_prompt"] = text
    return rows


def select_problems(value: str, count: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(count))
    selected = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))
    if not selected or any(i < 0 or i >= count for i in selected):
        raise ValueError(f"--problems {value!r} outside 0..{count - 1}")
    return list(dict.fromkeys(selected))


def extract_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    matches = re.findall(r"(?:final answer|answer is|final result)\D{0,40}(-?\d+)", text, re.I)
    if matches:
        return matches[-1].strip()
    return None


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q + 0.5))]


def counter_delta(before: dict, after: dict, keys: tuple[str, ...]) -> dict:
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a") as output:
        output.write(json.dumps(row) + "\n")
        output.flush()
        os.fsync(output.fileno())


def existing_results(path: str) -> list[dict]:
    output = Path(path)
    if not output.is_file():
        return []
    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    if any(row.get("kind") == "summary" for row in rows):
        raise ValueError(f"{path} already contains a completed suite summary")
    requests = [row for row in rows if row.get("kind") == "request"]
    problems = [row["problem"] for row in requests]
    if len(problems) != len(set(problems)):
        raise ValueError(f"{path} contains duplicate problem rows")
    return requests


def measure_request(args, origin: str, model_id: str, index: int, row: dict, sampling: dict) -> dict:
    before = get_json(f"{origin}/v1/stats")
    telemetry = GpuTelemetry()
    telemetry.start()
    try:
        result = stream_generate(origin, model_id, row["_eval_prompt"], sampling, args)
    finally:
        gpu_telemetry = telemetry.stop()
    after = get_json(f"{origin}/v1/stats")
    stamps = result["stamps"]
    usage = result["usage"]
    completion = int(usage["completion_tokens"])
    steps = max(0, completion - 1)
    decode_seconds = stamps[-1] - stamps[0] if len(stamps) >= 2 else 0.0
    gaps = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
    answer = extract_answer(result["text"])
    expected = str(row.get("answer", "")).strip()
    measured = {
        "kind": "request",
        "problem": index,
        "expected": expected,
        "answer": answer,
        "correct": answer == expected,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": completion,
        "decode_steps": steps,
        "decode_seconds": decode_seconds,
        "decode_tok_s": steps / decode_seconds if decode_seconds else 0.0,
        "ttft_ms": (stamps[0] - result["t0"]) * 1e3 if stamps else 0.0,
        "inter_token_ms_p50": percentile(gaps, 0.50),
        "inter_token_ms_p95": percentile(gaps, 0.95),
        "output_sha1": hashlib.sha1(result["text"].encode()).hexdigest()[:12],
        "finish_reason": result["finish_reason"],
        "ended_by_eos": result["finish_reason"] == "stop",
        "vram_gib": after.get("vram_bytes", 0) / 2**30,
        "gpu_telemetry": gpu_telemetry,
    }
    if args.include_output:
        measured["output_text"] = result["text"]

    before_moe, after_moe = before.get("moe") or {}, after.get("moe") or {}
    if after_moe:
        measured["moe"] = counter_delta(
            before_moe,
            after_moe,
            (
                "active", "missing", "fetched", "calls", "sparse_prefill_layers",
                "sparse_prefill_routes", "sparse_prefill_unique_rows",
                "sparse_prefill_fallback_layers", "shared_expert_overlap_calls",
            ),
        )
        before_disk, after_disk = before_moe.get("disk") or {}, after_moe.get("disk") or {}
        if after_disk:
            measured["moe"]["disk"] = counter_delta(
                before_disk,
                after_disk,
                ("cache_hits", "cache_misses", "cache_evictions", "cache_bypasses",
                 "read_ops", "logical_bytes", "physical_bytes", "read_seconds"),
            )
            for key in (
                "cache_capacity_entries", "cache_capacity_bytes", "cache_allocated_bytes",
                "cache_occupancy_entries", "cache_occupancy_bytes",
                "staging_allocated_bytes", "host_allocated_bytes", "read_workers",
                "cache_policy", "staging_buffers",
            ):
                measured["moe"]["disk"][key] = after_disk.get(key, 0)
    return measured


def summarize(args, results: list[dict], sampling: dict) -> dict:
    tps = [row["decode_tok_s"] for row in results if row["decode_steps"]]
    ttft = [row["ttft_ms"] for row in results]
    total_steps = sum(row["decode_steps"] for row in results)
    total_decode = sum(row["decode_seconds"] for row in results)
    disk_keys = (
        "cache_hits", "cache_misses", "cache_evictions", "cache_bypasses",
        "read_ops", "logical_bytes", "physical_bytes", "read_seconds",
    )
    disk = {
        key: sum(row.get("moe", {}).get("disk", {}).get(key, 0) for row in results)
        for key in disk_keys
    }
    requests = disk["cache_hits"] + disk["cache_misses"]
    gpu_rows = [row["gpu_telemetry"] for row in results if row.get("gpu_telemetry")]
    summary = {
        "kind": "summary",
        "model": args.model,
        "backend": args.backend,
        "problems": len(results),
        "correct": sum(row["correct"] for row in results),
        "accuracy": statistics.mean(row["correct"] for row in results),
        "sampling": sampling,
        "max_output_tokens": args.decode,
        "ended_by_eos": sum(row["ended_by_eos"] for row in results),
        "hit_token_cap": sum(not row["ended_by_eos"] for row in results),
        "mean_decode_tok_s": statistics.mean(tps),
        "token_weighted_decode_tok_s": total_steps / total_decode if total_decode else 0.0,
        "mean_ttft_ms": statistics.mean(ttft),
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "mean_inter_token_ms_p50": statistics.mean(row["inter_token_ms_p50"] for row in results),
        "mean_inter_token_ms_p95": statistics.mean(row["inter_token_ms_p95"] for row in results),
        "first_request": {key: results[0][key] for key in ("decode_tok_s", "ttft_ms")},
        "steady_requests": {
            "mean_decode_tok_s": statistics.mean(row["decode_tok_s"] for row in results[1:])
            if len(results) > 1 else 0.0,
            "mean_ttft_ms": statistics.mean(row["ttft_ms"] for row in results[1:])
            if len(results) > 1 else 0.0,
        },
        "disk": {
            **disk,
            "physical_gib": disk["physical_bytes"] / 2**30,
            "effective_gib_s": (
                disk["physical_bytes"] / 2**30 / disk["read_seconds"]
                if disk["read_seconds"] else 0.0
            ),
            "host_cache_hit_rate": disk["cache_hits"] / requests if requests else 0.0,
        },
        "gpu_telemetry": {
            "power_w_avg": statistics.mean(row["power_w_avg"] for row in gpu_rows),
            "power_w_peak": max(row["power_w_peak"] for row in gpu_rows),
            "temperature_c_peak": max(row["temperature_c_peak"] for row in gpu_rows),
            "request_energy_j_total_est": sum(
                row["request_energy_j_est"] for row in gpu_rows
            ),
        } if gpu_rows else {},
        "config": {
            "host_cache_gb": args.host_cache_gb,
            "nvfp4_backend": args.nvfp4_backend,
            "hybrid_fetch": args.hybrid_fetch,
            "cpu_threads": args.cpu_threads,
            "disk_read_workers": int(os.getenv("FREETOKEN_DISK_READ_WORKERS", "16")),
            "cache_policy": args.cache_policy,
            "prefill_overlap": not args.disable_prefill_overlap,
            "prefill_hit_d2d": args.prefill_hit_d2d,
            "prefill_sparse_max_tokens": args.prefill_sparse_max_tokens,
            "shared_expert_overlap": args.shared_expert_overlap,
            "memory_ratio": args.mem_ratio,
            "cuda_graph": not args.no_graph,
        },
        "provenance": runtime_provenance(),
    }
    return summary


def main() -> int:
    args = parse_args()
    rows = load_rows(args.aime)
    selected = select_problems(args.problems, len(rows))
    results = existing_results(args.json_out)
    completed = {row["problem"] for row in results}
    remaining = [index for index in selected if index not in completed]
    if completed - set(selected):
        raise ValueError(
            f"{args.json_out} contains problems outside current selection: "
            f"{sorted(completed - set(selected))}"
        )
    sampling, sampling_source = resolve_sampling(args.model, args.greedy)
    if not remaining:
        summary = summarize(args, results, sampling)
        append_jsonl(args.json_out, summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix="bench-aime-suite-", suffix=".log")
    cmd = serve_cmd(args, args.backend, port)
    print(
        f"[suite] {len(selected)} AIME25 problems ({len(remaining)} remaining), "
        f"max_tokens={args.decode}, "
        f"sampling={sampling} <- {sampling_source}\n[suite] server log: {log_path}",
        flush=True,
    )
    with os.fdopen(fd, "wb") as log_file:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_file), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            for position, index in enumerate(remaining, len(completed) + 1):
                measured = measure_request(args, origin, model_id, index, rows[index], sampling)
                append_jsonl(args.json_out, measured)
                results.append(measured)
                print(
                    f"[suite] {position}/{len(selected)} problem={index} "
                    f"answer={measured['answer']!r}/{measured['expected']!r} "
                    f"tokens={measured['completion_tokens']} tps={measured['decode_tok_s']:.3f} "
                    f"ttft={measured['ttft_ms'] / 1000:.1f}s",
                    flush=True,
                )
        finally:
            stop_server(proc)
            pump.join(timeout=10)
    summary = summarize(args, results, sampling)
    summary["server_log"] = log_path
    append_jsonl(args.json_out, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
