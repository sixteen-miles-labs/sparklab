#!/usr/bin/env python3
"""Run the W1-W4 quality workloads described by arXiv:2608.16157v1.

W1 is a native AIME client. W2-W4 execute an explicit JSON scenario so private or
licensed task artifacts do not leak into this repository. The runner observes the
SparkLab request ring and emits the paper's per-request decode-rate and TTFT metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PAPER = "arXiv:2608.16157v1"
AIME_REPO = "math-ai/aime25"
AIME_FILE = "test.jsonl"
BOXED_INSTRUCTION = "Put your final answer in \\boxed{...}."
CLOUD_CREDENTIALS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN", "OPENAI_API_KEY",
    "GEMINI_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "XAI_API_KEY",
    "OPENROUTER_API_KEY",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, choices=("W1", "W2", "W3", "W4"))
    parser.add_argument("--model", required=True, help="model registry key or served model id")
    parser.add_argument("--weight-format", required=True, help="for example NVFP4, BF16, or MXFP4")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--request-timeout", type=float, default=1800)

    w1 = parser.add_argument_group("W1 AIME options")
    w1.add_argument("--aime", type=Path, help="local AIME JSONL; otherwise download math-ai/aime25")
    w1.add_argument("--problems", default=None, help="all, comma list, or inclusive range such as 0-4")
    w1.add_argument("--max-tokens", type=int, default=32768)
    w1.add_argument("--temperature", type=float)
    w1.add_argument("--top-p", type=float)
    w1.add_argument("--seed", type=int)
    w1.add_argument("--include-output", action="store_true")

    agent = parser.add_argument_group("W2-W4 agent options")
    agent.add_argument("--scenario", type=Path, help="scenario JSON containing argv-only steps")
    agent.add_argument("--step-timeout", type=float, default=3600)
    agent.add_argument("--allow-partial-scenario", action="store_true",
                       help="smoke only: allow fewer turns than the paper protocol")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.workload == "W1" and args.scenario:
        parser.error("--scenario applies only to W2-W4")
    if args.workload != "W1" and not args.scenario:
        parser.error(f"{args.workload} requires --scenario")
    if args.allow_partial_scenario and args.mode != "smoke":
        parser.error("--allow-partial-scenario is valid only with --mode smoke")
    return args


def get_json(url: str, timeout: float = 10) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {url}")
    return value


def post_stream(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.monotonic()
    first_token_at: float | None = None
    last_token_at: float | None = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            visible = delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            if (visible or thought) and first_token_at is None:
                first_token_at = time.monotonic()
            if visible or thought:
                last_token_at = time.monotonic()
            if visible:
                content.append(visible)
            if thought:
                reasoning.append(thought)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    finished = time.monotonic()
    if first_token_at is None:
        first_token_at = finished
    if last_token_at is None:
        last_token_at = first_token_at
    completion_tokens = int(usage.get("completion_tokens") or 0)
    decode_seconds = max(0.0, last_token_at - first_token_at)
    decode_steps = max(0, completion_tokens - 1)
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish_reason,
        "ttft_seconds": first_token_at - started,
        "duration_seconds": finished - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_steps / decode_seconds if decode_seconds else 0.0,
    }


def load_models() -> dict[str, dict[str, Any]]:
    return json.loads((ROOT / "models.json").read_text())


def resolve_model(value: str) -> tuple[str, dict[str, Any] | None]:
    entry = load_models().get(value)
    return (str(entry["served_model"]), entry) if entry else (value, None)


def server_model(base_url: str) -> str:
    payload = get_json(f"{base_url.rstrip('/')}/v1/models")
    models = payload.get("data") or []
    if not models or not isinstance(models[0], dict) or not models[0].get("id"):
        raise RuntimeError("server reports no model")
    return str(models[0]["id"])


def load_aime(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("install huggingface_hub or pass --aime") from exc
        path = Path(hf_hub_download(AIME_REPO, AIME_FILE, repo_type="dataset"))
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no AIME rows in {path}")
    return rows


def select_indices(selection: str, count: int) -> list[int]:
    if selection.lower() == "all":
        return list(range(count))
    selected: list[int] = []
    for item in selection.split(","):
        item = item.strip()
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(item))
    selected = list(dict.fromkeys(selected))
    if not selected or any(index < 0 or index >= count for index in selected):
        raise ValueError(f"problem selection must be within 0..{count - 1}")
    return selected


def extract_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    fallback = re.findall(r"(?:final answer|answer is|final result)\D{0,40}(-?\d+)", text, re.I)
    return fallback[-1].strip() if fallback else None


def request_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates: list[float] = []
    ttfts: list[float] = []
    for row in rows:
        completion = row.get("completion_tokens")
        duration_ms = row.get("duration_ms")
        ttft_ms = row.get("ttft_ms")
        if isinstance(ttft_ms, (int, float)):
            ttfts.append(float(ttft_ms) / 1000)
        if isinstance(completion, int) and isinstance(duration_ms, (int, float)):
            decode_seconds = max(0.0, (float(duration_ms) - float(ttft_ms or 0)) / 1000)
            if decode_seconds:
                rates.append(max(0, completion - 1) / decode_seconds)
    return {
        "request_count": len(rows),
        "mean_decode_tokens_per_second": statistics.mean(rates) if rates else 0.0,
        "mean_ttft_seconds": statistics.mean(ttfts) if ttfts else 0.0,
        "max_ttft_seconds": max(ttfts) if ttfts else 0.0,
    }


def provenance(base_url: str, registry_entry: dict[str, Any] | None) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    health: dict[str, Any]
    try:
        health = get_json(f"{base_url.rstrip('/')}/health")
    except (OSError, ValueError, urllib.error.URLError):
        health = {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision or None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "server": health,
        "model_registry": registry_entry,
    }


def base_result(args: argparse.Namespace, served_model: str, entry: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "paper": PAPER,
        "workload": args.workload,
        "model": args.model,
        "served_model": served_model,
        "weight_format": args.weight_format.upper(),
        "mode": args.mode,
        "base_url": args.base_url.rstrip("/"),
        "provenance": provenance(args.base_url, entry),
    }


def run_w1(args: argparse.Namespace, served_model: str, entry: dict[str, Any] | None) -> dict[str, Any]:
    rows = load_aime(args.aime)
    selection = args.problems or ("0" if args.mode == "smoke" else "all")
    indices = select_indices(selection, len(rows))
    requests: list[dict[str, Any]] = []
    for index in indices:
        row = rows[index]
        prompt = str(row.get("problem") or row.get("prompt") or "")
        if "boxed" not in prompt.lower():
            prompt = f"{prompt}\n\n{BOXED_INSTRUCTION}"
        body: dict[str, Any] = {
            "model": served_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        for key in ("temperature", "top_p", "seed"):
            value = getattr(args, key)
            if value is not None:
                body[key] = value
        measured = post_stream(
            f"{args.base_url.rstrip('/')}/v1/chat/completions", body, args.request_timeout
        )
        final_text = measured["content"] if measured["reasoning"] else (
            measured["content"] + measured["reasoning"]
        )
        expected = str(row.get("answer", "")).strip()
        answer = extract_answer(final_text)
        result = {
            "problem": index,
            "expected": expected,
            "answer": answer,
            "correct": answer == expected,
            "prompt_tokens": int(measured["usage"].get("prompt_tokens") or 0),
            "completion_tokens": int(measured["usage"].get("completion_tokens") or 0),
            "ttft_seconds": measured["ttft_seconds"],
            "duration_seconds": measured["duration_seconds"],
            "decode_seconds": measured["decode_seconds"],
            "decode_tokens_per_second": measured["decode_tokens_per_second"],
            "finish_reason": measured["finish_reason"],
            "output_sha256": hashlib.sha256(final_text.encode()).hexdigest(),
        }
        if args.include_output:
            result["reasoning_output"] = measured["reasoning"]
            result["output"] = measured["content"]
        requests.append(result)
        print(
            f"W1 {len(requests)}/{len(indices)} problem={index} "
            f"answer={answer!r}/{expected!r} tok/s={result['decode_tokens_per_second']:.2f}",
            flush=True,
        )
    rates = [row["decode_tokens_per_second"] for row in requests]
    ttfts = [row["ttft_seconds"] for row in requests]
    result = base_result(args, served_model, entry)
    result.update({
        "quality": {
            "passed": all(row["correct"] for row in requests),
            "correct": sum(row["correct"] for row in requests),
            "total": len(requests),
            "accuracy": statistics.mean(row["correct"] for row in requests),
            "gate": "final extracted answer equals the AIME reference answer",
        },
        "metrics": {
            "request_count": len(requests),
            "mean_decode_tokens_per_second": statistics.mean(rates),
            "mean_ttft_seconds": statistics.mean(ttfts),
            "max_ttft_seconds": max(ttfts),
        },
        "requests": requests,
        "configuration": {
            "dataset": f"{AIME_REPO}/{AIME_FILE}",
            "problems": selection,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
    })
    return result


def render(value: str, variables: dict[str, str]) -> str:
    try:
        return value.format_map(variables)
    except KeyError as exc:
        raise ValueError(f"unknown scenario template variable {exc.args[0]!r}") from exc


def validate_scenario(workload: str, scenario: dict[str, Any], allow_partial: bool) -> None:
    workloads = json.loads((ROOT / "workloads.json").read_text())
    expected = int(workloads[workload]["turns"])
    steps = scenario.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("scenario steps must be a non-empty array")
    if len(steps) != expected and not allow_partial:
        raise ValueError(f"{workload} requires exactly {expected} scripted turns, got {len(steps)}")
    if workload in {"W2", "W3"} and not isinstance(scenario.get("evaluator"), dict):
        raise ValueError(f"{workload} requires an evaluator for the produced patch")
    for number, step in enumerate(steps, 1):
        if not isinstance(step, dict) or not isinstance(step.get("argv"), list) or not step["argv"]:
            raise ValueError(f"step {number} requires a non-empty argv array")
        if not all(isinstance(arg, str) for arg in step["argv"]):
            raise ValueError(f"step {number} argv values must be strings")


def run_command(
    spec: dict[str, Any], variables: dict[str, str], env: dict[str, str], cwd: Path,
    timeout: float, include_output: bool,
) -> dict[str, Any]:
    argv = [render(item, variables) for item in spec["argv"]]
    command_env = env.copy()
    for key, value in (spec.get("env") or {}).items():
        command_env[str(key)] = render(str(value), variables)
    started = time.monotonic()
    try:
        process = subprocess.run(
            argv,
            cwd=cwd,
            env=command_env,
            input=render(str(spec.get("stdin", "")), variables) or None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(spec.get("timeout", timeout)),
            check=False,
        )
        result = {
            "executable": argv[0],
            "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
            "exit_code": process.returncode,
            "duration_seconds": time.monotonic() - started,
            "output_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
        }
        if include_output:
            result["argv"] = argv
            result["output_tail"] = process.stdout[-4000:]
        return result
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        result = {
            "executable": argv[0],
            "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
            "exit_code": None,
            "timed_out": True,
            "duration_seconds": time.monotonic() - started,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        }
        if include_output:
            result["argv"] = argv
            result["output_tail"] = output[-4000:]
        return result


def request_cursor(base_url: str) -> int:
    # An empty match returns the ring's all-time next cursor. Starting from zero
    # with limit=1 would return 1 whenever the ring already contained requests.
    return int(get_json(
        f"{base_url.rstrip('/')}/v1/requests?since=2147483647&limit=1"
    ).get("next_cursor", 0))


def requests_since(base_url: str, cursor: int) -> tuple[list[dict[str, Any]], int, bool]:
    start_cursor = cursor
    rows: list[dict[str, Any]] = []
    while True:
        payload = get_json(f"{base_url.rstrip('/')}/v1/requests?since={cursor}&limit=512")
        batch = payload.get("entries") or []
        if not isinstance(batch, list):
            raise ValueError("invalid /v1/requests response")
        rows.extend(row for row in batch if isinstance(row, dict))
        next_cursor = int(payload.get("next_cursor", cursor))
        if next_cursor == cursor or len(batch) < 512:
            break
        cursor = next_cursor
    return rows, next_cursor, next_cursor - start_cursor == len(rows)


def run_agent(args: argparse.Namespace, served_model: str, entry: dict[str, Any] | None) -> dict[str, Any]:
    scenario = json.loads(args.scenario.read_text())
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be a JSON object")
    validate_scenario(args.workload, scenario, args.allow_partial_scenario)
    workspace = Path(str(scenario.get("workspace", "."))).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"scenario workspace is not a directory: {workspace}")
    variables = {
        "base_url": args.base_url.rstrip("/"),
        "openai_base_url": f"{args.base_url.rstrip('/')}/v1",
        "model": served_model,
        "workspace": str(workspace),
        "prompt": "",
        "step": "",
    }
    env = os.environ.copy()
    for key in CLOUD_CREDENTIALS:
        env.pop(key, None)
    env.update({
        "SPARKLAB_BASE_URL": variables["base_url"],
        "OPENAI_BASE_URL": variables["openai_base_url"],
        "OPENAI_API_KEY": "sparklab-local",
        "ANTHROPIC_BASE_URL": variables["base_url"],
        "ANTHROPIC_API_KEY": "sparklab-local",
        "ANTHROPIC_AUTH_TOKEN": "sparklab-local",
        "ANTHROPIC_MODEL": served_model,
        "SPARKLAB_MODEL": served_model,
    })
    cursor = request_cursor(args.base_url)
    requests: list[dict[str, Any]] = []
    request_capture_complete = True
    steps: list[dict[str, Any]] = []
    for index, spec in enumerate(scenario["steps"], 1):
        variables["step"] = str(index)
        variables["prompt"] = str(spec.get("prompt", ""))
        measured = run_command(
            spec, variables, env, workspace, args.step_timeout, args.include_output
        )
        measured["step"] = index
        steps.append(measured)
        new_requests, cursor, complete = requests_since(args.base_url, cursor)
        requests.extend(new_requests)
        request_capture_complete = request_capture_complete and complete
        print(f"{args.workload} turn {index}/{len(scenario['steps'])} exit={measured['exit_code']}", flush=True)
        if measured["exit_code"] != int(spec.get("expected_exit_code", 0)):
            break
    evaluator_result = None
    if len(steps) == len(scenario["steps"]) and all(
        row["exit_code"] == int(spec.get("expected_exit_code", 0))
        for row, spec in zip(steps, scenario["steps"])
    ) and isinstance(scenario.get("evaluator"), dict):
        evaluator = scenario["evaluator"]
        evaluator_result = run_command(
            evaluator, variables, env, workspace, args.step_timeout, args.include_output
        )
        evaluator_result["expected_exit_code"] = int(evaluator.get("expected_exit_code", 0))
    workloads = json.loads((ROOT / "workloads.json").read_text())
    paper_turns = int(workloads[args.workload]["turns"])
    steps_ok = len(steps) == len(scenario["steps"]) and all(
        row["exit_code"] == int(spec.get("expected_exit_code", 0))
        for row, spec in zip(steps, scenario["steps"])
    )
    evaluator_ok = evaluator_result is None or (
        evaluator_result["exit_code"] == evaluator_result["expected_exit_code"]
    )
    result = base_result(args, served_model, entry)
    result.update({
        "scenario": {
            "file": args.scenario.name,
            "sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
            "name": scenario.get("name"),
            "metadata": scenario.get("metadata") or {},
        },
        "quality": {
            "passed": steps_ok and evaluator_ok,
            "turns_completed": len(steps),
            "scenario_turns": len(scenario["steps"]),
            "paper_turns_required": paper_turns,
            "evaluator_passed": evaluator_ok if evaluator_result is not None else None,
        },
        "metrics": {**request_metrics(requests), "request_capture_complete": request_capture_complete},
        "requests": requests,
        "steps": steps,
        "evaluator": evaluator_result,
    })
    return result


def main() -> int:
    args = parse_args()
    served_model, entry = resolve_model(args.model)
    actual_model = server_model(args.base_url)
    if served_model != actual_model:
        raise RuntimeError(
            f"requested model {served_model!r}, but server reports {actual_model!r}; "
            "use the server's model id or restart the server"
        )
    if entry and args.weight_format.upper() not in {
        str(value).upper() for value in entry["supported_weight_formats"]
    }:
        raise ValueError(
            f"{args.model} registry does not list weight format {args.weight_format!r}"
        )
    result = run_w1(args, served_model, entry) if args.workload == "W1" else run_agent(
        args, served_model, entry
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "quality_passed": result["quality"]["passed"],
        "metrics": result["metrics"],
    }, indent=2))
    return 0 if result["quality"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
