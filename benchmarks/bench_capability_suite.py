"""Auditable complete-checkpoint reasoning, tool-parser, and coding-agent probes.

The suite keeps one real server alive for all probes and writes one atomic JSON result.
It intentionally validates model behavior, not just parser unit fixtures:

* reasoning must arrive on the reasoning channel and a correct answer on content;
* a declared function must arrive as a structured OpenAI tool call with typed args;
* a generated Python function must pass an AST safety policy and executable tests.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from bench_decode_moe import (
    checkpoint_provenance,
    free_port,
    gb10_evidence,
    get_json,
    pump_output,
    runtime_provenance,
    serve_cmd,
    stop_server,
    stream_generate,
    wait_ready,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--json", dest="json_out", required=True)
    p.add_argument("--backend", default="offload", choices=("offload", "cpu", "hybrid"))
    p.add_argument("--storage", default="disk", choices=("ram", "disk"))
    p.add_argument("--attention-backend", default="auto")
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--cache-type", choices=("naive", "radix"), default="naive")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--nvfp4-backend", default="triton")
    p.add_argument("--host-cache-gb", type=float, default=2.0)
    p.add_argument("--cache", type=int, default=0)
    p.add_argument("--cache-rate", type=float, default=None)
    p.add_argument("--cache-policy", default="lru")
    p.add_argument("--hybrid-fetch", type=int, default=-1)
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--mem-ratio", type=float, default=0.9)
    p.add_argument("--num-tokens", type=int, default=4096)
    p.add_argument("--disable-prefill-overlap", action="store_true")
    p.add_argument("--prefill-hit-d2d", action="store_true")
    p.add_argument("--prefill-sparse-max-tokens", type=int, default=0)
    p.add_argument("--shared-expert-overlap", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--server-timeout", type=float, default=1800)
    p.set_defaults(collect_moe_stats=True, normal_eos=True, decode=512)
    return p.parse_args()


def post_json(url: str, body: dict, timeout: float = 1800) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
    return (blocks[-1] if blocks else text).strip()


def validate_clamp_program(source: str) -> tuple[bool, str]:
    """Execute only a tightly constrained, import-free ``clamp`` function."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "expected exactly one function definition"
    fn = tree.body[0]
    if fn.name != "clamp" or [arg.arg for arg in fn.args.args] != ["value", "lower", "upper"]:
        return False, "expected clamp(value, lower, upper)"
    allowed = (
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.If, ast.Return,
        ast.Compare, ast.Name, ast.Load, ast.Constant, ast.Lt, ast.LtE, ast.Gt,
        ast.GtE, ast.Eq,
    )
    unexpected = next((type(node).__name__ for node in ast.walk(tree) if not isinstance(node, allowed)), None)
    if unexpected:
        return False, f"program contains an unsupported AST node: {unexpected}"
    namespace: dict = {"__builtins__": {}}
    exec(compile(tree, "<model-clamp>", "exec"), namespace, namespace)
    clamp = namespace["clamp"]
    cases = [
        ((5, 0, 10), 5), ((-2, 0, 10), 0), ((12, 0, 10), 10),
        ((3.5, 3.5, 3.5), 3.5),
    ]
    try:
        for values, expected in cases:
            if clamp(*values) != expected:
                return False, f"failed case {values}"
    except Exception as exc:  # the AST is constrained; report functional failures
        return False, f"execution failed: {type(exc).__name__}: {exc}"
    return True, "all clamp cases passed"


def main() -> int:
    args = parse_args()
    output = Path(args.json_out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    from sparklab.catalog import get_recipe

    recipe = get_recipe(args.recipe)
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix="bench-capability-suite-", suffix=".log")
    cmd = serve_cmd(args, args.backend, port)
    started = time.monotonic()
    with os.fdopen(fd, "wb") as log_file:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_file), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]

            reasoning_args = SimpleNamespace(**vars(args))
            reasoning_args.decode = 384
            reasoning = stream_generate(
                origin,
                model_id,
                "Compute 37 * 43. Explain your reasoning, then give the visible final answer as FINAL=1591.",
                {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
                reasoning_args,
            )
            reasoning_ok = bool(
                reasoning["reasoning_text"]
                and "FINAL=1591" in reasoning["content_text"].replace(" ", "")
            )

            tool_body = {
                "model": model_id,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Use the multiply tool exactly once to multiply 17 by 23. "
                        "Do not calculate it yourself and do not answer with prose."
                    ),
                }],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "multiply",
                        "description": "Multiply two integers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                            "additionalProperties": False,
                        },
                    },
                }],
                "tool_choice": "auto",
                "temperature": 0.0,
                "max_tokens": 384,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            tool_response = post_json(f"{origin}/v1/chat/completions", tool_body)
            message = tool_response["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            tool_args = {}
            if len(calls) == 1:
                try:
                    tool_args = json.loads(calls[0]["function"]["arguments"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    pass
            tool_ok = bool(
                len(calls) == 1
                and calls[0].get("function", {}).get("name") == "multiply"
                and tool_args == {"a": 17, "b": 23}
            )

            coding_body = {
                "model": model_id,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Write a Python function clamp(value, lower, upper). Return value "
                        "when it is in range, lower when below, and upper when above. "
                        "Output exactly one Python code block containing only that function. "
                        "Use comparisons and returns only: no imports, calls, decorators, or prose."
                    ),
                }],
                "temperature": 0.0,
                "max_tokens": 384,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            coding_response = post_json(f"{origin}/v1/chat/completions", coding_body)
            coding_message = coding_response["choices"][0]["message"]
            code = extract_python(coding_message.get("content") or "")
            coding_ok, coding_detail = validate_clamp_program(code)
            stats = get_json(f"{origin}/v1/stats")
        finally:
            stop_server(proc)
            pump.join(timeout=10)

    result = {
        "schema_version": "1.0",
        "kind": "capability_suite",
        "recipe": {
            "slug": recipe.slug,
            "recipe_version": recipe.recipe_version,
            "revision": recipe.revision,
        },
        "engine": runtime_provenance(),
        "checkpoint": checkpoint_provenance(args.model),
        "platform": gb10_evidence(args.model),
        "configuration": {
            "backend": args.backend,
            "storage": args.storage,
            "attention_backend": args.attention_backend,
            "prefill_sparse_max_tokens": args.prefill_sparse_max_tokens,
            "prefill_hit_d2d": args.prefill_hit_d2d,
        },
        "validation": {
            "reasoning_parser": reasoning_ok,
            "tool_parser": tool_ok,
            "coding_agent_task": coding_ok,
        },
        "reasoning": {
            "reasoning_characters": len(reasoning["reasoning_text"]),
            "content": reasoning["content_text"],
            "finish_reason": reasoning["finish_reason"],
        },
        "tool": {
            "calls": calls,
            "finish_reason": tool_response["choices"][0].get("finish_reason"),
        },
        "coding": {
            "source": code,
            "detail": coding_detail,
            "finish_reason": coding_response["choices"][0].get("finish_reason"),
        },
        "memory": {
            "vram_bytes": stats.get("vram_bytes"),
            "bounded_by_recipe": (
                isinstance(stats.get("vram_bytes"), int)
                and recipe.runtime_memory is not None
                and stats["vram_bytes"] <= recipe.runtime_memory["total_bytes"]
            ),
        },
        "duration_seconds": time.monotonic() - started,
        "server_log": log_path,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if all(result["validation"].values()) and result["memory"]["bounded_by_recipe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
