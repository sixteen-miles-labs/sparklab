#!/usr/bin/env python3
"""Probe Inkling's chat API; retain responses and exact-usage streaming timings."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import statistics
import time
import urllib.request

MODEL = "thinkingmachines/Inkling-2-Mini"


def request(base_url, body):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({"model": MODEL, **body}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=600)


def chat(base_url, messages, **kwargs):
    with request(base_url, {
        "messages": messages, "temperature": 0, "max_tokens": 512,
        "chat_template_kwargs": {"reasoning_effort": "none"}, **kwargs,
    }) as response:
        return json.load(response)


def stream(base_url, max_tokens):
    started = time.perf_counter()
    times, content, reasoning = [], [], []
    usage, finish, done = None, None, False
    with request(base_url, {
        "messages": [{"role": "user", "content": (
            "Explain how a hash table works, including collisions, resizing, and its time "
            "complexity. Use a concrete example and give a detailed, self-contained explanation."
        )}],
        "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "none"},
        "stream": True, "stream_options": {"include_usage": True},
    }) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            if line[5:].strip() == "[DONE]":
                done = True
                break
            event = json.loads(line[5:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if text or thought:
                    times.append(time.perf_counter() - started)
                    content.append(text)
                    reasoning.append(thought)
    elapsed = time.perf_counter() - started
    if not done or not times or not usage or usage.get("completion_tokens", 0) < 2:
        raise RuntimeError("Incomplete stream or missing exact token usage")
    return {
        "ttft_seconds": times[0], "request_seconds": elapsed, "usage": usage,
        "decode_tokens_per_second": (usage["completion_tokens"] - 1) / (times[-1] - times[0]),
        "completion_tokens_per_second": usage["completion_tokens"] / elapsed,
        "content": "".join(content), "reasoning": "".join(reasoning),
        "finish_reason": finish,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.trials < 1 or args.max_tokens < 2:
        parser.error("--trials must be positive and --max-tokens must be at least 2")
    result = {
        "model": MODEL, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "checks": {}, "trials": [],
        "provenance": json.loads(args.provenance.read_text()) if args.provenance else {},
        "method": {"concurrency": 1, "temperature": 0, "reasoning_effort": "none",
                   "max_tokens": args.max_tokens, "warmup_requests": 1,
                   "decode_metric": "(exact completion_tokens - 1) / (last nonempty delta - first nonempty delta)",
                   "scope": "Smoke checks only; timing on a shared GPU is not an isolated benchmark."},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def check(name, messages, validate, **kwargs):
        try:
            response = chat(args.base_url, messages, **kwargs)
            passed = bool(validate(response["choices"][0]["message"]))
            result["checks"][name] = {"passed": passed, "response": response}
        except Exception as exc:
            result["checks"][name] = {"passed": False, "error": str(exc)}
        save()

    save()
    try:
        check("text", [{"role": "user", "content": "What is 17 * 19? Reply with only the integer."}],
              lambda m: m.get("content", "").strip() == "323")
        check("json", [{"role": "user", "content": 'Return JSON with exactly the keys "answer" (323) and "ok" (true).'}],
              lambda m: json.loads(m["content"]) == {"answer": 323, "ok": True},
              response_format={"type": "json_object"})
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }}]

        def valid_tool(message):
            calls = message.get("tool_calls") or []
            return any(c["function"]["name"] == "get_weather" and
                       json.loads(c["function"]["arguments"]).get("city", "").lower() == "toronto" for c in calls)

        tool_prompt = [{"role": "user", "content": "Use get_weather to check the current weather in Toronto."}]
        check("tool", tool_prompt,
              valid_tool, tools=tools, tool_choice="auto")
        if result["checks"]["tool"]["passed"]:
            message = result["checks"]["tool"]["response"]["choices"][0]["message"]
            calls = message["tool_calls"]
            # A deterministic test fixture, not a live weather lookup.
            tool_messages = [{"role": "tool", "tool_call_id": c["id"],
                              "content": '{"temperature_c":17,"conditions":"sunny"}'} for c in calls]
            check("tool_roundtrip", [*tool_prompt, message, *tool_messages],
                  lambda m: "17" in (m.get("content") or ""), tools=tools, tool_choice="none")
        from PIL import Image, ImageDraw

        picture = Image.new("RGB", (384, 384), "white")
        ImageDraw.Draw(picture).rectangle((64, 64, 320, 320), fill="red")
        data = io.BytesIO()
        picture.save(data, format="PNG")
        check("image", [{"role": "user", "content": [
            {"type": "text", "text": "What color is the large square? Reply with one word."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()}},
        ]}], lambda m: "red" in m.get("content", "").lower())
        result["warmup"] = stream(args.base_url, 32)
        save()
        for _ in range(args.trials):
            result["trials"].append(stream(args.base_url, args.max_tokens))
            save()
        result["summary"] = {
            name + "_median": statistics.median(t[name] for t in result["trials"])
            for name in ("ttft_seconds", "decode_tokens_per_second", "completion_tokens_per_second")
        }
        result["status"] = "passed" if all(c["passed"] for c in result["checks"].values()) else "failed_checks"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
    save()
    print(json.dumps({"status": result["status"], "checks": {k: v["passed"] for k, v in result["checks"].items()},
                      "summary": result.get("summary")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
