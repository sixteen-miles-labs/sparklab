"""Small multi-image and tool-call integration checks."""

import argparse
import json
from pathlib import Path

import requests
from compare import image_url


def run(base):
    body = {
        "model": "bonsai2",
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    rows = []
    images = [
        {
            "type": "text",
            "text": "Name the square colors in the first and second images, in that order. Reply with only the two colors.",
        },
        {"type": "image_url", "image_url": {"url": image_url("FIRST", "red")}},
        {"type": "image_url", "image_url": {"url": image_url("SECOND", "blue")}},
    ]
    data = requests.post(
        base + "/v1/chat/completions",
        json={**body, "messages": [{"role": "user", "content": images}]},
        timeout=120,
    ).json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    rows.append(
        {
            "case": "two_images",
            "passed": "red" in content.lower()
            and "blue" in content.lower()
            and content.lower().index("red") < content.lower().index("blue"),
            "response": data,
        }
    )
    tool = {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": "Use the multiply tool to multiply 6 and 7. Do not calculate it yourself.",
        }
    ]
    data = requests.post(
        base + "/v1/chat/completions",
        json={**body, "messages": messages, "tools": [tool]},
        timeout=120,
    ).json()
    msg = data.get("choices", [{}])[0].get("message", {})
    calls = msg.get("tool_calls", [])
    passed = False
    if len(calls) == 1:
        call = calls[0]
        passed = call["function"]["name"] == "multiply" and json.loads(
            call["function"]["arguments"]
        ) == {"a": 6, "b": 7}
    rows.append({"case": "tool_call", "passed": passed, "response": data})
    if passed:
        messages += [
            msg,
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": "42"},
            {"role": "user", "content": "Reply with only the integer result."},
        ]
        data = requests.post(
            base + "/v1/chat/completions",
            json={**body, "messages": messages, "tools": [tool]},
            timeout=120,
        ).json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        )
        rows.append(
            {
                "case": "tool_roundtrip",
                "passed": content.strip() == "42",
                "response": data,
            }
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.url)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print([(r["case"], r["passed"]) for r in result])
