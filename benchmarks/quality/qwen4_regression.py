"""Focused Qwen4 regression checks, not broad quality certification.

Runs against an existing local OpenAI-compatible server. MGSM uses a fixed,
SHA-pinned 100-case subset; provide the official EN/ZH TSVs with --data-dir.
"""

import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path


def cases():
    result = []
    for i in range(24):
        a, b = 23 + 7 * i, 11 + 3 * i
        expected = a * b
        wording = f"Compute {a} * {b}." if i % 2 == 0 else f"计算 {a} 乘以 {b}。"
        result.append(
            (
                f"multiply-{i}",
                wording + " Reply only with FINAL=<integer>.",
                str(expected),
                "final",
            )
        )
    for i in range(12):
        a, b, c = 107 + 19 * i, 7 + i, 3 + 2 * i
        result.append(
            (
                f"remainder-{i}",
                f"What is the remainder when ({a} * {b} + {c}) is divided by 17? Reply only with FINAL=<integer>.",
                str((a * b + c) % 17),
                "final",
            )
        )
    for i in range(8):
        key = f"violet-{7319 + 137 * i}"
        filler = "The record contains ordinary inventory notes. " * 120
        result.append(
            (
                f"recall-{i}",
                f"{filler}\nThe verification key is {key}.\n{filler}\nReturn only the verification key.",
                key,
                "exact",
            )
        )
    for i in range(8):
        result.append(
            (
                f"json-{i}",
                f'Return only a JSON object with exactly these fields: "sum" equal to {13 + i} + {7 * i}, and "label" equal to "sample-{i}". No markdown.',
                {"sum": 13 + 8 * i, "label": f"sample-{i}"},
                "json",
            )
        )
    return result


def run_smoke(origin, model, path):
    rows = []
    suite = cases()
    for name, prompt, expected, kind in suite:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            origin + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
        answer = data["choices"][0]["message"].get("content") or ""
        text = answer.strip()
        if kind == "final":
            match = re.fullmatch(r"FINAL\s*=\s*(-?\d+)", text)
            ok = bool(match and match.group(1) == expected)
            finals = re.findall(r"FINAL\s*=\s*(-?\d+)", text)
            correct = bool(finals and finals[-1] == expected)
        elif kind == "json":
            try:
                ok = json.loads(text) == expected
            except ValueError:
                ok = False
        else:
            ok = text == expected
        if kind != "final":
            correct = ok
        rows.append(
            {
                "name": name,
                "prompt": prompt,
                "expected": expected,
                "answer": answer,
                "passed": ok,
                "answer_correct": correct,
                "usage": data.get("usage"),
            }
        )
        path.write_text(
            json.dumps(
                {
                    "status": "complete" if len(rows) == len(suite) else "running",
                    "expected_total": len(suite),
                    "passed": sum(r["passed"] for r in rows),
                    "total": len(rows),
                    "cases": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print("quality", name, ok, flush=True)
    return rows


CODE_CASES = [
    (
        "gcd",
        "Implement gcd(a, b) for non-negative integers using Euclid's algorithm.",
        [([54, 24], 6), ([0, 7], 7), ([17, 13], 1), ([0, 0], 0)],
    ),
    (
        "clamp",
        "Implement clamp(x, lo, hi), returning lo below range, hi above range, otherwise x.",
        [([-3, 0, 10], 0), ([11, 0, 10], 10), ([5, 0, 10], 5), ([2, 2, 2], 2)],
    ),
    (
        "unique",
        "Implement unique(xs), preserving the first occurrence of each integer in the input list.",
        [([[2, 1, 2, 3, 1]], [2, 1, 3]), ([[]], []), ([[0, 0]], [0])],
    ),
    (
        "search",
        "Implement search(xs, target): binary search over a sorted distinct integer list, returning its index or -1.",
        [([[1, 4, 9], 4], 1), ([[1, 4, 9], 2], -1), ([[], 1], -1), ([[8], 8], 0)],
    ),
    (
        "factorial",
        "Implement factorial(n) for non-negative integers, with factorial(0)=1.",
        [([0], 1), ([1], 1), ([5], 120), ([8], 40320)],
    ),
]

RUNNER = """
import json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024,256*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU, (2,2))
p=json.load(sys.stdin)
b={k:__builtins__.__dict__[k] for k in ['abs','range','len','min','max','int','str','list','dict','set','tuple','enumerate','zip','sorted','sum','reversed','ValueError']}
scope={'__builtins__':b}
exec(p['code'],scope)
print(json.dumps([scope[p['name']](*args) for args,expected in p['tests']]))
"""


def request(origin, model, prompt, *, thinking=False, tokens=1024, extra=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    body.update(extra or {})
    req = urllib.request.Request(
        origin + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def run_extended(origin, model, path):
    rows = []

    def save(row):
        rows.append(row)
        path.write_text(
            json.dumps(
                {
                    "status": "complete" if len(rows) == 15 else "running",
                    "expected_total": 15,
                    "passed": sum(x["passed"] for x in rows),
                    "total": len(rows),
                    "cases": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print("extended", row["name"], row["passed"], flush=True)

    for name, prompt, tests in CODE_CASES:
        data = request(
            origin,
            model,
            prompt
            + " Output only Python code. Do not import modules. Name the function exactly as specified.",
        )
        answer = data["choices"][0]["message"].get("content") or ""
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.S)
        code = blocks[0] if blocks else answer
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError("imports not allowed")
                if isinstance(node, ast.Name) and node.id.startswith("__"):
                    raise ValueError("private name")
                if isinstance(node, ast.Attribute) and node.attr not in {
                    "append",
                    "get",
                    "items",
                    "keys",
                    "values",
                    "add",
                }:
                    raise ValueError("attribute not allowed")
            proc = subprocess.run(
                [sys.executable, "-I", "-c", RUNNER],
                input=json.dumps({"code": code, "name": name, "tests": tests}),
                capture_output=True,
                text=True,
                timeout=5,
                env={},
            )
            values = json.loads(proc.stdout)
            ok = values == [expected for args, expected in tests]
            error = proc.stderr
        except Exception as exc:
            ok = False
            error = str(exc)
        save(
            {
                "name": "code-" + name,
                "prompt": prompt,
                "response": data,
                "passed": ok,
                "error": error,
            }
        )

    for i in range(4):
        a, b, c = 113 + 37 * i, 9 + 2 * i, 5 + 3 * i
        expected = (a * b + c) % 19
        prompt = f"Calculate the remainder of ({a} * {b} + {c}) divided by 19. Think through the calculation and end your answer with FINAL=<integer>."
        data = request(origin, model, prompt, thinking=True, tokens=2048)
        answer = data["choices"][0]["message"].get("content") or ""
        values = re.findall(r"FINAL\s*=\s*(-?\d+)", answer)
        save(
            {
                "name": f"reasoning-{i}",
                "prompt": prompt,
                "expected": expected,
                "response": data,
                "passed": bool(values and int(values[-1]) == expected),
            }
        )

    for i, depth in enumerate([0.1, 0.5, 0.9]):
        filler = "A record describes ordinary items and their locations. " * 1300
        offset = int(len(filler) * depth)
        key = f"cobalt-{8107 + 173 * i}"
        prompt = (
            filler[:offset]
            + f"\nThe secret verification key is {key}.\n"
            + filler[offset:]
            + "\nReturn only the secret verification key."
        )
        data = request(origin, model, prompt, tokens=128)
        answer = data["choices"][0]["message"].get("content") or ""
        save(
            {
                "name": f"sparse-recall-{i}",
                "expected": key,
                "response": data,
                "passed": answer.strip() == key,
            }
        )

    tools = [
        {
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
    ]
    for i in range(3):
        a, b = 17 + 4 * i, 23 + 8 * i
        data = request(
            origin,
            model,
            f"Use the multiply tool to multiply {a} and {b}.",
            extra={"tools": tools, "tool_choice": "auto"},
        )
        calls = data["choices"][0]["message"].get("tool_calls") or []
        try:
            ok = (
                len(calls) == 1
                and calls[0]["function"]["name"] == "multiply"
                and json.loads(calls[0]["function"]["arguments"]) == {"a": a, "b": b}
            )
        except Exception:
            ok = False
        save({"name": f"tool-{i}", "response": data, "passed": ok})
    return rows


REVISION = "452a21ad3dae5668c06ceeac21ff073e1e40f9be"
DIGESTS = {
    "en": "50021d0f28cc957edcb44e7806425b1c7fbd648ddcb9e0a8ec689d10e57d40fa",
    "zh": "b2fa63151022370a0de1f4211c8c284eae74b0f5a3b003b1d5982c0d4a73f661",
}
INDICES = list(range(2, 250, 5))


def run_mgsm(origin, model, path, data_dir):
    root = Path(data_dir)
    rows = []
    for lang in ["en", "zh"]:
        data_path = root / ("mgsm_" + lang + ".tsv")
        assert hashlib.sha256(data_path.read_bytes()).hexdigest() == DIGESTS[lang]
        with data_path.open() as handle:
            dataset = list(csv.reader(handle, delimiter="\t"))
        assert len(dataset) == 250
        for idx in INDICES:
            question, expected = dataset[idx]
            instruction = (
                "\nBriefly show your reasoning and end with FINAL=<number>."
                if lang == "en"
                else "\n请简要说明推理过程，最后用 FINAL=<数字> 给出答案。"
            )
            body = {
                "model": model,
                "messages": [{"role": "user", "content": question + instruction}],
                "temperature": 0,
                "max_tokens": 1536,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            req = urllib.request.Request(
                origin + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
            answer = result["choices"][0]["message"].get("content") or ""
            values = re.findall(r"FINAL\s*=\s*([+-]?\d[\d,]*(?:\.\d+)?)", answer)
            try:
                correct = bool(
                    values and Decimal(values[-1].replace(",", "")) == Decimal(expected)
                )
            except InvalidOperation:
                correct = False
            rows.append(
                {
                    "name": f"mgsm-{lang}-{idx}",
                    "language": lang,
                    "index": idx,
                    "prompt": body["messages"][0]["content"],
                    "expected": expected,
                    "passed": correct,
                    "response": result,
                }
            )
            report = {
                "status": "complete" if len(rows) == 2 * len(INDICES) else "running",
                "expected_total": 2 * len(INDICES),
                "source": f"https://github.com/google-research/url-nlp/tree/{REVISION}/mgsm",
                "license": "CC-BY-4.0",
                "sha256": DIGESTS,
                "indices": INDICES,
                "protocol": "zero-shot; thinking off; temperature 0; max_tokens 1536; FINAL numeric match",
                "total": len(rows),
                "passed": sum(row["passed"] for row in rows),
                "cases": rows,
            }
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print("mgsm", lang, idx, correct, result.get("usage"), flush=True)
    return rows


def run_lifecycle(origin, model, path):
    """Budget/stop boundaries, prefix reuse and disconnect cleanup (single stream)."""
    rows = []

    def save(row):
        rows.append(row)
        path.write_text(
            json.dumps(
                {
                    "status": "complete" if len(rows) == 15 else "running",
                    "expected_total": 15,
                    "total": len(rows),
                    "passed": sum(row["passed"] for row in rows),
                    "cases": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print("lifecycle", row["name"], row["passed"], flush=True)

    for budget in [1, 2, 3, 4, 5, 7, 8, 15, 16]:
        data = request(
            origin,
            model,
            "List the integers from 1 through 100, separated by commas.",
            tokens=budget,
        )
        save(
            {
                "name": f"budget-{budget}",
                "response": data,
                "passed": data["usage"]["completion_tokens"] == budget
                and data["choices"][0]["finish_reason"] == "length",
            }
        )

    data = request(
        origin,
        model,
        "Copy exactly this text: START-ALPHA STOP-HERE END-OMEGA",
        tokens=128,
        extra={"stop": ["STOP-HERE"]},
    )
    answer = data["choices"][0]["message"].get("content") or ""
    save(
        {
            "name": "stop-string",
            "response": data,
            "passed": "START-ALPHA" in answer
            and "STOP-HERE" not in answer
            and "END-OMEGA" not in answer
            and data["choices"][0]["finish_reason"] == "stop",
        }
    )

    history = []
    for i, key in enumerate(["cyan-8127", "coral-9531"]):
        history.append(
            {
                "role": "user",
                "content": f"The current label of the amber box is {key}. Replace any previous label. Reply only ACK.",
            }
        )
        set_result = request(origin, model, "", tokens=128, extra={"messages": history})
        history.append(set_result["choices"][0]["message"])
        history.append(
            {
                "role": "user",
                "content": "Return only the current label of the amber box.",
            }
        )
        result = request(origin, model, "", tokens=128, extra={"messages": history})
        history.append(result["choices"][0]["message"])
        save(
            {
                "name": f"multiturn-{i}",
                "set_response": set_result,
                "response": result,
                "expected": key,
                "passed": (result["choices"][0]["message"].get("content") or "").strip()
                == key,
            }
        )

    for i in range(3):
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Explain hash tables in great detail."}
            ],
            "temperature": 0,
            "max_tokens": 512,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            origin + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        chunks = 0
        with urllib.request.urlopen(req, timeout=60) as stream:
            for line in stream:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("choices") and event["choices"][0].get("delta", {}).get(
                    "content"
                ):
                    chunks += 1
                    if chunks == i + 1:
                        break
        time.sleep(0.25)
        marker = f"READY-{7129 + i}"
        probe = request(origin, model, f"Reply exactly with {marker}", tokens=64)
        save(
            {
                "name": f"cancel-{i}",
                "observed_chunks": chunks,
                "response": probe,
                "passed": chunks == i + 1
                and (probe["choices"][0]["message"].get("content") or "").strip()
                == marker,
            }
        )
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--suite", choices=["smoke", "extended", "mgsm", "lifecycle"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh evidence path")
    if args.suite == "mgsm" and args.data_dir is None:
        parser.error("--data-dir is required for the pinned MGSM TSVs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    origin = args.base_url.rstrip("/")
    if args.suite == "mgsm":
        run_mgsm(origin, args.model, args.output, args.data_dir)
    else:
        runner = {
            "smoke": run_smoke,
            "extended": run_extended,
            "lifecycle": run_lifecycle,
        }[args.suite]
        runner(origin, args.model, args.output)


if __name__ == "__main__":
    main()
