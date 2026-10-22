#!/usr/bin/env python3
"""Independent long-prefix factual recall; does not qualify tool sessions or speed."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

try:
    from benchmarks import validate_runtime_sessions as sessions
except ModuleNotFoundError:
    import validate_runtime_sessions as sessions


def validate_case(case, budget):
    body = case["response"]
    if body.get("error"):
        raise ValueError("request failed")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected one answer")
    choice = choices[0]
    message = choice["message"]
    if choice.get("finish_reason") != "stop" or message.get("tool_calls"):
        raise ValueError("incomplete answer or unexpected tool call")
    if message.get("role") != "assistant":
        raise ValueError("invalid answer role")
    expected = sessions.FACTS[case["key"]]
    if sessions.canonical(sessions.strict_json(message["content"])) != sessions.canonical(expected):
        raise ValueError("fact/value/source oracle mismatch")
    usage = body["usage"]
    prompt, output, total = (usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if (
        any(type(n) is not int for n in (prompt, output, total))
        or prompt != case["length"]
        or prompt != case["rendered"]["prompt_tokens"]
        or not 1 <= output <= budget
        or total != prompt + output
    ):
        raise ValueError("rendered prompt or completed usage mismatch")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if type(cached) is not int or not 0 <= cached <= prompt:
        raise ValueError("invalid cached usage")
    return sessions.signature(message)


def validate_report(report):
    work = report["workload"]
    if (
        not work["lengths"] or len(set(work["lengths"])) != len(work["lengths"])
        or any(type(n) is not int or n not in (8192, 32768) for n in work["lengths"])
        or type(work["output_budget"]) is not int or not 1 <= work["output_budget"] <= 1024
    ):
        raise ValueError("invalid recall envelope")
    expected = [(length, key, phase) for length in work["lengths"] for key in sessions.FACTS for phase in ("seed", "repeat")]
    if report.get("error") or report.get("complete") is not True or len(report["cases"]) != len(expected):
        raise ValueError("recall work is incomplete")
    observed = [(c["length"], c["key"], c["phase"]) for c in report["cases"]]
    if observed != expected:
        raise ValueError("missing, duplicate or reordered recall cases")
    for case in report["cases"]:
        if case.get("error") or case.get("validated") is not True:
            raise ValueError("failed recall case")
        if validate_case(case, work["output_budget"]) != case["output_signature"]:
            raise ValueError("stored answer signature disagrees with response")
        if report.get("requires_exact_hits") and case["phase"] == "repeat":
            cached = (case["response"]["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            if cached != case["length"] - 1:
                raise ValueError("repeat did not reuse the exact unaligned safe-suffix endpoint")
    if report.get("requires_exact_hits"):
        stats = report["native_stats"]
        exact = stats["native_runtime"]["exact_prefix"]
        if stats["requests"]["active"] != 0 or exact["published"] <= 0 or exact["hits"] <= 0:
            raise ValueError("exact path was not exercised and drained")


def compare(baseline, candidate):
    if baseline["identity"] != candidate["identity"] or baseline["workload"] != candidate["workload"]:
        raise ValueError("model or workload identity differs")
    for report in (baseline, candidate):
        validate_report(report)
    for a, b in zip(baseline["cases"], candidate["cases"]):
        if a["rendered"] != b["rendered"] or a["output_signature"] != b["output_signature"]:
            raise ValueError("rendered prompt or strict answer differs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--require-exact-hits", action="store_true")
    p.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768])
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--max-context", type=int, default=65536)
    p.add_argument("--timeout", type=float, default=300)
    args = p.parse_args()
    if (
        args.output.exists()
        or len(set(args.lengths)) != len(args.lengths)
        or not args.lengths
        or any(n not in (8192, 32768) for n in args.lengths)
        or not 1 <= args.output_tokens <= 1024
        or max(args.lengths) + args.output_tokens > args.max_context
        or not 0 < args.max_context <= 131072
        or not math.isfinite(args.timeout) or args.timeout <= 0
    ):
        p.error("invalid recall envelope or evidence output already exists")
    render = sessions.tokenizer_renderer(args.tokenizer)
    report = {
        "schema_version": "1.0", "kind": "independent-prefix-recall", "complete": False,
        "identity": sessions.strict_json(args.identity.read_text()),
        "workload": {"lengths": args.lengths, "output_budget": args.output_tokens,
            "max_context": args.max_context, "facts": sessions.FACTS,
            "system_sha256": sessions.digest(sessions.SYSTEM),
            "session_harness_sha256": hashlib.sha256(Path(sessions.__file__).read_bytes()).hexdigest(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "cases": [], "comparison_passed": False, "performance_qualification": False,
        "requires_exact_hits": args.require_exact_hits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    for length in args.lengths:
        for key in sessions.FACTS:
            try:
                messages = sessions.exact_recall_messages(render, args.model, args.output_tokens, key, length)
                payload = sessions.request_payload(args.model, messages, args.output_tokens)
            except Exception as exc:
                for phase in ("seed", "repeat"):
                    report["cases"].append({"length":length,"key":key,"phase":phase,
                        "validated":False,"error":f"{type(exc).__name__}: {exc}"})
                save()
                continue
            for phase in ("seed", "repeat"):
                case = {"length": length, "key": key, "phase":phase, "validated": False, "error": None}
                report["cases"].append(case)
                try:
                    case.update(request=payload, rendered=render(payload))
                    case["response"] = sessions.http_chat(args.base_url, payload, args.timeout)
                    case["output_signature"] = validate_case(case, args.output_tokens)
                    case["validated"] = True
                except Exception as exc:
                    case["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    save()
    report["complete"] = all(c["validated"] for c in report["cases"])
    try:
        if args.require_exact_hits:
            deadline = time.monotonic() + 10
            while True:
                with urllib.request.urlopen(args.base_url.rstrip("/")+"/v1/stats", timeout=10) as response:
                    report["native_stats"] = sessions.strict_json(response.read((2<<20)+1).decode())
                if report["native_stats"]["requests"]["active"] == 0:
                    break
                if time.monotonic() >= deadline:
                    raise ValueError("request ownership did not drain")
                time.sleep(0.05)
        validate_report(report)
        if args.baseline:
            compare(sessions.strict_json(args.baseline.read_text()), report)
            report["comparison_passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    save()
    print(json.dumps({"complete": report["complete"], "comparison_passed": report["comparison_passed"], "error": report.get("error")}))
    return 0 if report["complete"] and not report.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
