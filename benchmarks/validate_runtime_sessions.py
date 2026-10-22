#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic session/tool/branch/recall correctness; no performance qualification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import urllib.error
import urllib.request

FACTS = {
    "k0": {"fact": "k0", "value": "AX-4317", "source": "P001"},
    "k1": {"fact": "k1", "value": "2031-06-19", "source": "P500"},
    "k2": {"fact": "k2", "value": 7341, "source": "P999"},
}
LEGAL_RECORD = """Wholly fictional case SYN-1042. Use only these four documents.
L003, document date 2031-05-03: Juniper Labs (supplier) states "export delivered on 2031-04-29". This is an unverified party statement.
L001, document date 2031-04-02: Oak Ltd (client) and Juniper Labs sign a service order. The original delivery deadline is 2031-04-30.
L004, document date 2031-05-04: Both parties agree to the action "corrected export", due 2031-05-08. This agreement determines no fault. No court finding is present in the record.
L002, document date 2031-05-02: Oak Ltd states "headers were missing". This is an unverified party statement.
"""
LEGAL_SUMMARY = {
    "case_id": "SYN-1042",
    "original_deadline": {"value": "2031-04-30", "source": "L001"},
    "positions": [
        {
            "party": "Oak Ltd",
            "statement": "headers were missing",
            "status": "unverified",
            "source": "L002",
        },
        {
            "party": "Juniper Labs",
            "statement": "export delivered on 2031-04-29",
            "status": "unverified",
            "source": "L003",
        },
    ],
    "remedy": {
        "action": "corrected export",
        "deadline": "2031-05-08",
        "source": "L004",
    },
    "court_finding": None,
}
LEGAL_CHRONOLOGY = {
    "documents": [
        {"date": "2031-04-02", "source": "L001"},
        {"date": "2031-05-02", "source": "L002"},
        {"date": "2031-05-03", "source": "L003"},
        {"date": "2031-05-04", "source": "L004"},
    ]
}
LEGAL_CASES = (
    (
        "legal-summary",
        "LEGAL SUMMARY: Return one object with exactly case_id, original_deadline (value and source), positions (client then supplier, each with party, statement, status, source), remedy (action, deadline, source), and court_finding. Copy the quoted statements and action literally. Preserve unverified status; use JSON null for an absent court finding. Distinguish the original deadline from the corrective deadline.",
        LEGAL_SUMMARY,
    ),
    (
        "legal-chronology",
        'LEGAL CHRONOLOGY: Return exactly {"documents":[...]} for all four documents in ascending document-date order, each with exactly date and source. Use document dates, not dates mentioned inside a party statement or a deadline.',
        LEGAL_CHRONOLOGY,
    ),
)
SYSTEM = {
    "role": "system",
    "content": "Use only the synthetic record or tool results. Return exactly the requested JSON, without markdown or commentary. Preserve each value type and source label.",
}
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_fact",
        "description": "Look up one fact in the synthetic record.",
        "parameters": {
            "type": "object",
            "properties": {"fact": {"type": "string", "enum": list(FACTS)}},
            "required": ["fact"],
            "additionalProperties": False,
        },
    },
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strict_json(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite JSON number")

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=constant)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def record_text(repeats=0, tail=0):
    padding = " Archived note: no additional facts." * repeats
    return f"P001: fact k0 has value AX-4317.\n{padding}{' x' * tail}\nP500: fact k1 has value 2031-06-19.\n{padding}\nP999: fact k2 has integer value 7341."


def question(key):
    return {
        "role": "user",
        "content": f"Return fact {key} as an object with exactly fact, value and source. Copy the literal value and source from the record.",
    }


def lookup_fact(arguments):
    if set(arguments) != {"fact"} or arguments["fact"] not in FACTS:
        raise ValueError("unsupported synthetic lookup arguments")
    return copy.deepcopy(FACTS[arguments["fact"]])


def request_payload(model, messages, budget, tool=False):
    result = {
        "model": model,
        "messages": copy.deepcopy(messages),
        "max_tokens": budget,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tool:
        result.update(tools=[TOOL], tool_choice="required")
    return result


def tokenizer_renderer(path):
    """Use the same request conversion and tokenizer as the native worker."""
    if not Path(path).exists():
        raise ValueError("a local serving tokenizer artifact is required")
    from sparklab.message import TokenizeMsg
    from sparklab.serving.api_models import ChatCompletionRequest
    from sparklab.serving.openai_api import chat_request_to_genspec
    from sparklab.tokenizer.tokenize import TokenizeManager
    from sparklab.utils import load_tokenizer

    manager = TokenizeManager(load_tokenizer(str(path)))

    def render(payload):
        # http_chat serializes canonical JSON. Preserve that wire ordering in
        # nested tool-schema dictionaries: Jinja may render their insertion
        # order, affecting both prompt token counts and hashes.
        payload = strict_json(canonical(payload))
        spec = chat_request_to_genspec(
            ChatCompletionRequest.model_validate(payload), {}
        )
        ids = manager.tokenize(
            [
                TokenizeMsg(
                    uid=0,
                    text=spec.messages,
                    sampling_params=spec.sampling_params,
                    tools=spec.template_tools,
                    chat_template_kwargs=spec.chat_template_kwargs,
                )
            ]
        )[0].tolist()
        return {"prompt_tokens": len(ids), "token_ids_sha256": digest(ids)}

    return render


def exact_recall_messages(render, model, budget, key, target):
    def candidate(n, tail=0):
        messages = [
            SYSTEM,
            {"role": "user", "content": record_text(n, tail)},
            question(key),
        ]
        return messages, render(request_payload(model, messages, budget))[
            "prompt_tokens"
        ]

    base = candidate(0)[1]
    slope = max(1, candidate(1)[1] - base)
    low, high = 0, min(target, max(1, (target - base) // slope + 2))
    while candidate(high)[1] < target and high < target:
        high = min(target, 2 * high)
    if base > target or candidate(high)[1] < target:
        raise ValueError(
            "requested recall length is outside the bounded text construction"
        )
    while low < high:
        mid = (low + high + 1) // 2
        if candidate(mid)[1] <= target:
            low = mid
        else:
            high = mid - 1
    for tail in range(65):
        messages, length = candidate(low, tail)
        if length == target:
            return messages
    raise ValueError(
        "cannot construct the exact rendered recall length with this tokenizer"
    )


def http_chat(base, payload, timeout):
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=canonical(payload).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read((2 << 20) + 1)
        if len(raw) > 2 << 20:
            raise ValueError("response exceeds 2 MiB evidence bound")
        body = strict_json(raw.decode())
        if not isinstance(body, dict):
            raise ValueError("response must be a JSON object")
        if response.status != 200:
            body["error"] = body.get("error") or f"HTTP {response.status}: {body}"
            body["http_status"] = response.status
        return body


def signature(message):
    # IDs are protocol-generated, not sampled text. Their actual linkage is
    # validated during the tool loop; rendered token hashes still include them
    # if the model template does so.
    return {
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": [call["function"] for call in message.get("tool_calls") or []],
    }


class SessionRun:
    def __init__(
        self,
        base,
        model,
        identity,
        render,
        *,
        turns=20,
        budget=256,
        max_context=4096,
        recall_lengths=(),
        legal_case=False,
        timeout=180,
    ):
        if type(turns) is not int or not 20 <= turns <= 50:
            raise ValueError("turns must be 20..50")
        if type(budget) is not int or not 1 <= budget <= 1024:
            raise ValueError("output budget must be 1..1024")
        if type(max_context) is not int or not budget < max_context <= 131072:
            raise ValueError("invalid declared context envelope")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if (
            len(recall_lengths) > 2
            or len(set(recall_lengths)) != len(recall_lengths)
            or any(
                n not in (8192, 32768) or n + budget > max_context
                for n in recall_lengths
            )
        ):
            raise ValueError(
                "recall lengths must be distinct 8192/32768 and fit with output space"
            )
        if not isinstance(identity, dict) or not identity:
            raise ValueError("effective model identity is required")
        if type(legal_case) is not bool:
            raise ValueError("legal_case must be a boolean")
        self.legal_case = legal_case
        self.base, self.model, self.identity, self.render = (
            base,
            model,
            copy.deepcopy(identity),
            render,
        )
        self.turns, self.budget, self.max_context, self.recall_lengths, self.timeout = (
            turns,
            budget,
            max_context,
            tuple(recall_lengths),
            timeout,
        )
        self.records = []
        self.progress = None
        self.finished = False
        self.used_tool_ids = set()
        self.planned = (
            [
                name
                for i in range(turns)
                for name in (
                    [f"turn-{i}.tool", f"turn-{i}.answer"]
                    if i % 4 == 3
                    else [f"turn-{i}.answer"]
                )
            ]
            + ["branch-amended", "branch-parent", "summary"]
            + ([name for name, _, _ in LEGAL_CASES] if legal_case else [])
            + [f"recall-{n}.{key}" for n in recall_lengths for key in FACTS]
        )

    def skip(self, name, reason):
        self.records.append({"case": name, "attempted": False, "error": reason})

    def step(self, name, messages, expected, tool=False):
        # Match wire dictionary order even for templates iterating tool schemas.
        payload = strict_json(
            canonical(request_payload(self.model, messages, self.budget, tool))
        )
        observation = {
            "case": name,
            "attempted": True,
            "request": payload,
            "expected": expected,
            "started": time.monotonic(),
            "error": None,
            "validated": False,
        }
        self.records.append(observation)
        if self.progress:
            self.progress(self.report())
        try:
            rendered = self.render(payload)
            observation["rendered"] = rendered
            if (
                type(rendered["prompt_tokens"]) is not int
                or not 0 < rendered["prompt_tokens"] <= self.max_context - self.budget
            ):
                raise ValueError("rendered prompt plus output exceeds declared context")
            body = http_chat(self.base, payload, self.timeout)
            observation["response"] = body
            if body.get("error"):
                raise ValueError(f"server error: {body['error']}")
            choices = body.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("expected one completion")
            choice = choices[0]
            usage = body.get("usage") or {}
            if (
                usage.get("prompt_tokens") != rendered["prompt_tokens"]
                or type(usage.get("prompt_tokens")) is not int
            ):
                raise ValueError("server/local prompt token counts differ")
            count = usage.get("completion_tokens")
            if type(count) is not int or not 0 < count <= self.budget:
                raise ValueError("invalid completion usage")
            if (
                type(usage.get("total_tokens")) is not int
                or usage["total_tokens"] != usage["prompt_tokens"] + count
            ):
                raise ValueError("usage totals do not reconcile")
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            if type(cached) is not int or not 0 <= cached <= usage["prompt_tokens"]:
                raise ValueError("invalid cached usage")
            message = choice["message"]
            if message.get("role") != "assistant":
                raise ValueError("invalid assistant role")
            if tool:
                calls = message.get("tool_calls") or []
                if (
                    choice.get("finish_reason") != "tool_calls"
                    or len(calls) != 1
                    or (
                        message.get("content") is not None
                        and (
                            not isinstance(message["content"], str)
                            or message["content"].strip()
                        )
                    )
                ):
                    raise ValueError("expected one tool call and no unparsed content")
                call = calls[0]
                if (
                    call.get("type") != "function"
                    or call["function"]["name"] != "lookup_fact"
                    or strict_json(call["function"]["arguments"])
                    != {"fact": expected["fact"]}
                ):
                    raise ValueError("incorrect tool name or arguments")
                call_id = call.get("id")
                if (
                    not isinstance(call_id, str)
                    or not call_id
                    or call_id in self.used_tool_ids
                ):
                    raise ValueError("missing or reused tool call ID")
                self.used_tool_ids.add(call_id)
            else:
                if choice.get("finish_reason") != "stop" or message.get("tool_calls"):
                    raise ValueError("incomplete answer or unexpected tool call")
                if canonical(strict_json(message["content"])) != canonical(expected):
                    raise ValueError("JSON fact/value/source oracle mismatch")
            observation["output_signature"] = signature(message)
            observation["validated"] = True
            return copy.deepcopy(
                {
                    k: v
                    for k, v in message.items()
                    if k in ("role", "content", "reasoning_content", "tool_calls")
                }
            )
        except Exception as exc:
            observation["error"] = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            observation["ended"] = time.monotonic()
            if self.progress:
                self.progress(self.report())

    def run(self):
        history = [SYSTEM, {"role": "user", "content": record_text()}]
        blocked = False
        parent = None
        for i in range(self.turns):
            key = list(FACTS)[i % 3]
            expected = FACTS[key]
            name = f"turn-{i}"
            if blocked:
                if i % 4 == 3:
                    self.skip(name + ".tool", "prior session step failed")
                self.skip(name + ".answer", "prior session step failed")
                continue
            history.append(question(key))
            if i % 4 == 3:
                history[-1]["content"] = (
                    "First call lookup_fact for fact "
                    + key
                    + ". Then return only its fact/value/source JSON result."
                )
                reply = self.step(name + ".tool", history, expected, True)
                if reply is None:
                    self.skip(name + ".answer", "tool step failed")
                    blocked = True
                    continue
                # Preserve the actual returned ID in both assistant and tool
                # messages. This is an executed tool loop, not a forged history.
                history.append(reply)
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": reply["tool_calls"][0]["id"],
                        "content": canonical(
                            lookup_fact(
                                strict_json(
                                    reply["tool_calls"][0]["function"]["arguments"]
                                )
                            )
                        ),
                    }
                )
            reply = self.step(name + ".answer", history, expected)
            if reply is None:
                blocked = True
            else:
                history.append(reply)
                if i == self.turns // 2 - 1:
                    parent = copy.deepcopy(history)
        amended = {"fact": "k0", "value": "AX-8879", "source": "AMENDMENT-A"}
        for name, query, expected in [
            (
                "branch-amended",
                {
                    "role": "user",
                    "content": "For this branch only, AMENDMENT-A replaces fact k0 with value AX-8879. Return its fact/value/source JSON.",
                },
                amended,
            ),
            ("branch-parent", question("k0"), FACTS["k0"]),
        ]:
            if parent is None:
                self.skip(name, "shared parent was not completed")
            else:
                self.step(name, [*copy.deepcopy(parent), query], expected)
        if blocked:
            self.skip("summary", "session did not complete")
        else:
            self.step(
                "summary",
                [
                    *history,
                    {
                        "role": "user",
                        "content": 'SUMMARY: Return exactly {"facts":[...]} with all three original facts, each having fact/value/source, in k0,k1,k2 order. Do not use branch amendments.',
                    },
                ],
                {"facts": list(FACTS.values())},
            )
        if self.legal_case:
            for name, query, expected in LEGAL_CASES:
                self.step(
                    name,
                    [
                        SYSTEM,
                        {"role": "user", "content": LEGAL_RECORD},
                        {"role": "user", "content": query},
                    ],
                    expected,
                )
        for n in self.recall_lengths:
            for key in FACTS:
                name = f"recall-{n}.{key}"
                try:
                    messages = exact_recall_messages(
                        self.render, self.model, self.budget, key, n
                    )
                except Exception as exc:
                    self.records.append(
                        {
                            "case": name,
                            "attempted": False,
                            "error": f"exact recall construction failed: {exc}",
                        }
                    )
                else:
                    self.step(name, messages, FACTS[key])
        self.finished = True
        return self.report()

    def report(self):
        attempted = [r for r in self.records if r["attempted"]]
        passed = sum(
            r.get("validated", False) and r["error"] is None for r in attempted
        )
        return {
            "schema_version": "1.0",
            "finalized": self.finished,
            "identity": self.identity,
            "workload": {
                "turns": self.turns,
                "output_budget": self.budget,
                "max_context": self.max_context,
                "recall_lengths": list(self.recall_lengths),
                "facts": FACTS,
                "client_concurrency": 1,
                **(
                    {
                        "legal_case": {
                            "version": "synthetic-case-v1",
                            "record_sha256": digest(LEGAL_RECORD),
                            "oracle_sha256": digest([LEGAL_SUMMARY, LEGAL_CHRONOLOGY]),
                        }
                    }
                    if self.legal_case
                    else {}
                ),
            },
            "planned_cases": self.planned,
            "observations": self.records,
            "summary": {
                "planned": len(self.planned),
                "attempted": len(attempted),
                "passed": passed,
                "failed": len(attempted) - passed,
                "not_run": len(self.planned) - len(attempted),
                "complete": passed == len(self.planned),
            },
        }


def compare(baseline, candidate):
    if (
        baseline["identity"] != candidate["identity"]
        or baseline["workload"] != candidate["workload"]
    ):
        raise ValueError("effective model or workload identity differs")
    if baseline["planned_cases"] != candidate["planned_cases"]:
        raise ValueError("planned cases differ")
    if not all(
        r.get("finalized") and r["summary"]["complete"] for r in (baseline, candidate)
    ):
        raise ValueError(
            "comparison requires fully completed, independently validated runs"
        )
    for report in (baseline, candidate):
        if len(report["observations"]) != len(report["planned_cases"]) or not all(
            r.get("validated") and r.get("attempted") and r.get("error") is None
            for r in report["observations"]
        ):
            raise ValueError("missing, failed or incomplete observations")
    a = {r["case"]: r for r in baseline["observations"]}
    b = {r["case"]: r for r in candidate["observations"]}
    if set(a) != set(b) or set(a) != set(baseline["planned_cases"]):
        raise ValueError("missing or duplicate observations")
    unmatched = [name for name in a if a[name]["rendered"] != b[name]["rendered"]]
    different = [
        name for name in a if a[name]["output_signature"] != b[name]["output_signature"]
    ]
    work = [
        name
        for name in a
        if a[name]["response"]["usage"]["completion_tokens"]
        != b[name]["response"]["usage"]["completion_tokens"]
    ]
    return {
        "matched_rendered_inputs": not unmatched,
        "unmatched_cases": unmatched,
        "strict_output_match": not different,
        "different_outputs": different,
        "equal_completed_output_counts": not work,
        "different_output_counts": work,
        "passed": not (unmatched or different or work),
        "note": "Correctness comparison only; no paired performance/quality/soak qualification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--recall-lengths", type=int, nargs="*", default=[])
    parser.add_argument(
        "--legal-case",
        action="store_true",
        help="add fictional attributed case summary and document chronology",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    render = tokenizer_renderer(args.tokenizer)
    runner = SessionRun(
        args.base_url,
        args.model,
        json.loads(args.identity.read_text()),
        render,
        turns=args.turns,
        budget=args.output_tokens,
        max_context=args.max_context,
        recall_lengths=args.recall_lengths,
        legal_case=args.legal_case,
        timeout=args.timeout,
    )
    if args.baseline and args.output.resolve() == args.baseline.resolve():
        parser.error("output must not overwrite its baseline")
    baseline = strict_json(args.baseline.read_text()) if args.baseline else None
    provenance = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "base_url": args.base_url,
        "served_model": args.model,
        "tokenizer_path": str(args.tokenizer),
        "feature_env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                (
                    "SPARKLAB_EXACT_",
                    "SPARKLAB_PREFIX_",
                    "SPARKLAB_SCHEDULER_",
                    "SPARKLAB_DFLASH_",
                )
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(report):
        report.update(label=args.label, provenance=provenance)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    runner.progress = save
    report = None
    try:
        report = runner.run()
    finally:
        report = report or runner.report()
        if baseline is not None:
            try:
                report["comparison"] = compare(baseline, report)
            except Exception as exc:
                report["comparison"] = {"passed": False, "error": str(exc)}
        save(report)
    print(json.dumps(report["summary"]))
    if "comparison" in report:
        print(json.dumps(report["comparison"]))
    if (
        not report["summary"]["complete"]
        or not report.get("comparison", {"passed": True})["passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
