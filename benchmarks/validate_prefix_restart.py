#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate native restart/disable evidence from raw bounded idle dispatch logs."""

import argparse
import json
from pathlib import Path
import re

LOG_LIMIT = 16 << 20
MODES = ("cold", "publish", "restore", "rollback")


def read_measurements(path):
    path = Path(path)
    if path.stat().st_size > LOG_LIMIT:
        raise ValueError("diagnostic log exceeds 16 MiB evidence bound")
    result = {"metrics": None, "prefix": None}
    counts = {"metrics": 0, "prefix": 0}
    with path.open("rb") as source:
        raw = source.read(LOG_LIMIT + 1)
    if len(raw) > LOG_LIMIT:
        raise ValueError("diagnostic log exceeds 16 MiB evidence bound")
    contents = raw.decode(errors="replace")
    lines = contents.splitlines()
    for index, line in enumerate(lines):
        for marker, key in [
            ("Runtime measurements: ", "metrics"),
            ("Prefix measurements: ", "prefix"),
        ]:
            if marker not in line:
                continue
            text = line.split(marker, 1)[1].lstrip()
            try:
                value, end = json.JSONDecoder().raw_decode(text)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not contents.endswith("\n"):
                    continue  # observer raced the final log write; retry later
                raise ValueError("malformed complete diagnostic JSON line") from None
            if re.sub(r"\x1b\[[0-9;]*m", "", text[end:]).strip():
                raise ValueError("trailing data in diagnostic record")
            if not isinstance(value, dict):
                raise ValueError("diagnostic record must be an object")
            result[key] = value
            counts[key] += 1
    result["snapshot_counts"] = counts
    return result


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def dispatch_work(metrics):
    trace = metrics.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("missing raw dispatch trace")
    total = 0
    decoding = False
    for event in trace:
        if event.get("phase") not in ("prefill", "decode"):
            raise ValueError("unexpected non-target dispatch phase")
        if integer(event.get("requests"), "dispatch requests") != 1:
            raise ValueError("requires serial one-request dispatch evidence")
        tokens = integer(event.get("tokens"), "dispatch token count")
        if not tokens:
            raise ValueError("empty dispatch in completed request evidence")
        if event["phase"] == "prefill":
            if decoding:
                raise ValueError("prefill dispatch follows decode in serial request")
            total += tokens
        else:
            if not total:
                raise ValueError("decode dispatch precedes prompt processing")
            decoding = True
    counters = metrics.get("counters") or {}
    if integer(counters.get("prefill_computed_tokens"), "computed tokens") != total:
        raise ValueError(
            "raw prefill trace and counter disagree or trace was truncated"
        )
    return total, integer(counters.get("prefix_reused_tokens"), "reused tokens")


def inventory(root):
    records = {}
    root = Path(root)
    for path in root.glob("*/*.pxc"):
        if len(records) >= 32:
            raise ValueError("restart fixture record bound exceeded")
        stat = path.stat()
        records[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return records


def verify(report):
    runs = report.get("runs", {})
    if set(runs) != set(MODES):
        raise ValueError("requires cold, publish, restore and rollback boots")
    prefix = integer(report.get("prefix_tokens"), "prefix length")
    query = integer(report.get("query_tokens"), "query length")
    budget = integer(report.get("output_tokens"), "output budget")
    if not 0 < prefix < query or not budget:
        raise ValueError("invalid continuation envelope")
    for key in ("prefix_ids_sha256", "query_ids_sha256", "query_prefix_ids_sha256"):
        if not isinstance(report.get(key), str) or not re.fullmatch(
            "[0-9a-f]{64}", report[key]
        ):
            raise ValueError("missing effective token identity")
    if report["prefix_ids_sha256"] != report["query_prefix_ids_sha256"]:
        raise ValueError("stored tokens are not an exact prefix of the query")
    identity = runs["cold"]["identity"]
    geometry = runs["cold"]["geometry"]
    if (
        not isinstance(identity, dict)
        or not identity
        or not isinstance(geometry, dict)
        or not geometry
    ):
        raise ValueError("model identity and measured cache geometry are required")
    computed = {}
    for mode in MODES:
        run = runs[mode]
        if (
            run.get("mode") != mode
            or run["identity"] != identity
            or run["geometry"] != geometry
        ):
            raise ValueError("boot/profile identity or cache capacities differ")
        if run.get("exit_code") != 143:
            raise ValueError("owned server did not complete ordinary SIGTERM cleanup")
        if run.get("forced_stop") or run.get("semaphore_warning"):
            raise ValueError("forced shutdown or semaphore warning")
        enabled = mode in ("publish", "restore")
        if (
            run.get("exact_enabled") is not enabled
            or run.get("disk_enabled") is not enabled
        ):
            raise ValueError("feature selection does not match boot role")
        startup = run.get("startup_metrics")
        if (
            startup is None
            or integer(
                startup.get("counters", {}).get("prefill_computed_tokens", 0),
                "startup computed tokens",
            )
            != 0
            or startup.get("trace") != []
        ):
            raise ValueError("boot was already warmed by model requests")
        computed[mode], reused = dispatch_work(run["metrics"])
        expected_input = prefix if mode == "publish" else query
        expected_output = 1 if mode == "publish" else budget
        decodes = [e for e in run["metrics"]["trace"] if e["phase"] == "decode"]
        if len(decodes) != expected_output - 1 or any(
            e["tokens"] != 1 for e in decodes
        ):
            raise ValueError("target decode dispatches do not cover the output work")
        if (
            integer(
                run["metrics"].get("counters", {}).get("committed_output_tokens"),
                "committed output count",
            )
            != expected_output
        ):
            raise ValueError("dispatch snapshot does not cover all completed output")
        reply = run["reply"]
        usage = reply.get("usage") or {}
        if (
            integer(usage.get("prompt_tokens"), "prompt usage") != expected_input
            or integer(usage.get("completion_tokens"), "output usage")
            != expected_output
        ):
            raise ValueError("request usage does not match declared work")
        if (
            integer(usage.get("total_tokens"), "total usage")
            != expected_input + expected_output
        ):
            raise ValueError("usage total does not reconcile")
        choices = reply.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or choices[0].get("finish_reason") != "length"
            or not isinstance(choices[0].get("text"), str)
        ):
            raise ValueError("incomplete forced-budget completion")
        if run.get("prefix_ids_sha256") != report.get("prefix_ids_sha256") or run.get(
            "query_ids_sha256"
        ) != report.get("query_ids_sha256"):
            raise ValueError("effective token IDs changed across boots")
        if mode == "restore":
            if computed[mode] != query - prefix or reused != prefix:
                raise ValueError("restore did not dispatch exactly the uncached suffix")
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if integer(cached, "restored cached usage") != prefix:
                raise ValueError("restored cached usage differs from dispatch")
        elif computed[mode] != expected_input or reused:
            raise ValueError(
                "cold/publication/disabled boot unexpectedly reused prefix work"
            )
        store = run.get("prefix")
        if enabled:
            if not isinstance(store, dict) or store.get("disabled"):
                raise ValueError("prefix store is unavailable")
            for key in (
                "unreleased_tickets",
                "completed_pending_fence",
                "staging_bytes",
                "restore_waiters",
                "private_restore_pages",
                "private_restore_slots",
            ):
                if integer(store.get(key), key) != 0:
                    raise ValueError("prefix resources did not drain")
            if integer(store.get("records"), "retained records") < 1:
                raise ValueError("no retained record evidence")
            counter = "writes_completed" if mode == "publish" else "restores_completed"
            if integer(store.get("counters", {}).get(counter, 0), counter) < 1:
                raise ValueError("required prefix operation did not complete")
            if mode == "restore":
                before = run.get("startup_prefix") or {}
                if integer(before.get("records", 0), "recovered startup records") < 1:
                    raise ValueError(
                        "restart did not recover records before its first request"
                    )
        elif store is not None:
            raise ValueError("disabled boot still owns a prefix store")
    text = runs["cold"]["reply"]["choices"][0]["text"]
    if not text or any(
        runs[mode]["reply"]["choices"][0]["text"] != text
        for mode in ("restore", "rollback")
    ):
        raise ValueError("strict cold/restore/rollback output mismatch")
    before = report.get("records_before_rollback")
    if not before or before != report.get("records_after_rollback"):
        raise ValueError("disabled rollback mutated retained checkpoint files")
    return {
        "passed": True,
        "computed_prefill_tokens": computed,
        "reused_tokens": prefix,
        "strict_output_equal": True,
        "note": "Restart/disable correctness only; no performance, broad quality or soak claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.evidence.read_text())
    print(json.dumps(verify(report), indent=2))


if __name__ == "__main__":
    main()
