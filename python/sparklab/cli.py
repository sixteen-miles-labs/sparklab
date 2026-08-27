"""Spark Lab's GB10 product CLI.

Legacy engine commands delegate to the compatibility-stable ``freetoken``
implementation. Product commands live here and avoid importing torch unless the
requested operation needs hardware inspection or inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO


def _print_help(file: TextIO) -> None:
    print(
        """usage: sparklab <command> [args]

Spark Lab runs frontier open-weight models on one NVIDIA GB10.

Product commands:
  doctor      Check GB10, CUDA 13, unified memory, storage, and dependencies
  models      Show the recipe-backed Fast, Frontier, and Research portfolio
  status      Show the persistent engine status

Engine commands (FreeToken-compatible):
  serve       Start the OpenAI/Anthropic-compatible API server
  shell       Chat with a running server
  ctl         Query and manage a running server
  daemon      Run or control the persistent engine supervisor
  launch      Configure and launch an agent against Spark Lab
  checkpoint  Convert a Hugging Face checkpoint to FTW
  bench       Run a hardware benchmark (currently: bench bw)

The legacy `ft` command remains supported during the staged migration.
Use `sparklab <command> --help` for command-specific options.""",
        file=file,
    )


def _gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / (1 << 30):.1f} GiB"


def _run_doctor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sparklab doctor",
        description="Inspect this machine against Spark Lab's NVIDIA GB10 profile.",
    )
    parser.add_argument(
        "--storage-path",
        default=".",
        help="Model/checkpoint filesystem to inspect (default: current directory)",
    )
    parser.add_argument("--json", action="store_true", help="Emit the versioned JSON report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero for warnings as well as failed requirements",
    )
    args = parser.parse_args(argv)

    from freetoken.platform import assess_gb10, collect_gb10_snapshot

    report = assess_gb10(collect_gb10_snapshot(args.storage_path))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Spark Lab doctor: {report['status']}")
        print(f"  GPU:       {report['runtime']['gpu'] or 'not detected'}")
        capability = report["runtime"]["compute_capability"]
        capability_text = ".".join(str(value) for value in capability) if capability else "unknown"
        print(f"  CUDA/SM:   {report['runtime']['cuda'] or 'unavailable'} / {capability_text}")
        print(
            f"  Memory:    {_gib(report['memory']['available_bytes'])} available; "
            f"{_gib(report['memory']['safe_available_bytes'])} safe after reserve"
        )
        print(
            f"  Storage:   {_gib(report['storage']['free_bytes'])} free on "
            f"{report['storage']['block_device'] or 'unknown device'}"
        )
        for check in report["checks"]:
            marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[check["status"]]
            print(f"  [{marker}] {check['name']}: {check['observed']}")
        print("  Recommendations:")
        for recommendation in report["recommendations"]:
            print(f"    - {recommendation}")
    if not report["ready"]:
        return 1
    if args.strict and report["warnings"]:
        return 1
    return 0


def _run_models(argv: list[str]) -> int:
    from sparklab.catalog import STATUSES, TIERS, load_catalog, select_recipes

    parser = argparse.ArgumentParser(
        prog="sparklab models",
        description="List versioned GB10 model recipes without implying certification.",
    )
    parser.add_argument("--tier", choices=TIERS)
    parser.add_argument("--status", choices=STATUSES)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    recipes = select_recipes(load_catalog(), tier=args.tier, status=args.status)
    if args.json:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "product": "Spark Lab",
                    "platform": "gb10",
                    "recipes": [recipe.to_dict() for recipe in recipes],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print("Spark Lab model portfolio — NVIDIA GB10")
    if not recipes:
        print("No recipes match the selected filters.")
        return 0
    print(f"{'TIER':<10} {'STATUS':<13} {'RECIPE':<24} MODEL")
    for recipe in recipes:
        print(
            f"{recipe.intended_tier.upper():<10} {recipe.status.upper():<13} "
            f"{recipe.slug:<24} {recipe.model}"
        )
    if not any(recipe.status == "certified" for recipe in recipes):
        print("\nNo recipe is certified yet; current labels are admission status, not promises.")
    return 0


def _run_serve(argv: list[str]) -> int:
    from freetoken.server import launch_server

    launch_server(argv=argv, prog="sparklab serve")
    return 0


def _run_shell(argv: list[str]) -> int:
    from freetoken.shell import main

    return main(argv, prog="sparklab shell")


def _run_ctl(argv: list[str]) -> int:
    from freetoken.control_cli import main

    return main(argv, prog="sparklab ctl")


def _run_daemon(argv: list[str]) -> int:
    from freetoken.daemon import main

    return main(argv, prog="sparklab daemon")


def _run_status(argv: list[str]) -> int:
    from freetoken.daemon.client import main

    return main(["status", *argv], prog="sparklab status")


def _run_launch(argv: list[str]) -> int:
    from freetoken.launch import main

    return main(argv, prog="sparklab launch")


def _run_checkpoint(argv: list[str]) -> int:
    from freetoken.checkpoint.__main__ import main

    return main(argv, prog="sparklab checkpoint")


def _run_bench(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: sparklab bench bw [args]")
        print("\nSubcommands:\n  bw   Benchmark CPU MoE and GPU expert-transfer bandwidth")
        return 0 if argv else 2
    if argv[0] != "bw":
        print(f"unknown sparklab bench subcommand: {argv[0]}", file=sys.stderr)
        return 2
    from freetoken.moe.benchbw import main

    return main(argv[1:], prog="sparklab bench bw")


COMMANDS = {
    "doctor": _run_doctor,
    "models": _run_models,
    "status": _run_status,
    "serve": _run_serve,
    "shell": _run_shell,
    "ctl": _run_ctl,
    "daemon": _run_daemon,
    "launch": _run_launch,
    "checkpoint": _run_checkpoint,
    "bench": _run_bench,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_help(sys.stderr)
        return 2
    if args[0] in {"-h", "--help"}:
        _print_help(sys.stdout)
        return 0
    if args[0] in {"-V", "--version"}:
        from sparklab import __version__

        print(f"Spark Lab {__version__} (FreeToken engine)")
        return 0
    command = COMMANDS.get(args[0])
    if command is None:
        print(f"unknown sparklab command: {args[0]}", file=sys.stderr)
        _print_help(sys.stderr)
        return 2
    return command(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
