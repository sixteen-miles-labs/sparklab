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
  plan        Show storage and unified-memory admission for one recipe
  pull        Acquire an immutable model revision, optionally preparing FTW
  run         Start a recipe-backed server after fail-closed GB10 admission
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


def _recipe(slug: str):
    from sparklab.catalog import get_recipe

    try:
        return get_recipe(slug)
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown Spark Lab recipe: {slug}") from exc


def _run_plan(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sparklab plan",
        description="Plan disk artifacts and unified-memory admission without loading a model.",
    )
    parser.add_argument("recipe", type=_recipe)
    parser.add_argument("--root", help="Spark Lab state root (default: ~/.sparklab)")
    parser.add_argument("--prepare", action="store_true", help="Include FTW preparation space")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from freetoken.platform import collect_gb10_snapshot
    from sparklab.planner import plan_artifacts, plan_runtime

    artifacts = plan_artifacts(args.recipe, root=args.root, include_prepared=args.prepare)
    memory = plan_runtime(args.recipe, collect_gb10_snapshot(artifacts.root))
    payload = {
        "schema_version": "1.0",
        "recipe": args.recipe.slug,
        "artifacts": artifacts.to_dict(),
        "runtime": memory.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Spark Lab plan: {args.recipe.slug}")
        required = artifacts.required_bytes
        print(f"  Storage: {_gib(required)} required; {_gib(artifacts.free_bytes)} free")
        print(f"  Download/preparation ready: {'yes' if artifacts.ready else 'no'}")
        print(
            f"  Runtime: {_gib(memory.required_bytes)} required; "
            f"{_gib(memory.usable_bytes)} safely usable"
        )
        print(f"  Runtime ready: {'yes' if memory.ready else 'no'}")
        for reason in (*artifacts.reasons, *memory.reasons):
            print(f"    - {reason}")
    return 0 if artifacts.ready and memory.ready else 1


def _run_pull(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sparklab pull",
        description="Acquire a recipe's exact Hugging Face revision with resumable downloads.",
    )
    parser.add_argument("recipe", type=_recipe)
    parser.add_argument("--root", help="Spark Lab state root (default: ~/.sparklab)")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Also convert the completed source checkpoint into its FTW execution artifact",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from sparklab.acquire import AcquisitionError, acquire_recipe

    try:
        result = acquire_recipe(
            args.recipe,
            root=args.root,
            prepare=args.prepare,
            dry_run=args.dry_run,
        )
    except AcquisitionError as exc:
        print(f"sparklab pull: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        plan = result["artifact_plan"]
        action = "would acquire" if args.dry_run else "acquired"
        print(
            f"Spark Lab {action} {args.recipe.model}@{args.recipe.revision[:12]} "
            f"at {plan['source_path']}"
        )
        if args.prepare:
            print(f"  FTW: {plan['prepared_path']}")
    return 0


def _run_recipe(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sparklab run",
        description="Start a recipe only after its checkpoint and GB10 memory plan pass.",
    )
    parser.add_argument("recipe", type=_recipe)
    parser.add_argument("--root", help="Spark Lab state root (default: ~/.sparklab)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    from freetoken.platform import collect_gb10_snapshot
    from sparklab.runtime import RuntimePlanError, plan_invocation

    try:
        invocation = plan_invocation(
            args.recipe,
            collect_gb10_snapshot("."),
            root=args.root,
            extra_args=tuple(extra),
        )
    except RuntimePlanError as exc:
        print(f"sparklab run: {exc}", file=sys.stderr)
        return 1
    if args.json or args.dry_run:
        if args.json:
            print(json.dumps(invocation.to_dict(), indent=2, sort_keys=True))
        else:
            print("sparklab serve " + " ".join(invocation.arguments))
        return 0

    from freetoken.server import launch_server

    if args.recipe.status != "certified":
        print(
            f"WARNING: {args.recipe.slug} is {args.recipe.status}, not certified.",
            file=sys.stderr,
        )
    launch_server(argv=list(invocation.arguments), prog=f"sparklab run {args.recipe.slug}")
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
    "plan": _run_plan,
    "pull": _run_pull,
    "run": _run_recipe,
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
