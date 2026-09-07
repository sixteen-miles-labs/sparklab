import pytest
import json

from benchmarks.bench_decode_moe import (
    load_problem,
    parse_args,
    parse_speculative_summary,
    serve_cmd,
)


def test_preformatted_benchmark_prompt_does_not_receive_math_instruction(tmp_path):
    path = tmp_path / "prompts.jsonl"
    prompt = "Write a Python interval-merging function."
    path.write_text(json.dumps({"prompt": prompt, "answer": ""}) + "\n")
    assert load_problem(str(path), 0) == (prompt, "")
    path.write_text(json.dumps({"problem": "Compute 2 + 2.", "answer": "4"}) + "\n")
    text, answer = load_problem(str(path), 0)
    assert "boxed" in text
    assert answer == "4"


def test_decode_benchmark_forwards_speculative_options():
    args = parse_args([
        "--model", "/tmp/model",
        "--speculative-method", "dspark",
        "--speculative-tokens", "7",
        "--draft-sample-method", "probabilistic",
    ])

    command = serve_cmd(args, "offload", 12345)

    assert command[command.index("--speculative-method") + 1] == "dspark"
    assert command[command.index("--speculative-tokens") + 1] == "7"
    assert command[command.index("--draft-sample-method") + 1] == "probabilistic"


def test_decode_benchmark_defaults_to_gpu_first_unified_memory_residency():
    args = parse_args(["--model", "/tmp/model", "--storage", "disk"])

    command = serve_cmd(args, "offload", 12345)

    assert args.host_cache_gb == 0
    assert command[command.index("--moe-host-cache-gb") + 1] == "0.0"
    assert "--disable-moe-prefill-overlap" in command


def test_aime_suite_can_use_shared_server_launcher(monkeypatch):
    from pathlib import Path
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from benchmarks import bench_aime_suite

    monkeypatch.setattr(sys, "argv", [
        "bench_aime_suite.py", "--model", "/tmp/model", "--json", "/tmp/result.jsonl",
        "--min-memory-gib", "12",
    ])
    args = bench_aime_suite.parse_args()
    command = serve_cmd(args, "offload", 12345)
    assert args.normal_eos and args.min_memory_gib == 12
    assert command[command.index("--speculative-method") + 1] == "auto"
    assert command[command.index("--speculative-tokens") + 1] == "0"
    assert command[command.index("--dsv4-kv-storage") + 1] == "bf16"


def test_decode_benchmark_keeps_overlap_with_an_explicit_host_cache():
    args = parse_args([
        "--model", "/tmp/model",
        "--storage", "disk",
        "--host-cache-gb", "4",
    ])

    command = serve_cmd(args, "offload", 12345)

    assert "--disable-moe-prefill-overlap" not in command


def test_decode_benchmark_extracts_final_speculative_summary():
    log = """
MTP summary: steps=7, accepted=4/7 (57.1%), outputs=6, target_forwards=3, outputs/target=2.00
MTP summary: steps=7, accepted=190/252 (75.4%), outputs=256, target_forwards=66, outputs/target=3.88
"""

    summary = parse_speculative_summary(log)

    assert summary == {
        "steps": 7,
        "accepted": 190,
        "drafted": 252,
        "acceptance_rate": pytest.approx(0.754),
        "outputs": 256,
        "target_forwards": 66,
        "outputs_per_target_forward": pytest.approx(3.88),
    }
