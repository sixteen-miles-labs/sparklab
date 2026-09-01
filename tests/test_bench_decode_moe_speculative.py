import pytest

from benchmarks.bench_decode_moe import (
    parse_args,
    parse_speculative_summary,
    serve_cmd,
)


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
