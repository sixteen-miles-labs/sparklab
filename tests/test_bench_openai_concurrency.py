from benchmarks.bench_openai_concurrency import RequestResult, _percentile, _run_trial


class _FakeClient:
    def generate(self, prompt: str, output_tokens: int) -> RequestResult:
        assert "Benchmark request" in prompt
        return RequestResult(
            prompt_tokens=10,
            completion_tokens=output_tokens,
            ttft_seconds=0.1,
            elapsed_seconds=0.2,
            output_sha256="hash",
        )


def test_percentile_interpolates() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 3.8499999999999996


def test_run_trial_reports_aggregate_requests() -> None:
    result = _run_trial(_FakeClient(), concurrency=4, trial=2, output_tokens=32)

    assert result["completion_tokens"] == 128
    assert result["prompt_tokens"] == 40
    assert len(result["requests"]) == 4
    assert result["aggregate_output_tokens_per_second"] > 0
    assert result["ttft_p95_seconds"] == 0.1
    assert result["e2e_p95_seconds"] == 0.2
