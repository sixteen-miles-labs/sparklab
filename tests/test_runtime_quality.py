import pytest
import torch

from benchmarks.runtime_quality import GreedyLogitTrace, first_divergence


def test_diagnostic_exposes_first_rank_change_and_exact_ties():
    left, right = GreedyLogitTrace(), GreedyLogitTrace()
    left.observe(
        torch.tensor([[0.0, 4.0, 1.0], [2.0, 2.0, 0.0]]),
        torch.tensor([1, 0]),
        phase="decode",
    )
    right.observe(
        torch.tensor([[0.0, 4.0, 1.0], [2.0, 2.25, 0.0]]),
        torch.tensor([1, 1]),
        phase="verify",
    )
    left.validate_output([1, 0])
    right.validate_output([1, 1])
    result = first_divergence(left, right)
    assert result["position"] == result["equal_preceding_output_tokens"] == 1
    assert result["reference"]["maximum_ties"] == 2
    assert result["reference"]["top_margin"] == 0
    assert result["candidate"]["maximum_ties"] == 1
    assert result["candidate"]["top_margin"] == 0.25
    assert result["reference"]["logits_sha256"] != result["candidate"]["logits_sha256"]


def test_diagnostic_is_bounded_and_checks_emission_accounting():
    trace = GreedyLogitTrace(limit=2)
    trace.observe(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]), phase="verify"
    )
    with pytest.raises(ValueError, match="bound exceeded"):
        trace.observe(torch.tensor([[1.0, 0.0]]), torch.tensor([0]), phase="decode")
    with pytest.raises(ValueError, match="completed output"):
        trace.validate_output([1, 0])
    assert len(trace.tokens) == 2
    assert first_divergence(trace, trace) is None


@pytest.mark.parametrize(
    "values,ids",
    [
        ([[0.0, float("nan")]], [0]),
        ([[1.0, 0.0]], [1]),
        ([[1.0, 0.0]], [9]),
        ([[1.0, 0.0]], [0, 0]),
    ],
)
def test_invalid_evidence_cannot_train_or_pass_as_a_diagnostic(values, ids):
    trace = GreedyLogitTrace()
    with pytest.raises(ValueError):
        trace.observe(torch.tensor(values), torch.tensor(ids), phase="verify")
    assert trace.tokens == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("drafts", [[1, 2], [1, 0], [0, 2]])
def test_actual_greedy_verifier_records_only_the_emitted_prefix(device, drafts):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA diagnostic transfer gate")
    from sparklab.core import SamplingParams
    from sparklab.runtime.engine.engine import _verify_speculative_tokens

    logits = torch.tensor(
        [[0.0, 4.0, 1.0], [0.0, 1.0, 3.0], [2.0, 1.0, 0.0]], device=device
    )
    proposed = torch.tensor(drafts, dtype=torch.int32, device=device)
    accepted, chosen = _verify_speculative_tokens(
        logits, proposed, None, SamplingParams(temperature=0.0)
    )
    trace = GreedyLogitTrace(limit=3)
    before = logits.clone()
    trace.observe(logits, chosen, phase="verify")
    expected = [1, 2, 0] if drafts == [1, 2] else [1, 2] if drafts == [1, 0] else [1]
    trace.validate_output(expected)
    assert len(trace.tokens) == accepted + 1
    assert torch.equal(logits, before)
    logits.fill_(-99)
    assert trace.tokens[0]["top_logits"][0] == [1, 4.0]


def test_rejected_diagnostic_is_atomic_and_rows_do_not_alias():
    trace = GreedyLogitTrace()
    with pytest.raises(ValueError, match="selection disagrees"):
        trace.observe(
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]), torch.tensor([0, 0]), phase="verify"
        )
    assert trace.tokens == []
