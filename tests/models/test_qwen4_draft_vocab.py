import pytest
import torch

from sparklab.models.qwen4_exp.draft_vocab import DraftVocabulary, draft_token_ids


def test_budget_preserves_control_tokens_and_has_no_duplicates():
    assert draft_token_ids(100, 6, [0, 98, 99, 99]) == [0, 1, 2, 3, 98, 99]
    assert draft_token_ids(8, 8, [0, 7]) == list(range(8))
    assert draft_token_ids(8, 2, [6, 7]) == [6, 7]


@pytest.mark.parametrize("budget,required", [(0, []), (11, []), (2, [1, 2, 3]), (5, [-1]), (5, [10])])
def test_invalid_budget_or_required_token_fails_closed(budget, required):
    with pytest.raises(ValueError):
        draft_token_ids(10, budget, required)


@pytest.mark.parametrize("with_bias", [False, True])
def test_reduced_head_maps_ids_and_never_mutates_target(with_bias):
    torch.manual_seed(91)
    weight = torch.randn(16, 8)
    original = weight.clone()
    bias = torch.randn(16) if with_bias else None
    ids = draft_token_ids(16, 7, [14, 15])
    head = DraftVocabulary(weight, bias, ids)
    hidden = torch.randn(3, 8)
    full_logits = torch.nn.functional.linear(hidden, weight, bias)
    expected = torch.tensor(ids)[full_logits[:, ids].argmax(-1)]
    torch.testing.assert_close(head.select(hidden), expected, rtol=0, atol=0)
    torch.testing.assert_close(weight, original, rtol=0, atol=0)
    assert head.weight.data_ptr() != weight.data_ptr()


def test_excluded_winner_only_changes_draft_not_target():
    weight = torch.arange(10, dtype=torch.float32).view(10, 1)
    head = DraftVocabulary(weight, None, [0, 1, 2])
    hidden = torch.ones(1, 1)
    assert head.select(hidden).item() == 2
    assert torch.nn.functional.linear(hidden, weight).argmax(-1).item() == 9


def test_model_draft_selector_defaults_to_full_head():
    from types import SimpleNamespace
    from sparklab.models.qwen4_exp.model import Qwen4ExpForCausalLM

    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    weight = torch.arange(10, dtype=torch.float32).view(10, 1)
    model.lm_head = SimpleNamespace(weight=weight, bias=None, tied_embedding=None)
    model._draft_vocab = None
    hidden = torch.ones(1, 1)
    assert model._select_draft_token(hidden).item() == 9
    model._draft_vocab = DraftVocabulary(weight, None, [0, 1, 2])
    assert model._select_draft_token(hidden).item() == 2
    assert torch.nn.functional.linear(hidden, model.lm_head.weight).argmax(-1).item() == 9
