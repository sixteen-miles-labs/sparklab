"""Endpoint eligibility follows actual rendered tokens, never role/character offsets."""

import torch
import pytest
from jinja2.exceptions import TemplateError
from sparklab.core import SamplingParams
from sparklab.message import TokenizeMsg
from sparklab.tokenizer.tokenize import TokenizeManager


class Tokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, messages, **_):
        tags = {"user": "U", "assistant": "A", "system": "S"}
        return "".join(tags[m["role"]] + m["content"] for m in messages) + "A"

    def encode(self, text, return_tensors=None, **_):
        ids = [ord(c) for c in text]
        return torch.tensor([ids]) if return_tensors else ids


def test_only_exact_prefix_of_final_render_is_forwarded(monkeypatch):
    monkeypatch.setenv("SPARKLAB_EXACT_PREFIX", "1")
    manager = TokenizeManager(Tokenizer())
    msg = TokenizeMsg(
        1,
        [
            {"role": "user", "content": "abc"},
            {"role": "assistant", "content": "def"},
            {"role": "user", "content": "ghi"},
        ],
        SamplingParams(),
    )
    full = manager.tokenize([msg])[0].tolist()
    assert msg.checkpoint_endpoints == (5, len(full))
    # The assistant-boundary render ends in a fresh assistant header; it is NOT
    # a prefix of the complete render, which continues with a user header.
    assert 9 not in msg.checkpoint_endpoints


@pytest.mark.parametrize("error", [TemplateError, ValueError])
def test_optional_system_only_boundary_cannot_break_a_valid_full_request(monkeypatch, error):
    class RequiresUser(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if not any(m["role"] == "user" for m in messages):
                raise error("No user query found in messages.")
            return super().apply_chat_template(messages, **kwargs)

    manager = TokenizeManager(RequiresUser())
    text = [{"role":"system", "content":"rules"}, {"role":"user", "content":"fact"}]
    monkeypatch.setenv("SPARKLAB_EXACT_PREFIX", "0")
    cold = manager.tokenize([TokenizeMsg(1, text, SamplingParams())])[0]
    monkeypatch.setenv("SPARKLAB_EXACT_PREFIX", "1")
    message = TokenizeMsg(2, text, SamplingParams())
    hot = manager.tokenize([message])[0]
    assert torch.equal(cold, hot)
    assert message.checkpoint_endpoints == (len(hot),)
    # Invalid complete requests must still fail, including when cache is off.
    for enabled in ("0", "1"):
        monkeypatch.setenv("SPARKLAB_EXACT_PREFIX", enabled)
        with pytest.raises(error, match="No user query"):
            manager.tokenize([TokenizeMsg(3, text[:1], SamplingParams())])


def test_partial_probe_does_not_hide_runtime_failures(monkeypatch):
    class FailingProbe(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if len(messages) == 1:
                raise RuntimeError("unexpected renderer failure")
            return super().apply_chat_template(messages, **kwargs)

    monkeypatch.setenv("SPARKLAB_EXACT_PREFIX", "1")
    manager = TokenizeManager(FailingProbe())
    message = TokenizeMsg(1, [{"role":"system","content":"rules"},
        {"role":"user","content":"fact"}], SamplingParams())
    with pytest.raises(RuntimeError, match="unexpected renderer"):
        manager.tokenize([message])
