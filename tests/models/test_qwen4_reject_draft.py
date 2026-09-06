from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sparklab.models.qwen4_exp.model import Qwen4ExpForCausalLM


@pytest.mark.parametrize("width", [3, 4, 5])
@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize("qwen4", [False, True])
def test_optimizations_keep_supported_draft_width_limit(
    monkeypatch, width, experimental, qwen4
):
    from sparklab.runtime.engine.engine import _adjust_speculative_config
    from tests.models.test_qwen4_exp import _config, parse_config

    monkeypatch.setenv("SPARKLAB_QWEN4_MTP4", "1" if experimental else "0")
    model_config = parse_config(_config())
    if not qwen4:
        object.__setattr__(model_config, "qwen4_exp_args", None)
        object.__setattr__(model_config, "mtp_num_hidden_layers", 1)
    allowed = width <= (4 if qwen4 and experimental else 3)
    config = SimpleNamespace(
        model_config=model_config,
        speculative_tokens=width,
        max_running_req=1,
        cache_type="radix",
        cuda_graph_bs=[],
        cuda_graph_max_bs=0,
    )

    def adjust():
        _adjust_speculative_config(
            config, lambda name, value: setattr(config, name, value)
        )

    if allowed:
        adjust()
        assert config.model_config.speculative_tokens == width
    else:
        with pytest.raises(ValueError, match="MTP supports at most"):
            adjust()


@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
@pytest.mark.parametrize("raises", [False, True])
def test_prefix_proposal_hides_rejected_tokens_and_restores_features(
    monkeypatch, accepted, raises
):
    req = SimpleNamespace(cached_len=7, device_len=11, max_device_len=64)
    batch = SimpleNamespace(
        reqs=[req],
        input_ids=torch.tensor([10, 20, 30, 40]),
        positions=torch.arange(7, 11),
        out_loc=torch.arange(17, 21),
    )
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    hidden = torch.arange(24).reshape(4, 6)
    model._mtp_target_hidden = hidden
    prepared = []
    backend = SimpleNamespace(prepare_metadata=lambda b: prepared.append(b))
    monkeypatch.setattr(
        "sparklab.models.qwen4_exp.model.get_global_ctx",
        lambda: SimpleNamespace(attn_backend=backend),
    )
    correction = torch.tensor([99])

    def propose(prefix, token):
        assert prefix is prepared[0]
        assert prefix.reqs[0] is not req
        assert prefix.reqs[0].cached_len == 7
        assert prefix.reqs[0].device_len == 7 + accepted
        assert prefix.reqs[0].max_device_len == 64
        assert prefix.input_ids.tolist() == batch.input_ids[:accepted].tolist()
        assert prefix.positions.tolist() == list(range(7, 7 + accepted))
        assert prefix.out_loc.tolist() == list(range(17, 17 + accepted))
        assert torch.equal(model._mtp_target_hidden, hidden[:accepted])
        assert token is correction
        if raises:
            raise RuntimeError("test failure")
        return token

    model.propose_mtp = propose
    if raises:
        with pytest.raises(RuntimeError, match="test failure"):
            model.propose_mtp_prefix(batch, correction, accepted)
    else:
        assert model.propose_mtp_prefix(batch, correction, accepted) is correction
    assert model._mtp_target_hidden is hidden
    assert req.device_len == 11
    assert batch.input_ids.numel() == 4


@pytest.mark.parametrize("accepted", [0, 5])
def test_invalid_accepted_prefix_fails_closed(accepted):
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    with pytest.raises(ValueError):
        model.propose_mtp_prefix(
            SimpleNamespace(input_ids=torch.zeros(4)), torch.tensor([1]), accepted
        )


@pytest.mark.parametrize("tied", [False, True])
def test_fp8_draft_head_does_not_replace_target_weights(monkeypatch, tied):
    monkeypatch.setenv("SPARKLAB_QWEN4_DRAFT_HEAD", "fp8")
    monkeypatch.setenv("SPARKLAB_QWEN4_DRAFT_VOCAB_SIZE", "0")
    original = torch.arange(48, dtype=torch.bfloat16).reshape(8, 6) / 48
    head = SimpleNamespace(weight=original.clone(), tp_size=1)
    if tied:
        lm_head = SimpleNamespace(tied_embedding=head, bias=None)
    else:
        lm_head = head
        lm_head.tied_embedding = None
        lm_head.bias = None
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model.lm_head = lm_head
    model.model = SimpleNamespace(embed_tokens=head)
    model._mtp = SimpleNamespace(load_sidecar=lambda *_: None)
    model._mtp_path = "unused-test-sidecar"
    model._draft_vocab = None
    model._draft_fp8_head = None
    model.prepare_for_runtime()
    weight, scale = model._draft_fp8_head
    assert weight.dtype == torch.float8_e4m3fn
    assert weight.shape == original.shape
    assert scale.shape == (original.shape[0],)
    assert weight.data_ptr() != head.weight.data_ptr()
    assert torch.equal(head.weight, original)
    assert head.weight.dtype == torch.bfloat16

    def reference(hidden, actual_weight, actual_scale, bias):
        assert actual_weight is weight and actual_scale is scale and bias is None
        return hidden.float() @ (weight.float() * scale[:, None]).T

    monkeypatch.setattr(
        "sparklab.kernels.triton.fp8_pertensor_linear.fp8_pertensor_linear", reference
    )
    assert model._select_draft_token(torch.ones(1, 6)).tolist() == [7]
    assert torch.equal(head.weight, original)


def test_mtp_proposal_does_not_override_sampled_requests(monkeypatch):
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._mtp = object()
    model._mtp_target_hidden = torch.ones(1, 6)
    batch = SimpleNamespace(
        size=1,
        reqs=[
            SimpleNamespace(
                sampling_params=SimpleNamespace(is_greedy=False),
            )
        ],
    )
    monkeypatch.setattr(
        "sparklab.models.qwen4_exp.model.get_global_ctx", lambda: object()
    )
    assert model.propose_mtp(batch, torch.tensor([1])) is None


@pytest.mark.parametrize(
    "width,accepted", [(w, a) for w in (3, 4) for a in range(w + 1)]
)
@pytest.mark.parametrize("mode", ["enabled", "disabled", "missing_hook", "replay"])
@pytest.mark.parametrize("light_snapshot", [False, True])
def test_engine_redraft_preserves_verified_outputs_and_state(
    monkeypatch, width, accepted, mode, light_snapshot
):
    """Exercise the real verifier/engine branch, stubbing only device execution."""
    from sparklab.runtime.engine.engine import Engine

    monkeypatch.setenv(
        "SPARKLAB_QWEN4_REJECT_DRAFT", "0" if mode == "disabled" else "1"
    )
    monkeypatch.setenv(
        "SPARKLAB_QWEN4_LIGHT_VERIFY_SNAPSHOT", "1" if light_snapshot else "0"
    )
    engine = Engine.__new__(Engine)
    engine.stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: engine.stream)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda _: None)
    )
    engine.config = SimpleNamespace(
        speculative_tokens=width,
        speculative_method="mtp",
        model_config=SimpleNamespace(qwen4_exp_args=object()),
    )
    engine.ctx = SimpleNamespace(forward_batch=lambda _: nullcontext())
    engine.cpu_moe_executor = None
    engine.mtp_graph_runner = None
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda _: False)
    engine.mtp_stats = dict(
        target_forwards=0,
        drafted=0,
        accepted=0,
        outputs=0,
        replay_calls=0,
        replay_tokens=0,
    )
    events = []
    engine.linear_state_pool = SimpleNamespace(
        num_slots=4,
        verify_steps=0 if mode == "replay" else width + 1,
        copy_from=lambda *args: events.append(("copy", *args)),
        snapshot_verify_inputs=lambda *args: events.append(("light", *args)),
        commit_verify_prefix=lambda *args: events.append(("commit", *args)),
    )
    req = SimpleNamespace(
        table_idx=1,
        cached_len=7,
        device_len=7 + width + 1,
        linear_slot_idx=None,
        mamba_ping_pong=None,
        speculative_draft_probs=None,
        sampling_params=SimpleNamespace(is_greedy=True),
        can_decode=True,
    )
    batch = SimpleNamespace(
        is_prefill=False,
        is_verify=True,
        size=1,
        reqs=[req],
        verify_cached_lens=(7,),
        input_ids=torch.arange(1, width + 2) * 10,
        cache_verify_states=False,
    )
    truth = [20, 30, 40, 50][:width] + [99]
    truth[accepted] = 99
    logits = torch.zeros(width + 1, 100)
    logits[torch.arange(width + 1), torch.tensor(truth)] = 1
    proposals = torch.arange(51, 51 + width, dtype=torch.int32)

    def prefix_propose(active_batch, correction, length):
        assert active_batch is batch
        assert correction.tolist() == [99]
        assert length == accepted + 1
        # Recurrent state must be committed before generating new drafts.
        assert ("commit", 3, 1, length) in events
        events.append(("redraft", length))
        return proposals

    engine.model = SimpleNamespace(
        begin_external_inputs=lambda _: None, forward=lambda: logits
    )
    if mode != "missing_hook":
        engine.model.propose_mtp_prefix = prefix_propose
    engine._propose_speculative = lambda *_: proposals
    engine._replay_verified_prefix = lambda *args: events.append(("replay", args[1]))

    output = engine.forward_batch(batch, None)
    assert events[0] == (
        "light" if light_snapshot and mode != "replay" else "copy",
        1,
        3,
    )
    expected = [20, 30, 40, 50][:accepted] + [99]
    assert output.next_tokens_cpu.tolist() == expected
    assert output.next_tokens_gpu.tolist() == expected
    assert output.token_counts == (accepted + 1,)
    assert output.managed_writes
    assert req.cached_len == 7 + accepted + 1
    assert req.device_len == req.cached_len + 1
    assert batch.speculative_write[:2] == (1, req.cached_len)
    assert int(batch.speculative_write[2]) == 99
    assert engine.mtp_stats["accepted"] == accepted
    assert engine.mtp_stats["outputs"] == accepted + 1
    should_redraft = accepted < width and mode == "enabled"
    assert any(event[0] == "redraft" for event in events) == should_redraft
    if should_redraft or accepted == width:
        assert req.speculative_drafts is proposals
    else:
        assert req.speculative_drafts is None
    assert engine.mtp_stats["replay_calls"] == int(
        mode == "replay" and accepted < width
    )
