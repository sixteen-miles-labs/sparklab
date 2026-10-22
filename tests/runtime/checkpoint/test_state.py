"""Copy/identity oracles for complete state images, not real-model quality gates."""

from types import SimpleNamespace

import pytest
import torch
from sparklab.models.config import LinearGatedDeltaGroupConfig
from sparklab.runtime.kvcache.mha_pool import MHAKVCache
from sparklab.runtime.kvcache.linear_state_pool import LinearStatePool
from sparklab.runtime.checkpoint.state import (
    Compatibility,
    MHARecurrentAdapter,
    eligible_endpoints,
)


def identity(**changes):
    config = dict(
        artifact="ftw-fingerprint",
        precision="bf16-fp32",
        model_config="config-hash",
        tokenizer="tokenizer-hash",
        template="template-hash",
        state_layout="mha-gdn-v1",
        cache_geometry={"page_size": 4},
        continuation="target-only",
        scope="single-instance-test",
    )
    return Compatibility.from_config({**config, **changes})


def pools(monkeypatch, device="cpu", page_size=4):
    monkeypatch.setattr(
        "sparklab.runtime.kvcache.mha_pool.get_tp_info", lambda: SimpleNamespace(size=1)
    )
    kv = MHAKVCache(1, 2, 4, 12, page_size, torch.bfloat16, torch.device(device))
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 1),
        num_key_heads=1,
        num_value_heads=1,
        key_head_dim=4,
        value_head_dim=4,
        conv_kernel_dim=3,
        output_gate=True,
    )
    linear = LinearStatePool(group, 8, torch.bfloat16, torch.device(device), tp_size=1)
    linear.ensure_aux_state("ple", (3,), dtype=torch.int64, fill=-1)
    return kv, linear, MHARecurrentAdapter(kv, linear)


def fill_state(kv, linear, slot, n):
    # Independently reconstruct the exact endpoint, not a rounded page boundary.
    kv._kv_buffer.copy_(
        torch.arange(kv._kv_buffer.numel(), device=kv.device).reshape(
            kv._kv_buffer.shape
        )
    )
    linear.conv_states[:, slot].fill_(n)
    linear.recurrent_states[:, slot].fill_(n * n)
    linear._aux_states["ple"][slot] = torch.tensor([n - 2, n - 1, n], device=kv.device)


@pytest.mark.parametrize("n", [1, 3, 4, 5, 7, 63, 64, 65])
def test_exact_copy_includes_tail_conv_recurrent_and_auxiliary_state(monkeypatch, n):
    # Change the pool page count for large endpoints while preserving actual P=4.
    kv, linear, adapter = pools(monkeypatch)
    if n >= 48:
        kv.rebuild(32)
    source, destination = linear.alloc(2)
    fill_state(kv, linear, source, n)
    mapping = torch.arange(n, dtype=torch.int32)
    image = adapter.snapshot(
        identity(), list(range(n)), mapping, source, max_bytes=1 << 20
    )
    expected = image.inspect()
    # Poison all destination state, including inactive rows and the boundary page.
    kv._kv_buffer.fill_(-99)
    linear.clear_slots([destination])
    target = torch.arange(n, dtype=torch.int32) + 4
    adapter.restore(image, identity(), target, destination)
    torch.testing.assert_close(
        kv._kv_buffer.flatten(2, 3).index_select(2, target.long()),
        expected["kv"],
        rtol=0,
        atol=0,
    )
    assert torch.equal(linear.conv_states[:, destination], expected["conv"])
    assert torch.equal(linear.recurrent_states[:, destination], expected["recurrent"])
    assert torch.equal(linear._aux_states["ple"][destination], expected["aux.ple"])
    assert kv._kv_buffer.flatten(2, 3)[:, :, 0].eq(-99).all()
    assert image.consumed == n and not image.supports_full_prompt_resume
    # A branch never aliases the shared parent image.
    linear.recurrent_states[:, destination].zero_()
    kv._kv_buffer.zero_()
    assert torch.equal(image.inspect()["kv"], expected["kv"])
    assert torch.equal(image.inspect()["recurrent"], expected["recurrent"])


def test_incompatibility_and_missing_components_fail_before_destination_mutation(
    monkeypatch,
):
    kv, linear, adapter = pools(monkeypatch)
    a, b = linear.alloc(2)
    fill_state(kv, linear, a, 5)
    image = adapter.snapshot(
        identity(), [1, 2, 3, 4, 5], torch.arange(5), a, max_bytes=1 << 20
    )
    kv._kv_buffer.fill_(77)
    for altered in [
        identity(scope="other"),
        identity(template="changed"),
        identity(artifact="other"),
    ]:
        with pytest.raises(ValueError, match="identity"):
            adapter.restore(image, altered, torch.arange(5), b)
        assert kv._kv_buffer.eq(77).all()
    linear.ensure_aux_state("new-state", (2,), dtype=torch.int64, fill=0)
    with pytest.raises(ValueError, match="layout"):
        adapter.restore(image, identity(), torch.arange(5), b)
    assert kv._kv_buffer.eq(77).all()


def test_insufficient_budget_skips_without_allocating_or_changing_state(monkeypatch):
    kv, linear, adapter = pools(monkeypatch)
    slot = linear.alloc(1)[0]
    kv._kv_buffer.fill_(19)
    assert (
        adapter.snapshot(identity(), [1], torch.tensor([0]), slot, max_bytes=0) is None
    )
    assert kv._kv_buffer.eq(19).all()
    with pytest.raises(ValueError, match="mapping"):
        adapter.snapshot(identity(), [1, 2], torch.tensor([0]), slot, max_bytes=1 << 20)
    with pytest.raises(ValueError, match="draft"):
        MHARecurrentAdapter(kv, linear, continuation="mtp")


def test_message_boundaries_require_exact_rendered_token_prefixes():
    assert eligible_endpoints(
        [1, 2, 3, 4], [[1], [1, 2], [1, 2], [1, 9], [1, 2, 3, 4, 5]], terminal=3
    ) == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        eligible_endpoints([1], terminal=2)
    with pytest.raises(ValueError):
        Compatibility.from_config({"artifact": "a"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer gate")
def test_cuda_transfer_preserves_exact_bytes_and_private_branches(monkeypatch):
    kv, linear, adapter = pools(monkeypatch, device="cuda")
    src, dst = linear.alloc(2)
    fill_state(kv, linear, src, 5)
    image = adapter.snapshot(
        identity(), [1, 2, 3, 4, 5], torch.arange(5), src, max_bytes=1 << 20
    )
    with torch.cuda.stream(torch.cuda.Stream()):
        adapter.restore(image, identity(), torch.arange(5) + 8, dst)
        torch.cuda.current_stream().synchronize()  # test observation, not production copy path
    assert torch.equal(linear.conv_states[:, src], linear.conv_states[:, dst])
    assert torch.equal(linear._aux_states["ple"][src], linear._aux_states["ple"][dst])
    assert torch.equal(
        kv._kv_buffer.flatten(2, 3)[:, :, :5], kv._kv_buffer.flatten(2, 3)[:, :, 8:13]
    )
