"""P1 unit: LinearStatePool free-list allocator (alloc/free/clear_slots/copy_from).
CPU-only, fast — pure slot bookkeeping + state copy/zero, no kernels."""
from __future__ import annotations

import pytest
import torch

from sparklab.runtime.kvcache.linear_state_pool import LinearStatePool
from sparklab.models.config import LinearGatedDeltaGroupConfig


def _pool(num_slots=8, device="cpu"):
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1),
        num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    return LinearStatePool(group=group, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device(device), tp_size=1)


def test_alloc_free_roundtrip():
    pool = _pool(num_slots=8)
    assert pool.num_free_slots == 7          # slots 1..7 (slot 0 = padding)
    a = pool.alloc(3)
    assert len(set(a)) == 3 and all(1 <= s <= 7 for s in a)
    assert pool.padding_slot not in a        # slot 0 never allocated
    assert pool.num_free_slots == 4
    pool.free(a)
    assert pool.num_free_slots == 7
    # int and tensor free forms
    s = pool.alloc(1)[0]
    pool.free(s)
    s2 = pool.alloc(2)
    pool.free(torch.tensor(s2, dtype=torch.long))
    assert pool.num_free_slots == 7


def test_alloc_exhaustion_raises():
    pool = _pool(num_slots=4)                # 3 allocatable
    pool.alloc(3)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.alloc(1)


def test_clear_slots_zeros_all_layers():
    pool = _pool(num_slots=6)
    s = pool.alloc(1)[0]
    pool.conv_states[:, s] = 1.5
    pool.recurrent_states[:, s] = 2.0
    pool.clear_slots([s])
    assert pool.conv_states[:, s].abs().sum() == 0
    assert pool.recurrent_states[:, s].abs().sum() == 0


def test_copy_from_snapshot():
    pool = _pool(num_slots=6)
    src, dst = pool.alloc(2)
    torch.manual_seed(0)
    pool.conv_states[:, src] = torch.randn_like(pool.conv_states[:, src])
    pool.recurrent_states[:, src] = torch.randn_like(pool.recurrent_states[:, src])
    pool.copy_from(src, dst)
    assert torch.equal(pool.conv_states[:, dst], pool.conv_states[:, src])
    assert torch.equal(pool.recurrent_states[:, dst], pool.recurrent_states[:, src])


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
def test_light_snapshot_commit_never_reads_snapshot_recurrent_state(length):
    pool = _pool(num_slots=4)
    pool.enable_verify_transactions(5)
    src, dst = 1, 2
    torch.manual_seed(25)
    pool.conv_states.normal_()
    pool.recurrent_states.normal_()
    aux = pool.ensure_aux_state("ple", (2, 7), torch.bfloat16)
    aux.normal_()
    aux_inputs = pool.ensure_aux_verify_inputs("ple")
    aux_inputs.normal_()
    pool.verify_recurrent_states.normal_()
    pool.verify_conv_inputs.normal_()
    conv_before = pool.conv_states[:, src].clone()
    aux_before = aux[src].clone()
    rec_before = pool.recurrent_states[:, src].clone()
    pool.recurrent_states[:, dst].fill_(float("nan"))

    pool.snapshot_verify_inputs(src, dst)
    assert torch.equal(pool.conv_states[:, dst], conv_before)
    assert torch.equal(aux[dst], aux_before)
    assert torch.isnan(pool.recurrent_states[:, dst]).all()
    assert torch.equal(pool.recurrent_states[:, src], rec_before)
    pool.conv_states[:, src].fill_(99)
    aux[src].fill_(99)
    pool.commit_verify_prefix(dst, src, length)

    expected_conv = torch.cat((conv_before, pool.verify_conv_inputs[:, :length].transpose(1, 2)), -1)
    expected_aux = torch.cat((aux_before, aux_inputs[:length].T), -1)
    assert torch.equal(pool.conv_states[:, src], expected_conv[..., -conv_before.shape[-1]:])
    assert torch.equal(aux[src], expected_aux[..., -aux_before.shape[-1]:])
    assert torch.equal(pool.recurrent_states[:, src], pool.verify_recurrent_states[:, 0, length - 1])


def test_light_snapshot_requires_transactions():
    with pytest.raises(RuntimeError, match="prefix transactions"):
        _pool().snapshot_verify_inputs(1, 2)


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_commit_verify_prefix(length):
    pool = _pool(num_slots=6)
    snapshot, live = pool.alloc(2)
    pool.enable_verify_transactions(4)
    pool.conv_states[:, snapshot] = torch.arange(
        pool.conv_states[:, snapshot].numel(), dtype=torch.bfloat16
    ).reshape_as(pool.conv_states[:, snapshot])
    pool.verify_conv_inputs.copy_(
        torch.arange(pool.verify_conv_inputs.numel(), dtype=torch.bfloat16).reshape_as(
            pool.verify_conv_inputs
        )
        + 1000
    )
    pool.verify_recurrent_states.copy_(
        torch.arange(
            pool.verify_recurrent_states.numel(), dtype=pool.recurrent_states.dtype
        ).reshape_as(pool.verify_recurrent_states)
    )

    old = pool.conv_states[:, snapshot].clone()
    inputs = pool.verify_conv_inputs.clone()
    expected_conv = torch.cat((old.transpose(1, 2), inputs), dim=1)[
        :, length : length + old.shape[-1]
    ].transpose(1, 2)
    pool.commit_verify_prefix(snapshot, live, length)

    assert torch.equal(pool.conv_states[:, live], expected_conv)
    assert torch.equal(
        pool.recurrent_states[:, live], pool.verify_recurrent_states[:, 0, length - 1]
    )


if __name__ == "__main__":
    test_alloc_free_roundtrip()
    test_alloc_exhaustion_raises()
    test_clear_slots_zeros_all_layers()
    test_copy_from_snapshot()
    print("LinearStatePool allocator unit: PASS")


@pytest.mark.parametrize("consumer", ["copy", "free", "clear", "reset", "view", "owner", "rebuild"])
def test_deferred_commit_materializes_before_other_consumers(consumer):
    pool = _pool()
    pool.enable_verify_transactions(5)
    pool.enable_deferred_verify_commits()
    pool.verify_recurrent_states.normal_()
    pool.verify_conv_inputs.zero_()
    expected = pool.verify_recurrent_states[:, 0, 2].clone()
    pool.commit_verify_prefix(2, 1, 3)
    assert pool._pending_verify_state == (1, 2)
    assert not torch.equal(pool.recurrent_states[:, 1], expected)
    if consumer == "copy": pool.copy_from(1, 3)
    elif consumer == "free": pool.free(1)
    elif consumer == "clear": pool.clear_slots([1])
    elif consumer == "reset": pool.reset(1)
    elif consumer == "view": pool.recurrent_state(0, 1)
    elif consumer == "owner": pool.prepare_verify_state(3)
    else: pool.rebuild(10)
    assert pool._pending_verify_state is None
    assert pool.cached_initial_state_step.item() == -1
    if consumer in {"clear", "reset", "rebuild"}:
        assert pool.recurrent_states[:, 1].count_nonzero() == 0
    else:
        torch.testing.assert_close(pool.recurrent_states[:, 1], expected, atol=0, rtol=0)
    if consumer == "copy":
        torch.testing.assert_close(pool.recurrent_states[:, 3], expected, atol=0, rtol=0)
