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
