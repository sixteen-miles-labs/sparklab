from __future__ import annotations

from types import SimpleNamespace

import torch

import freetoken.models.qwen3_5_moe.weight as weight


class _FakeBank:
    def __init__(self, shape, dtype, state):
        self.tensor = torch.empty(shape, dtype=dtype)
        self.nbytes = self.tensor.numel() * self.tensor.element_size()
        self._state = state
        self._released = False
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])

    def release(self):
        if not self._released:
            self._released = True
            self._state["live"] -= 1


class _FakeReader:
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.closed = False

    def get(self, name):
        if name.endswith("weight_scale_inv"):
            shape = (
                (self.intermediate_size // 128, self.hidden_size // 128)
                if ".down_proj." not in name
                else (self.hidden_size // 128, self.intermediate_size // 128)
            )
            return torch.ones(shape, dtype=torch.bfloat16)
        shape = (
            (self.intermediate_size, self.hidden_size)
            if ".down_proj." not in name
            else (self.hidden_size, self.intermediate_size)
        )
        return torch.zeros(shape, dtype=torch.float8_e4m3fn)

    def close(self):
        self.closed = True


def test_serial_fp8_conversion_allocates_and_releases_one_layer_at_a_time(monkeypatch):
    layers, experts, hidden, intermediate, dense = 3, 1, 128, 128, 2
    state = {"live": 0, "peak": 0, "allocations": 0}
    reader = _FakeReader(hidden, intermediate)

    def alloc_banks(specs):
        state["allocations"] += 1
        return {
            name: _FakeBank(shape, dtype, state)
            for name, (shape, dtype) in specs.items()
        }

    monkeypatch.setattr(weight, "_moe_dims", lambda config: (
        layers, experts, hidden, intermediate, dense
    ))
    monkeypatch.setattr(weight, "_expert_reader", lambda *args: reader)
    monkeypatch.setattr(
        weight,
        "get_tp_info",
        lambda: SimpleNamespace(size=1, is_primary=lambda: True),
    )
    monkeypatch.setattr("freetoken.moe.host_banks.alloc_banks", alloc_banks)

    seen = []

    def sink(layer_id, banks):
        # A later layer must not be allocated until the previous layer has been handed off.
        assert state["live"] == len(banks) == 4
        seen.append(layer_id)
        for bank in banks.values():
            bank.release()

    sources = weight._build_fp8_expert_banks(
        "unused",
        SimpleNamespace(fp8_block_scale_dtype="bfloat16"),
        dummy=False,
        parallel=False,
        pin=True,
        layer_sink=sink,
    )

    assert seen == [0, 1, 2]
    assert state == {"live": 0, "peak": 4, "allocations": layers}
    assert sources == {
        "gate_up": [],
        "gate_up_scale": [],
        "down": [],
        "down_scale": [],
    }
    assert reader.closed


def test_serial_fp8_conversion_releases_partial_layer_when_reader_fails(monkeypatch):
    state = {"live": 0, "peak": 0, "allocations": 0}

    class FailingReader(_FakeReader):
        def get(self, name):
            raise RuntimeError("broken shard")

    reader = FailingReader(128, 128)

    def alloc_banks(specs):
        state["allocations"] += 1
        return {
            name: _FakeBank(shape, dtype, state)
            for name, (shape, dtype) in specs.items()
        }

    monkeypatch.setattr(weight, "_moe_dims", lambda config: (2, 1, 128, 128, 0))
    monkeypatch.setattr(weight, "_expert_reader", lambda *args: reader)
    monkeypatch.setattr(
        weight,
        "get_tp_info",
        lambda: SimpleNamespace(size=1, is_primary=lambda: True),
    )
    monkeypatch.setattr("freetoken.moe.host_banks.alloc_banks", alloc_banks)

    try:
        weight._build_fp8_expert_banks(
            "unused",
            SimpleNamespace(fp8_block_scale_dtype="bfloat16"),
            dummy=False,
            parallel=False,
            pin=True,
            layer_sink=lambda *_: None,
        )
    except RuntimeError as exc:
        assert str(exc) == "broken shard"
    else:
        raise AssertionError("expected the fake shard failure")

    assert state == {"live": 0, "peak": 4, "allocations": 1}
    assert reader.closed
