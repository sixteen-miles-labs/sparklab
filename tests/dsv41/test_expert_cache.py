from types import SimpleNamespace

import torch

from sparklab.models.deepseek_v41.expert_cache import ExpertBank


def _metadata(args):
    result = {}
    for layer in range(2):
        for expert in range(3):
            root = f"layers.{layer}.ffn.experts.{expert}"
            for projection, shape in {
                "w1": (args.moe_inter_dim, args.dim // 2),
                "w3": (args.moe_inter_dim, args.dim // 2),
                "w2": (args.dim, args.moe_inter_dim // 2),
            }.items():
                result[root + f".{projection}.weight"] = (0, 0, shape, torch.uint8)
                result[root + f".{projection}.scale"] = (
                    0, 0, (shape[0], shape[1] // 16), torch.uint8
                )
    return result


def test_expert_bank_lru_and_gate_up_order(monkeypatch):
    args = SimpleNamespace(dim=64, moe_inter_dim=32, n_activated_experts=2,
                           n_routed_experts=3, n_layers=2)
    store = SimpleNamespace(device=torch.device("cpu"), metadata=_metadata(args))

    def read(name):
        shape = store.metadata[name][2]
        value = 3 if ".w3." in name else 2 if ".w2." in name else 1
        return torch.full(shape, value, dtype=torch.uint8)

    store._read_host = read
    monkeypatch.setenv("SPARKLAB_DSV41_EXPERT_CACHE_GB", "1")
    bank = ExpertBank(args, store)
    bank.capacity = 2
    assert bank.slots(0, torch.tensor([[2, 1]])).tolist() == [[0, 1]]
    assert torch.all(bank.gate_up[0, :32] == 1)
    assert torch.all(bank.gate_up[0, 32:] == 3)
    assert torch.all(bank.down[0] == 2)
    assert bank.slots(0, torch.tensor([[1, 0]])).tolist() == [[1, 0]]
    assert (0, 2) not in bank.entries


def test_expert_bank_deduplicates_batch_misses(monkeypatch):
    args = SimpleNamespace(dim=64, moe_inter_dim=32, n_activated_experts=2,
                           n_routed_experts=3, n_layers=2)
    store = SimpleNamespace(device=torch.device("cpu"), metadata=_metadata(args))
    calls = []

    def read(name):
        calls.append(name)
        return torch.zeros(store.metadata[name][2], dtype=torch.uint8)

    store._read_host = read
    monkeypatch.setenv("SPARKLAB_DSV41_EXPERT_CACHE_GB", "1")
    bank = ExpertBank(args, store)
    bank.capacity = 9
    slots = bank.slots(0, torch.tensor([[2, 1], [2, 1], [0, 2]]))
    assert slots.tolist() == [[0, 1], [0, 1], [2, 0]]
    assert len(calls) == 3 * 6


def test_expert_bank_does_not_evict_rows_used_by_current_batch(monkeypatch):
    args = SimpleNamespace(dim=64, moe_inter_dim=32, n_activated_experts=1,
                           n_routed_experts=3, n_layers=2)
    store = SimpleNamespace(device=torch.device("cpu"), metadata=_metadata(args))
    store._read_host = lambda name: torch.zeros(store.metadata[name][2], dtype=torch.uint8)
    monkeypatch.setenv("SPARKLAB_DSV41_EXPERT_CACHE_GB", "1")
    bank = ExpertBank(args, store)
    bank.capacity = 2
    assert bank.slots(0, torch.tensor([[0], [1]])).tolist() == [[0], [1]]
    slots = bank.slots(0, torch.tensor([[1], [2]]))
    assert slots.tolist() == [[1], [0]]
    assert bank.entries == {(0, 1): 1, (0, 2): 0}
