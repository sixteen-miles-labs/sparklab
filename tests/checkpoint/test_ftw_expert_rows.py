import pytest
import torch

from freetoken.checkpoint.ftw import FTWReader, FTWWriter, layer_bank_entry_name


def _write_ftw(path, tensors):
    writer = FTWWriter(str(path), shard_limit=4096 * 3)
    for name, tensor in tensors:
        writer.add_tensor(name, tensor, kind="experts_bank")
    writer.finalize({"expert_bank_num_layers": 2, "quant_format": "test"})


def test_per_layer_expert_rows_round_trip_across_shards(tmp_path):
    layers = []
    tensors = []
    for layer_id in range(2):
        value = torch.arange(5 * 777, dtype=torch.int32).view(5, 777) + layer_id * 100_000
        layers.append(value)
        tensors.append((layer_bank_entry_name("packed", layer_id), value))
    tensors.append(("gate_up_alpha", torch.ones(10, dtype=torch.float32)))
    _write_ftw(tmp_path, tensors)

    reader = FTWReader(str(tmp_path))
    try:
        index = reader.expert_row_descriptors(num_layers=2)
        assert len(index) == 2 * 5
        assert all(key[2] == "packed" for key in index)
        for layer_id in range(2):
            for expert_id in range(5):
                desc = index[(layer_id, expert_id, "packed")]
                assert desc.read_off % 4096 == 0
                assert desc.read_nbytes % 4096 == 0
                assert 0 <= desc.head_pad < 4096
                assert torch.equal(reader.read_expert_row(desc), layers[layer_id][expert_id])
    finally:
        reader.close()


def test_flat_expert_rows_round_trip(tmp_path):
    flat = torch.arange(2 * 3 * 513, dtype=torch.int16).view(6, 513)
    _write_ftw(tmp_path, [("weight", flat)])

    reader = FTWReader(str(tmp_path))
    try:
        index = reader.expert_row_descriptors(num_layers=2)
        assert len(index) == 6
        for layer_id in range(2):
            for expert_id in range(3):
                got = reader.read_expert_row(index[(layer_id, expert_id, "weight")])
                assert torch.equal(got, flat[layer_id * 3 + expert_id])
    finally:
        reader.close()


def test_expert_row_index_rejects_missing_layer(tmp_path):
    value = torch.zeros(2, 16, dtype=torch.uint8)
    _write_ftw(tmp_path, [(layer_bank_entry_name("packed", 0), value)])

    reader = FTWReader(str(tmp_path))
    try:
        with pytest.raises(ValueError, match=r"expected \[0, 1\]"):
            reader.expert_row_descriptors(num_layers=2)
    finally:
        reader.close()
