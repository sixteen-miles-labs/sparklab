import json

import pytest
import torch
from safetensors.torch import save_file

from sparklab.models.qwen4_exp.ple import RawNGramStore, SafetensorNGramStore


@pytest.mark.parametrize("kind", ["raw", "safetensors"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_large_lookup_preserves_rows_scale_and_async_order(tmp_path, kind, dtype):
    rows, dim = 5003, 4
    values = ((torch.arange(rows * dim).reshape(rows, dim) % 81 - 40) / 4).to(dtype)
    if kind == "raw":
        data = values.view(torch.uint8).numpy().tobytes()
        (tmp_path / "table.bin").write_bytes(data)
        manifest = {"file": "table.bin", "dtype": str(dtype).removeprefix("torch."),
                    "rows": rows, "dim": dim, "nbytes": len(data), "weight_scale": .125}
        store = RawNGramStore(str(tmp_path), manifest, dim)
    else:
        prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        tensors = {prefix + ".shard_0.weight": values[:2501].contiguous(),
                   prefix + ".shard_1.weight": values[2501:].contiguous(),
                   prefix + ".weight_scale": torch.tensor([.125], dtype=torch.bfloat16)}
        save_file(tensors, tmp_path / "table.safetensors")
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {k: "table.safetensors" for k in tensors}}))
        store = SafetensorNGramStore(str(tmp_path), 2, dim)
    store._row_cache_capacity = 64  # exercise eviction between mixed cached/missing lookups
    torch.manual_seed(45)
    ids = torch.cat((torch.randperm(rows), torch.tensor([5002, 0, 2500, 2501, 3])))
    expected = (values.float() * .125).to(torch.bfloat16)[ids]
    try:
        torch.testing.assert_close(store.lookup(ids), expected, rtol=0, atol=0)
        torch.testing.assert_close(store.lookup_async(ids).result(), expected, rtol=0, atol=0)
        assert len(store._row_cache) <= 64
    finally:
        store.close()


def test_large_lookup_propagates_short_read_without_caching_partial_result(tmp_path):
    rows, dim = 5003, 4
    (tmp_path / "table.bin").write_bytes(bytes(rows * dim * 2 - 1))
    store = RawNGramStore(str(tmp_path), {
        "file": "table.bin", "dtype": "bfloat16", "rows": rows,
        "dim": dim, "nbytes": rows * dim * 2,
    }, dim)
    try:
        with pytest.raises(OSError, match="short n-gram row read"):
            store.lookup(torch.arange(rows))
        assert not store._row_cache
    finally:
        store.close()
