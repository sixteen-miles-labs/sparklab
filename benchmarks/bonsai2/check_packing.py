"""Check real checkpoint blocks against Prism's compiled CPU dequantizers."""

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch
from sparklab.kernels.triton.bonsai import embedding
from sparklab.models.gguf.reader import iter_gguf_tensors


def run(model, library):
    lib = ctypes.CDLL(str(library))
    names = {
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ffn_gate.weight",
        "blk.3.attn_q.weight",
    }
    rows = []
    for t in iter_gguf_tensors(str(model)):
        if t.name not in names:
            continue
        kind = t.ggml_type
        raw = t.packed()[:16].contiguous()
        width = t.shape[-1]
        ref = np.empty((16, width), dtype=np.float32)
        fn = getattr(lib, "dequantize_row_" + {142: "pq2_0", 143: "ptq1_0"}[kind])
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(raw.data_ptr(), ref.ctypes.data, ref.size)
        got = embedding(torch.arange(16, device="cuda"), raw.cuda(), kind, width)
        expected = torch.from_numpy(ref).to(device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        rows.append(
            {
                "tensor": t.name,
                "type": kind,
                "values": ref.size,
                "exact_bf16_match": True,
            }
        )
    assert len(rows) == len(names)
    return {"model": model.name, "reference_library": str(library), "checks": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.model, a.library)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
