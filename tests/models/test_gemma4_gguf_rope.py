"""gemma4's full-attention layers are partial-rotary. llama.cpp does not say so in the
metadata -- it writes rope.dimension_count = full head_dim and carries the factor as the
rope_freqs.weight divisor -- so the GGUF config path has to recover it, or the tail dims
that must be identity get rotated (5 of 30 layers, error growing with position)."""
from __future__ import annotations

import glob
import os

import pytest
import torch

GGUF_GLOB = os.environ.get("SPARKLAB_GEMMA4_GGUF_GLOB", "")


def _shim(path):
    from sparklab.models.gguf.config import GgufConfigShim
    from sparklab.models.gguf.reader import load_gguf_metadata

    md = load_gguf_metadata(path)
    return GgufConfigShim(architectures=["Gemma4GGUFForCausalLM"], model_path=path,
                          model_type="gemma4", metadata=md, vocab_size=262144,
                          tie_word_embeddings=True), md


@pytest.mark.skipif(not glob.glob(GGUF_GLOB), reason="SPARKLAB_GEMMA4_GGUF_GLOB not set to a local gemma4 GGUF")
def test_full_rotary_dim_recovered_from_rope_freqs():
    from sparklab.models.gemma4.gguf import _full_rotary_dim

    shim, md = _shim(glob.glob(GGUF_GLOB)[0])
    head_dim = int(md["gemma4.attention.key_length"])
    # The metadata alone would say "rotate everything"; rope_freqs says otherwise.
    assert int(md["gemma4.rope.dimension_count"]) == head_dim
    assert _full_rotary_dim(shim, head_dim) == head_dim // 4


def test_proportional_rope_matches_llama_cpp_rope_freqs_division():
    """ggml applies theta/rope_freqs[j] (ops.cpp), with 1e30 zeroing the unrotated tail.
    SparkLab's "proportional" branch must produce the same table."""
    from sparklab.layers.rotary import get_rope

    head_dim, n_rot, base, max_position = 512, 128, 1_000_000.0, 512
    rope = get_rope(head_dim=head_dim, rotary_dim=n_rot, max_position=max_position,
                    base=base, rope_scaling=(("rope_type", "proportional"),))

    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    rope_freqs = torch.cat([torch.ones(n_rot // 2),
                            torch.full((head_dim // 2 - n_rot // 2,), 1e30)])
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float), theta / rope_freqs)
    expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    torch.testing.assert_close(rope._cos_sin_cache, expected, rtol=0, atol=1e-6)
