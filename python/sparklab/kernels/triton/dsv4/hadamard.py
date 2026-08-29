"""Normalized Walsh-Hadamard transform for DeepSeek-V4's indexer (``rotate_activation``).

The reference applies ``fast_hadamard_transform.hadamard_transform(x, scale=d**-0.5)``
to the indexer's query and (rotate) compressor KV before FP4 quant -- the standard
normalized Hadamard transform on the last (power-of-two) dim. We compute it as a
cached-matrix matmul in FP32 (then cast back), which is numerically the WHT that
``fast_hadamard_transform`` implements; for ``d=128`` the matrix is tiny so this is
both faithful and fast.
"""

from __future__ import annotations

import torch

_HCACHE: dict = {}


def _hadamard_matrix(d: int, device: torch.device) -> torch.Tensor:
    """Sylvester-ordered Hadamard matrix (+-1), fp32 (the ``d**-0.5`` normalization is
    applied *after* the matmul to bit-match ``fast_hadamard_transform``, which scales the
    WHT output rather than the +-1 basis)."""
    key = (d, str(device))
    H = _HCACHE.get(key)
    if H is None:
        assert (d & (d - 1)) == 0, f"Hadamard dim must be a power of 2, got {d}"
        H = torch.ones(1, 1, device=device, dtype=torch.float32)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        _HCACHE[key] = H
    return H


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalized Hadamard transform on the last dim (matches ``rotate_activation`` =
    ``fast_hadamard_transform.hadamard_transform(x, scale=d**-0.5)``, validated bit-exact).

    Computed in FP32 (WHT then scale) and cast back to ``x``'s dtype (reference is bf16)."""
    d = x.shape[-1]
    H = _hadamard_matrix(d, x.device)
    y = (x.float().reshape(-1, d) @ H) * (d ** -0.5)
    return y.reshape(x.shape).to(x.dtype)


__all__ = ["hadamard_transform"]
