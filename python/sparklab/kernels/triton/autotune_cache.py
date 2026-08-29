"""Version-gated ``cache_results`` for the pure-triton fallback kernels.

Without result caching, ``@triton.autotune`` re-benchmarks the whole config grid
on the first launch of every distinct key in every new process -- which lands
inside CUDA-graph capture / warmup and costs seconds at startup (worst on hybrid
models that touch many of these kernels). ``cache_results=True`` persists the
winning config per key to triton's on-disk cache and reloads it next time. The
on-disk key already folds in the triton version, the device/backend target hash,
and the kernel source hash, so a cached config is never reused across a different
GPU arch, triton build, or edited kernel.

``cache_results`` was only added to ``triton.autotune`` in newer releases; gate on
the signature so these fallbacks still import on older triton (there it is a
no-op). Mirrors kernel/fla/utils.py for the vendored FLA kernels.
"""

from __future__ import annotations

import inspect
import os

import triton

_SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)

# Escape hatch: SPARKLAB_TRITON_CACHE_RESULTS=0 forces a fresh sweep every run.
_CACHE_RESULTS = os.getenv("SPARKLAB_TRITON_CACHE_RESULTS", "1") == "1"

# Splat into each @triton.autotune(...): {"cache_results": True} on new triton,
# {} on old triton so the kwarg is simply absent.
autotune_cache_kwargs = (
    {"cache_results": _CACHE_RESULTS} if _SUPPORTS_AUTOTUNE_CACHE else {}
)

__all__ = ["autotune_cache_kwargs"]
