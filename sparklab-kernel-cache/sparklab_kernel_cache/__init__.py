from __future__ import annotations

from pathlib import Path

try:
    from ._build_meta import __version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"

jit_cache_dir = Path(__file__).parent / "jit_cache"


def get_jit_cache_dir() -> Path:
    return jit_cache_dir


__all__ = ["__version__", "get_jit_cache_dir", "jit_cache_dir"]
