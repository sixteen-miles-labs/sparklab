"""Spark Lab environment-variable compatibility helpers.

Product-prefixed values win. Legacy FreeToken variables remain supported so the
staged rebrand does not change existing engine behavior.
"""

from __future__ import annotations

import os
from typing import TypeVar

PRODUCT_PREFIX = "SPARKLAB_"
LEGACY_PREFIX = "FREETOKEN_"

T = TypeVar("T")


def product_name(legacy_name: str) -> str:
    if not legacy_name.startswith(LEGACY_PREFIX):
        raise ValueError(f"legacy environment name must start with {LEGACY_PREFIX}")
    return PRODUCT_PREFIX + legacy_name.removeprefix(LEGACY_PREFIX)


def getenv_compat(legacy_name: str, default: T = None) -> str | T:
    """Return SPARKLAB_* first, then the corresponding FREETOKEN_* value."""
    value = os.environ.get(product_name(legacy_name))
    if value is not None:
        return value
    return os.environ.get(legacy_name, default)


__all__ = ["LEGACY_PREFIX", "PRODUCT_PREFIX", "getenv_compat", "product_name"]
