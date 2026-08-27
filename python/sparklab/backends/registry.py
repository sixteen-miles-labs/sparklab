"""Explicit built-in runtime registry.

Beta 0.1 intentionally does not auto-load third-party entry points. An explicit registry
keeps import and execution provenance auditable while the adapter API is still evolving.
"""

from __future__ import annotations

from collections.abc import Iterable

from .base import BackendError, RuntimeBackend

_BACKENDS: dict[str, RuntimeBackend] = {}
_BUILTINS_LOADED = False


def register_backend(backend: RuntimeBackend, *, replace: bool = False) -> None:
    backend_id = backend.backend_id.strip().lower()
    if not backend_id or backend_id != backend.backend_id:
        raise BackendError(f"invalid normalized backend id: {backend.backend_id!r}")
    if backend_id in _BACKENDS and not replace:
        raise BackendError(f"backend {backend_id!r} is already registered")
    _BACKENDS[backend_id] = backend


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .native import NativeBackend

    register_backend(NativeBackend())
    _BUILTINS_LOADED = True


def get_backend(backend_id: str) -> RuntimeBackend:
    _load_builtins()
    try:
        return _BACKENDS[backend_id]
    except KeyError as exc:
        raise BackendError(f"unknown Spark Lab runtime backend: {backend_id!r}") from exc


def list_backends() -> tuple[RuntimeBackend, ...]:
    _load_builtins()
    return tuple(_BACKENDS[key] for key in sorted(_BACKENDS))


def registered_backend_ids() -> Iterable[str]:
    _load_builtins()
    return tuple(sorted(_BACKENDS))


__all__ = ["get_backend", "list_backends", "register_backend", "registered_backend_ids"]
