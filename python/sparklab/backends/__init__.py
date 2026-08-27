"""Runtime backend contracts and the built-in Spark Lab backend registry."""

from .base import (
    ArtifactValidation,
    BackendCapabilities,
    BackendError,
    BackendLaunchPlan,
    RuntimeBackend,
    RuntimeRequest,
)
from .registry import get_backend, list_backends, register_backend

__all__ = [
    "ArtifactValidation",
    "BackendCapabilities",
    "BackendError",
    "BackendLaunchPlan",
    "RuntimeBackend",
    "RuntimeRequest",
    "get_backend",
    "list_backends",
    "register_backend",
]
