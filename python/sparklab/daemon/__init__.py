"""``sparklab daemon`` — the SparkLab engine supervisor (persistent, torch-free control plane).

This package MUST NOT import torch / CUDA / flashinfer / sgl_kernel / any model or kernel code,
directly or transitively. That includes ``sparklab.serving.*`` (its ``__init__`` pulls
torch) and ``sparklab.utils.*`` (its ``__init__`` pulls transformers). Keep this module's imports
to ``main`` only, which itself imports lazily.
"""

from __future__ import annotations

import sys


def main(argv=None, *, prog: str = "sparklab daemon") -> int:
    """Single entry for ``sparklab daemon``. A leading client verb (``sparklab daemon status`` …) controls a
    running daemon; anything else (bare, or server flags like ``--host``/``--port``, which is what
    the systemd unit uses) runs the daemon server."""
    from .client import CLIENT_VERBS

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in CLIENT_VERBS:
        from .client import main as _client_main

        return _client_main(args, prog=prog)

    from .server import main as _server_main

    return _server_main(args, prog=prog)


__all__ = ["main"]
