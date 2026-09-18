"""Diagnostic-only server: capture one requested prefill's final-token logits.

Set BONSAI_CAPTURE=/tmp/path, then create /tmp/path.request after the server is
ready. The next prefill saves /tmp/path.pt and removes the request marker.
This instrumentation lives outside the inference package.
"""

import os
from pathlib import Path

import torch
from sparklab.core import get_global_ctx
from sparklab.models.bonsai2.model import TernaryHead

_original = TernaryHead.forward
_capture = Path(os.environ["BONSAI_CAPTURE"])


def forward(self, x):
    result = _original(self, x)
    if get_global_ctx().batch.is_prefill and _capture.with_suffix(".request").exists():
        torch.save(result[-1].detach().float().cpu(), _capture.with_suffix(".pt"))
        _capture.with_suffix(".request").unlink()
    return result


TernaryHead.forward = forward

if __name__ == "__main__":
    from sparklab.serving import launch_server

    launch_server()
