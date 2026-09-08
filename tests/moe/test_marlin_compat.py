"""Compatibility and artifact-selection failures must be caught before serving."""
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from sparklab.moe import nvfp4_backends as backends


@pytest.fixture(autouse=True)
def clear_marlin_api():
    backends._marlin_moe_api.cache_clear()
    yield
    backends._marlin_moe_api.cache_clear()


@pytest.mark.parametrize("modern", [False, True])
def test_marlin_donor_signature_and_activation(monkeypatch, modern):
    class Activation(Enum):
        SILU = "silu"

    def old_forward(*, activation="silu", gating_output):
        assert activation == "silu" and gating_output is None

    def new_forward(*, activation=Activation.SILU):
        assert activation is Activation.SILU

    def import_donor(name):
        if modern and name.endswith(".fused_marlin_moe"):
            raise ModuleNotFoundError(name=name)
        return SimpleNamespace(fused_marlin_moe=new_forward if modern else old_forward)

    monkeypatch.setattr(backends, "import_module", import_donor)
    forward, kwargs, dtype = backends._marlin_moe_api()
    forward(**kwargs)
    assert dtype == (torch.float32 if modern else torch.bfloat16)


def test_marlin_import_does_not_hide_broken_dependencies(monkeypatch):
    def broken_import(name):
        raise ModuleNotFoundError(name="missing_donor_dependency")

    monkeypatch.setattr(backends, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError) as error:
        backends._marlin_moe_api()
    assert error.value.name == "missing_donor_dependency"


@pytest.mark.parametrize("requested", ["marlin", "flashinfer"])
def test_ftw_loader_rejects_backend_that_does_not_match_stored_banks(monkeypatch, requested):
    from sparklab.checkpoint import ftw
    from sparklab.moe.expert_banks import ExpertBanks, load_expert_banks

    monkeypatch.setattr(ftw, "is_ftw_checkpoint", lambda _: True)
    monkeypatch.setattr(ftw, "load_ftw_banks", lambda *a, **kw: ExpertBanks("nvfp4", {}))
    config = SimpleNamespace(num_moe_layers=2, nvfp4_backend=requested)
    with pytest.raises(ValueError, match="Convert the original checkpoint"):
        load_expert_banks("prepared", config, device=torch.device("cpu"), dtype=torch.bfloat16)
