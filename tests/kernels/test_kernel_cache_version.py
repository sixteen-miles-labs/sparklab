"""Version pairing between the runtime and the sparklab-kernel-cache wheel.

The comparator lives in sparklab.kernels.utils; this pins the matrix for the stamped
version scheme (runtime `0.1.1+g<sha>`, cache `0.1.1+cu130.g<sha>`) introduced by
scripts/build-release-wheels.sh."""

from types import SimpleNamespace

import pytest

from sparklab.kernels import _toolchain
from sparklab.kernels import utils
from sparklab.kernels.utils import _kernel_cache_version_ok


@pytest.mark.parametrize(
    ("cache", "runtime", "ok"),
    [
        # Unstamped pairs (the pre-stamp world) behave as before.
        ("0.1.1", "0.1.1", True),
        ("0.1.1+cu130", "0.1.1", True),
        ("0.2.0+cu130", "0.1.1", False),
        ("0.1.10+cu130", "0.1.1", False),  # prefix of the release string, not the release
        # Stamped pairs: same build passes...
        ("0.1.1+cu130.g3f01615c9", "0.1.1+g3f01615c9", True),
        # ...and a runtime/cache pair from two different builds is exactly the
        # mismatch this scheme exists to catch (bare release numbers agree!).
        ("0.1.1+cu130.gffc111e2e", "0.1.1+g3f01615c9", False),
        # One-sided stamps are tolerated (a dev build against a release wheel and
        # vice versa) -- only the release part is compared then.
        ("0.1.1+cu130", "0.1.1+g3f01615c9", True),
        ("0.1.1+cu130.g3f01615c9", "0.1.1", True),
        ("0.1.1", "0.1.1+g3f01615c9", True),
        # A `g...` token must be g+hex to count as a stamp; anything else is an
        # ordinary local segment and stays out of the comparison.
        ("0.1.1+cu130.gabcdefgh", "0.1.1+g3f01615c9", True),
        # The release part must still match even when stamps agree.
        ("0.2.0+cu130.g3f01615c9", "0.1.1+g3f01615c9", False),
    ],
)
def test_kernel_cache_version_matrix(cache: str, runtime: str, ok: bool) -> None:
    assert _kernel_cache_version_ok(cache, runtime) is ok


def _cache_package(version: str, tmp_path):
    return SimpleNamespace(
        __version__=version,
        get_jit_cache_dir=lambda: str(tmp_path),
    )


def test_mismatched_cache_falls_back_to_jit(monkeypatch, tmp_path) -> None:
    utils._WARNED_CACHE_ISSUES.clear()
    package = _cache_package("0.1.2+cu130", tmp_path)
    monkeypatch.setattr(utils.importlib, "import_module", lambda _: package)
    monkeypatch.setattr(utils, "_sparklab_version", lambda: "0.1.3")

    with pytest.warns(RuntimeWarning, match="ignoring the incompatible cache"):
        assert utils._kernel_cache_dir() is None


def test_mismatched_cache_is_actionable_when_jit_is_disabled(monkeypatch, tmp_path) -> None:
    utils._WARNED_CACHE_ISSUES.clear()
    package = _cache_package("0.1.2+cu130", tmp_path)
    monkeypatch.setattr(utils.importlib, "import_module", lambda _: package)
    monkeypatch.setattr(utils, "_sparklab_version", lambda: "0.1.3")
    monkeypatch.setenv(utils.DISABLE_JIT_ENV, "1")

    with pytest.raises(RuntimeError, match="same SparkLab release.*JIT fallback is disabled"):
        utils._kernel_cache_dir()


def test_wrong_cuda_cache_falls_back_to_jit(monkeypatch, tmp_path) -> None:
    utils._WARNED_CACHE_ISSUES.clear()
    package = _cache_package("0.1.3+cu120", tmp_path)
    monkeypatch.setattr(utils.importlib, "import_module", lambda _: package)
    monkeypatch.setattr(utils, "_sparklab_version", lambda: "0.1.3")
    monkeypatch.setattr(_toolchain, "torch_cuda_major", lambda: 13)

    with pytest.warns(RuntimeWarning, match="built for CUDA 12.x"):
        assert utils._kernel_cache_dir() is None
