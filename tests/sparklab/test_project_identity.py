from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_public_package_metadata_belongs_to_sixteenmiles_labs():
    pyproject = _read("pyproject.toml")
    kernel_project = _read("sparklab-kernel-cache/pyproject.toml")

    for text in (pyproject, kernel_project):
        assert 'name = "SixteenMiles Labs"' in text
        assert "https://github.com/sixteen-miles-labs/sparklab" in text
    assert 'description = "SparkLab:' in pyproject
    assert 'license-files = ["LICENSE", "NOTICE"]' in pyproject


def test_release_workflows_are_guarded_for_the_current_repository():
    release = _read(".github/workflows/release.yml")
    nightly = _read(".github/workflows/nightly-wheels.yml")
    publisher = _read("scripts/publish-wheels.sh")

    for workflow in (release, nightly):
        assert "github.repository == 'sixteen-miles-labs/sparklab'" in workflow
        assert "github.repository == 'FlashML-org/FreeToken'" not in workflow
    assert "actions/attest@" in release
    assert "SHA256SUMS" in release
    assert "sixteen-miles-labs/sparklab" in publisher
    assert 'gh release create "$TAG"' in publisher


def test_release_wheels_target_dgx_spark_arm64():
    release = _read(".github/workflows/release.yml")
    nightly = _read(".github/workflows/nightly-wheels.yml")
    builder = _read("scripts/ci/manylinux-build.sh")
    publisher = _read("scripts/publish-wheels.sh")

    for workflow in (release, nightly):
        assert "runs-on: [self-hosted, linux, ARM64, engine-build]" in workflow
    assert "manylinux*aarch64.whl" in release
    assert "linux_aarch64" in nightly
    assert "pytorch/manylinuxaarch64-builder:cuda13.0@sha256:" in builder
    assert "*linux_aarch64*) echo linux_aarch64" in publisher


def test_sparklab_is_the_primary_installer_and_service_identity():
    installer = _read("install.sh")
    service = _read("python/sparklab/daemon/sparklab.service")

    assert 'ln -sf "$SPARKLAB_BIN" "$BIN_DIR/sparklab"' in installer
    assert '"$BIN_DIR/ft"' not in installer
    assert "SPARKLAB_INSTALL_ROOT" in installer
    assert "ExecStart=%h/.local/bin/sparklab daemon" in service
    assert "Description=SparkLab engine supervisor" in service


def test_only_the_sparklab_python_namespace_is_packaged():
    assert (ROOT / "python" / "sparklab").is_dir()
    assert not (ROOT / "python" / "freetoken").exists()
    pyproject = _read("pyproject.toml")
    assert 'name = "sparklab"' in pyproject
    assert 'sparklab = "sparklab.cli:main"' in pyproject
    assert '\nft = ' not in pyproject


def test_community_health_files_are_present():
    for relative in (
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "GOVERNANCE.md",
        "RELEASING.md",
        "SECURITY.md",
        "NOTICE",
        "CITATION.cff",
        ".gitleaks.toml",
        ".github/PULL_REQUEST_TEMPLATE.md",
    ):
        assert (ROOT / relative).is_file(), relative
