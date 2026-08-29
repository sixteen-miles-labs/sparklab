from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_public_package_metadata_belongs_to_sixteenmiles_labs():
    pyproject = _read("pyproject.toml")
    kernel_project = _read("freetoken-kernel-cache/pyproject.toml")

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


def test_sparklab_is_the_primary_installer_and_service_identity():
    installer = _read("install.sh")
    service = _read("python/freetoken/daemon/sparklab.service")

    assert 'ln -sf "$SPARKLAB_BIN" "$BIN_DIR/sparklab"' in installer
    assert 'ln -sf "$FT_BIN" "$BIN_DIR/ft"' in installer
    assert "SPARKLAB_INSTALL_ROOT" in installer
    assert "ExecStart=%h/.local/bin/sparklab daemon" in service
    assert "Description=SparkLab engine supervisor" in service


def test_community_health_files_are_present():
    for relative in (
        "BRANDING.md",
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
