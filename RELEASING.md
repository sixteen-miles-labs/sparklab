# Release and package policy

Public releases are produced by **SixteenMiles Labs** from this repository. Oakmind AI is
the parent organization and legal steward. SparkLab incorporates work derived from the
FreeToken research project; that ancestry is preserved in `NOTICE` and project citations.

## Current package boundary

- PyPI distribution and Python namespace: `sparklab`
- Command: `sparklab`
- Native runtime: `sparklab.*`
- Companion binary package: `sparklab-kernel-cache`
- GitHub release repository: `sixteen-miles-labs/sparklab`
- Future container namespace: `ghcr.io/sixteen-miles-labs/sparklab`

## Tagged release procedure

1. Start from a clean, reviewed commit on the protected default branch.
2. Update and test the SparkLab version. `vX.Y.Z` must equal
   `python/sparklab/version.py` exactly.
3. Run focused tests, the supported product suite, package builds, and `twine check`.
4. Review licenses, `NOTICE`, model-recipe revisions, evidence links, and release notes.
5. Create and push the signed tag. The protected release workflow builds the wheels.
6. Approve the `pypi` environment only after inspecting the immutable build artifacts.
7. Verify the generated `SHA256SUMS` and, for public releases, the GitHub
   build-provenance attestation.
8. Edit the draft GitHub release notes, document limitations and migrations, then publish.
9. Install from the published artifacts on a clean supported GB10 system and run the
   documented smoke test.

Never build or upload a formal release from a maintainer workstation. PyPI credentials or
trusted-publisher bindings must be scoped to this repository and protected environment.

## One-time external setup

Repository owners must configure these outside the source tree:

- protect the default branch and release tags;
- restrict the `pypi` and `testpypi` environments to release managers;
- configure a project-scoped PyPI token or, preferably, PyPI trusted publishing;
- enable GitHub private vulnerability reporting;
- reserve the SixteenMiles Labs package, container, and Hugging Face namespaces;
- create a SixteenMiles Labs Hugging Face organization before moving model repositories;
- preserve immutable model revisions and redirects during any repository transfer.

The canonical rolling `beta` release lives in `sixteen-miles-labs/sparklab`. The publisher
uses `SPARKLAB_WEB_REPO` only for controlled release automation; formal releases must not
target an external upstream repository.
