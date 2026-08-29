# Contributing to SparkLab

Thank you for contributing to SparkLab. The project is maintained
by **SixteenMiles Labs**, a research lab under **Oakmind AI**.

## Before opening a change

1. Search existing issues and pull requests.
2. Open a design issue before changing public APIs, package names, checkpoint formats,
   model-support claims, or compatibility policy.
3. Keep changes focused. Do not include model weights, credentials, benchmark raw streams,
   generated build products, or unrelated formatting rewrites.

Security vulnerabilities must follow [SECURITY.md](SECURITY.md), not the public issue
tracker.

## Development workflow

SparkLab uses trunk-based development. Create a short-lived branch, open a pull request
against `main`, and merge only after review and required checks pass. There is no
long-lived `develop` branch; `main` remains the source for rolling beta builds and signed
releases.

Every pull request requires a passing DCO sign-off check and the hosted CPU test suite.
During the single-maintainer beta, approval is not required because GitHub does not allow
an author to approve their own pull request. Enable one approving review, with stale
approvals dismissed by new commits, as soon as a second maintainer joins. GPU, checkpoint,
and release tests run only in trusted environments when the change requires them; pull
requests from forks never execute on the self-hosted GB10 runner.

Direct pushes to `main` are prohibited. Maintainers use the same pull-request workflow as
other contributors.

## Development setup

On the supported NVIDIA GB10 environment:

```bash
git clone https://github.com/sixteen-miles-labs/sparklab.git
cd sparklab
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run the smallest relevant test set while developing, then the affected package suite. For
product-layer changes, run:

```bash
.venv/bin/python -m pytest -q tests/sparklab
```

GPU, checkpoint, and slow tests have explicit markers and environment requirements; see
[tests/README.md](tests/README.md).

Contributors can run SparkLab from source and build native development wheels on their
own machines; release credentials and the trusted runner are not required. On a supported
DGX Spark, `scripts/build-release-wheels.sh` produces local `linux_aarch64` wheels, while
`scripts/ci/manylinux-build.sh` uses the architecture-matched container when a
PyPI-compatible development wheel is needed. Formal wheels are built only from protected
`main` or a protected release tag. Fork pull requests never run either release workflow.

## Pull requests

- Explain the user-visible outcome and important tradeoffs.
- Add or update tests for behavior changes.
- Update documentation, recipes, evidence, and migration notes when their contract changes.
- Do not claim model quality, performance, capacity, or certification without checked-in,
  versioned evidence for the exact checkpoint and recipe.
- Disclose generated code or content when it materially affects review, and verify that you
  have the right to contribute every submitted file.
- Prefer a squash merge for a focused change. Preserve the DCO sign-off in the resulting
  commit; use a rebase merge when a meaningful, fully signed commit series should remain.

## Developer Certificate of Origin

Contributions use the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
Sign off every commit:

```bash
git commit -s -m "area: describe the change"
```

The sign-off certifies that you created the contribution or have the right to submit it
under this project's Apache-2.0 license. It is not a copyright assignment to
SixteenMiles Labs or Oakmind AI.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md) and the
[governance policy](GOVERNANCE.md).
