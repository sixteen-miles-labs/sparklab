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

## Pull requests

- Explain the user-visible outcome and important tradeoffs.
- Add or update tests for behavior changes.
- Update documentation, recipes, evidence, and migration notes when their contract changes.
- Do not claim model quality, performance, capacity, or certification without checked-in,
  versioned evidence for the exact checkpoint and recipe.
- Disclose generated code or content when it materially affects review, and verify that you
  have the right to contribute every submitted file.

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
