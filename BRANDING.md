# Project identity

SparkLab is developed by **[SixteenMiles Labs](https://github.com/sixteen-miles-labs)**,
a research lab under **[Oakmind AI](https://oakmind.ai/)**.

Use these names consistently on public surfaces:

| Name | Role | Primary namespace |
|---|---|---|
| **Oakmind AI** | Parent organization, legal stewardship, and commercial support | [oakmind.ai](https://oakmind.ai/) |
| **SixteenMiles Labs** | Open-source research lab, publisher, and community steward | [GitHub](https://github.com/sixteen-miles-labs), [X](https://x.com/16MilesLabs) |
| **SparkLab** | User-facing GB10 product, CLI, recipes, deployment, and documentation | `sparklab` |
| **FreeToken** | Inference engine, Python package, FTW format, and research attribution | `freetoken` |

Model releases are currently published through the
[Oakmind AI Hugging Face account](https://huggingface.co/oakmindai). Future model
repositories should use a SixteenMiles Labs Hugging Face organization once that namespace
and its access controls are ready. Existing repositories should not be duplicated or moved
without a migration plan that preserves immutable revisions and user links.

## Packaging and deployment

- Public source, issues, release notes, and build provenance belong to SixteenMiles Labs.
- The current PyPI distribution remains `freetoken` for compatibility and installs both
  the primary `sparklab` command and the legacy `ft` alias.
- A future `sparklab` distribution must be introduced as a separate, versioned migration;
  it must not silently replace the engine package or break `freetoken.*` imports.
- Container images, when released, use `ghcr.io/sixteen-miles-labs/sparklab`.
- Services, logs, configuration, and new deployment documentation use `sparklab`; legacy
  `ft` and `FREETOKEN_*` surfaces remain compatibility aliases during the migration.
- Copyright or trademark claims must use the verified legal owner. Do not infer a legal
  entity suffix from the public Oakmind AI name.

The standard attribution line is:

> SparkLab is developed by SixteenMiles Labs, a research lab under Oakmind AI.
