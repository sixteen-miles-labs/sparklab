# Project identity

SparkLab is developed by **[SixteenMiles Labs](https://github.com/sixteen-miles-labs)**,
a research lab under **[Oakmind AI](https://oakmind.ai/)**.

Use these names consistently on public surfaces:

| Name | Role | Primary namespace |
|---|---|---|
| **Oakmind AI** | Parent organization, legal stewardship, and commercial support | [oakmind.ai](https://oakmind.ai/) |
| **SixteenMiles Labs** | Open-source research lab, publisher, and community steward | [GitHub](https://github.com/sixteen-miles-labs), [X](https://x.com/16MilesLabs) |
| **SparkLab** | Open-source GB10 inference system, Python package, CLI, recipes, runtime, and documentation | `sparklab` |
| **FreeToken** | Upstream research and source-code ancestry | [upstream project](https://github.com/FlashML-org/FreeToken) |

Model releases are currently published through the
[Oakmind AI Hugging Face account](https://huggingface.co/oakmindai). Future model
repositories should use a SixteenMiles Labs Hugging Face organization once that namespace
and its access controls are ready. Existing repositories should not be duplicated or moved
without a migration plan that preserves immutable revisions and user links.

## Packaging and deployment

- Public source, issues, release notes, and build provenance belong to SixteenMiles Labs.
- The PyPI distribution and Python namespace are `sparklab`.
- The companion prebuilt-kernel distribution is `sparklab-kernel-cache`.
- The supported command, environment-variable prefix, cache path, and service name are
  `sparklab`, `SPARKLAB_*`, `~/.cache/sparklab`, and `sparklab.service`.
- Container images, when released, use `ghcr.io/sixteen-miles-labs/sparklab`.
- FTW remains the established checkpoint-format name and preserves its v1 on-disk identifiers.
- FreeToken attribution belongs in `NOTICE`, credits, and research citations rather than
  in SparkLab package or runtime namespaces.
- Copyright or trademark claims must use the verified legal owner. Do not infer a legal
  entity suffix from the public Oakmind AI name.

The standard attribution line is:

> SparkLab is developed by SixteenMiles Labs, a research lab under Oakmind AI.
