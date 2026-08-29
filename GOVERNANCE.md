# Governance

SparkLab and its FreeToken engine are open-source projects stewarded by
**SixteenMiles Labs**, a research lab under **Oakmind AI**.

## Roles

- **Contributors** propose issues, documentation, tests, code, recipes, and benchmark
  evidence.
- **Reviewers** evaluate changes in areas where they have demonstrated context.
- **Maintainers** merge changes, manage labels and releases, and enforce project policy.
- **Release managers** verify version, provenance, security, licensing, and artifact gates.
- **Security responders** coordinate private vulnerability reports and disclosure.

Roles are earned through sustained, constructive work. Maintainer and release access is
granted by the SixteenMiles Labs organization owners using least privilege and protected
environments. Employment by Oakmind AI does not by itself bypass review or release gates.

## Decisions

Routine changes use pull-request review and lazy consensus. A maintainer may merge when
required tests pass, relevant reviewers have had a reasonable opportunity to respond, and
no unresolved technical or policy objection remains.

Changes to public APIs, model claims, licenses, governance, security posture, package
names, artifact formats, or compatibility promises require a public design issue or RFC.
The decision and rationale must be recorded in the repository. Maintainers may make an
expedited security decision privately and publish an explanation after coordinated
disclosure.

## Releases

Release authority belongs to designated SixteenMiles Labs release managers. No individual
may publish from an unreviewed workstation build. Tagged releases must be produced by the
protected workflow described in [RELEASING.md](RELEASING.md).

## Project and corporate boundaries

SixteenMiles Labs owns public technical governance and community participation. Oakmind AI
provides organizational backing, legal stewardship, and commercial support. Commercial
priorities may inform the roadmap but do not alter the Apache-2.0 terms of accepted
contributions.

This governance document can be changed through the same public RFC process described
above.
