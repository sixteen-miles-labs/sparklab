# Security policy

## Supported versions

Security fixes target the latest published release and the default branch. Older releases
may receive a fix when practical, but are not supported unless their release notes say so.

## Report a vulnerability

Do not open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/sixteen-miles-labs/freetoken/security/advisories/new)
to report vulnerabilities to the SixteenMiles Labs security responders.

Include, when available:

- affected commit or release;
- impact and realistic threat model;
- reproduction steps or proof of concept;
- whether credentials, model artifacts, user prompts, or network access are involved;
- any suggested mitigation or disclosure constraint.

Please avoid accessing other users' data, degrading shared infrastructure, or publishing
details before a coordinated fix is available. The project will acknowledge the report,
assess impact, coordinate remediation, and credit reporters who want attribution. Response
timing depends on severity and maintainer availability; this policy does not promise a
fixed service-level agreement.

## Security boundaries

SparkLab is local-first, but a server bound beyond loopback must be treated as a network
service and protected by the operator. Model checkpoints and FTW artifacts are executable
inputs to native kernels and loaders; acquire them from pinned repositories and verify the
recorded revision and manifest. Release wheels should be verified against their GitHub
artifact attestation and `SHA256SUMS` before deployment.
