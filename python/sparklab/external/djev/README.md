# Upstream provenance

`upstream.py` is the vLLM structured diffusion example from
`examples/features/structured_diffusion/structured_server.py` at
`1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`, Apache-2.0.
Original SHA-256: `7cd9aa0081090c064eaac28db0f54f812749eeb3ae787d7f7653d2e35d8a938f`.
The only local modification replaces `pybase64` with standard-library `base64`.
A test verifies the original hash after undoing that import substitution.

`gateway.py` restricts the exposed API to structured reads, bounds request size,
validates the model/seed, and checks upstream readiness. `supervisor.py` manages
both child process groups. The upstream example's raw chat proxy, TLS helper,
and direct main entry point are not used.
