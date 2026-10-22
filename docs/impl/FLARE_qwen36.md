# Qwen3.6 target-only FLARE validation

Latest: the authorized paired performance attempt fails strict 32K branch output
parity (13/14 outputs match, no incomplete requests). Zero pairs qualify.
Independent finite baseline/memory/disk pressure collections pass, including 63
completed requests and 21 cancellations; the disk path performs real restores.
A benchmark-only HTTP busy handling fix passes 37 checks. The primary owner is
inactive and all six owned servers exited 143. See
[performance/pressure results](FLARE_qwen36_performance.md) and
`qwen36-validation-v3.json`. Earlier results below retain their recorded scopes.


The user stopped the root model-owner service for the long-test window. At the
final audit it was active again (11:50:50 EDT); all owned native tests/servers had
finished and it was left untouched.
Disk free is over 600 GiB. With explicit authorization,
[issue 99](https://github.com/sixteen-miles-labs/sparklab/issues/99) was published
and the three-adapter rejection fix was applied. Forced selection is rejected,
not silently downgraded to auto; the original failed tool session is retained.
Source adapter checks: 162 passes. Real HTTP checks reject 18 stream/nonstream
constraints without consuming UIDs; supported generation continues normally.

Eight complete-state 8K/32K memory/disk branch cases pass in 224.24 seconds
(`qwen36-long-exact-state.json` and its log). The 32K four-boot gate also passes:
computed prefill 32769/32768/1/32769, reuse of 32768 tokens, strict output match,
and four exits of 143 (`qwen36-restart-32k-1/evidence.json`). Natural factual
recall is being checked separately from that synthetic state/restart evidence.

The first independent recall baseline answered all six long questions correctly.
Its candidate exposed an optional-boundary renderer bug before HTTP execution:
system-only partial histories are not valid Qwen conversations. The tokenizer
now omits unsupported optional boundaries after validating the full request;
full-request and unexpected runtime errors remain authoritative. Prior artifact:
two negative tests fail, two controls pass. Focused source/verifier: 17 pass.
The final wheel, `wheels-partial-render/`, matches all 436 production modules;
SHA-256 `5a70962d6d84554029da8982b634947cc812be5bbb16f349bce07504ea42e46e`.
Its CPU/MoE selection passes 2339, skips 241 and deselects six in 246.33 seconds.
The stronger paired recall run passes 12 seed/repeat cases at 8K and 32K with
strict output signatures and equal completed token counts. All repeat requests
reuse the exact unaligned N-1 endpoint (8191/32767 tokens). Final native counters
show 21 publications, six exact hits and zero readers/private restore pages or
slots. Both servers exit 143; the candidate's additional 16 HTTP branch controls
also pass. Reports are `qwen36-recall-{baseline,candidate}-partialfix-1/recall.json`.

The same wheel's four-protocol streaming/stage smoke passes in 26.60 seconds:
five HTTP samples reconcile as four admissions/generations/completions plus one
validation error, with no invalid stage intervals or active requests. Evidence:
`qwen36-protocol-stages-api99-1/observations.json`. No timing overhead or pressure
benefit is claimed.

A separate supported-auto arithmetic workflow completed one actual multiplication
call and consumed its real tool result, but its baseline answered the next turn
directly instead of calling the tool. The 20-round strict tool workflow therefore
remains incomplete and its candidate was not run (`qwen36-auto-tools-baseline-2/`).
The earlier first attempt exposed a harness reference error: schema dictionary
ordering before tokenization differed from the canonical JSON sent by http_chat.
Reference rendering now uses the actual wire ordering, with 49 focused harness
checks passing. All 24 prior factual-recall counts/hashes were independently
rechecked unchanged (`recall-wire-order-reconciliation.json`); the historical
provider source is retained for provenance. No prompt or call oracle was weakened.
All earlier wheel evidence retains its original artifact scope. The paragraphs
below preserve earlier observations and should not be read as current blockers.

This extends the experimental validation matrix; it does not replace the original
Qwen3.8/DFlash quality gate. The local Qwen3.6-35B-A3B Triton FTW at
`$HOME/.sparklab/models/qwen3.6-35b-a3b/prepared/0.5.0` has fingerprint
`bda7268c3e0afd7b`. Metadata validation confirms four complete shards, 1071
indexed tensors and 22,580,867,072 physical bytes, including unused MTP weights.
Payload hashes/numerical correctness are not proven by metadata validation.

Use the **current host wheel**, not the released `0.6.0` Marlin container.
The new internal test profile `qwen36-triton-resident` checks the Qwen MoE
architecture and native NVFP4 format, then selects Triton, offload GPU execution,
disk backing, full immutable expert preload, no CUDA graphs and a 0.85 memory
ratio. Disk backing streams expert weights once at startup; it avoids keeping
duplicate complete pinned RAM and GPU expert banks in GB10's physical pool.
Expert-weight loading and conversation-prefix retention remain separate stores.
No speculative path is enabled; fixed depth is zero.

The profile is shared by exact-state, paired session and four-boot restart
fixtures. Their default profile remains unchanged. Actual parsed CLI options are
checked in CPU tests, including the valueless `--moe-preload-all` flag and zero
speculation. Existing session resource overrides still apply equally to both
arms and are recorded in launch identity.

## Current observations

The first guarded baseline attempt passed artifact metadata validation but
declined to boot when free GPU memory changed from 30.21 to **29.94 GiB**.
A second attempt declined at **29.66 GiB**. Both are preflight blocks, not failed
model execution and not model correctness passes. Raw launch/evidence files are
under `$HOME/results/flare-runtime/qwen36-baseline-*`.
The fixed **30 GiB** pre-boot floor remains; it may be raised after measured
resource budgeting. Do not terminate unrelated services without permission.

The initial shared-profile/session/restart CPU selection passed **74 tests**.
The first real CLI test used the wrong tool-parser name in its test control;
correcting it yielded 43 passes (`qwen36-profile-cli-corrected.log`). The first
quiet baseline launch then exposed an older fixture error: internal
`max_extend_tokens` had been rendered as the nonexistent `--max-extend-tokens`.
All full fixture profiles now share canonical CLI spelling and pass the actual
parser for default/Qwen profiles and one/two-request geometries: the final
profile/session/restart CPU selection is **78 passed** in 16.40 seconds
(`qwen36-fixture-cli-final.log`). These selections overlap and are not summed.
Production Python modules are unchanged from the current spawn-gap wheel.

After authorized VLLM container stop, the corrected target-only baseline reached
ready in 30.06 seconds, returned `ready`, and exited normally with code 143.
Measured free GPU memory was 68.26 GiB before boot and 42.04 GiB at readiness.
These observations are not peak allocation or throughput qualification.
`qwen36-baseline-quiet-2/evidence.json` and `server.log` preserve the raw run.

The complete-model same-partition state matrix passed **32 cases** across all
eight short endpoints, both suffix branches and memory/disk paths, in 58.68
seconds (`qwen36-exact-state-retry.log`, `qwen36-exact-state.json`). All cases
report state and output equality. An earlier invocation skipped without printing
its reason and is not counted as a pass. The HTTP branch/rebuild smoke separately
passed **16 cases**, including exact cached-token accounting and greedy output
equality against cold controls (`qwen36-api-exact-1/branches.json`). Its server
also exited normally with code 143. Short synthetic equality does not establish
long-session or diverse-content quality.

The actual four-boot restart gate **passed in 109.45 seconds**
(`qwen36-restart-1.log`, `qwen36-restart-1/evidence.json`). Its independent raw
dispatch verifier observed computed prefill counts **66 / 65 / 1 / 66** for
cold/publish/restore/disabled rollback. Restore reused 65 tokens, strict output
equality passed, rollback preserved the record inventory, and all four server
exit codes were 143. This closes the short Qwen3.6 restart/disable experiment,
not long-prefix factual recall, other models, performance or endurance.

The paired 20-turn application fixture **failed on the baseline**, before any
candidate execution (`qwen36-sessions-1.log`, `qwen36-sessions-1/baseline.json`).
At `turn-3.tool`, the model returned the known fact as JSON instead of an actual
lookup_fact call. The strict oracle rejected it. Of 30 planned cases, six were
attempted: five passed, one failed, and 24 were not run. The five passing cases
include three initial facts and the independent legal summary/chronology cases;
this is not a completed 20-turn session. The baseline server exited 143.

Cold-reset diagnostics replayed that request with both `required` and `auto`
tool choice and with original versus explicit tool-first system instructions.
All four still returned plain JSON instead of a call
(`qwen36-tool-diagnostic-1/evidence.json`). Therefore a prompt-conflict explanation
alone is not established. Source inspection shows that `required` currently
exposes/parses tools without constraining the decoder to emit a call. No API
semantics were changed to hide this result, and no failing tool oracle was
weakened. A second diagnostic run reproduced all four failures and passed both
minimal lookup and multiplication controls (`qwen36-tool-diagnostic-2/evidence.json`).
Thus parsing/tool emission works on those minimal requests, while required-call
behavior fails in the session context. These controls do not replace the failed
baseline or establish candidate parity. Both diagnostic servers exited 143.

Completed-test temporary prefix records were hashed and removed to recover
1,051,793,716 bytes; case evidence and logs remain. The inventory is
`qwen36-exact-record-cleanup.json`. Another 167,451,510 bytes of inactive pip
wheel-download cache bodies were removed after wheel-content and open-holder
checks (`qwen36-wheel-download-cache-cleanup.json`); installed environments were
preserved. Disk free remains close to the restart fixture's 3 GiB floor.

VLLM is owned by `grove-beta-next-primary-model.service`, which recreated its
container after direct stop. The replacement was also stopped with authorization.
Stopping the systemd owner itself requires unavailable privileged authentication;
the user has been asked to stop that exact service. Do not stop retrieval or
other platform roles. Any correctness run must retain its per-boot memory guard;
a controlled performance/soak window is not established merely by stopping a
container once.

At this handoff no owned native test/server remains live and no VLLM container is
present. The model-owner systemd unit remains in its restart cycle; the environment
cannot authenticate a stop or restore of that root-managed service. The requested
operator action is `sudo systemctl stop grove-beta-next-primary-model.service`.
When the test window is finished, the service can be resumed by its operator with
`sudo systemctl start grove-beta-next-primary-model.service`; inspect its health
and approved-startup requirements if its existing restart failures persist. No
privileged service configuration, restart policy or other role was changed.

## Reproduce after establishing a quiet test window

From the FLARE worktree, use a fresh evidence directory. Keep disk free above
3 GiB before the restart fixture; its owned store is bounded to 2 GiB.

```sh
export PYTHONPATH="$HOME/results/flare-runtime/wheels-partial-render/extracted"
export SPARKLAB_DISABLE_JIT=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SPARKLAB_DISABLE_OVERLAP_SCHEDULING=1
export QWEN36_TARGET="$HOME/.sparklab/models/qwen3.6-35b-a3b/prepared/0.5.0"
export QWEN36_RUN="$HOME/results/flare-runtime/qwen36-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$QWEN36_RUN"
```

First establish target-only baseline startup, a short completion, geometry and
ordinary shutdown with the owned external `qwen36_smoke.py` runner. It records
preflight/ready/after memory, metadata, full launch arguments and raw server log.
Its `baseline` and `exact` modes boot separately and preserve failures. It is a
small correctness experiment, not a performance or complete-state certificate.

Then run exact-state continuation using identical consumed partitions:

```sh
SPARKLAB_EXACT_TEST_MODEL="$QWEN36_TARGET" \
SPARKLAB_EXACT_TEST_MODEL_PROFILE=qwen36-triton-resident \
SPARKLAB_EXACT_TEST_MIN_FREE_GIB=30 \
SPARKLAB_EXACT_TEST_EVIDENCE="$QWEN36_RUN/exact.json" \
"$HOME/projects/sparklab/.venv/bin/python" -m pytest -q -rs tests/e2e/test_exact_checkpoints.py
```

This checks endpoints 1/15/16/17/31/63/64/65, two suffix branches and memory/disk
paths. It must conserve private tails, KV and all registered GDN state, compare
independent state/output controls, and retain raw failures. Long 8K/32K runs need
separate cache sizing and streamed hashing; no short result qualifies them.

Run target-only paired application and restart gates independently:

```sh
SPARKLAB_SESSION_TEST_MODEL="$QWEN36_TARGET" \
SPARKLAB_SESSION_MODEL_PROFILE=qwen36-triton-resident \
SPARKLAB_SESSION_PACKAGE_PATH="$PYTHONPATH" \
SPARKLAB_SESSION_EVIDENCE_ROOT="$QWEN36_RUN/sessions" \
SPARKLAB_SESSION_MIN_FREE_GIB=30 SPARKLAB_SESSION_LEGAL_CASE=1 \
"$HOME/projects/sparklab/.venv/bin/python" -m pytest -q -rs tests/e2e/test_runtime_sessions.py

SPARKLAB_RESTART_TEST_MODEL="$QWEN36_TARGET" \
SPARKLAB_RESTART_MODEL_PROFILE=qwen36-triton-resident \
SPARKLAB_RESTART_PACKAGE_PATH="$PYTHONPATH" \
SPARKLAB_RESTART_EVIDENCE_ROOT="$QWEN36_RUN/restart" \
SPARKLAB_RESTART_MIN_FREE_GIB=30 \
"$HOME/projects/sparklab/.venv/bin/python" -m pytest -q -rs tests/e2e/test_prefix_restart.py
```

Cross-partition greedy differences fail their quality gate even when snapshot
bytes match. Do not weaken parity/answer oracles, silently enable MTP, substitute
the container, or interpret a skipped fixture as a pass. After correctness,
collect matched timing, cancellation/pressure and integrated endurance evidence
using the original plan and [benchmark methodology](FLARE_benchmarks.md).
