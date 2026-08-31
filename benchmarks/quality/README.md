# Paper quality benchmark (W1-W4)

This directory implements the quality gates and serving metrics described in
[FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://arxiv.org/abs/2608.16157)
(arXiv:2608.16157v1, Section 5.1).

The paper defines these workloads:

| Option | Workload | Harness | Full-run quality gate |
|---|---|---|---|
| W1 | AIME math reasoning | native streaming client | extracted final answer matches the reference |
| W2 | SWE-bench coding, three scripted turns | OpenCode | scenario evaluator accepts the produced patch |
| W3 | same SWE issue, native Anthropic protocol | Claude Code | scenario evaluator accepts the produced patch |
| W4 | email/calendar agent, 13 fixed turns | OpenClaw | all 13 turns (and optional evaluator) succeed |

For every workload the result records request-level data and the paper's serving metrics:
per-request decode tok/s and TTFT. Agent trajectories naturally diverge, so the runner does
not use total wall-clock time as a cross-model score.

## Important reproducibility boundary

The paper and its public repository do not publish the selected SWE-bench instance, three
turns, gold-patch evaluator, W4 mailbox kit, or thirteen-turn script. They also do not state
the exact AIME edition in Section 5.1. This implementation uses the public FreeToken repo's
existing `math-ai/aime25/test.jsonl` convention for W1. W2-W4 accept explicit scenario files
so private/licensed artifacts can be supplied without inventing data. A result embeds the
scenario SHA-256 and metadata; it should be called a paper reproduction only when the authors'
exact artifacts are used.

## Start Qwen3.6 NVFP4 on SparkLab

```bash
sparklab serve \
  --model "$HOME/.sparklab/models/qwen3.6-35b-a3b/prepared/0.2.0" \
  --host 127.0.0.1 \
  --port 18080 \
  --served-model-name qwen3.6-35b-a3b \
  --max-running-requests 1 \
  --max-seq-len-override 32768 \
  --num-tokens 32832 \
  --cuda-graph-max-bs 1 \
  --moe-backend offload \
  --moe-cache-rate 1.0 \
  --nvfp4-backend triton \
  --moe-prefill-hit-d2d
```

## W1: AIME

One-problem smoke validation (not a paper-quality score):

```bash
python benchmarks/quality/run_quality.py \
  --workload W1 \
  --model qwen3.6-35b-a3b \
  --weight-format NVFP4 \
  --base-url http://127.0.0.1:18080 \
  --mode smoke \
  --problems 0 \
  --max-tokens 4096 \
  --output benchmarks/quality/results/qwen3.6-35b-a3b-nvfp4-w1-smoke.json
```

Full AIME-25 quality run:

```bash
python benchmarks/quality/run_quality.py \
  --workload W1 \
  --model qwen3.6-35b-a3b \
  --weight-format NVFP4 \
  --base-url http://127.0.0.1:18080 \
  --mode full \
  --problems all \
  --max-tokens 32768 \
  --output benchmarks/quality/results/qwen3.6-35b-a3b-nvfp4-w1-full.json
```

Omit `--temperature`, `--top-p`, and `--seed` to use the checkpoint/server sampling
defaults. Pass `--aime /path/to/test.jsonl` for an offline or pinned dataset copy.

## W2-W4: real agent harnesses

Copy and fill the appropriate file in `scenarios/`. Scenario commands are argv arrays;
the runner never invokes a shell. Supported substitutions are `{base_url}`,
`{openai_base_url}`, `{model}`, `{workspace}`, `{prompt}`, and `{step}`.
Rendered argv and client stdout are hashed but not stored by default, which avoids placing
repository or mailbox contents in a result. Pass `--include-output` only when the command and
output are safe to retain.

Configure the real client against the already-running server, then execute the scenario:

```bash
sparklab launch opencode --server http://127.0.0.1:18080 --config
cp benchmarks/quality/scenarios/w2.template.json /tmp/w2.json
# Fill /tmp/w2.json with the real repository, turns, instance id, and evaluator.
python benchmarks/quality/run_quality.py \
  --workload W2 --model qwen3.6-35b-a3b --weight-format NVFP4 \
  --base-url http://127.0.0.1:18080 --scenario /tmp/w2.json \
  --output benchmarks/quality/results/qwen3.6-35b-a3b-nvfp4-w2.json
```

Use `sparklab launch claude ... --config` with `w3.template.json`, and
`sparklab launch openclaw ... --config` with `w4.template.json`. Full W2/W3 runs require
exactly three steps and a passing evaluator. Full W4 runs require exactly thirteen steps.
`--mode smoke --allow-partial-scenario` is available only to validate client wiring and is
always labeled as smoke in the output.

## Supported models and validation

`models.json` contains the three paper models and their reported weight formats. An arbitrary
served model id is also accepted, but its weight format remains mandatory so results cannot
silently mix quantizations.

```bash
python -m pytest -q benchmarks/quality/test_quality.py
python -m json.tool benchmarks/quality/models.json >/dev/null
python -m json.tool benchmarks/quality/workloads.json >/dev/null
```
