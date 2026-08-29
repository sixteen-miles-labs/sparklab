# SparkLab CLI reference

```
sparklab <command> [args]
```

| Command | Purpose |
|---|---|
| `sparklab doctor` | Validate GB10, CUDA 13, unified memory, storage, and dependencies |
| `sparklab models` | List versioned Fast, Frontier, and Research recipes |
| `sparklab plan` | Check recipe artifact space and runtime-memory admission |
| `sparklab pull` | Resume a pinned checkpoint acquisition and optionally prepare FTW |
| `sparklab run` | Launch a prepared recipe after fail-closed GB10 admission |
| `sparklab gate` | Evaluate versioned full-model evidence against product tier gates |
| `sparklab status` | Show the persistent engine status |
| `sparklab serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `sparklab shell` | Chat with a server in the terminal |
| `sparklab ctl` | Query and manage a running server over HTTP |
| `sparklab launch` | Configure and launch a coding agent against a server |
| `sparklab checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `sparklab bench bw` | Benchmark CPU vs expert-transfer bandwidth |

`sparklab --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## sparklab doctor

```bash
sparklab doctor [--storage-path PATH] [--json] [--strict]
```

The human report explains every platform failure and warning. The JSON report is
schema-versioned for installers and CI. A supported GB10 can still be not ready—for
example, if swap is already in use or less than the operational memory reserve is
available.

## sparklab models

```bash
sparklab models [--tier fast|frontier|research] \
  [--status certified|preview|experimental] \
  [--role primary|fallback] [--json]
```

Tier is the intended product layer. Portfolio role separates the primary lineup from
retained fallbacks. Status records how much of the target layer has been proven. Human
output groups models under one tier-description separator row and reports accepted GB10
performance; JSON output includes the underlying evidence-bound performance fields.

## sparklab plan

```bash
sparklab plan <recipe> [--root PATH] [--prepare] [--from-source] [--json]
```

Reports source/prepared artifact paths, required and free storage, and the
recipe's unified-memory admission result without loading weights. A non-zero
exit means at least one admission check failed or still lacks measured data.

## sparklab pull

```bash
sparklab pull <recipe> [--root PATH] [--prepare] [--from-source] [--dry-run] [--json]
```

Downloads an immutable Hugging Face revision into the SparkLab state root.
Downloads are resumable. With `--prepare`, a recipe-pinned prebuilt FTW artifact
is preferred when available; otherwise SparkLab downloads and converts the
source checkpoint. `--from-source` forces local conversion, and `--dry-run`
performs planning only.

## sparklab run

```bash
sparklab run <recipe> [--root PATH] [--dry-run] [--json] [-- <extra serve args>]
```

Resolves the prepared artifact and recipe-owned runtime flags, then refuses to
launch unless the checkpoint manifest and GB10 memory plan pass. Experimental
recipes remain visibly non-certified when launched for engineering work.

## sparklab gate

```bash
sparklab gate <recipe> <evidence.json> [--tier fast|frontier|research] [--json]
```

Fails closed if evidence belongs to another recipe version/revision or omits
full-model identity, correctness, parser, coding-task, memory, or tier-specific
performance/context/stability/NVMe proof. Passing this command is necessary but
does not itself edit or promote a catalog recipe.

## sparklab serve

```bash
sparklab serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV) |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |

### MoE offload

See the [FTW and NVMe execution notes](models.md#ftw-and-nvme-execution) for
the model-portfolio context behind offloaded execution.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-backend` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid with a `sparklab bench bw` profile |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family backends) |
| `--moe-cache-policy` | `lru` | `lru`, or borrowable `layer_lru` protection applied to both the GPU slot cache and disk host LRU |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`: which MoE layers decode on CPU (`3,7,11`, a count, or a fraction) |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--moe-prefill-sparse-max-tokens N` | 0/off | For short prefills up to `N` tokens, route first and stage only unique active experts through the persistent cache |
| `--moe-shared-expert-overlap` | off | Disk mode: overlap supported models' resident shared-expert CUDA work with routed-row staging |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |

## sparklab shell

```bash
sparklab shell                                    # attach to a running server
sparklab shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## sparklab ctl

```bash
sparklab ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## sparklab launch

```bash
sparklab launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## sparklab checkpoint

```bash
sparklab checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--nvfp4-backend triton] [--shard-gib 8] [--device cuda:0]
```

Converts an HF safetensors checkpoint to FTW, the engine's self-contained
fast-load format; point `sparklab serve --model` at the output dir. `--moe-backend
offload` (default) packs experts into offload banks; `--moe-backend triton`
keeps them dense for resident serving. See the
[FTW and NVMe execution notes](models.md#ftw-and-nvme-execution). NVFP4 layouts
are backend-owned, so choose the
same `--nvfp4-backend` at conversion and serve time; `auto` selects by GPU,
while `flashinfer` forces the SM12x b12x layout.

## sparklab bench bw

```bash
sparklab bench bw                       # once per machine
sparklab bench bw --dtype nvfp4,bf16    # only the formats you serve
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and
writes a profile (`~/.cache/sparklab/benchbw.json`) that `sparklab serve
--moe-backend auto` and `--moe-hybrid-max-fetch -1` read. Profiles are keyed on
expert format + GPU name, so a profile from different hardware is ignored
rather than misapplied. Selection flags: `--dtype`, `--model`, `--formats`,
`--isa`; decision rule: `--threshold` (default 2.0 — recommend hybrid when CPU
bandwidth > 2× PCIe).
