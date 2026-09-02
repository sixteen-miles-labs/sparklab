# SparkLab quick start

Install the SparkLab distribution first; see [install.md](install.md). The supported
production target is one NVIDIA GB10.

## 1. Inspect the machine

```bash
sparklab doctor --storage-path /path/to/models
```

Resolve every failed requirement before loading a model. Warnings identify
conditions requiring review, such as device-mapper storage whose NVMe backing
cannot be proven automatically. The JSON form is intended for installers and CI:

```bash
sparklab doctor --storage-path /path/to/models --json
```

## 2. Choose a recipe

```bash
sparklab models
sparklab models --tier frontier
sparklab models --json
```

Recipe status is separate from intended tier. `experimental` and `preview`
entries do not carry the latency and stability promise of `certified`.

## 3. Plan and acquire a recipe

First check both artifact space and runtime admission. `pull` is resumable and
uses the immutable revision recorded in the recipe. `--prepare` downloads a
pinned prebuilt FTW artifact when the recipe publishes one, otherwise it builds
the self-contained FTW artifact locally. Use `--from-source` to force local
conversion.

```bash
sparklab plan qwen3.8-flash-next --prepare
sparklab pull qwen3.8-flash-next --prepare
```

Then launch the exact prepared recipe:

```bash
sparklab run qwen3.8-flash-next
```

The server is ready when the log reports that the API is listening on
`127.0.0.1:1919`.

Experimental recipes print a warning and remain non-certified. For an
uncataloged checkpoint or native-runtime experimentation, use the direct serving
command:

```bash
sparklab serve --model /path/to/checkpoint
```

## 4. Send a request

```bash
curl http://127.0.0.1:1919/v1/models

curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "served-model-id",
    "messages": [{"role": "user", "content": "Explain unified memory."}],
    "max_tokens": 256,
    "stream": true
  }'
```

SparkLab serves the OpenAI Chat Completions and Responses APIs and the
Anthropic Messages API.

## 5. Use the terminal or a coding agent

Use SparkLab's built-in terminal chat:

```bash
sparklab shell
```

Ground the terminal chat in local text or Markdown documents:

```bash
sparklab shell --documents /path/to/documents
```

SparkLab retrieves relevant excerpts for each turn and asks the model to cite
the source file and chunk. The repository includes
[`docs/sample-insurance-users.md`](sample-insurance-users.md) as synthetic test
data.

Or launch any supported coding-agent framework against the running server:

```bash
sparklab launch codex
sparklab launch claude
sparklab launch dsh
sparklab launch hermes
sparklab launch opencode
sparklab launch openclaw
```

`sparklab launch` discovers the served model through `/v1/models`, configures
the selected agent for SparkLab's local API, and starts it. If the agent CLI is
missing, SparkLab offers to install it. Cloud API credentials are removed from
the child process so it cannot silently fall back to a paid endpoint.

Preview the configuration without changing files, or configure an agent
without starting it:

```bash
sparklab launch codex --dry-run
sparklab launch codex --config
```

Arguments after `--` are passed directly to the selected agent:

```bash
sparklab launch codex -- --full-auto
```

The agent framework still provides its terminal interface, file tools, and
approval workflow, while the model's inference comes from SparkLab. Available
model-specific features can therefore differ from the framework's hosted
service.
