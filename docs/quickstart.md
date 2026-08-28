# Spark Lab quick start

This early product path assumes the compatibility distribution is installed; see
[install.md](install.md). The supported production target is one NVIDIA GB10.

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
uncataloged checkpoint or engine-level experimentation, use the compatibility
surface directly:

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

Spark Lab serves the OpenAI Chat Completions and Responses APIs and the
Anthropic Messages API.

## 5. Use the terminal or a coding agent

```bash
sparklab shell
sparklab launch codex
```

The legacy `ft` forms remain valid. See [migration.md](migration.md).
