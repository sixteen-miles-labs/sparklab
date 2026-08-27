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

## 3. Launch the compatibility engine

Recipe-backed `sparklab pull` and `sparklab run` are still being implemented. In
this first migration slice, use the Spark Lab alias for the existing engine:

```bash
sparklab serve --model /path/to/checkpoint
```

The server is ready when the log reports that the API is listening on
`127.0.0.1:1919`.

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
