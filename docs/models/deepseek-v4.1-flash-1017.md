# DeepSeek V4.1 Flash 1017 — native Research

`deepseek-v4.1-flash-1017` adds the [publisher's 1017 update](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
as a separate **Experimental** recipe. It pins revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`; the full source snapshot is
510,311,045,295 bytes (about 475 GiB). Its configuration matches the original
V4.1 release: 552B total parameters and 8–16B active per token.

```bash
sparklab plan deepseek-v4.1-flash-1017
sparklab pull deepseek-v4.1-flash-1017
sparklab run deepseek-v4.1-flash-1017 --dry-run
sparklab run deepseek-v4.1-flash-1017
```

Use a source installation and local NVMe storage. Omit `--prepare`: this decoder
reads original safetensors directly. The [native V4.1 run instructions](deepseek-v4.1-flash.md)
describe the shared disk streaming, expert cache and API behavior. Use
`deepseek-ai/DeepSeek-V4.1-Flash-1017` as the API model name.

The decoder inspects the snapshot's `inference/model.py` without executing it to
select the 1017 FP8 activation rounding before the grouped attention output
projection. This preserves the original checkpoint's arithmetic and works when
the snapshot directory is renamed. Prompt rendering loads the pinned snapshot's
`encoding/encoding.py`, including its updated tool conversation handling.

Execution remains text-only, TP=1, one request, eager BF16 computation and a
2,048-token total context. Vision, prefix reuse and CUDA graphs are unsupported. DSpark is unvalidated
for 1017 and disabled in this recipe. The 96 GiB memory reservation reuses the original model's cache
allocation for identical geometry; it has not been measured on this checkpoint.

Local regression tests cover activation rounding, sequential decoding, prefill,
recipe acquisition constraints and catalog integration. Full-checkpoint inference,
quality, tool generation, performance and endurance for 1017 remain unverified.
Earlier V4.1 benchmark results do not transfer to this recipe.
