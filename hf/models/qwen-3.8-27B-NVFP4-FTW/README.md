# Qwen3.8-27B NVFP4 FTW publishing

`MODEL_CARD.md` is the canonical Hub README for
`oakmindai/Qwen3.8-27B-NVFP4-FTW`.

Validate the prepared target without network writes:

```bash
python hf/models/qwen-3.8-27B-NVFP4-FTW/push_weights.py \
  /path/to/prepared/qwen3.8-27b --validate-only
```

Omit `--validate-only` to publish to the empty destination. The publisher checks
the pinned fingerprint, shard sizes, tensor bounds, required metadata, and model
card before uploading. It publishes all files in one Hub commit with a parent
revision guard, then checks remote sizes and LFS hashes. An existing populated
destination is refused to prevent accidental artifact replacement.

The public index replaces the workstation-local source path with upstream
provenance. Source-safetensors checksum files are excluded because they do not
describe the FTW payload. The prepared local weights and metadata are not modified.
The optional DFlash2 draft is documented but is not uploaded here.
