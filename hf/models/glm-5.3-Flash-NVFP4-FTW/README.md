# GLM-5.3 Flash NVFP4 FTW upload

This helper publishes Spark Lab's prepared GLM-5.3 Flash artifact to
[`oakmindai/GLM-5.3-Flash-NVFP4-FTW`](https://huggingface.co/oakmindai/GLM-5.3-Flash-NVFP4-FTW).

The default source is the validated Spark Lab artifact:

```text
~/.sparklab/models/glm-5.3-flash/prepared/0.3.1
```

Upload or resume it with:

```bash
huggingface-cli login
HF_XET_HIGH_PERFORMANCE=1 \
  python hf/models/glm-5.3-Flash-NVFP4-FTW/push_weights.py --create
```

The uploader verifies the FTW index, fingerprint, total byte count, and all 24 shards
before publishing. Hugging Face's large-folder upload state is resumable when the same
command is run again.
