# GLM-5.3 Flash NVFP4 FTW upload

This helper publishes SparkLab's prepared GLM-5.3 Flash artifact to
[`oakmindai/GLM-5.3-Flash-NVFP4-FTW`](https://huggingface.co/oakmindai/GLM-5.3-Flash-NVFP4-FTW).

SparkLab source: <https://github.com/sixteen-miles-labs/sparklab>

The default source is the validated SparkLab artifact:

```text
~/.sparklab/models/glm-5.3-flash/prepared/0.3.2
```

Upload or resume it with:

```bash
huggingface-cli login
HF_XET_HIGH_PERFORMANCE=1 \
  python hf/models/glm-5.3-Flash-NVFP4-FTW/push_weights.py --create
```

The uploader verifies the FTW index, fingerprint, total byte count, and all 23 shards
before publishing. Hugging Face's large-folder upload state is resumable when the same
command is run again.
