# Qwen3.8 Flash Next NVFP4 FTW upload

This helper publishes the prepared SparkLab FTW artifact to
[`oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW`](https://huggingface.co/oakmindai/Qwen3.8-Flash-Next-NVFP4-FTW).

SparkLab source: <https://github.com/sixteen-miles-labs/sparklab>

The default source is SparkLab's prepared artifact:

```text
~/.sparklab/models/qwen3.8-flash-next/prepared/0.5.0
```

Upload it with:

```bash
huggingface-cli login
HF_XET_HIGH_PERFORMANCE=1 \
  python hf/models/qwen-3.8-Flash-Next-NVFP4-FTW/push_weights.py --create
```

Pass a directory as the first argument to upload a different copy. The uploader
checks for the FTW index, shards, and n-gram artifacts before starting, and its
upload state is resumable when the same command is run again.
