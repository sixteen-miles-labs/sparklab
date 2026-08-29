# Qwen3.8 Flash Next FP8 FTW upload

Upload the Qwen FP8 FTW model directory to
`oakmindai/Qwen3.8-Flash-Next-FP8-FTW`:

SparkLab source: <https://github.com/sixteen-miles-labs/sparklab>

```bash
huggingface-cli login
python hf/models/qwen-3.8-Flash-Next-FP8-FTW/push_qwen_nvfp4_ftw.py \
  /path/to/qwen-nvfp4-ftw
```

Alternatively, set `HF_TOKEN` to a Hugging Face write token instead of logging in.
The uploader is resumable, so rerunning the same command continues an interrupted
upload. It excludes Git metadata, caches, Python bytecode, and temporary files.

If the destination repository has not been created yet, add `--create` (and
optionally `--private`). Run the script with `--help` for all options.
