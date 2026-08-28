# Qwen3.6 35B-A3B NVFP4 FTW upload

Upload a local model directory to
`oakmindai/Qwen3.6-35B-A3B-NVFP4-FTW`:

```bash
huggingface-cli login
python hf/models/qwen-3.6-35B-A3B-NVFP4-FTW/push_weights.py \
  /path/to/qwen3.6-35b-a3b-nvfp4-ftw
```

Alternatively, set `HF_TOKEN` to a Hugging Face write token. The large-folder
uploader resumes interrupted uploads when the same command is run again.

If the repository does not exist yet, add `--create` (and optionally
`--private`). Run the script with `--help` for all options.
