# Kimi K3 NVFP4 FTW upload

This helper publishes SparkLab's validated Kimi K3 artifact to
[`oakmindai/Kimi-K3-NVFP4-FTW`](https://huggingface.co/oakmindai/Kimi-K3-NVFP4-FTW).

The default source is the validated local artifact:

```text
~/.sparklab/models/kimi-k3/prepared/0.2.0
```

Authenticate, validate without uploading, then start or resume the upload:

```bash
hf auth login
python hf/models/kimi-k3-NVFP4-FTW/push_weights.py --validate-only

HF_XET_HIGH_PERFORMANCE=1 \
  python hf/models/kimi-k3-NVFP4-FTW/push_weights.py --create --workers 8
```

The uploader checks the FTW identity, fingerprint, total byte count, all 194 indexed
shards, and the absence of stale shard files before publishing. It excludes the local
source model card and index from the bulk transfer, then commits this directory's public
model card together with an index whose machine-local source path has been redacted.

Hugging Face's large-folder state is stored under the artifact's `.cache/huggingface`
directory, so rerunning the same command resumes completed hashing, pre-upload, and commit
work. The model repository is public; do not use `--private` for the production upload.
