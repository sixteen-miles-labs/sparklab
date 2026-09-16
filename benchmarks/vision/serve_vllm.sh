#!/usr/bin/env bash
# Run from the repository root. Supply the pinned NVIDIA source snapshot.
set -euo pipefail
source_dir=${1:?Usage: serve_vllm.sh SOURCE_DIR target-or-recipe}
profile=${2:?Usage: serve_vllm.sh SOURCE_DIR target-or-recipe}
common=(/model --host 127.0.0.1 --port 18362 --served-model-name qwen36-vision-bench
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.5 --moe-backend marlin --attention-backend flashinfer
  --no-enable-prefix-caching --mm-processor-cache-gb 0
  --limit-mm-per-prompt '{"image":4,"video":0}'
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":1048576}}')
case "$profile" in
  target) extra=(--kv-cache-dtype bfloat16 --enforce-eager) ;;
  recipe) extra=(--kv-cache-dtype fp8 --max-cudagraph-capture-size 8
    --mm-encoder-tp-mode data --async-scheduling --enable-chunked-prefill
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}') ;;
  *) echo 'Profile must be target or recipe' >&2; exit 2 ;;
esac
docker run -d --name "sparklab-vision-bench-vllm-$profile" --gpus all --network host \
  --shm-size 16g -v "$source_dir:/model:ro" vllm/vllm-openai:v0.28.0 "${common[@]}" "${extra[@]}"
