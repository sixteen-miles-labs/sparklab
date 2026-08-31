#!/usr/bin/env bash
set -euo pipefail

framework=${1:?usage: setup_framework.sh FRAMEWORK}
root=${SPARKLAB_FRAMEWORK_ROOT:-$HOME/.local/share/sparklab-frameworks}
mkdir -p "$root"

case "$framework" in
  vllm)
    uv venv --python 3.12 "$root/vllm"
    uv pip install --python "$root/vllm/bin/python" 'vllm==0.28.0'
    ;;
  sglang)
    uv venv --python 3.12 "$root/sglang"
    uv pip install --python "$root/sglang/bin/python" 'sglang[all]==0.5.18'
    ;;
  ktransformers)
    uv venv --python 3.11 "$root/ktransformers"
    uv pip install --python "$root/ktransformers/bin/python" 'ktransformers==0.7.0.post1'
    ;;
  llama.cpp)
    if [[ ! -d "$root/llama.cpp/.git" ]]; then
      git clone https://github.com/ggml-org/llama.cpp "$root/llama.cpp"
    fi
    git -C "$root/llama.cpp" pull --ff-only
    cmake -S "$root/llama.cpp" -B "$root/llama.cpp/build" \
      -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
    # Keep CUDA compilation below the host-memory safety margin. Unbounded
    # parallel compilation previously contributed to a kernel OOM on this host.
    cmake --build "$root/llama.cpp/build" --config Release --target llama-server -j "${MAX_JOBS:-4}"
    ;;
  ollama)
    mkdir -p "$root/ollama/bin"
    archive=$(mktemp)
    curl -fL https://ollama.com/download/ollama-linux-arm64.tar.zst -o "$archive"
    tar --zstd -xf "$archive" -C "$root/ollama"
    rm -f "$archive"
    ;;
  freetoken)
    src="$root/freetoken-src"
    if [[ ! -d "$src/.git" ]]; then
      git clone https://github.com/FlashML-org/FreeToken.git "$src"
    fi
    git -C "$src" fetch origin 4b94bdc38a46a4dfe534e8793126160d56904c44
    git -C "$src" checkout --detach 4b94bdc38a46a4dfe534e8793126160d56904c44
    uv venv --python 3.12 "$root/freetoken"
    uv pip install --python "$root/freetoken/bin/python" -e "$src[accel]"
    ;;
  *)
    echo "unknown framework: $framework" >&2
    exit 2
    ;;
esac
