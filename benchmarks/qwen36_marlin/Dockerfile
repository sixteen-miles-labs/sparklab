# Qwen3.6 vLLM donor environment, separate from the native torch 2.11 install.
# Build from the repository root. The base and added packages are pinned.
FROM vllm/vllm-openai:v0.28.0@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
WORKDIR /workspace
RUN uv pip install --system flashlib==0.3.0 gguf==0.19.0 prompt-toolkit==3.0.53 wcwidth==0.8.3
COPY pyproject.toml setup.py README.md LICENSE NOTICE ./
COPY python ./python
RUN MAX_JOBS=4 python3 setup.py egg_info build_ext --inplace
ENV PYTHONPATH=/workspace/python SPARKLAB_DISABLE_KERNEL_CACHE=1
ENTRYPOINT ["python3", "-m", "sparklab"]
