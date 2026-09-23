# Cached ARM64/CUDA 13 base; upgrade using the official wheel at a pinned commit.
FROM vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
RUN curl -fL --retry 3 'https://wheels.vllm.ai/e9757321527ca1ecd514c07c1418dd2c53da3d19/vllm-0.29.1rc1.dev573%2Bge97573215-cp38-abi3-manylinux_2_28_aarch64.whl' \
      -o /tmp/vllm-0.29.1rc1.dev573+ge97573215-cp38-abi3-manylinux_2_28_aarch64.whl \
    && echo '1204dfb590c9fbefab368b6878516629bada48693157bb46f4f15cf20b8813e0  /tmp/vllm-0.29.1rc1.dev573+ge97573215-cp38-abi3-manylinux_2_28_aarch64.whl' | sha256sum -c -
COPY containers/djev/normalize_sbsa.py /tmp/normalize_sbsa.py
RUN --mount=type=cache,target=/opt/uv/cache env -u UV_OVERRIDE uv pip install --system \
      /tmp/vllm-0.29.1rc1.dev573+ge97573215-cp38-abi3-manylinux_2_28_aarch64.whl \
      huggingface-hub==1.32.0 instanttensor==0.2.0 nvidia-nccl-cu13==2.29.7 \
      'https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18.post1/flashinfer_cubin-0.6.18.post1-py3-none-any.whl#sha256=bbacb5b8bbf429e43bf2740bb45551d981f41d0462842e36ee6fb3e3452763ab' \
    && python3 /tmp/normalize_sbsa.py \
    && python3 -m pip check \
    && rm /tmp/vllm-*.whl
# The cached base carries an incompatible optional FlashInfer AOT/JIT bundle.
# Keep version checks enabled and compile against the installed version.
RUN uv pip uninstall --system flashinfer-jit-cache
COPY python/sparklab/__init__.py python/sparklab/version.py /opt/sparklab/python/sparklab/
COPY python/sparklab/external /opt/sparklab/python/sparklab/external
ENV PYTHONPATH=/opt/sparklab/python
# The NVRTC wheel provides nvrtc.h, but its other runtime headers conflict with
# this image's CUDA 13.0 nvcc. Expose only the standalone NVRTC header.
RUN mkdir -p /opt/sparklab/nvrtc-include \
    && cp /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/nvrtc.h /opt/sparklab/nvrtc-include/ \
    && ln -s libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so
ENV CPATH=/opt/sparklab/nvrtc-include
# FlashInfer uses Ninja for first-run kernels. Eight parallel nvcc jobs exhausted
# even a 64 GiB cgroup during model warmup on GB10.
ENV MAX_JOBS=1
ENV HF_HUB_OFFLINE=1
LABEL org.sparklab.backend="djev" org.sparklab.djev.upstream="1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8" org.sparklab.vllm.commit="e9757321527ca1ecd514c07c1418dd2c53da3d19"
ENTRYPOINT ["python3", "-m", "sparklab.external.djev.supervisor"]
