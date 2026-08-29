# sparklab-kernel-cache

`sparklab-kernel-cache` is a companion wheel for `sparklab`. It ships prebuilt
TVM FFI kernel shared libraries so common runtime paths can load `.so` files
directly instead of compiling with nvcc on first use.

## Build Both Wheels

Use the top-level helper to build the runtime wheel and the matching kernel-cache
wheel in one command:

```bash
scripts/build-release-wheels.sh
```

By default, artifacts are written to `dist/`:

```text
dist/sparklab-<version>-cp312-cp312-linux_x86_64.whl
dist/sparklab_kernel_cache-<version>+cu130-py3-none-linux_x86_64.whl
```

Useful knobs:

```bash
SPARKLAB_BUILD_OUT_DIR=/tmp/sparklab-dist \
SPARKLAB_BUILD_PYTHON=.venv/bin/python \
scripts/build-release-wheels.sh
```

To build only a subset of the cache for quick checks:

```bash
SPARKLAB_KERNEL_CACHE_SPECS=sparklab__store_1024_128_1_false \
scripts/build-release-wheels.sh
```

## Wheel Contents

The cache wheel contains one directory per kernel:

```text
sparklab_kernel_cache/
  jit_cache/
    sparklab__store_1024_128_1_false/
      sparklab__store_1024_128_1_false.so
```

At runtime, `sparklab.kernels.utils.load_jit()` and `load_aot()` look for
`sparklab_kernel_cache.get_jit_cache_dir()` and load
`<jit_cache>/<kernel_name>/<kernel_name>.so` before falling back to JIT.

## Install

`install.sh` installs both wheels. Pass both explicitly:

```bash
SPARKLAB_WHEEL=dist/sparklab-0.1.1-cp312-cp312-linux_x86_64.whl \
SPARKLAB_KERNEL_CACHE_WHEEL=dist/sparklab_kernel_cache-0.1.1+cu130-py3-none-linux_x86_64.whl \
bash install.sh
```

If the cache wheel is next to the runtime wheel, `install.sh` can auto-detect the
sibling `sparklab_kernel_cache-*.whl` when only `SPARKLAB_WHEEL` is set.

For release validation, disable runtime JIT:

```bash
SPARKLAB_DISABLE_JIT=1 sparklab serve --model <path>
```

With this flag, any cache miss fails immediately instead of compiling at runtime.
