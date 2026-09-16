# Install SparkLab

## Requirements

- NVIDIA GB10 with 128 GB coherent unified memory
- ARM64 Linux / DGX OS, driver r580+, CUDA 13 toolkit
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install the release

For the complete 0.1.3 release with its matching prebuilt CUDA kernel cache, use
the managed installer. It creates a fresh environment and verifies the runtime,
kernel-cache, and CUDA versions before reporting success.
The complete accelerated environment downloads several gigabytes of CUDA and
FlashInfer packages on a cold cache.

```bash
curl -fsSL https://raw.githubusercontent.com/sixteen-miles-labs/sparklab/v0.1.3/install.sh | bash
```

Use PyPI directly when a CUDA 13 toolkit with `nvcc` is available:

```bash
uv venv && source .venv/bin/activate
uv pip install "sparklab[accel]"
```

CUDA kernels are JIT-compiled on first use and need a CUDA 13 toolkit with `nvcc`
on `PATH`.
If this environment contains a kernel-cache wheel from another SparkLab release,
SparkLab ignores it with a warning and uses JIT compilation. Install the matching
cache wheel or remove the stale package to avoid that compilation.

The `sparklab` distribution and package metadata are maintained by SixteenMiles Labs.

## Method 2: Install from source

```bash
git clone https://github.com/sixteen-miles-labs/sparklab.git && cd sparklab
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Verify

```bash
source .venv/bin/activate
sparklab --version
sparklab doctor --storage-path ~/models
sparklab models
```

Then follow the [quick start](quickstart.md) to plan, download, and launch a recipe
before sending an API request.

## Optional persistent supervisor

The source tree includes a SparkLab systemd user unit.

```bash
mkdir -p ~/.config/systemd/user
cp python/sparklab/daemon/sparklab.service ~/.config/systemd/user/
```

Before enabling the service, edit `~/.config/systemd/user/sparklab.service` so
`ExecStart` uses the absolute path to your environment's `sparklab` executable
(shown by `command -v sparklab` after activation). The reference unit defaults to
`~/.local/bin/sparklab`, which `install.sh` provides; the virtual-environment installs
above do not create that link.

```bash
systemctl --user daemon-reload
systemctl --user enable --now sparklab
systemctl --user status sparklab
```

See the [supervisor guide](../python/sparklab/daemon/README.md) for configuration.
