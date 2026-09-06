# Install SparkLab

## Requirements

- NVIDIA GB10 with 128 GB coherent unified memory
- ARM64 Linux / DGX OS, driver r580+, CUDA 13 toolkit
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

Use this for the published release. For the current Qwen3.8-Flash-Next recipe and
unreleased runtime changes documented in this checkout, use
[Method 2](#method-2-install-from-source). The released 0.1.2 wheel lacks the current
Qwen3.8-Flash-Next checkpoint support.

```bash
uv venv && source .venv/bin/activate
uv pip install "sparklab[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

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
