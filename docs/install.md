# Install SparkLab

## Requirements

- NVIDIA GB10 with 128 GB coherent unified memory
- ARM64 Linux / DGX OS, driver r580+, CUDA 13 toolkit
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

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
sparklab serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).

## Optional persistent supervisor

The source tree includes a SparkLab systemd user unit.

```bash
mkdir -p ~/.config/systemd/user
cp python/sparklab/daemon/sparklab.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sparklab
systemctl --user status sparklab
```

The packaged installer links `sparklab` into `~/.local/bin` by default. If you install
into a checkout-local virtual environment, edit the unit's `ExecStart` first.
