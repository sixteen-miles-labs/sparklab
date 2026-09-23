# DiffusionGemma structured decisions (DJev)

Experimental, source-only integration of the Jev-like decision API from
[vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250). SparkLab
acquires the pinned NVFP4 checkpoint and runs vLLM plus its structured-decision
gateway in one supervised container. This uses vLLM diffusion execution.

## Install and run

From a SparkLab source checkout with the normal GB10 dependencies installed:

```bash
docker build -f containers/djev/Dockerfile -t sparklab-djev:gb10-v1 .
sparklab pull djev
sparklab run djev
# Choose another localhost port:
sparklab run djev -- --port 9011
```

Docker with the NVIDIA Container Toolkit is required. Build the container before
launching; no package or model conversion is needed. The image downloads roughly
10 GB of compressed layers in addition to the approximately 18.9 GB model.
SparkLab reserves 80 GiB for GPU allocations, host processes, and first-launch
compilation, plus its normal host safety reserve. The container separately caps
cgroup-accounted memory at 64 GiB and prohibits cgroup swap; that limit does not
cover every CUDA allocation on GB10. Stop other model servers if
the available-memory check fails.

Compiled kernels are retained in the Docker volume `sparklab-djev-gb10-v1-cache`
across launches; checkpoint files are mounted read-only. A clean first launch
can spend tens of minutes compiling CUDA kernels with one compiler job. The
supervisor allows up to one hour for this initial startup.

The gateway listens at `http://127.0.0.1:8011`. vLLM is private inside the
container. Interrupting the launcher stops both services; if either service
fails, the supervisor stops the other. `/health` returns 503 if vLLM is not
ready, `/v1/models` identifies the checkpoint, and `/metrics` exposes vLLM's
Prometheus metrics. Bindings other than localhost are not supported by this recipe.

## Make a decision

```bash
curl http://127.0.0.1:8011/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
    "state": {"ticket": "Everything is down and our customer demo starts in ten minutes."},
    "questions": {
      "urgent": {
        "type": "noul",
        "instructions": "Does this need an immediate response?"
      },
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"support": "An outage or technical problem", "billing": "A payment problem"}
      }
    },
    "samples": 1
  }'
```

Question types are `noul` (yes/no), `choice` (a map of alternatives), and `score`
(an ordered list of levels). The gateway converts alternatives into single-token
labels and checks their positions in the answer template. `samples: "auto"`
(the default) takes more noise draws when its entropy heuristic exceeds the
threshold; a fixed sample count is useful for reproducible latency measurements.
`depends_on`, `ask_if`, and `alone` control staged decisions. `steps` controls
canvas denoising; the default is one. `think` requests an optional reasoning
pass and adds latency.

Images can be supplied as data URLs in `images`, or multipart form data with a
JSON `request` part and image file parts. Requests are limited to 16 MiB. The
underlying model accepts images; vision quality is experimental.

`/v1/chat/completions` is a **schema-based decision interface**, not ordinary
chat: the system message contains the JSON question schema and the user message
contains the state. There are no Responses or Anthropic endpoints in this recipe.

## Pins and limits

- Model: `nvidia/diffusiongemma-26B-A4B-it-NVFP4` at
  `ec4ff3df205028f4e81c954c2227f9312b3ec2ea`.
- Container base: `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`.
- vLLM engine: official ARM64 wheel at `e9757321527ca1ecd514c07c1418dd2c53da3d19`
  (`0.29.1rc1.dev573+ge97573215`), SHA-256
  `1204dfb590c9fbefab368b6878516629bada48693157bb46f4f15cf20b8813e0`.
  The build upgrades the base runtime and verifies dependency consistency.
- Structured gateway: vLLM commit `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`;
  only the base64 import is changed. Lifecycle and request handling are wrapped
  separately by SparkLab.
- Default canvas: 32 tokens; up to 32 active vLLM sequences; maximum context
  8,192 tokens; KV cache budget 2 GiB using the checkpoint's FP8 cache scheme. Larger schemas may be split into reads.
- Returned probabilities are normalized over the provided alternatives. Confidence
  and repeated-sample standard errors are **not calibrated accuracy estimates**.
  A model can confidently choose an incorrect answer.
- Benchmark this endpoint in requests/second and decisions/second, with latency,
  schema, input length, sample policy, and concurrency recorded. Chat decode
  tokens/second does not describe this workload.

The container build disables the base image's NCCL installer override and pins
NCCL to PyTorch's declared version. NVIDIA's cuSPARSELT 0.8.1 ARM wheel has an
`sbsa` tag inside its metadata despite its `aarch64` filename; the build checks
the library's ELF architecture, corrects that metadata and its RECORD checksum,
and runs `pip check`. It does not alter cuSPARSELT executable code. The stale
optional FlashInfer JIT-cache package from the base is removed; version checks
remain enabled, and matching kernels compile on the first launch. The build
sets `MAX_JOBS=1` for FlashInfer's Ninja compiler: parallel CUDA compilation
previously exhausted the container's memory during first launch. It exposes
only `nvrtc.h` from the CUDA runtime wheel, so that wheel's other headers do
not override the CUDA 13.0 toolkit, and adds the missing unversioned NVRTC
linker name for the toolkit library.

A warmed GB10 run measured 9.88 requests/s at one client and 62.84 requests/s
at 32 clients for three decisions per request, with zero HTTP errors. The
synthetic accuracy and all request details are in the
[DJev benchmark record](../../benchmarks/djev/README.md). First launch compiles
and caches FlashInfer kernels; subsequent launches reuse the cache.
