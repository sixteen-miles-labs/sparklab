# Bonsai 2 native decode optimization

Measured September 18, 2026, on the same DGX Spark. The default PTQ1 native decoder is faster after reusing packed ternary bytes inside GEMV and enabling CUDA graphs for text-only decode. The Q8 vision tower remains loaded; image requests retain eager execution.

| Workload | Native before | Native after | Gain | Prism reference |
|---|---:|---:|---:|---:|
| short | 19.75 tok/s | 26.95 tok/s | 36.5% | 33.42 tok/s |
| long | 18.73 tok/s | 25.22 tok/s | 34.7% | 30.42 tok/s |

The before/reference values are the preceding [isolated runs](ISOLATION.md), not the older runs with vLLM resident. The optimized native runs also had no other model serving process resident. vLLM remained stopped. Short uses 96 input / 128 output tokens and three trials after one warmup. Long uses the same first three frozen 8K requests, 256 output tokens, and one warmup. Values are client-observed medians; this is not a full 30-request qualification.

Long TTFT was 9.030 s versus 8.798 s previously; this change targets decode, not large-prefill performance.

## Changes and measurement

- The [32-step CUDA profile](results/optimization/profile-before.txt) attributed 91.77% of decode GPU time to `_gemv`. Graph replay alone only reached 20.28 tok/s.
- PTQ1 GEMV now loads each packed byte once and reuses it for its five trits (four in the tail). It reduces within each group before applying the FP16 group scale, using BF16 activations and FP32 accumulation. Resident weights and the activation precision are unchanged. PQ2 retains its prior GEMV implementation.
- The final tile is 16 output rows by eight 128-weight groups. Wider tiles that looked promising in repeated single-matrix microbenchmarks fell to 20.24 tok/s end-to-end and were rejected. A 32-bit packed-word alternative was also slower than the selected byte-reuse kernel and was discarded.
- Only the Bonsai vision configuration opts into text decode graphs. The graph eligibility check rejects image requests both before and after multimodal position metadata is assembled, so their image embeddings and MRoPE stay on the eager path. Other Qwen vision profiles retain their existing graph policy.
- The recipe now selects `cuda_graph_max_bs=1`. Its weight files and revision are unchanged.

## Validation and numerical limits

- 154 CPU tests passed; one optional Gemma GGUF fixture test skipped. 12 GPU tests passed, including representative widths and the small non-power-of-two fallback.
- Text/image integration: 8/8. Multi-image and tool-call round-trip: 3/3. Checks also exercise returning to text graph replay after image requests.
- Five diagnostic prompts retain 5/5 next-token argmax agreement with Prism. Top-100 log-probability mean absolute errors range from 0.0269 to 0.0514; raw reports are included.
- A full-answer check of AIME-25 problem 0 passed in both optimized native and Prism: both returned `70` and stopped normally, using 676 and 733 completion tokens respectively at a 2,048-token budget. This is one additional reasoning check, not an accuracy benchmark.
- The reduction order changes FP32 rounding. Outputs are **not bit-identical**: the AIME speed probe generates different reasoning wording, consistently across all three trials. This is not evidence of general quality parity. No additional activation quantization was introduced.

The rebuilt wheel and the normal recipe pull/run path also passed. The recipe
captured a size-one graph, served queued image/text clients, and resumed text
generation after the image request.

[Source hashes and validation summary](results/optimization/provenance.json).

## Reproduction

Use the same pinned models and packages as the [initial comparison](README.md). Launch the current recipe, or use the documented direct PTQ1 server command with `--cuda-graph-max-bs 1`. For the final numerical diagnostics we used `capture_server.py` with the same model/server flags; the hook only captures a prefill when explicitly armed.

Repeat `bench_single_stream.py` and `latency.py` with the commands in the initial comparison. Run a single backend at a time for timing. The final result files are `results/optimization/final-short.json` and `final-long.json`.

The repeatable GEMV microbenchmark is:

```bash
PYTHONPATH=python python benchmarks/bonsai2/bench_gemv.py \
  --model /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --output gemv.json
# Wider tile candidates:
PYTHONPATH=python python benchmarks/bonsai2/bench_gemv.py \
  --model /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --rows 32 64 --output wide.json
```

For a CUDA decode trace, use `profile_server.py` in place of the server entrypoint, then touch `/tmp/bonsai-profile.request` and send a request with at least 33 generated tokens. The diagnostic wrapper profiles 32 graph replays and writes `/tmp/bonsai-profile.txt` and `/tmp/bonsai-profile.json`. Never use profiled request timing as throughput evidence.
