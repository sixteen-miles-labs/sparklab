# Previous Qwen3.8-27B Inferact profile

Recipe 0.3.0 used `Inferact/Qwen3.8-27B-NVFP4` at revision
`6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`, with native W4A16 prefill and
optional DFlash2-12. These commands reproduce that profile independently of the
current recipe. It remains an experimental, text-only configuration for one GB10.

```bash
hf download Inferact/Qwen3.8-27B-NVFP4 \
  --revision 6128240ebaf4eaa7bad2b3d1c72c37d677c5f462 \
  --local-dir /path/to/inferact-source
sparklab checkpoint --model /path/to/inferact-source \
  --out /path/to/inferact-prepared --nvfp4-backend triton --device cuda:0

sparklab serve --model /path/to/inferact-prepared \
  --served-model-name qwen27-inferact \
  --nvfp4-backend triton --nvfp4-prefill-backend w4a16 \
  --cuda-graph-max-bs 1 --cache-type radix --page-size 16 \
  --max-running-requests 1 --max-seq-len-override 65536 --num-tokens 65536 \
  --attention-backend triton --host 127.0.0.1 --port 1919
```

For the previous speculative profile, download its pinned draft:

```bash
hf download maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal \
  --revision bd7a934213c47a9e7ef69eef36bb3325f47fd1f1 \
  --local-dir /path/to/dflash-draft
```

Append `--speculative-method dflash2 --speculative-tokens 12
--speculative-draft-model /path/to/dflash-draft` to the server command.
`SPARKLAB_DFLASH2_REJECT_DRAFT=1` and `SPARKLAB_DFLASH2_VERIFY_GRAPH=1` are the
measured defaults. Use `temperature: 0` in requests to exercise speculative decode;
sampled requests use target-only generation. Multiple clients queue behind one
active request.

The September 8 matched 30-request benchmark, with roughly 8K input tokens and
256 generated tokens per request, measured 8.92 decode tok/s and 7.10 s mean
time to first token for target-only serving. DFlash2-12 measured 22.05 tok/s and
7.47 s respectively. The earlier 45.88 tok/s short-prompt result used a different
workload and should not be compared directly with this benchmark.

See the [matched benchmark](../../benchmarks/gb10/QWEN38_27B_VLLM.md) and
[earlier speculative evidence](../../benchmarks/gb10/results/GB10-QWEN38-DFLASH-004.json).
W4A16 retains BF16 activations, but it does not establish quality equivalence with
unquantized weights or another engine. The earlier Inferact DFlash regression
screen passed 23 of 26 checks, including all four long-prompt cases; it failed two
JSON checks and one arithmetic check.
