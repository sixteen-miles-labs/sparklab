# DeepSeek V4.1 reduced reference fixture

`tiny/` contains synthetic random weights, not publisher model weights. Seed 137,
BF16, six layers, width 32, four experts, and two Engram layers. The configuration
also exercises shared KV and index sources, both compression ratios, a candidate
source, and a four-token sliding window.

`expected.json` (CPU) and `expected-gb10.json` (GB10) were generated with the publisher's `inference/model.py` at
`df42c109f1defefcbfcedbe7d905718a12266e40`, running eight sequential tokens. Its
TileLang entry points were replaced by SparkLab's Torch arithmetic helpers for
CPU/CUDA execution. CPU and CUDA BF16 reductions can choose different near-tied index candidates; each native run was compared against the reference on the same device. This independently checks the decoder wiring and state, but does
not establish kernel parity or full-checkpoint quality. Quantization has separate
known-value tests. The fixture's Engram embeddings remain FP8 with E8M0 scales.

Reference source and license:
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40/inference

The native engine also served this synthetic checkpoint through `/v1/completions`
on a GB10 during development. This is an integration smoke test only.
