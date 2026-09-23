# DiffusionGemma structured decision checks

Run the `djev` recipe as described in [model instructions](../../docs/models/djev.md),
then:

```bash
python benchmarks/djev/bench.py --output /tmp/djev-results.json
```

The benchmark uses eight fixed synthetic support tickets, three decisions per
request, one noise draw, and a 32-token canvas. Each concurrency level (1, 4, 16,
32) gets two warmup batches at that concurrency followed by 64 requests. A unique request ID changes each
state, while the question/schema prefix remains cacheable. The script records
successful requests/second, decisions/second, mean/p50/p95 end-to-end HTTP latency,
errors, and per-question correctness. It additionally checks four-draw and
automatic sampling, two denoising steps, and red/blue image decisions.

These are small functional and performance screens, not an accuracy certification
or a reproduction of the upstream author's undisclosed ticket corpus. Score
correctness rounds the model's expected zero-based severity to the nearest level.
Compare results only with the same request count, schema, inputs, canvas,
sampling policy, and concurrency. CPU HTTP isolation tests do not substitute for
real concurrent inference.

## GB10 result (2026-09-23)

Using the pinned runtime and model in [provenance](results/provenance.json),
the [measured results](results/gb10.json) give:

| Clients | Requests/s | Decisions/s | Mean latency | P95 latency | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.88 | 29.63 | 101 ms | 114 ms | 0 |
| 4 | 21.38 | 64.15 | 187 ms | 194 ms | 0 |
| 16 | 53.50 | 160.49 | 294 ms | 336 ms | 0 |
| 32 | 62.84 | 188.53 | 493 ms | 521 ms | 0 |

The synthetic urgent and route checks were correct on all 64 requests at each
concurrency; tone was correct on 56/64. The 32 concurrent isolation checks all
passed. Four-sample, auto-sample, two-step, and red/blue image checks succeeded;
both image colors were classified correctly. The cold first request took 3.64 s
after server readiness, before this warmed benchmark. This corpus differs from
the upstream author's benchmark, so the throughput values are not a direct
speed comparison.
