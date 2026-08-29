# radix prefix-cache tests

Three sibling classes — they share `RadixTreeNode` and `split_at`, and nothing else:

| class | second currency | module |
|---|---|---|
| `RadixPrefixCache` | none | `test_plain_radix.py` |
| `SWARadixCache` | sliding window | `test_swa_radix.py` |
| `HybridRadixCache` | GDN state snapshot | `test_hybrid_radix.py` |

`test_tree_and_harness.py` holds what belongs to no single class: the shared tree machinery
(`split_at` identity semantics, `key_fn`, `get_match_len`, `align_down`) covered once, and a
self-test that wraps a cache in a deliberately broken proxy and requires the battery to fire.

Every test is a named scenario. They are short because the harness does the checking: each op is
asked of the cache *and* of the reference model, the answers are compared field by field, and
`Session.check()` re-runs the whole invariant battery.

## The harness

Nothing here decides what to expect by reading the cache's own bookkeeping.

- `model.py` — the reference model, **keyed by page, not by token**. A token-level trie is wrong
  at `page_size > 1`: two different pages can share leading tokens, and the tree's reuse unit is a
  whole page. Keys are `tuple(ids[:(p+1) * page_size])`. It never reads the cache's counters or
  the node fields it exists to predict; the only things it learns from the implementation are
  opaque handles (SWA `swa_uuid`) and which of several equally-old LRU candidates was picked.
- `adapters.py` — normalizes the three genuinely different APIs onto one protocol, and holds the
  battery run after *every* op: counters recomputed from raw node fields, slot conservation,
  prefix closure, the class's own `check_integrity`, and agreement with the model.
- `driver.py` — the (class × page_size) geometries and `Session`, which asks one op of both the
  cache and the model, compares the answers, and runs the battery.
- `conftest.py` — the per-class geometry fixtures and a deterministic clock.

Slot ids are globally unique and never reused, so every returned tensor maps back to
`(path, page)` and the model can be maintained from public outputs alone.

**Use the deterministic clock for any ordering assertion.** `_tree_walk` takes one
`time.monotonic_ns()` per call and stamps every node it touches with that same value, so nodes
tie within a call and real-clock LRU assertions are flaky on a loaded box.

Every harness failure raises a `HarnessFailure` carrying a stable `tag` naming the check that
fired (`match.cached_len`, `conservation.kv`, `model.evict`, …) and both sides of the comparison.

## Mutation score

17 plausible regressions injected into the three implementations, one at a time: **17 caught**.

The 8-mutation gate — one per implementation bug class: page alignment on match and on insert,
index aliasing, both size transfers on lock, both windowed-safe boundary commits, snapshot dedup —
is carried by the named tests alone: **8/8**.

The last one to fall was `swa_radix_cache.py:108`, `match_since_tomb >= sliding_window_size` → `>`,
the boundary commit at a tombstone. ~8500 randomized runs never produced a counterexample and it was
written off here as probably equivalent; that was wrong. Randomized traces explore interleavings but
rarely land on an exact numeric boundary, and this needs a live run of *exactly* one window between
two tombstones. Reaching that shape takes both tombstone paths, because neither gets there alone:
`_stamp_path` restamps ancestors, so LRU can never age a root-side node past its own descendants
(the head tombstone has to come from `trim_head_swa`), and `evict_swa` unlinks free leaves rather
than tombstoning them, so the second tombstone has to be an internal node with a live node below it.
Under `>` the whole match collapses to 0 — silent extra prefill, not a wrong answer, which is why
nothing else noticed. `test_a_live_run_of_exactly_one_window_between_two_tombstones_is_reusable`
pins it; the sibling comparison on line 129 is caught by 14 tests.

That test is a regression gate, not independent evidence for the semantics: the reference model
makes the same `>=` choice (`run >= self.W`), so a spec of `>` would have both agreeing and both
wrong. The semantics come from elsewhere — sglang's byte-identical `_match_prefix_helper`, whose
docstring states the contract as "greater than or equal to the sliding window size", and the
arithmetic that a live run of exactly W covers a window of W.

Three further mutations were classified as equivalent while the suite was being built: `inc_lock`'s
`not cur.swa_tombstone`, `evict_swa`'s leaf `ref_count == 0`, and `_free_node_mamba`'s
`mamba_ref_count == 0` all guard states unreachable through the public API.
