"""Cache geometries, and the ``Session`` that drives one op at a time.

A ``Session`` pairs one real cache (through its adapter) with its own reference model: every op is
asked of both, the two answers are compared field by field, and the invariant battery runs on top.
Nothing here reads the cache's bookkeeping to decide what to expect -- expectations come from
``model.py``; the implementation is only ever *asked* and *checked*.
"""
from __future__ import annotations

import contextlib
import itertools
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .adapters import (Adapter, HybridAdapter, LockToken, PlainAdapter, SlotLedger, SWAAdapter,
                       UnsupportedOp, check_all, node_path_slots)
from .model import (ExpLock, HybridModel, ModelMismatch, PreconditionError, RefModel, SWAModel)


@contextlib.contextmanager
def deterministic_clock():
    """Patch the timestamp source the trees stamp nodes with.

    ``_tree_walk`` takes ONE ``time.monotonic_ns()`` read per call, so nodes tie *within* a call no
    matter what.  What a real clock adds is ties *between* calls (its resolution is coarser than a
    walk), which makes LRU assertions flaky; a strictly increasing counter removes those.
    """
    saved = time.monotonic_ns
    time.monotonic_ns = itertools.count(1_000_000_001).__next__   # type: ignore[assignment]
    try:
        yield
    finally:
        time.monotonic_ns = saved                                 # type: ignore[assignment]


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class CacheSpec:
    kind: str                     # "plain" | "swa" | "hybrid"
    page_size: int
    window: int = 0               # swa only

    @property
    def id(self) -> str:
        return f"{self.kind}-p{self.page_size}" + (f"-w{self.window}" if self.kind == "swa" else "")

    @property
    def unsupported(self) -> Optional[str]:
        """Why the class itself rejects this geometry, or None."""
        from sparklab.kernels.fla.chunk import CHUNK_SIZE
        if self.kind == "hybrid" and CHUNK_SIZE % self.page_size:
            return (f"HybridRadixCache requires CHUNK_SIZE({CHUNK_SIZE}) % "
                    f"page_size({self.page_size}) == 0")
        return None

    def build(self) -> Tuple[Adapter, RefModel]:
        dev, P = torch.device("cpu"), self.page_size
        if self.kind == "plain":
            from sparklab.runtime.kvcache.radix_cache import RadixPrefixCache
            return PlainAdapter(RadixPrefixCache(dev, page_size=P), P), RefModel(P)
        if self.kind == "swa":
            from sparklab.runtime.kvcache.swa_radix_cache import SWARadixCache
            w = self.window or P
            return (SWAAdapter(SWARadixCache(dev, page_size=P, sliding_window_size=w), P),
                    SWAModel(P, w))
        if self.kind == "hybrid":
            from sparklab.runtime.kvcache.hybrid_radix_cache import HybridRadixCache
            return HybridAdapter(HybridRadixCache(dev, page_size=P), P), HybridModel(P)
        raise ValueError(f"unknown cache kind {self.kind!r}")


#: The (class x page_size) matrix.  ``hybrid`` at page_size 128 is rejected by the class itself
#: (snapshots land on CHUNK_SIZE=64 boundaries) and is skipped by the fixtures.
ALL_SPECS: Tuple[CacheSpec, ...] = (
    CacheSpec("plain", 1), CacheSpec("plain", 4), CacheSpec("plain", 128),
    CacheSpec("swa", 1, window=4), CacheSpec("swa", 4, window=8), CacheSpec("swa", 128, window=128),
    CacheSpec("hybrid", 1), CacheSpec("hybrid", 4), CacheSpec("hybrid", 128),
)


def page_blocks(page_size: int, n_pages: int = 6) -> List[Tuple[int, ...]]:
    """A small alphabet of whole PAGES that deliberately share leading tokens -- at ``page_size >
    1`` exactly the shape that breaks a token-keyed model (two different pages with a common prefix
    must NOT share a tree edge), so scenarios build their keys out of these."""
    leads = max(1, n_pages // 2)
    return [(s + 1,) if page_size == 1 else ((s % leads) + 1,) + (7,) * (page_size - 2) + (s + 1,)
            for s in range(n_pages)]


# --------------------------------------------------------------------------- session
class Session:
    """Executes ops against one cache + its reference model, comparing every answer; ``check()``
    runs the whole invariant battery, which is why the scenario tests can be short."""

    def __init__(self, spec: CacheSpec) -> None:
        self.spec = spec
        self.P = spec.page_size
        self.ad, self.model = spec.build()
        self.kv = SlotLedger(1_000_000, "kv")
        self.second: Optional[SlotLedger] = (
            SlotLedger(9_000_000, "mamba") if self.ad.second_currency == "mamba" else None)

    def check(self) -> None:
        check_all(self.ad, self.model, self.kv, self.second)

    @staticmethod
    def _same(tag: str, what: str, got, want) -> None:
        """Every implementation-vs-model comparison: one stable tag each, both sides printed."""
        if got != want:
            raise ModelMismatch(tag, f"{what}: cache says {got!r}, model says {want!r}")

    # -- individual ops ------------------------------------------------------
    def do_match(self, ids: Sequence[int]):
        exp = self.model.match(ids)
        got = self.ad.match(ids)
        q = f"match_prefix({tuple(ids)})"
        self._same("match.cached_len", f"{q} reusable length", got.cached_len, exp.cached_len)
        self._same("match.indices", f"{q} kv indices", got.indices, exp.indices)
        self._same("match.node", f"{q} lock-target node's root path",
                   node_path_slots(got.node), exp.indices)
        self._same("match.second", f"{q} second-currency value", got.second, exp.second)
        return got, exp

    def do_insert(self, ids: Sequence[int], swa_evicted: int = 0, *,
                  slots: Optional[Sequence[int]] = None, reused_len: int = 0):
        if swa_evicted % self.P:               # validate BEFORE taking slots, so a rejected op
            raise PreconditionError(           # leaves no half-allocated state behind
                "precondition.swa_evicted",
                f"swa_evicted_seqlen={swa_evicted} must be a multiple of page_size={self.P}")
        if slots is None:
            slots = self.kv.take(len(ids))
        kw: Dict[str, Any] = {}
        if self.spec.kind == "swa":
            kw = {"swa_evicted": swa_evicted, "update_after": reused_len}
        elif self.spec.kind == "hybrid":
            assert self.second is not None
            kw = {"mamba": self.second.take(1)[0]}
        exp = self.model.insert(ids, slots, reused_len, **kw)
        got = self.ad.insert(ids, slots, **kw)
        q = f"insert({tuple(ids)})"
        self._same("insert.matched_len", f"{q} matched prefix", got.matched_len, exp.matched_len)
        self._same("insert.freed", f"{q} slots handed back", got.freed, exp.freed)
        self._same("insert.second_exists", f"{q} second_exists",
                   got.second_exists, exp.second_exists)
        # everything the caller keeps ownership of goes back to the ledger
        self.kv.release(list(exp.freed) + list(exp.dups)
                        + list(slots[(len(ids) // self.P) * self.P:]))
        if "mamba" in kw and exp.second_exists:
            assert self.second is not None
            self.second.release([kw["mamba"]])
        return got, exp

    def _lock(self, got, exp) -> Optional[Tuple[LockToken, ExpLock]]:
        if exp.group is None:                  # cold match: nothing to lock
            return None
        tok = self.ad.inc_lock(got.node)
        return tok, self.model.inc_lock(exp.group, tok.uuid)

    def do_lock(self, ids: Sequence[int]) -> Optional[Tuple[LockToken, ExpLock]]:
        return self._lock(*self.do_match(ids))

    def do_unlock(self, held: Optional[Tuple[LockToken, ExpLock]], skip_swa: bool = False) -> None:
        if held is None:
            return
        tok, lexp = held
        self.ad.dec_lock(tok, skip_swa=skip_swa)
        self.model.dec_lock(lexp, skip_swa=skip_swa)

    def _evict(self, which: str, n: int) -> None:
        got = getattr(self.ad, which)(n)
        getattr(self.model, which)(n, got.primary, got.second)
        self.kv.release(got.primary)
        if self.second is not None:
            self.second.release(got.second)

    def do_evict_full(self, n: int) -> None:
        if self.spec.kind == "plain":
            n = min(n, self.ad.counters()["full_evictable"])   # RadixPrefixCache.evict asserts
        self._evict("evict_full", n)

    def do_evict_second(self, n: int) -> None:
        self._evict("evict_second", n)

    def do_trim(self, ids: Sequence[int], keep_from: int) -> None:
        if not self.ad.supports_trim:
            raise UnsupportedOp(f"{self.ad.kind} has no trim_head_swa")
        exp = self.model.trim_head_swa(ids, keep_from)
        got = self.ad.trim_head_swa(ids, keep_from)
        self._same("trim.freed", f"trim_head_swa(keep_from={keep_from}) freed", got, exp)

    def do_request(self, ids: Sequence[int], prompt_pages: int, swa_evicted: int = 0) -> None:
        """A realistic request lifecycle: match a prompt, lock it, extend + commit, unlock.

        The only way the harness produces ``reused_len > 0`` inserts, and it keeps the schedulers'
        contract: the slots supplied for ``[0, reused_len)`` ARE the tree's own, so the cache must
        neither free them nor re-own them."""
        got, exp = self.do_match(tuple(ids[: prompt_pages * self.P]))
        held = self._lock(got, exp)
        slots = list(got.indices) + self.kv.take(len(ids) - got.cached_len)
        self.do_insert(ids, swa_evicted, slots=slots, reused_len=got.cached_len)
        self.do_unlock(held)
