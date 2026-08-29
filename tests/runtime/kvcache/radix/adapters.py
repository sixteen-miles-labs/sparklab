"""The implementation-facing side: one protocol over three sibling caches, plus the battery of
invariants checked over it after every op.

``RadixPrefixCache``, ``SWARadixCache`` and ``HybridRadixCache`` share only ``RadixTreeNode`` and
``split_at``; their public surfaces differ in names, argument lists and return shapes.  The
adapters normalize them onto one op vocabulary -- ``match``, ``insert``, ``inc_lock``/``dec_lock``,
``evict_full``/``evict_second``, ``trim_head_swa`` -- so the model, the battery and the session are
written once.  Everything crossing the boundary is a plain ``list[int]``; tensors never leak into
the model.

The battery: (a) the cache's maintained sizes vs the same sizes recomputed from raw node fields,
(b) slot conservation per currency ({live in the tree} + {handed back} == {handed out}, nothing
owned twice), (c) prefix closure (parent links, key registration, page alignment), (d) the class's
own ``check_integrity``, (e) node-for-node agreement with the page-keyed reference model.  (e) is
where regressions actually die; (a)-(d) are cheap tripwires that fire earlier and localize.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from .model import InvariantViolation, Path, Record, RefModel, sizes


class UnsupportedOp(Exception):
    """The class under test does not offer this op (e.g. ``evict_second`` on the plain radix)."""


def ids_tensor(ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.int64)


def slots_tensor(slots: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(slots), dtype=torch.int32)


@dataclass
class MatchOut:
    cached_len: int
    indices: List[int]
    node: Any                          # opaque implementation node; the lock target
    second: Optional[int] = None       # hybrid: the GDN snapshot slot to restore from


@dataclass
class InsertOut:
    matched_len: int
    freed: List[int]                   # slots the cache itself handed back (SWA only)
    second_exists: Optional[bool] = None


@dataclass
class EvictOut:
    primary: List[int]                 # full-KV page indices removed from the tree
    second: List[int]                  # swa-freed full indices / freed GDN snapshot slots


@dataclass
class LockToken:
    node: Any
    uuid: Optional[int] = None


# --------------------------------------------------------------------------- node introspection
def iter_nodes(root) -> Iterator[Tuple[Any, Any]]:
    """Every non-root node with its parent, depth-first."""
    stack = [root]
    while stack:
        n = stack.pop()
        for c in n.children.values():
            yield c, n
            stack.append(c)


def _up(node, field: str) -> List[List[int]]:
    """One tensor field of every node from the root down to (and including) ``node``."""
    out = []
    while not node.is_root():
        out.append(getattr(node, field).tolist())
        node = node.parent
    return out[::-1]


def node_end_path(node) -> Path:
    """The node's globally unique end boundary: every key token from the root down to it."""
    return tuple(t for k in _up(node, "_key") for t in k)


def node_path_slots(node) -> List[int]:
    return [s for v in _up(node, "value") for s in v]


# --------------------------------------------------------------------------- adapters
class Adapter:
    """The op vocabulary (per subclass) plus the introspection all three share, their nodes having
    the same layout."""

    kind = "base"
    second_currency: Optional[str] = None
    supports_trim = False

    def __init__(self, cache, page_size: int) -> None:
        self.cache = cache
        self.page_size = page_size

    def evict_second(self, n: int) -> EvictOut:
        raise UnsupportedOp(f"{self.kind} has no second currency")

    def counters(self) -> Dict[str, int]:
        """The cache's own INCREMENTALLY MAINTAINED running sizes."""
        raise NotImplementedError

    @property
    def root(self):
        # RadixPrefixCache calls it ``root_node``; the two siblings call it ``root``.
        return self.cache.root if hasattr(self.cache, "root") else self.cache.root_node

    def nodes(self) -> List[Any]:
        return [n for n, _ in iter_nodes(self.root)]

    def records(self) -> List[Tuple[Path, Record]]:
        """One record per node keyed by its end boundary -- the shape the model also emits."""
        return [(node_end_path(n),
                 {"length": n.length, "slots": tuple(n.value.tolist()), "ref": n.ref_count,
                  "tomb": bool(n.swa_tombstone), "swa_ref": n.swa_ref_count,
                  "swa_uuid": n.swa_uuid, "mamba": n.mamba_value,
                  "mamba_ref": n.mamba_ref_count})
                for n in self.nodes()]

    def structure(self) -> Dict[Path, Record]:
        return dict(self.records())

    def recomputed(self, records=None) -> Dict[str, int]:
        """The same sizes recomputed from raw node fields (the counter model)."""
        return sizes((r for _, r in (self.records() if records is None else records)),
                     self.second_currency)

    def tree_slots(self) -> List[int]:
        return [s for n in self.nodes() for s in n.value.tolist()]


class PlainAdapter(Adapter):
    """RadixPrefixCache.  Wart worth knowing: ``cuda_handle.get_matched_indices()`` raises on a
    cold miss (``torch.cat`` of an empty list), hence the ``cached_len == 0`` guard below."""

    kind = "plain"

    def match(self, ids: Sequence[int]) -> MatchOut:
        res = self.cache.match_prefix(ids_tensor(ids))
        h = res.cuda_handle
        idx = [] if h.cached_len == 0 else h.get_matched_indices().tolist()
        # ``mamba_value`` rides on every MatchResult; the model expects None from this class.
        return MatchOut(int(h.cached_len), idx, h.node, res.mamba_value)

    def insert(self, ids, slots, **kw) -> InsertOut:
        res = self.cache.insert_prefix(ids_tensor(ids), slots_tensor(slots))
        return InsertOut(int(res.cached_len), [], None)

    def inc_lock(self, node) -> LockToken:
        from sparklab.runtime.kvcache.radix_cache import RadixCacheHandle
        self.cache.lock_handle(RadixCacheHandle(0, node))
        return LockToken(node)

    def dec_lock(self, token: LockToken, skip_swa: bool = False) -> None:
        from sparklab.runtime.kvcache.radix_cache import RadixCacheHandle
        self.cache.lock_handle(RadixCacheHandle(0, token.node), unlock=True)

    def evict_full(self, n: int) -> EvictOut:
        return EvictOut(self.cache.evict(n).tolist(), [])

    def counters(self) -> Dict[str, int]:
        si = self.cache.size_info
        return {"full_evictable": si.evictable_size, "full_protected": si.protected_size}


class SWAAdapter(Adapter):
    kind = "swa"
    second_currency = "swa"
    supports_trim = True

    def match(self, ids: Sequence[int]) -> MatchOut:
        m = self.cache.match_prefix(ids_tensor(ids))
        return MatchOut(int(m.cached_len), m.kv_indices.tolist(), m.node)

    def insert(self, ids, slots, *, swa_evicted: int = 0, update_after: int = 0, **kw) -> InsertOut:
        matched, freed = self.cache.insert(ids_tensor(ids), slots_tensor(slots),
                                           swa_evicted_seqlen=swa_evicted,
                                           update_kv_after_len=update_after)
        return InsertOut(int(matched), freed.tolist(), None)

    def inc_lock(self, node) -> LockToken:
        return LockToken(node, self.cache.inc_lock(node))

    def dec_lock(self, token: LockToken, skip_swa: bool = False) -> None:
        self.cache.dec_lock(token.node, token.uuid, skip_swa=skip_swa)

    def evict_full(self, n: int) -> EvictOut:
        return self._evicted(self.cache.evict_full(n))

    def evict_second(self, n: int) -> EvictOut:
        return self._evicted(self.cache.evict_swa(n))

    @staticmethod
    def _evicted(ev) -> EvictOut:
        return EvictOut(ev.kv_indices.tolist(), ev.swa_indices.tolist())

    def trim_head_swa(self, ids: Sequence[int], keep_from: int) -> List[int]:
        return self.cache.trim_head_swa(ids_tensor(ids), keep_from).tolist()

    def counters(self) -> Dict[str, int]:
        c = self.cache
        return {"full_evictable": c.full_evictable, "full_protected": c.full_protected,
                "swa_evictable": c.swa_evictable, "swa_protected": c.swa_protected}


class HybridAdapter(Adapter):
    kind = "hybrid"
    second_currency = "mamba"

    def match(self, ids: Sequence[int]) -> MatchOut:
        m = self.cache.match_prefix(ids_tensor(ids))
        return MatchOut(int(m.cached_len), m.kv_indices.tolist(), m.node, m.mamba_value)

    def insert(self, ids, slots, *, mamba: int, **kw) -> InsertOut:
        matched, exists = self.cache.insert(ids_tensor(ids), slots_tensor(slots), mamba)
        return InsertOut(int(matched), [], bool(exists))

    def inc_lock(self, node) -> LockToken:
        self.cache.inc_lock(node)
        return LockToken(node)

    def dec_lock(self, token: LockToken, skip_swa: bool = False) -> None:
        self.cache.dec_lock(token.node)

    def evict_full(self, n: int) -> EvictOut:
        return self._evicted(self.cache.evict_full(n))

    def evict_second(self, n: int) -> EvictOut:
        return self._evicted(self.cache.evict_mamba(n))

    @staticmethod
    def _evicted(ev) -> EvictOut:
        return EvictOut(ev.kv_indices.tolist(), list(ev.mamba_slots))

    def counters(self) -> Dict[str, int]:
        c = self.cache
        return {"full_evictable": c.full_evictable, "full_protected": c.full_protected,
                "mamba_evictable": c.mamba_evictable, "mamba_protected": c.mamba_protected}


# --------------------------------------------------------------------------- the battery
class SlotLedger:
    """Globally unique, never-reused slot ids, plus the book of what has been handed back."""

    def __init__(self, base: int = 1_000_000, name: str = "kv") -> None:
        self.name = name
        self._next = base
        self.handed: List[int] = []
        self.handed_set: set = set()
        self.free: set = set()

    def take(self, count: int) -> List[int]:
        out = list(range(self._next, self._next + count))
        self._next += count
        self.handed.extend(out)
        self.handed_set.update(out)
        return out

    def release(self, slots: Sequence[int]) -> None:
        for s in slots:
            if s not in self.handed_set:
                raise InvariantViolation(f"conservation.{self.name}",
                                         f"slot {s} was handed back but never handed out")
            if s in self.free:
                raise InvariantViolation(f"conservation.{self.name}",
                                         f"DOUBLE FREE: slot {s} was handed back twice")
            self.free.add(s)

    def in_use(self) -> set:
        return self.handed_set - self.free


def check_counter_model(ad: Adapter, records=None) -> None:
    got, want = ad.counters(), ad.recomputed(records)
    if got != want:
        diff = {k: (got.get(k), want.get(k)) for k in set(got) | set(want)
                if got.get(k) != want.get(k)}
        raise InvariantViolation("counters", f"maintained sizes drifted from the raw-node "
                                             f"recomputation (counter vs raw): {diff}")


def check_conservation(name: str, live: Sequence[int], ledger: SlotLedger) -> None:
    """{live in the tree} + {handed back} == {handed out}, no slot in two places at once."""
    live_set = set(live)
    for suffix, bad, why in (
            (".aliasing", [s for s, c in Counter(live).items() if c > 1], "owned by two nodes"),
            ("", live_set & ledger.free, "BOTH live in the tree and handed back"),
            ("", ledger.handed_set - live_set - ledger.free, "LEAKED: neither live nor handed back"),
            ("", live_set - ledger.handed_set, "live in the tree but never handed out")):
        if bad:
            raise InvariantViolation(f"conservation.{name}{suffix}",
                                     f"{len(bad)} slot(s) {sorted(bad)[:8]} are {why}")


def check_prefix_closure(ad: Adapter) -> None:
    P = ad.page_size
    seen = set()
    for node, parent in iter_nodes(ad.root):
        for tag, ok, why in (
                ("cycle", id(node) not in seen, "is reachable twice"),
                ("parent", node._parent is parent,
                 "has a _parent other than the node whose children map reached it"),
                ("key", parent.children.get(ad.cache.key_fn(node._key)) is node,
                 "is not registered under its own first page key"),
                ("page_align", node.length % P == 0,
                 f"has length {node.length}, not a multiple of page_size={P}"),
                ("key_value", len(node._key) == len(node.value),
                 f"has {len(node._key)} key tokens but {len(node.value)} values")):
            if not ok:
                raise InvariantViolation(f"structure.{tag}", f"node {node_end_path(node)} {why}")
        seen.add(id(node))


def check_class_integrity(ad: Adapter) -> None:
    try:
        ad.cache.check_integrity()
    except AssertionError as exc:
        raise InvariantViolation("check_integrity", f"{ad.kind}.check_integrity(): {exc}") from exc


def check_model_agreement(ad: Adapter, model: RefModel, records=None) -> None:
    impl = ad.structure() if records is None else dict(records)
    ref = model.structure()
    if set(impl) != set(ref):
        raise InvariantViolation(
            "model.structure",
            f"tree node set differs from the page-keyed model: {len(impl)} impl nodes vs "
            f"{len(ref)} model nodes; only in cache: {sorted(set(impl) - set(ref))[:4]}; "
            f"only in model: {sorted(set(ref) - set(impl))[:4]}")
    for end, got in impl.items():
        if got != ref[end]:
            diff = {k: (got[k], ref[end][k]) for k in got if got[k] != ref[end][k]}
            raise InvariantViolation("model.node", f"node ending at {end} disagrees with the "
                                                   f"model (field: cache vs model): {diff}")
    got_c, want_c = ad.counters(), model.counters()
    if got_c != want_c:
        diff = {k: (got_c.get(k), want_c.get(k)) for k in set(got_c) | set(want_c)
                if got_c.get(k) != want_c.get(k)}
        raise InvariantViolation("model.counters",
                                 f"running sizes disagree with the model: {diff}")


def check_all(ad: Adapter, model: RefModel, kv_ledger: SlotLedger,
              second_ledger: Optional[SlotLedger] = None) -> None:
    recs = ad.records()
    check_prefix_closure(ad)
    check_counter_model(ad, recs)
    check_class_integrity(ad)
    check_conservation("kv", [s for _, r in recs for s in r["slots"]], kv_ledger)
    if second_ledger is not None:            # GDN snapshots: their own currency, their own ledger
        check_conservation("mamba", [r["mamba"] for _, r in recs if r["mamba"] is not None],
                           second_ledger)
    check_model_agreement(ad, model, recs)
