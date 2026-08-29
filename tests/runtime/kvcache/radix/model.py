"""Page-keyed reference model for the three radix prefix caches.

THE MODEL IS KEYED BY PAGE, NOT BY TOKEN.  A token-level trie is demonstrably wrong once
``page_size > 1``: the tree's reuse unit is a whole page and two *different* pages may share
leading tokens, so a token trie would predict sharing (and an index) that the real tree never
produces.  Every page is identified by its page-aligned prefix ``tuple(ids[:(p + 1) * page_size])``
-- a globally unique name for "this page, reached this way".

This is a full, independent re-implementation of the three caches at page granularity.  It never
reads the implementation's bookkeeping to decide what to expect; the only things it learns from the
implementation are *opaque handles* (SWA ``swa_uuid`` values) and, for eviction, which of several
equally-old LRU candidates was picked (ties are inherent -- see ``_pick_victim``).

``RefModel`` is RadixPrefixCache (one currency) plus the machinery ``SWAModel`` (a per-node ``tomb``
bit; the window keeps no slot of its own, its KV rides the full page indices) and ``HybridModel``
(an optional GDN snapshot slot per node) extend.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Path = Tuple[int, ...]
Record = Dict[str, object]

# SWARadixCache multiplies its logical clock by this so one event can stamp a whole path with
# strictly decreasing values.  The model mirrors the formula so LRU *order* is identical.
_EVENT_STRIDE = 1 << 24


class HarnessFailure(AssertionError):
    """Every harness-detected failure carries a stable ``tag``, so a caller can tell 'the same bug'
    from 'some other bug'."""

    def __init__(self, tag: str, message: str) -> None:
        super().__init__(f"[{tag}] {message}")
        self.tag = tag
        self.detail = message


class ModelMismatch(HarnessFailure):
    """The implementation disagreed with the independent page-keyed model."""


class InvariantViolation(HarnessFailure):
    """A universal invariant (accounting / conservation / structure) broke."""


class PreconditionError(HarnessFailure):
    """The caller asked for something outside the cache's documented contract."""


def _fail(tag: str, msg: str) -> "ModelMismatch":
    return ModelMismatch(tag, msg)


def sizes(records: Iterable[Record], currency: Optional[str]) -> Dict[str, int]:
    """The running sizes, recomputed from raw per-node records.

    One formula over two independently produced sets of records: the cache's nodes
    (``Adapter.records``) and the model's own.  Recomputing rather than maintaining is the point --
    an incremental counter bug on either side then has nothing to hide behind.
    """
    recs = list(records)
    out = {"full_evictable": sum(r["length"] for r in recs if r["ref"] == 0),
           "full_protected": sum(r["length"] for r in recs if r["ref"] > 0)}
    if currency == "swa":
        live = [r for r in recs if not r["tomb"]]
        out["swa_evictable"] = sum(r["length"] for r in live if r["swa_ref"] == 0)
        out["swa_protected"] = sum(r["length"] for r in live if r["swa_ref"] > 0)
    elif currency == "mamba":
        live = [r for r in recs if r["mamba"] is not None]
        out["mamba_evictable"] = sum(1 for r in live if r["mamba_ref"] == 0)
        out["mamba_protected"] = sum(1 for r in live if r["mamba_ref"] > 0)
    return out


# --------------------------------------------------------------------------- trie primitives
class MGroup:
    """One radix *node*: a contiguous run of pages sharing a single tree edge.

    ``paths[i]`` is the page-aligned prefix ending at the i-th page, so ``paths[-1]`` names the
    node's end boundary; ``slots`` holds ``page_size`` slot ids per page, flattened."""

    __slots__ = ("paths", "slots", "stamp", "ref", "tomb", "swa_ref", "swa_uuid",
                 "mamba", "mamba_ref")

    def __init__(self, paths: Sequence[Path], slots: Sequence[int], stamp: int):
        self.paths: List[Path] = list(paths)
        self.slots: List[int] = list(slots)
        self.stamp = stamp
        self.ref = 0
        self.tomb = False
        self.swa_ref = 0
        self.swa_uuid: Optional[int] = None
        self.mamba: Optional[int] = None
        self.mamba_ref = 0

    length = property(lambda self: len(self.slots))
    n_pages = property(lambda self: len(self.paths))
    end = property(lambda self: self.paths[-1])
    start = property(lambda self: self.paths[0])

    def record(self) -> Record:
        return {"length": self.length, "slots": tuple(self.slots), "ref": self.ref,
                "tomb": self.tomb, "swa_ref": self.swa_ref, "swa_uuid": self.swa_uuid,
                "mamba": self.mamba, "mamba_ref": self.mamba_ref}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"MGroup(end={self.end}, len={self.length}, ref={self.ref}, tomb={self.tomb},"
                f" swa_ref={self.swa_ref}, mamba={self.mamba})")


class PageTrie:
    """Page-keyed trie.  ``kids[path][page_key] = child_path``; the root's path is ``()``.

    Page paths are absolute, so splitting a node never renames a page -- only the grouping changes.
    That is what makes the page keying cheap *and* faithful."""

    def __init__(self, page_size: int) -> None:
        self.P = page_size
        self.page_group: Dict[Path, MGroup] = {}
        self.kids: Dict[Path, Dict[Path, Path]] = {(): {}}
        self._groups: List[MGroup] = []
        #: Branch-coverage counters, so a scenario test can assert it really reached the branch it
        #: claims to test (``assert model.events["insert.revive_whole"] == 1``).
        self.events: Counter = Counter()

    def child(self, path: Path, key: Path) -> Optional[Path]:
        return self.kids.get(path, {}).get(key)

    def groups(self) -> List[MGroup]:
        return list(self._groups)

    def parent_group(self, g: MGroup) -> Optional[MGroup]:
        pp = g.start[: -self.P]
        return None if not pp else self.page_group[pp]

    def is_leaf(self, g: MGroup) -> bool:
        return not self.kids.get(g.end)

    def walk(self, ids: Sequence[int]) -> List[Path]:
        """Longest page-aligned prefix of ``ids`` present in the trie, as a list of page paths."""
        out: List[Path] = []
        cur: Path = ()
        for i in range(len(ids) // self.P):
            nxt = self.kids.get(cur, {}).get(tuple(ids[i * self.P: (i + 1) * self.P]))
            if nxt is None:
                break
            out.append(nxt)
            cur = nxt
        return out

    def path_slots(self, g: Optional[MGroup]) -> List[int]:
        """Slots from the root down to (and including) ``g``."""
        chain: List[MGroup] = []
        while g is not None:
            chain.append(g)
            g = self.parent_group(g)
        return [s for node in reversed(chain) for s in node.slots]

    def path_len(self, g: Optional[MGroup]) -> int:
        return 0 if g is None else len(g.end)

    def add_group(self, parent: Optional[MGroup], keys: Sequence[Path],
                  slots: Sequence[int], stamp: int) -> MGroup:
        assert keys, "cannot add an empty node"
        base: Path = () if parent is None else parent.end
        paths, cur = [], base
        for k in keys:
            cur = cur + k
            paths.append(cur)
        g = MGroup(paths, slots, stamp)
        for p in paths:
            self.page_group[p] = g
        self.kids.setdefault(base, {})[keys[0]] = paths[0]
        for i in range(len(paths) - 1):
            self.kids.setdefault(paths[i], {})[keys[i + 1]] = paths[i + 1]
        self._groups.append(g)
        self.events["node.add"] += 1
        return g

    def split(self, g: MGroup, n_pages: int) -> MGroup:
        """Mirror ``RadixTreeNode.split_at``: a NEW prefix node is created and ``g`` is mutated
        into the suffix, so handles held on ``g`` still name the suffix -- what lock/unlock symmetry
        depends on.  split_at copies the full ref, the swa lock and the tombstone; the
        window-boundary uuid MIGRATES root-side; the GDN snapshot stays on the suffix (a snapshot
        cannot be split)."""
        assert 0 < n_pages < g.n_pages, f"bad split {n_pages} of {g.n_pages}"
        pre = MGroup(g.paths[:n_pages], g.slots[: n_pages * self.P], g.stamp)
        pre.ref, pre.tomb, pre.swa_ref, pre.swa_uuid = g.ref, g.tomb, g.swa_ref, g.swa_uuid
        g.swa_uuid = None
        g.paths = g.paths[n_pages:]
        g.slots = g.slots[n_pages * self.P:]
        for p in pre.paths:
            self.page_group[p] = pre
        self._groups.insert(self._groups.index(g), pre)
        self.events["node.split"] += 1
        return pre

    def remove(self, g: MGroup) -> Optional[MGroup]:
        """Unlink a LEAF node; returns its parent node (None if the parent is the root)."""
        assert self.is_leaf(g), f"refusing to remove non-leaf {g}"
        parent = self.parent_group(g)
        base: Path = () if parent is None else parent.end
        del self.kids[base][g.start[len(base):]]
        for p in g.paths:
            del self.page_group[p]
            self.kids.pop(p, None)
        self._groups.remove(g)
        self.events["node.remove"] += 1
        return parent


# --------------------------------------------------------------------------- expectations
@dataclass
class ExpMatch:
    cached_len: int
    indices: List[int]
    group: Optional[MGroup]
    second: Optional[int] = None          # hybrid: the GDN snapshot slot to restore from


@dataclass
class ExpInsert:
    matched_len: int
    freed: List[int]                      # what the cache itself returns (SWA only; [] elsewhere)
    dups: List[int]                       # what the CALLER must free by convention (see _commit)
    adopted: List[int]                    # supplied slots the tree took ownership of
    second_exists: Optional[bool] = None  # hybrid dedup flag


@dataclass
class ExpLock:
    group: Optional[MGroup]
    uuid: Optional[int] = None


# --------------------------------------------------------------------------- plain KV radix
class RefModel:
    """RadixPrefixCache's semantics, plus the trie handling and eviction skeleton the two
    dual-currency models extend."""

    second_currency: Optional[str] = None

    def __init__(self, page_size: int) -> None:
        self.P = page_size
        self.trie = PageTrie(page_size)
        self.clk = 0

    @property
    def events(self) -> Counter:
        return self.trie.events

    def _tic(self) -> int:
        """One tick of the (deterministic) wall clock the plain/hybrid trees stamp with."""
        self.clk += 1
        return self.clk

    def _stride_tick(self) -> int:
        """One tick of SWARadixCache's logical clock (``_tick``)."""
        self.clk += 1
        return self.clk * _EVENT_STRIDE

    # -- walking ------------------------------------------------------------
    def _cover(self, ids: Sequence[int]) -> List[MGroup]:
        """Nodes covering the longest page-aligned prefix of ``ids``, splitting the last node when
        the boundary lands inside it (mirrors ``_tree_walk`` / ``_walk`` / ``_insert_helper``)."""
        chain = self.trie.walk(ids)
        out: List[MGroup] = []
        i = 0
        while i < len(chain):
            g = self.trie.page_group[chain[i]]
            cov = min(g.n_pages, len(chain) - i)
            if cov < g.n_pages:
                g = self.trie.split(g, cov)
            out.append(g)
            i += cov
        return out

    def _stamped_cover(self, ids: Sequence[int]) -> List[MGroup]:
        """``_tree_walk`` stamps every node it matched -- and the root-side half of one it split --
        with a single clock read, so those nodes tie by construction."""
        gs = self._cover(ids)
        tic = self._tic()
        for g in gs:
            g.stamp = tic
        return gs

    def _keys_of(self, ids: Sequence[int], start_tok: int, end_tok: int) -> List[Path]:
        return [tuple(ids[t: t + self.P]) for t in range(start_tok, end_tok, self.P)]

    # -- ops ----------------------------------------------------------------
    def match(self, ids: Sequence[int]) -> ExpMatch:
        gs = self._stamped_cover(ids)
        last = gs[-1] if gs else None
        return ExpMatch(self.trie.path_len(last), self.trie.path_slots(last), last)

    def _commit(self, ids: Sequence[int], slots: Sequence[int],
                reused_len: int) -> Tuple[Optional[MGroup], ExpInsert]:
        """Walk, then append whatever page-aligned suffix is missing.

        ``insert_prefix`` stores nothing for the already cached prefix, so the CALLER owns those
        duplicates -- production frees ``page_indices[old_reused_len:cached_len]`` (cache_req)."""
        insert_len = (len(ids) // self.P) * self.P
        ids, slots = list(ids[:insert_len]), list(slots[:insert_len])
        gs = self._stamped_cover(ids)
        node = gs[-1] if gs else None
        matched = self.trie.path_len(node)
        if matched != insert_len:
            node = self.trie.add_group(node, self._keys_of(ids, matched, insert_len),
                                       slots[matched:], self._tic())
        return node, ExpInsert(matched, [], slots[reused_len:matched], slots[matched:])

    def insert(self, ids: Sequence[int], slots: Sequence[int], reused_len: int = 0) -> ExpInsert:
        return self._commit(ids, slots, reused_len)[1]

    def inc_lock(self, g: Optional[MGroup], uuid: Optional[int] = None) -> ExpLock:
        cur = g
        while cur is not None:
            cur.ref += 1
            cur = self.trie.parent_group(cur)
        return ExpLock(g)

    def dec_lock(self, lock: ExpLock, skip_swa: bool = False) -> None:
        cur = lock.group
        while cur is not None:
            if cur.ref <= 0:
                raise _fail("model.dec_lock", f"full ref underflow at node end={cur.end}")
            cur.ref -= 1
            cur = self.trie.parent_group(cur)

    # -- reporting ----------------------------------------------------------
    def records(self) -> List[Tuple[Path, Record]]:
        """One record per node, keyed by its (globally unique) end boundary."""
        return [(g.end, g.record()) for g in self.trie.groups()]

    def structure(self) -> Dict[Path, Record]:
        return dict(self.records())

    def counters(self) -> Dict[str, int]:
        return sizes((r for _, r in self.records()), self.second_currency)

    # -- eviction plumbing --------------------------------------------------
    def _owner(self, slot: int, tag: str) -> MGroup:
        """Slot ids are globally unique and never reused, so a returned slot names exactly one
        node -- what lets the harness follow eviction without reading the tree."""
        for g in self.trie.groups():
            if slot in g.slots:
                return g
        raise _fail(tag, f"cache freed slot {slot} which the model does not own "
                         f"(already freed, or never handed out)")

    def _pick_victim(self, cands: List[MGroup], victim: MGroup, tag: str) -> None:
        """Assert the observed victim is a legal LRU choice.

        ``_tree_walk`` stamps every node it touches with ONE clock read, so several nodes can be
        exactly equally old and the heap breaks that tie arbitrarily.  We therefore demand not a
        *specific* victim but one whose stamp is minimal among the live candidates -- which still
        fails loudly on any real LRU regression (a fresher node evicted first)."""
        if victim not in cands:
            raise _fail(tag, f"evicted node end={victim.end} was not an eligible candidate "
                             f"(eligible: {[g.end for g in cands]})")
        if victim.stamp != min(g.stamp for g in cands):
            older = [(g.end, g.stamp) for g in cands if g.stamp < victim.stamp]
            raise _fail(tag, f"non-LRU victim end={victim.end} stamp={victim.stamp}; strictly "
                             f"older candidates existed: {older}")

    def _take(self, obs: Sequence[int], pos: int, g: MGroup, tag: str, what: str) -> int:
        """Consume exactly ``g``'s slots from the observed stream."""
        chunk = list(obs[pos: pos + g.length])
        if chunk != g.slots:
            raise _fail(tag, f"{what} freed {chunk} for node end={g.end}, model owns {g.slots}")
        return pos + g.length

    @staticmethod
    def _drained(obs: Sequence[int], pos: int, tag: str, what: str) -> None:
        if pos != len(obs):
            raise _fail(tag, f"{what} returned {len(obs) - pos} extra slot(s) {list(obs[pos:])} "
                             f"the model did not predict")

    def _cascadable(self, g: MGroup) -> bool:
        """May an exposed unlocked leaf be reclaimed outright?  Only a node that can never serve a
        second-currency reuse again -- so never, here."""
        return False

    def _second_free(self, g: MGroup, obs: Sequence[int], pos: int, tag: str) -> int:
        """Consume the second-currency slots the cache reported for ``g`` (none by default)."""
        return pos

    def _cascade(self, parent: Optional[MGroup], obs_kv: Sequence[int], pk: int,
                 cands: List[MGroup], tag: str, label: str):
        """Reclaim exposed ancestors carrying no second-currency value.  Returns ``(first
        ancestor that survived, kv position, full tokens reclaimed)``."""
        got = 0
        while (parent is not None and parent.ref == 0 and self.trie.is_leaf(parent)
               and self._cascadable(parent)):
            self.events[f"{label}.cascade"] += 1
            pk = self._take(obs_kv, pk, parent, tag, f"{label} cascade")
            got += parent.length
            if parent in cands:
                cands.remove(parent)
            parent = self.trie.remove(parent)
        return parent, pk, got

    def evict_full(self, n: int, obs_kv: Sequence[int], obs_second: Sequence[int] = ()) -> None:
        """The full-KV pass: unlocked LEAVES only, least recently used first."""
        tag = "model.evict"
        cands = [g for g in self.trie.groups() if g.ref == 0 and self.trie.is_leaf(g)]
        freed, pk, ps = 0, 0, 0
        while freed < n and cands:
            if pk >= len(obs_kv):
                raise _fail(tag, f"evict_full({n}) stopped after {freed} token(s) but "
                                 f"{len(cands)} evictable leaf node(s) remained")
            victim = self._owner(obs_kv[pk], tag)
            self._pick_victim(cands, victim, tag)
            pk = self._take(obs_kv, pk, victim, tag, "evict_full kv")
            freed += victim.length
            ps = self._second_free(victim, obs_second, ps, tag)
            cands.remove(victim)
            parent, pk, extra = self._cascade(self.trie.remove(victim), obs_kv, pk, cands,
                                              tag, "evict_full")
            freed += extra
            if (parent is not None and parent.ref == 0 and self.trie.is_leaf(parent)
                    and parent not in cands):
                self.events["evict.parent_exposed"] += 1
                cands.append(parent)
        self._drained(obs_kv, pk, tag, "evict_full kv")
        self._drained(obs_second, ps, tag, "evict_full second")


# --------------------------------------------------------------------------- SWA radix
class SWAModel(RefModel):
    """SWARadixCache: full KV + sliding-window currency.  The second currency stores no slot of its
    own -- a node is either swa-live or ``tomb`` (its swa KV was freed, its full KV survives)."""

    second_currency = "swa"

    def __init__(self, page_size: int, window: int) -> None:
        super().__init__(page_size)
        assert window > 0
        self.W = window
        self.uuid_owner: Dict[int, MGroup] = {}

    # -- match --------------------------------------------------------------
    def match(self, ids: Sequence[int]) -> ExpMatch:
        """Reusable only up to a boundary behind which the live (tombstone-free) run covers the
        whole window; the run accumulates across nodes."""
        gs = self._cover(ids)
        run = float("inf")                 # tombstone-free back to root => always reusable
        best_i, best_g = 0, None
        prev: Optional[MGroup] = None
        for i, g in enumerate(gs):
            if g.tomb:
                if run >= self.W:
                    best_i, best_g = i, prev
                run = 0
            else:
                run += g.length
            prev = g
        if run >= self.W:
            best_i, best_g = len(gs), (gs[-1] if gs else None)
        if best_i < len(gs):
            self.events["match.windowed_truncation"] += 1
        kv = [s for g in gs[:best_i] for s in g.slots]
        self._stamp_path(best_g)
        return ExpMatch(len(kv), kv, best_g)

    def _stamp_path(self, g: Optional[MGroup]) -> None:
        """Strictly DECREASING stamps toward the root, so the eviction heap reclaims near-root swa
        nodes first (``_stamp_path`` / sglang ``reset_node_and_parents_mru``)."""
        self.clk += 1
        base = self.clk * _EVENT_STRIDE
        off, cur = 0, g
        while cur is not None:
            cur.stamp = base - off
            off += 1
            cur = self.trie.parent_group(cur)

    # -- insert -------------------------------------------------------------
    def insert(self, ids: Sequence[int], slots: Sequence[int], reused_len: int = 0,
               swa_evicted: int = 0, update_after: int = 0) -> ExpInsert:
        # ``reused_len`` is accepted only for signature uniformity with the other two models: SWA
        # takes the same frontier as ``update_kv_after_len`` and returns its duplicates explicitly,
        # so there is nothing for the caller-side dup convention to compute.
        P = self.P
        insert_len = (len(ids) // P) * P
        ids, slots = list(ids[:insert_len]), list(slots[:insert_len])
        freed: List[int] = []
        adopted: List[int] = []
        chain = self.trie.walk(ids)
        node: Optional[MGroup] = None
        total, i = 0, 0
        while i < len(chain):
            g = self.trie.page_group[chain[i]]
            cov = min(g.n_pages, len(chain) - i)
            partial = cov < g.n_pages
            if partial:
                g = self.trie.split(g, cov)
            match_len = cov * P
            seg = slots[total: total + match_len]
            if update_after < total + match_len:
                if not g.tomb:
                    self.events["insert.dup_live"] += 1
                    freed.extend(seg)                  # live node: the tree's slots are canonical
                elif g.swa_ref != 0:
                    raise _fail("model.insert", f"tombstoned node end={g.end} holds a swa lock")
                elif g.ref > 0:
                    # A full-locked reader still gathers this node's CURRENT slots -> keep the
                    # tombstone and drop the incoming duplicate.
                    self.events["insert.locked_no_revive"] += 1
                    freed.extend(seg)
                elif swa_evicted <= total:
                    self.events["insert.revive_whole"] += 1
                    freed.extend(g.slots)              # branch 1: revive whole
                    g.slots = list(seg)
                    adopted.extend(seg)
                    g.tomb = False
                    g.stamp = self._stride_tick()
                elif swa_evicted < total + match_len:
                    self.events["insert.revive_tail"] += 1
                    start = swa_evicted - total         # branch 2: split, revive the live tail
                    self.trie.split(g, start // P)      # head stays tombstone; g := live tail
                    freed.extend(g.slots)
                    freed.extend(seg[:start])
                    g.slots = list(seg[start:])
                    adopted.extend(seg[start:])
                    g.tomb = False
                    g.stamp = self._stride_tick()
                else:
                    self.events["insert.keep_tombstone"] += 1
                    freed.extend(seg)                  # branch 3: still wholly out-of-window
            total += match_len
            node = g
            i += cov
            if partial:
                break
        matched_tokens = total
        if total < insert_len:
            boundary = max(0, min(swa_evicted, insert_len) - total)
            boundary = min(boundary, max(0, insert_len - total - P))  # never a tombstone leaf
            if boundary > 0:
                node = self.trie.add_group(node, self._keys_of(ids, total, total + boundary),
                                           slots[total: total + boundary], self._stride_tick())
                node.tomb = True
                self.events["insert.suffix_tombstone"] += 1
                adopted.extend(slots[total: total + boundary])
                total += boundary
            if total < insert_len:
                node = self.trie.add_group(node, self._keys_of(ids, total, insert_len),
                                           slots[total:insert_len], self._stride_tick())
                self.events["insert.suffix_live"] += 1
                adopted.extend(slots[total:insert_len])
        # SWA returns its duplicates explicitly, so there is no caller-side dup convention.
        return ExpInsert(matched_tokens, freed, [], adopted)

    # -- locking ------------------------------------------------------------
    def inc_lock(self, g: Optional[MGroup], uuid: Optional[int] = None) -> ExpLock:
        """``uuid`` is the opaque window handle the implementation just returned; the model checks
        it names the boundary node the window arithmetic says it must."""
        swa_locked = 0
        boundary: Optional[MGroup] = None
        cur = g
        while cur is not None:
            cur.ref += 1
            if swa_locked < self.W and not cur.tomb:
                cur.swa_ref += 1
                swa_locked += cur.length
                if swa_locked >= self.W:
                    boundary = cur
            cur = self.trie.parent_group(cur)
        tag = "model.inc_lock"
        if boundary is None:
            if uuid is not None:
                raise _fail(tag, f"inc_lock returned swa_uuid={uuid} but the locked path "
                                 f"({swa_locked} tokens) never covers the window {self.W}")
            return ExpLock(g, None)
        if uuid is None:
            raise _fail(tag, f"inc_lock returned no swa_uuid although the swa lock covered "
                             f"{swa_locked} >= window {self.W} at node end={boundary.end}")
        owner = self.uuid_owner.get(uuid)
        if boundary.swa_uuid is None and owner not in (None, boundary):
            raise _fail(tag, f"swa_uuid={uuid} already names node end={owner.end}, "
                             f"now returned for end={boundary.end}")
        if boundary.swa_uuid not in (None, uuid):
            raise _fail(tag, f"node end={boundary.end} owns swa_uuid={boundary.swa_uuid} but "
                             f"inc_lock returned {uuid}")
        boundary.swa_uuid = uuid
        self.uuid_owner[uuid] = boundary
        return ExpLock(g, uuid)

    def dec_lock(self, lock: ExpLock, skip_swa: bool = False) -> None:
        dec_swa = not skip_swa
        cur = lock.group
        while cur is not None:
            if cur.ref <= 0:
                raise _fail("model.dec_lock", f"full ref underflow at node end={cur.end}")
            cur.ref -= 1
            if dec_swa and not cur.tomb and cur.swa_ref > 0:
                cur.swa_ref -= 1
                if lock.uuid is not None and cur.swa_uuid == lock.uuid:
                    dec_swa = False           # released exactly this reader's own window
            cur = self.trie.parent_group(cur)

    # -- eviction -----------------------------------------------------------
    def _cascadable(self, g: MGroup) -> bool:
        return g.tomb                        # a tombstone leaf can never be matched through again

    def _second_free(self, g: MGroup, obs: Sequence[int], pos: int, tag: str) -> int:
        return pos if g.tomb else self._take(obs, pos, g, tag, "evict_full swa")

    def evict_second(self, n: int, obs_kv: Sequence[int], obs_swa: Sequence[int]) -> None:
        """``evict_swa(n)`` frees window tokens: tombstone in place, or unlink an unlocked leaf."""
        tag = "model.evict_swa"
        cands = [g for g in self.trie.groups() if not g.tomb and g.swa_ref == 0]
        freed, pk, ps = 0, 0, 0
        while freed < n and cands:
            if ps >= len(obs_swa):
                raise _fail(tag, f"evict_swa({n}) stopped after {freed} token(s) with "
                                 f"{len(cands)} unlocked live-swa node(s) still evictable")
            victim = self._owner(obs_swa[ps], tag)
            self._pick_victim(cands, victim, tag)
            ps = self._take(obs_swa, ps, victim, tag, "evict_swa swa")
            freed += victim.length
            cands.remove(victim)
            victim.tomb = True
            if self.trie.is_leaf(victim) and victim.ref == 0:
                self.events["evict_swa.leaf_free"] += 1
                pk = self._take(obs_kv, pk, victim, tag, "evict_swa kv")
                _, pk, _ = self._cascade(self.trie.remove(victim), obs_kv, pk, cands,
                                         tag, "evict_swa")
            else:
                self.events["evict_swa.tombstone_in_place"] += 1
        self._drained(obs_kv, pk, tag, "evict_swa kv")
        self._drained(obs_swa, ps, tag, "evict_swa swa")

    # -- retention ----------------------------------------------------------
    def trim_head_swa(self, ids: Sequence[int], keep_from: int) -> List[int]:
        """Tombstone the unlocked internal nodes below ``keep_from``; their full KV stays."""
        if keep_from <= 0:
            return []
        if keep_from % self.P:
            raise PreconditionError("precondition.keep_from",
                                    f"keep_from={keep_from} is not a multiple of page {self.P}")
        self.match(ids[:keep_from])         # the implementation re-matches (splits + stamps) first
        freed: List[int] = []
        node: Optional[MGroup] = None
        pos = 0
        while pos < keep_from:
            cp = self.trie.child(() if node is None else node.end,
                                 tuple(ids[pos: pos + self.P]))
            if cp is None:
                break
            g = self.trie.page_group[cp]
            if pos + g.length > keep_from:
                break
            if not g.tomb and g.swa_ref == 0 and not self.trie.is_leaf(g):
                self.events["trim.tombstone"] += 1
                freed.extend(g.slots)
                g.tomb = True
            node, pos = g, pos + g.length
        return freed


# --------------------------------------------------------------------------- hybrid (GDN) radix
class HybridModel(RefModel):
    """HybridRadixCache: full KV + an optional GDN snapshot slot at a node's END boundary."""

    second_currency = "mamba"

    def match(self, ids: Sequence[int]) -> ExpMatch:
        gs = self._stamped_cover(ids)
        for i, g in enumerate(reversed(gs)):  # deepest node that still owns a LIVE snapshot
            if g.mamba is not None:
                if i:
                    self.events["match.snapshot_truncation"] += 1
                return ExpMatch(self.trie.path_len(g), self.trie.path_slots(g), g, g.mamba)
        self.events["match.no_snapshot"] += 1
        return ExpMatch(0, [], None, None)

    def insert(self, ids: Sequence[int], slots: Sequence[int], reused_len: int = 0,
               *, mamba: int) -> ExpInsert:
        node, exp = self._commit(ids, slots, reused_len)
        exp.second_exists = node is None or node.mamba is not None
        if exp.second_exists:
            self.events["insert.snapshot_dedup"] += 1
        else:
            self.events["insert.snapshot_attach"] += 1
            node.mamba = mamba
        return exp

    def inc_lock(self, g: Optional[MGroup], uuid: Optional[int] = None) -> ExpLock:
        if g is not None and g.mamba is not None:
            g.mamba_ref += 1
        return super().inc_lock(g)

    def dec_lock(self, lock: ExpLock, skip_swa: bool = False) -> None:
        g = lock.group
        if g is not None and g.mamba is not None and g.mamba_ref > 0:
            g.mamba_ref -= 1
        super().dec_lock(lock)

    # -- eviction -----------------------------------------------------------
    def _cascadable(self, g: MGroup) -> bool:
        return g.mamba is None               # no snapshot left => nothing can resume from it

    def _second_free(self, g: MGroup, obs: Sequence[int], pos: int, tag: str) -> int:
        if g.mamba is None:
            return pos
        if pos >= len(obs) or obs[pos] != g.mamba:
            raise _fail(tag, f"expected snapshot slot {g.mamba} of node end={g.end} to be freed, "
                             f"got {list(obs[pos:pos + 1]) or 'nothing'}")
        g.mamba = None
        g.mamba_ref = 0
        return pos + 1

    def evict_second(self, n: int, obs_kv: Sequence[int], obs_mamba: Sequence[int]) -> None:
        """``evict_mamba(n)`` counts SNAPSHOTS, not tokens."""
        tag = "model.evict_mamba"
        cands = [g for g in self.trie.groups() if g.mamba is not None and g.mamba_ref == 0]
        freed, pk, pm = 0, 0, 0
        while freed < n and cands:
            if pm >= len(obs_mamba):
                raise _fail(tag, f"evict_mamba({n}) stopped after {freed} snapshot(s) with "
                                 f"{len(cands)} unlocked snapshot node(s) still evictable")
            owners = [g for g in cands if g.mamba == obs_mamba[pm]]
            if not owners:
                raise _fail(tag, f"evict_mamba freed snapshot slot {obs_mamba[pm]} which the model "
                                 f"does not consider an eligible snapshot")
            victim = owners[0]
            self._pick_victim(cands, victim, tag)
            cands.remove(victim)
            freed += 1
            if self.trie.is_leaf(victim) and victim.ref == 0:
                self.events["evict_mamba.leaf_free"] += 1
                pk = self._take(obs_kv, pk, victim, tag, "evict_mamba kv")
                pm = self._second_free(victim, obs_mamba, pm, tag)
                _, pk, _ = self._cascade(self.trie.remove(victim), obs_kv, pk, cands,
                                         tag, "evict_mamba")
            else:
                self.events["evict_mamba.tombstone_in_place"] += 1
                pm = self._second_free(victim, obs_mamba, pm, tag)
        self._drained(obs_kv, pk, tag, "evict_mamba kv")
        self._drained(obs_mamba, pm, tag, "evict_mamba mamba")
