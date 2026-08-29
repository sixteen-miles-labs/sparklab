"""RadixPrefixCache -- the single-currency (full KV) prefix cache.

Nothing here reads the cache's own bookkeeping to decide what to expect: every op goes through
``Session``, which computes the expectation from the page-keyed reference model first and then runs
the invariant battery (structure, counter model, slot conservation, model agreement).  What the
assertions in this module add on top are the *class-specific* contracts the shared battery cannot
state: which slots came back, what the running sizes are, what ``split_at`` does to node identity,
and the three ways the class refuses an illegal request.

Page size is parametrized over {1, 4} wherever it is load-bearing.  Slot ids are globally unique
and never reused, so ``session.kv.free`` is an exact, public record of what the cache handed back.
"""
from __future__ import annotations

import pytest

from sparklab.runtime.kvcache.radix_cache import RadixCacheHandle

from .adapters import ids_tensor, node_end_path, node_path_slots, slots_tensor
from .driver import CacheSpec, Session

PAGE_SPECS = (CacheSpec("plain", 1), CacheSpec("plain", 4))
P4 = CacheSpec("plain", 4)

EMPTY = {"full_evictable": 0, "full_protected": 0}


@pytest.fixture(params=PAGE_SPECS, ids=[s.id for s in PAGE_SPECS])
def sess(request) -> Session:
    """A plain radix cache + its reference model + slot ledger, at page_size 1 and 4."""
    return Session(request.param)


@pytest.fixture
def sess4() -> Session:
    """One session at page_size 4, for behaviours that do not depend on the page size."""
    return Session(P4)


def ev(session: Session):
    """Reference-model branch counters.  (The harness advertises ``Session.events`` but never
    defines the attribute; the Counter itself lives on the model.)"""
    return session.model.events


def page(P: int, tag: int) -> tuple:
    """One page.  At page_size > 1 every page starts with the same token: distinct pages that
    share leading tokens are exactly the shape a token-keyed tree gets wrong."""
    return (tag,) if P == 1 else (1,) + (7,) * (P - 2) + (tag,)


def seq(P: int, *tags: int) -> tuple:
    out: list = []
    for t in tags:
        out.extend(page(P, t))
    return tuple(out)


# --------------------------------------------------------------------------- match / insert
def test_cold_match_is_empty_and_a_commit_round_trips(sess):
    P = sess.P
    ids = seq(P, 1, 2, 3)

    cold, exp = sess.do_match(ids)
    assert (cold.cached_len, cold.indices) == (0, [])
    assert exp.group is None
    assert cold.node.is_root()               # the lock target of a cold match is the root
    assert sess.ad.counters() == EMPTY
    assert sess.ad.nodes() == []

    slots = sess.kv.take(len(ids))
    got, ins = sess.do_insert(ids, slots=slots)
    assert got.matched_len == 0              # nothing of this prefix was cached
    assert ins.adopted == slots              # the tree took every supplied slot

    warm, _ = sess.do_match(ids)
    assert (warm.cached_len, warm.indices) == (len(ids), slots)
    assert sess.ad.counters() == {"full_evictable": len(ids), "full_protected": 0}
    assert len(sess.ad.nodes()) == 1         # three pages, one radix node
    assert ev(sess)["node.add"] == 1 and ev(sess)["node.split"] == 0
    sess.check()


def test_cold_handle_get_matched_indices_raises(sess4):
    """WART: ``get_matched_indices`` on a cold handle is ``torch.cat([])`` -> ValueError."""
    P = sess4.P
    ids = seq(P, 1)

    res = sess4.ad.cache.match_prefix(ids_tensor(ids))
    assert res.cuda_handle.cached_len == 0
    with pytest.raises(ValueError):
        res.cuda_handle.get_matched_indices()
    assert sess4.ad.match(ids).indices == []          # every caller needs this cached_len==0 guard

    slots = sess4.kv.take(len(ids))
    sess4.do_insert(ids, slots=slots)
    warm, _ = sess4.do_match(ids)
    assert RadixCacheHandle(warm.cached_len, warm.node).get_matched_indices().tolist() == slots
    sess4.check()


def test_insert_drops_a_trailing_partial_page(sess):
    P = sess.P
    ids = seq(P, 1, 2) + (9,) * (P - 1)      # a ragged tail at page_size 4, nothing at 1
    kept = 2 * P
    slots = sess.kv.take(len(ids))

    _, ins = sess.do_insert(ids, slots=slots)
    assert ins.adopted == slots[:kept]

    m, _ = sess.do_match(ids)
    assert (m.cached_len, m.indices) == (kept, slots[:kept])
    # the partial page's slots stay the caller's -- insert_prefix truncates before it stores
    assert set(slots[kept:]) <= sess.kv.free
    assert not set(slots[kept:]) & set(sess.ad.tree_slots())
    sess.check()


def test_match_takes_a_ragged_length_and_stops_at_the_page_boundary(sess):
    """``match_prefix`` does NOT truncate its argument the way ``insert_prefix`` does: it accepts a
    raw length and rounds the match down per node."""
    P = sess.P
    ids = seq(P, 1, 2)
    slots = sess.kv.take(len(ids))
    sess.do_insert(ids, slots=slots)

    m, _ = sess.do_match(ids[: 2 * P - 1])   # one token short of the second page
    assert (m.cached_len, m.indices) == (P, slots[:P])
    assert ev(sess)["node.split"] == 1    # the boundary fell inside the node
    assert node_end_path(m.node) == ids[:P]
    sess.check()


def test_mid_node_divergence_rounds_down_to_the_page_boundary(sess):
    P = sess.P
    ids = seq(P, 1, 2)
    slots = sess.kv.take(len(ids))
    sess.do_insert(ids, slots=slots)

    alt = ids[:-1] + (999,)                  # identical for 2P-1 tokens, differs in the last
    m, _ = sess.do_match(alt)
    assert (m.cached_len, m.indices) == (P, slots[:P])
    if P > 1:                                # 2P-1 tokens compared equal, only P are reusable
        assert m.cached_len < len(ids) - 1
    assert ev(sess)["node.split"] == 1
    sess.check()


def test_pages_sharing_leading_tokens_do_not_share_a_tree_edge():
    """The tree's reuse unit is a whole page: two different pages with a common token prefix must
    get their own edges, or a match would hand back the wrong page's indices."""
    s = Session(P4)
    one, two = (1, 7, 7, 1), (1, 7, 7, 2)    # equal for 3 of 4 tokens, different pages
    s1, s2 = s.kv.take(4), s.kv.take(4)
    s.do_insert(one, slots=s1)
    s.do_insert(two, slots=s2)

    assert len(s.ad.root.children) == 2
    assert ev(s)["node.split"] == 0
    m1, _ = s.do_match(one)
    m2, _ = s.do_match(two)
    assert (m1.cached_len, m1.indices) == (4, s1)
    assert (m2.cached_len, m2.indices) == (4, s2)
    s.check()


def test_reinserting_a_cached_prefix_adopts_nothing(sess4):
    P = sess4.P
    ids = seq(P, 1, 2)
    first = sess4.kv.take(len(ids))
    sess4.do_insert(ids, slots=first)

    second = sess4.kv.take(len(ids))
    got, ins = sess4.do_insert(ids, slots=second)
    assert got.matched_len == len(ids)
    assert ins.adopted == []
    # insert_prefix stores nothing for an already cached prefix, and hands nothing back either:
    # the duplicates stay the CALLER's to free (scheduler/cache.py frees
    # page_indices[old_reused_len : cached_len]).
    assert ins.dups == second
    assert set(second) <= sess4.kv.free

    m, _ = sess4.do_match(ids)
    assert m.indices == first                # the tree kept its own slots
    assert len(sess4.ad.nodes()) == 1

    # The handle insert_prefix returns is the commit's lock target, and it names the TREE's own
    # indices for the whole inserted prefix -- never the supplied ones.
    fake = list(range(7_000_000, 7_000_000 + len(ids)))     # never registered with the ledger
    res = sess4.ad.cache.insert_prefix(ids_tensor(ids), slots_tensor(fake))
    assert res.cached_len == len(ids)        # all of it was already cached ...
    assert res.handle.cached_len == len(ids)  # ... and the handle still spans all of it
    assert res.handle.get_matched_indices().tolist() == first
    sess4.check()


def test_request_lifecycle_reuses_the_cached_prefix(sess):
    P = sess.P
    full = seq(P, 1, 2, 3)
    s_head = sess.kv.take(P)
    sess.do_insert(full[:P], slots=s_head)

    sess.do_request(full, prompt_pages=1)    # match -> lock -> extend+commit -> unlock
    assert sess.kv.free == set()             # a reused prefix duplicates nothing and leaks nothing

    m, _ = sess.do_match(full)
    assert m.cached_len == 3 * P
    assert m.indices[:P] == s_head           # the reused page still owns its original slots
    assert sess.ad.counters() == {"full_evictable": 3 * P, "full_protected": 0}
    assert ev(sess)["node.add"] == 2 and ev(sess)["node.split"] == 0
    sess.check()


# --------------------------------------------------------------------------- split mechanics
def test_split_turns_the_original_node_into_the_suffix(sess):
    P = sess.P
    ids = seq(P, 1, 2)
    slots = sess.kv.take(len(ids))
    sess.do_insert(ids, slots=slots)
    whole, _ = sess.do_match(ids)
    node = whole.node
    assert node.length == 2 * P

    pre, _ = sess.do_match(ids[:P])          # forces split_at(P)
    assert ev(sess)["node.split"] == 1
    assert len(sess.ad.nodes()) == 2
    assert pre.node is not node
    # split_at creates a NEW prefix node and keeps the original object as the SUFFIX, so a handle
    # taken before the split still names the same end boundary -- lock/unlock symmetry rides on it.
    assert node.length == P
    assert node.parent is pre.node
    assert node_end_path(pre.node) == ids[:P]
    assert node_end_path(node) == ids
    assert pre.node.value.tolist() == slots[:P]
    assert node.value.tolist() == slots[P:]
    assert node_path_slots(node) == slots

    again, _ = sess.do_match(ids)            # the split is transparent to a full match
    assert (again.cached_len, again.indices) == (2 * P, slots)
    assert again.node is node
    sess.check()


def test_lock_survives_a_split(sess):
    P = sess.P
    ids = seq(P, 1, 2)
    slots = sess.kv.take(len(ids))
    sess.do_insert(ids, slots=slots)

    held = sess.do_lock(ids)
    assert sess.ad.counters() == {"full_evictable": 0, "full_protected": 2 * P}

    sess.do_match(ids[:P])                   # splits the locked node in two
    assert ev(sess)["node.split"] == 1
    # split_at copies ref_count into the new prefix half, so the protected total is unchanged
    assert sess.ad.counters() == {"full_evictable": 0, "full_protected": 2 * P}
    sess.check()

    sess.do_unlock(held)                     # the handle points at the suffix; release walks up
    assert sess.ad.counters() == {"full_evictable": 2 * P, "full_protected": 0}
    sess.check()


# --------------------------------------------------------------------------- locking
def test_lock_accounting_walks_to_the_root(sess):
    P = sess.P
    head, tail = seq(P, 1), seq(P, 1, 2)
    sess.do_insert(head, slots=sess.kv.take(len(head)))
    sess.do_insert(tail, slots=sess.kv.take(len(tail)))
    assert len(sess.ad.nodes()) == 2
    assert sess.ad.counters() == {"full_evictable": 2 * P, "full_protected": 0}

    deep = sess.do_lock(tail)                # protects the leaf AND everything up to the root
    assert sess.ad.counters() == {"full_evictable": 0, "full_protected": 2 * P}
    again = sess.do_lock(tail)               # ref 2: still exactly one protected region
    assert sess.ad.counters() == {"full_evictable": 0, "full_protected": 2 * P}
    sess.do_unlock(again)
    assert sess.ad.counters() == {"full_evictable": 0, "full_protected": 2 * P}
    sess.do_unlock(deep)
    assert sess.ad.counters() == {"full_evictable": 2 * P, "full_protected": 0}

    shallow = sess.do_lock(head)             # a lock protects the root path, not the descendants
    assert sess.ad.counters() == {"full_evictable": P, "full_protected": P}
    sess.do_unlock(shallow)
    assert sess.ad.counters() == {"full_evictable": 2 * P, "full_protected": 0}
    sess.check()


# --------------------------------------------------------------------------- eviction
def test_eviction_is_lru_ordered_over_leaves(sess):
    P = sess.P
    a, b, c = seq(P, 1), seq(P, 2), seq(P, 3)
    sa, sb, sc = sess.kv.take(P), sess.kv.take(P), sess.kv.take(P)
    for ids, slots in ((a, sa), (b, sb), (c, sc)):
        sess.do_insert(ids, slots=slots)
    sess.do_match(a)                         # a becomes the most recently used

    sess.do_evict_full(P)
    assert set(sb) <= sess.kv.free           # b was the oldest
    assert not (set(sa) | set(sc)) & sess.kv.free

    sess.do_evict_full(P)
    assert set(sc) <= sess.kv.free
    still, _ = sess.do_match(a)
    assert (still.cached_len, still.indices) == (P, sa)
    gone, _ = sess.do_match(b)
    assert gone.cached_len == 0

    sess.do_evict_full(P)
    assert set(sa) <= sess.kv.free
    assert sess.ad.nodes() == [] and sess.ad.counters() == EMPTY
    sess.check()


def test_eviction_takes_the_leaf_before_the_parent(sess):
    P = sess.P
    head, tail = seq(P, 1), seq(P, 1, 2)
    s_head = sess.kv.take(P)
    sess.do_insert(head, slots=s_head)
    s_tail = sess.kv.take(2 * P)
    sess.do_insert(tail, slots=s_tail)
    leaf = s_tail[P:]                        # s_tail[:P] were duplicates of the head page

    sess.do_evict_full(P)                    # only leaves are collectible
    assert set(leaf) <= sess.kv.free
    assert not set(s_head) & sess.kv.free
    assert len(sess.ad.nodes()) == 1
    assert sess.ad.counters() == {"full_evictable": P, "full_protected": 0}
    m, _ = sess.do_match(head)
    assert (m.cached_len, m.indices) == (P, s_head)

    sess.do_evict_full(P)                    # now the parent is a leaf itself
    assert set(s_head) <= sess.kv.free
    assert sess.ad.nodes() == []
    sess.check()


def test_eviction_cascades_to_a_newly_childless_parent(sess):
    P = sess.P
    head, tail = seq(P, 1), seq(P, 1, 2)
    s_head = sess.kv.take(P)
    sess.do_insert(head, slots=s_head)
    s_tail = sess.kv.take(2 * P)
    sess.do_insert(tail, slots=s_tail)
    leaf = s_tail[P:]

    # ONE call: the parent is not collectible when the walk starts, it is pushed onto the heap
    # only after its last child is unlinked.
    sess.do_evict_full(2 * P)
    assert ev(sess)["evict.parent_exposed"] == 1
    assert set(leaf) <= sess.kv.free and set(s_head) <= sess.kv.free
    assert sess.ad.nodes() == [] and sess.ad.counters() == EMPTY
    sess.check()


def test_locked_nodes_are_never_evicted(sess):
    P = sess.P
    a, b = seq(P, 1), seq(P, 2)
    sa, sb = sess.kv.take(P), sess.kv.take(P)
    sess.do_insert(a, slots=sa)
    sess.do_insert(b, slots=sb)

    held = sess.do_lock(a)
    assert sess.ad.counters() == {"full_evictable": P, "full_protected": P}

    sess.do_evict_full(10 ** 6)              # clamped to the evictable size
    assert set(sb) <= sess.kv.free
    assert not set(sa) & sess.kv.free
    m, _ = sess.do_match(a)
    assert (m.cached_len, m.indices) == (P, sa)
    with pytest.raises(AssertionError):      # a locked node is not evictable capacity
        sess.ad.cache.evict(1)

    sess.do_unlock(held)
    assert sess.ad.counters() == {"full_evictable": P, "full_protected": 0}
    sess.do_evict_full(P)
    assert set(sa) <= sess.kv.free
    sess.check()


def test_evicting_more_than_evictable_raises_assertion_error(sess4):
    P = sess4.P
    ids = seq(P, 1)
    slots = sess4.kv.take(P)
    sess4.do_insert(ids, slots=slots)

    # BasePrefixCache.evict documents RuntimeError; the implementation asserts. Pin the REAL
    # type -- production is not being changed to match its docstring.
    with pytest.raises(AssertionError, match="Cannot evict"):
        sess4.ad.cache.evict(P + 1)

    sess4.check()                            # the guard fires before anything is unlinked
    m, _ = sess4.do_match(ids)
    assert (m.cached_len, m.indices) == (P, slots)


def test_evict_zero_is_a_no_op_and_evict_all_keeps_the_root(sess4):
    P = sess4.P
    assert sess4.ad.cache.evict(0).numel() == 0        # safe on an empty tree

    ids = seq(P, 1, 2)
    slots = sess4.kv.take(len(ids))
    sess4.do_insert(ids, slots=slots)
    assert sess4.ad.cache.evict(0).numel() == 0
    assert sess4.ad.tree_slots() == slots

    sess4.do_evict_full(len(ids))
    assert set(slots) <= sess4.kv.free
    root = sess4.ad.root
    assert root.is_root() and root.children == {}
    assert root.ref_count == 1                         # the root is protected, never a victim
    assert sess4.ad.counters() == EMPTY
    cold, _ = sess4.do_match(ids)
    assert cold.cached_len == 0
    sess4.check()


def test_reset_is_not_implemented_by_design(sess4):
    # The scheduler builds a fresh RadixPrefixCache rather than resetting one; this is a
    # deliberate refusal, not a gap.
    with pytest.raises(NotImplementedError):
        sess4.ad.cache.reset()
    sess4.check()


def test_insert_owns_its_indices_and_survives_caller_mutation(sess4: Session):
    """The tree must COPY the indices it stores. Slicing a tensor yields a view, so a cache that
    keeps ``indices[prefix_len:]`` unaliased serves whatever the caller's buffer later says -- and
    the caller (scheduler/cache.py) reuses its staging buffers. Mutation-battery gap: every other
    test hands the cache a fresh tensor and never writes to it again, so only an explicit
    write-after-insert can see this."""
    ids = list(range(1, 9))
    slots = sess4.kv.take(len(ids))
    caller_buffer = slots_tensor(slots)

    sess4.ad.cache.insert_prefix(ids_tensor(ids), caller_buffer)
    caller_buffer.fill_(-1)  # the caller reuses its staging buffer for the next request

    assert sess4.ad.match(ids).indices == slots
    assert -1 not in sess4.ad.tree_slots()
