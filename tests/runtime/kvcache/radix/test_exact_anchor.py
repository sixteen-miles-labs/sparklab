"""Extend the shared reference model for exact endpoints' full-page-only pins."""

import json
import random

import pytest

from .driver import Session, ALL_SPECS, CacheSpec
from .model import RefModel


def test_full_page_pin_does_not_release_another_readers_snapshot():
    spec = next(s for s in ALL_SPECS if s.kind == "hybrid" and not s.unsupported)
    session = Session(spec)
    ids = list(range(1, session.P * 2 + 1))
    node, group = session.do_insert_kv_only(ids)
    session.check()
    session.ad.cache.inc_lock_full(node)
    full_pin = RefModel.inc_lock(session.model, group)
    session.check()
    # A normal aligned snapshot is attached after the exact owner's KV pin.
    session.do_insert(ids)
    snapshot_pin = session.do_lock(ids)
    session.check()
    session.ad.cache.dec_lock_full(node)
    RefModel.dec_lock(session.model, full_pin)
    session.check()
    session.do_evict_second(100)
    session.check()  # reader still protects the aligned snapshot
    assert node.mamba_ref_count == 1
    session.do_unlock(snapshot_pin)
    session.do_evict_second(100)
    session.check()


@pytest.mark.parametrize("page_size", [1, 4, 16, 64])
@pytest.mark.parametrize("seed", [7, 73, 319, 901])
def test_seeded_exact_and_snapshot_owners_share_splits_and_eviction(
    page_size, seed, tmp_path
):
    session = Session(CacheSpec("hybrid", page_size))
    rng = random.Random(seed)
    words = [
        (1, 2, 3, 4),
        (1, 2, 5),
        (1, 6, 7, 8),
        (9, 10, 11),
        (1, 2),
        (1, 2, 3),
        (1,),
        (9, 10),
        (1, 6),
    ]
    full_pins, readers, trace = [], [], []
    actions = [
        "full_pin",
        "full_pin",
        "insert",
        "insert",
        "reader",
        "match",
        "evict_full",
        "evict_state",
        "unlock_full",
        "unlock_reader",
    ]
    try:
        for step in range(256):
            word = rng.choice(words)
            ids = [
                value * 1000 + offset for value in word for offset in range(page_size)
            ]
            ids += list(range(rng.randrange(page_size)))
            action = rng.choice(actions)
            if action == "full_pin" and len(full_pins) == 8:
                action = "unlock_full"
            if action == "reader" and len(readers) == 8:
                action = "unlock_reader"
            record = {"step": step, "action": action}
            trace.append(record)
            if action in ("insert", "full_pin", "reader", "match"):
                record["ids"] = ids
                if action == "full_pin":
                    full_pins.append(session.do_lock_full(ids))
                elif action == "insert":
                    session.do_insert(ids)
                elif action == "match":
                    session.do_match(ids)
                else:
                    held = session.do_lock(ids)
                    if held is not None:
                        readers.append(held)
            elif action in ("unlock_full", "unlock_reader"):
                pins = full_pins if action == "unlock_full" else readers
                if pins:
                    record["index"] = index = rng.randrange(len(pins))
                    held = pins.pop(index)
                    if action == "unlock_full":
                        session.do_unlock_full(held)
                    else:
                        session.do_unlock(held)
            elif action == "evict_full":
                record["tokens"] = count = rng.randrange(1, 5) * page_size
                session.do_evict_full(count)
            else:
                record["slots"] = count = rng.randrange(1, 4)
                session.do_evict_second(count)
            session.check()
        releases = [("full", held) for held in full_pins] + [
            ("reader", held) for held in readers
        ]
        rng.shuffle(releases)
        for kind, held in releases:
            if kind == "full":
                session.do_unlock_full(held)
            else:
                session.do_unlock(held)
            session.check()
        session.do_evict_full(1 << 30)
        session.do_evict_second(1 << 30)
        session.check()
        assert not session.kv.in_use() and not session.second.in_use()
        assert session.model.events["node.split"] > 0
        assert session.model.events["insert.snapshot_attach"] > 0
    except Exception as error:
        evidence = {
            "seed": seed,
            "page_size": page_size,
            "trace": trace,
            "error": f"{type(error).__name__}: {error}",
        }
        path = tmp_path / "exact-anchor-trace.json"
        path.write_text(json.dumps(evidence, indent=2) + "\n")
        error.add_note(f"Replay seed={seed}, page_size={page_size}; trace={path}")
        raise


@pytest.mark.parametrize("page_size", [1, 4, 16, 64])
def test_existing_snapshot_reader_survives_later_exact_owner_unlock(page_size):
    # Four-operation semantic reduction of the seeded mutation counterexample:
    # attach state, acquire its reader, pin KV independently, then release KV.
    session = Session(CacheSpec("hybrid", page_size))
    ids = list(range(1, page_size * 3 + 1))
    session.do_insert(ids)
    session.check()
    reader = session.do_lock(ids)
    session.check()
    exact = session.do_lock_full(ids)
    session.check()
    session.do_unlock_full(exact)
    session.check()
    session.do_evict_second(1 << 30)
    session.check()
    matched, _ = session.do_match(ids)
    assert matched.cached_len == len(ids) and matched.second is not None
    session.do_unlock(reader)
    session.do_evict_full(1 << 30)
    session.do_evict_second(1 << 30)
    session.check()
    assert not session.kv.in_use() and not session.second.in_use()
