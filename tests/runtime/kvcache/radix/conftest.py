"""Fixtures: the (class x page_size) matrix, and the deterministic clock.

The clock is AUTOUSE for the whole package.  ``_tree_walk`` stamps every node it touches with a
single ``time.monotonic_ns()`` read, so nodes tie within a call by construction; a real clock adds
ties *between* calls too (its resolution is coarser than one walk), which makes any LRU /
eviction-order assertion flaky.
"""
from __future__ import annotations

import pytest

from .driver import ALL_SPECS, CacheSpec, Session, deterministic_clock


@pytest.fixture(autouse=True)
def det_clock():
    with deterministic_clock():
        yield


def _matrix(kind: str):
    """A fixture over every geometry of one class; geometries the class itself rejects are skipped
    rather than silently dropped."""
    specs = [s for s in ALL_SPECS if s.kind == kind]

    @pytest.fixture(params=specs, ids=[s.id for s in specs])
    def fixture(request) -> CacheSpec:
        spec: CacheSpec = request.param
        if spec.unsupported:
            pytest.skip(spec.unsupported)
        return spec

    return fixture


plain_spec = _matrix("plain")
swa_spec = _matrix("swa")
hybrid_spec = _matrix("hybrid")


@pytest.fixture
def make_session():
    """Factory for tests that need a specific spec (or several sessions)."""
    return Session
