"""Exact hybrid endpoints backed by the existing radix/page/linear-state pools.

Full pages stay shared and immutable in HybridRadixCache. Each endpoint owns at
most one private tail page and one complete recurrent slot. No additional tensor
pool or model-weight copies are created. First integration is target-only and
non-overlap so publication occurs at the exact drained consumed endpoint.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar
import torch

from sparklab.runtime.kvcache.hybrid_radix_cache import HybridCacheHandle
from .state import MHARecurrentAdapter
from .config import exact_metadata_bytes


@dataclass
class Endpoint:
    tokens: tuple[int, ...]
    anchor: HybridCacheHandle
    state_slot: int
    tail: torch.Tensor | None
    ready: object | None
    readers: int = 0
    storage_key: str | None = None

    def complete(self):
        return self.ready is None or self.ready.query()


@dataclass(frozen=True)
class KVAnchor(HybridCacheHandle):
    full_only: ClassVar[bool] = True


@dataclass(frozen=True)
class ExactHandle(KVAnchor):
    endpoint: Endpoint


class ExactEndpointCache:
    def __init__(self, manager, kv_pool, *, max_records=8, max_metadata_bytes=None):
        if not manager.is_hybrid or not 1 <= max_records <= 128:
            raise ValueError("exact endpoints require a bounded hybrid radix cache")
        self.adapter = MHARecurrentAdapter(kv_pool, manager.linear_state_pool)
        self.manager = manager
        self.kv = kv_pool
        self.max_records = max_records
        self.max_metadata_bytes = (
            exact_metadata_bytes() if max_metadata_bytes is None else max_metadata_bytes
        )
        if not 1024 <= self.max_metadata_bytes <= 64 << 20:
            raise ValueError("invalid exact metadata bound")
        self.metadata_bytes = 0
        self.restoring_pages = 0
        self.restoring_states = 0
        self.on_publication = None
        self.entries = OrderedDict()
        self.published = 0
        self.skipped = 0
        self.hits = 0

    @property
    def max_consumed(self):
        return max((len(e.tokens) for e in self.entries.values()), default=0)

    @staticmethod
    def entry_metadata_bytes(consumed):
        return consumed * 64 + 512

    def match(self, tokens):
        tokens = tuple(tokens)
        matches = [
            entry
            for entry in self.entries.values()
            if len(entry.tokens) <= len(tokens)
            and entry.complete()
            and tokens[: len(entry.tokens)] == entry.tokens
        ]
        if not matches:
            return None
        entry = max(matches, key=lambda e: len(e.tokens))
        self.entries.move_to_end(entry.tokens)
        self.hits += 1
        return ExactHandle(
            len(entry.tokens), entry.anchor.node, entry.anchor.kv_indices, entry
        )

    def lock(self, handle):
        handle.endpoint.readers += 1

    def release(self, entry):
        if entry.readers <= 0:
            raise AssertionError("exact endpoint read pin underflow")
        entry.readers -= 1

    def evict_one(self):
        cm = self.manager
        for key, entry in self.entries.items():
            if entry.readers or not entry.complete():
                continue
            cm.unlock(entry.anchor)
            if entry.tail is not None:
                cm._free(entry.tail)
            cm.linear_state_pool.free(entry.state_slot)
            del self.entries[key]
            self.metadata_bytes -= self.entry_metadata_bytes(len(entry.tokens))
            return True
        return False

    def clear(self):
        while self.entries:
            if not self.evict_one():
                raise RuntimeError(
                    "cannot clear exact endpoints while GPU/read users remain"
                )

    def invalidate_rebuild(self):
        # The idle engine already fenced/reallocated pools; their free lists are
        # reset by CacheManager.rebuild. Never return old physical IDs to them.
        if (
            self.restoring_pages
            or self.restoring_states
            or any(e.readers for e in self.entries.values())
        ):
            raise AssertionError(
                "cache rebuild has live exact endpoint readers/restores"
            )
        self.entries.clear()
        self.metadata_bytes = 0

    @property
    def tail_pages(self):
        return sum(e.tail is not None for e in self.entries.values())

    def restore_tail(self, handle, table_idx):
        cm = self.manager
        entry = handle.endpoint
        full = entry.anchor.cached_len
        cm.page_table[table_idx, :full].copy_(entry.anchor.kv_indices)
        if entry.tail is None:
            return
        private = cm._page_to_token(cm._allocate(1))
        count = len(entry.tokens) - full
        buf = self.kv._kv_buffer.flatten(2, 3)
        source = entry.tail[:count].long()
        buf.index_copy_(2, private[:count].long(), buf.index_select(2, source))
        cm.page_table[table_idx, full : full + cm.page_size].copy_(private)

    def publish(self, req):
        cm = self.manager
        n = req.cached_len
        if (
            req.mm_embeds is not None
            or req.aborted
            or req.state_overadvanced
            or not n
            or n > len(req.input_ids)
            or req.linear_slot_idx is None
        ):
            self.skipped += 1
            return False
        metadata = self.entry_metadata_bytes(n)
        if metadata > self.max_metadata_bytes:
            self.skipped += 1
            return False
        tokens = tuple(req.input_ids[:n].tolist())
        if tokens in self.entries:
            self.entries.move_to_end(tokens)
            if self.on_publication is not None:
                self.on_publication(self.entries[tokens])
            return True
        while (
            len(self.entries) >= self.max_records
            or self.metadata_bytes + metadata > self.max_metadata_bytes
        ):
            if not self.evict_one():
                self.skipped += 1
                return False
        ps = cm.page_size
        full = n // ps * ps
        needs_tail = n != full
        def tail_fits():
            return not needs_tail or len(cm.free_slots) > cm.exact_future_pages

        # Publication never waits for an evictable GPU ticket or an allocation.
        # Pool pressure skips this endpoint; live inference keeps its resources.
        if cm.linear_state_pool.num_free_slots < 1 or (
            not tail_fits()
        ):
            if not self.evict_one():
                self.skipped += 1
                return False
        if cm.linear_state_pool.num_free_slots < 1 or (
            not tail_fits()
        ):
            self.skipped += 1
            return False
        slot = cm.linear_state_pool.alloc(1)[0]
        tail = cm._page_to_token(cm._allocate(1)) if needs_tail else None
        cm.linear_state_pool.copy_from(req.linear_slot_idx, slot)
        if tail is not None:
            buf = self.kv._kv_buffer.flatten(2, 3)
            source = cm.page_table[req.table_idx, full:n].long()
            buf.index_copy_(2, tail[: n - full].long(), buf.index_select(2, source))
        # Insert KV only: this node's recurrent state is NOT at the rounded key.
        # Exact recurrent state belongs to the Endpoint, never the aligned node.
        duplicate, node = cm.prefix_cache.insert_kv_only(
            req.input_ids[:full], cm.page_table[req.table_idx, :full]
        )
        indices = cm.prefix_cache._collect_kv(node)
        anchor = KVAnchor(full, node, indices)
        cm.lock(anchor)  # endpoint's immutable full-page ownership
        cm.lock(anchor)  # current request's ownership after canonicalization
        old = req.cache_handle
        cm.unlock(old)
        cm._free(cm.page_table[req.table_idx, old.cached_len : duplicate])
        cm.page_table[req.table_idx, :full].copy_(indices)
        req.cache_handle = anchor
        req.mamba_last_track_seqlen = None
        ready = None
        if cm.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(cm.device))
        self.entries[tokens] = Endpoint(tokens, anchor, slot, tail, ready)
        self.metadata_bytes += metadata
        self.published += 1
        if self.on_publication is not None:
            self.on_publication(self.entries[tokens])
        return True

    def adopt(self, tokens, indices, state_slot, *, ready=None, storage_key=None):
        """Adopt a validated private restore after GPU completion, without copies.

        False leaves every supplied page/slot owned by the transfer ticket. True
        transfers ownership to this cache and returns only duplicate full pages.
        """
        cm = self.manager
        tokens = tuple(tokens)
        # The caller transfers whole private pages, including the unused tail.
        # Validate metadata before mutating radix ownership. Never inspect CUDA
        # mapping values here: that would synchronize the model worker.
        expected = ((len(tokens) + cm.page_size - 1) // cm.page_size) * cm.page_size
        if (
            indices.ndim != 1
            or indices.numel() != expected
            or indices.dtype != torch.int32
            or indices.device != cm.device
            or type(state_slot) is not int
            or not 0 < state_slot < cm.linear_state_pool.num_slots
            or state_slot in cm.linear_state_pool._free_slots
        ):
            raise ValueError("restore does not own complete private pages/state slot")
        if (
            not tokens
            or tokens in self.entries
            or (ready is not None and not ready.query())
        ):
            return False
        metadata = self.entry_metadata_bytes(len(tokens))
        if metadata > self.max_metadata_bytes:
            return False
        while (
            len(self.entries) >= self.max_records
            or self.metadata_bytes + metadata > self.max_metadata_bytes
        ):
            if not self.evict_one():
                return False
        full = len(tokens) // cm.page_size * cm.page_size
        ids = torch.tensor(tokens, dtype=torch.int32, device="cpu")
        duplicate, node = cm.prefix_cache.insert_kv_only(ids[:full], indices[:full])
        anchor = KVAnchor(full, node, cm.prefix_cache._collect_kv(node))
        cm.lock(anchor)
        cm._free(indices[:duplicate])
        tail = indices[full:] if len(tokens) != full else None
        self.entries[tokens] = Endpoint(
            tokens, anchor, state_slot, tail, ready, storage_key=storage_key
        )
        self.metadata_bytes += metadata
        return True
