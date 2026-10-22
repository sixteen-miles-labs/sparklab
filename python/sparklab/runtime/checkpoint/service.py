"""Scheduler-owned bounded persistence lifecycle; all filesystem work is delegated.

The actor polls fences, owns private restore pages and waiters, and publishes only
validated complete state. Cancellation detaches a waiter, never a DMA/I/O owner.
"""

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
import math
import time

import torch
from sparklab.utils import align_ceil

from .async_store import AsyncPrefixStore
from .record import RecordError, RecordLimits
from .store import StoreFull, StoreBusy
from .transfer import GPUStateSource, GPUStateSink


@dataclass
class Restore:
    key: str
    tokens: tuple
    indices: object
    slot: int
    started: float
    waiters: dict = field(default_factory=dict)
    ticket: object = None


class PrefixRetention:
    def __init__(
        self,
        cache,
        compatibility,
        settings,
        *,
        prefill_cost,
        clock=time.monotonic,
        source_factory=GPUStateSource,
        sink_factory=GPUStateSink,
        store_factory=AsyncPrefixStore,
        max_waiters=128,
    ):
        if not 1 <= max_waiters <= 128:
            raise ValueError("invalid restore waiter bound")
        self.cache = cache
        self.cm = cache.manager
        self.compatibility = compatibility
        self.settings = settings
        self.prefill_cost = prefill_cost
        self.clock = clock
        self.source_factory = source_factory
        self.sink_factory = sink_factory
        self.max_waiters = max_waiters
        self.store = store_factory(
            settings.root,
            compatibility,
            max_tickets=settings.tickets,
            quota_bytes=settings.quota_bytes,
            max_records=settings.max_records,
            limits=RecordLimits(
                min(settings.quota_bytes, 4 << 30), settings.staging_bytes
            ),
        )
        self.writes = {}
        self.restore = None  # one inbound private image; coalesced bounded waiters
        self.completed = []
        self.costs = OrderedDict()
        self.stats = Counter()
        self.disabled = False
        self._closed = False
        cache.on_publication = self.publish

    @property
    def outstanding(self):
        return self.store.outstanding

    def _record_cost(self, kind, n, seconds):
        bucket = (kind, 1 << (max(1, n) - 1).bit_length())
        if bucket not in self.costs:
            if len(self.costs) >= 32:
                self.costs.popitem(last=False)
            self.costs[bucket] = deque(maxlen=32)
        self.costs.move_to_end(bucket)
        self.costs[bucket].append(seconds)

    def _estimate(self, kind, n):
        values = self.costs.get((kind, 1 << (max(1, n) - 1).bit_length()))
        if not values:
            return None
        ordered = sorted(values)
        return ordered[math.ceil(0.9 * len(ordered)) - 1]

    def publish(self, endpoint):
        if self.disabled or self._closed:
            return
        # Publication has frozen the endpoint. Its copy producer may still be
        # pending: the source lease retains it and the I/O stream waits on that
        # event. Requiring query() here would silently skip most GPU endpoints.
        if self.outstanding >= self.settings.tickets:
            self.stats["publication_skipped_pressure"] += 1
            return
        key = endpoint.storage_key
        if key is None:
            key = self.compatibility.prefix_key(endpoint.tokens)
            endpoint.storage_key = key
        if key in self.writes or any(e.key == key for e in self.store.descriptors()):
            return
        source = self.source_factory(
            self.cache,
            endpoint,
            self.compatibility,
            chunk_bytes=self.settings.staging_bytes,
        )
        ticket = self.store.submit_write(source)
        if ticket is None:
            source.discard_unsubmitted()
            self.stats["publication_skipped_pressure"] += 1
            return
        self.writes[key] = (ticket, self.clock(), len(endpoint.tokens))
        self.stats["writes_submitted"] += 1

    def _worth_restoring(self, entry, busy, computed_tokens):
        if self.settings.policy == "always":
            return True
        restore = self._estimate("read", entry.consumed)
        recompute = self.prefill_cost(computed_tokens)
        queue_cost = 0.0
        for _, _, length in self.writes.values():
            cost = self._estimate("write", length)
            if cost is None:
                return False  # unknown contention is not free
            queue_cost += cost
        if restore is None or recompute is None:
            # Explicit calibration only on an otherwise idle path, one probe.
            return (
                self.settings.policy == "calibrate"
                and not busy
                and not self.outstanding
            )
        return restore + queue_cost < recompute

    def maybe_restore(self, pending, *, reserved_tokens=0, busy=False):
        """Return True while this request waits; never waits for I/O or a fence."""
        if pending.restore_waiting:
            return True
        if (
            pending.restore_attempted
            or pending.chunked_req is not None
            or pending.mm_embeds is not None
        ):
            return False
        pending.restore_attempted = True
        if self.disabled or self._closed or not self.store.ready.is_set():
            return False
        memory_length = self.cm.match_req(pending).cuda_handle.cached_len
        descriptors = [
            e
            for e in self.store.descriptors()
            if memory_length < e.consumed < pending.input_len
        ]
        if not descriptors:
            return False
        maximum = max(e.consumed for e in descriptors)
        tokens = tuple(pending.input_ids[:maximum].tolist())
        keys = self.compatibility.prefix_keys(tokens, (e.consumed for e in descriptors))
        candidates = [e for e in descriptors if keys[e.consumed] == e.key]
        if not candidates:
            return False
        entry = max(candidates, key=lambda e: e.consumed)
        if self.restore is not None:
            group = self.restore
            if (
                group.key == entry.key
                and not group.ticket.cancelled
                and len(group.waiters) < self.max_waiters
            ):
                group.waiters[pending.uid] = pending
                pending.restore_waiting = True
                self.stats["restores_coalesced"] += 1
                return True
            return False
        if self.outstanding >= self.settings.tickets or not self._worth_restoring(
            entry, busy, entry.consumed - memory_length
        ):
            self.stats["restore_recompute_selected"] += 1
            return False
        pages = (entry.consumed + self.cm.page_size - 1) // self.cm.page_size
        # Private snapshot plus the live suffix/output must fit. This is an
        # admission check, never permission to overdraw an existing pool.
        need = (
            pages * self.cm.page_size
            + align_ceil(
                entry.consumed % self.cm.page_size
                + pending.input_len - entry.consumed + pending.output_len,
                self.cm.page_size,
            )
            + reserved_tokens
        )
        if need > self.cm.available_size:
            self.stats["restore_skipped_capacity"] += 1
            return False
        self.cm.ensure_mamba_slots(4)
        if self.cm.linear_state_pool.num_free_slots < 4:
            return False
        indices = self.cm._page_to_token(self.cm._allocate(pages))
        slot = self.cm.linear_state_pool.alloc(1)[0]
        self.cache.restoring_pages += pages
        self.cache.restoring_states += 1
        ready = None
        if self.cm.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.cm.device))
        group = Restore(
            entry.key, tokens[: entry.consumed], indices, slot, self.clock()
        )
        try:
            sink = self.sink_factory(
                self.cache.adapter,
                self.compatibility,
                group.tokens,
                indices,
                slot,
                chunk_bytes=self.settings.staging_bytes,
                allocation_ready=ready,
            )
            ticket = self.store.submit_read(entry.key, sink)
        except BaseException:
            self._return_private(group)
            raise
        if ticket is None:
            self._return_private(group)
            return False
        group.ticket = ticket
        group.waiters[pending.uid] = pending
        pending.restore_waiting = True
        self.restore = group
        self.stats["restores_submitted"] += 1
        return True

    def detach(self, uid):
        group = self.restore
        if group is not None:
            pending = group.waiters.pop(uid, None)
            if pending is not None:
                pending.restore_waiting = False
            if not group.waiters:
                self.store.cancel(group.ticket)

    def _return_private(self, group, *, adopted=False):
        self.cache.restoring_pages -= group.indices.numel() // self.cm.page_size
        self.cache.restoring_states -= 1
        if not adopted:
            self.cm._free(group.indices)
            self.cm.linear_state_pool.free(group.slot)

    def poll(self):
        if self.store.failure is not None or (
            self.store.store is not None and self.store.store.failed
        ):
            self.disabled = True
            self.stats["worker_unavailable"] = 1
        group = self.restore
        if (
            group is not None
            and not group.ticket.cancelled
            and self.clock() - group.started >= self.settings.timeout_seconds
        ):
            for waiter in group.waiters.values():
                waiter.restore_waiting = False
            group.waiters.clear()
            self.store.cancel(group.ticket)
            self.stats["restore_timeout"] += 1
        self.completed.extend(self.store.poll())
        remaining = []
        unexpected = None
        for ticket in self.completed:
            payload = ticket.payload
            if not payload.transfer_complete():
                remaining.append(ticket)
                continue
            if ticket.kind == "write":
                key = payload.endpoint.storage_key
                _, started, length = self.writes.pop(key)
                payload.release_source()
                if ticket.error is None and ticket.result is not None:
                    self._record_cost("write", length, self.clock() - started)
                    self.stats["writes_completed"] += 1
            else:
                group = self.restore
                if group is None or group.ticket is not ticket:
                    raise AssertionError("restore completion lost scheduler ownership")
                adopted = False
                if (
                    ticket.error is None
                    and not ticket.cancelled
                    and payload.validated
                    and group.waiters
                ):
                    adopted = self.cache.adopt(
                        group.tokens, group.indices, group.slot, storage_key=group.key
                    )
                    self._record_cost(
                        "read", len(group.tokens), self.clock() - group.started
                    )
                    self.stats["restores_completed"] += int(adopted)
                self._return_private(group, adopted=adopted)
                for waiter in group.waiters.values():
                    waiter.restore_waiting = False
                self.restore = None
            error = ticket.error
            if error is not None:
                self.stats[ticket.kind + "_failures"] += 1
                if isinstance(error, RecordError):
                    self.stats["record_validation_failures"] += 1
                    if "checksum" in str(error):
                        self.stats["checksum_failures"] += 1
                if not isinstance(
                    error, (OSError, RecordError, StoreFull, StoreBusy, KeyError)
                ):
                    unexpected = error
            self.store.release(ticket)
        self.completed = remaining
        if unexpected is not None:
            raise RuntimeError("unexpected prefix transfer failure") from unexpected

    def snapshot(self):
        """Bounded actor/index metadata only; no prompt/path/key metric labels."""
        entries = self.store.descriptors()
        tickets = [item[0] for item in self.writes.values()]
        if self.restore is not None:
            tickets.append(self.restore.ticket)
        staging = 0
        for ticket in tickets:
            buffer = getattr(ticket.payload, "_buffer", None)
            if buffer is not None:
                staging += buffer.numel() * buffer.element_size()
        root = getattr(getattr(self.store.store, "root_quota", None), "usage", None)
        return {
            "counters": dict(self.stats),
            "worker_observation": self.store.observation,
            "records": len(entries),
            "disk_bytes": sum(e.nbytes for e in entries),
            "root_observed_bytes": root.nbytes if root is not None else 0,
            "root_observed_records": root.records if root is not None else 0,
            "root_observed_namespaces": root.namespaces if root is not None else 0,
            "root_observed_at": root.observed_at if root is not None else None,
            "service_costs": [
                {
                    "kind": kind,
                    "tokens": n,
                    "samples": len(values),
                    "p90_seconds": self._estimate(kind, n),
                }
                for (kind, n), values in self.costs.items()
            ],
            "unreleased_tickets": self.outstanding,
            "completed_pending_fence": len(self.completed),
            "staging_bytes": staging,
            "restore_waiters": len(self.restore.waiters)
            if self.restore is not None
            else 0,
            "private_restore_pages": self.cache.restoring_pages,
            "private_restore_slots": self.cache.restoring_states,
            "disabled": self.disabled,
        }

    def close(self, timeout=5.0):
        """Control-only shutdown; pools must remain alive if this returns False."""
        self._closed = True
        self.cache.on_publication = None
        if self.restore is not None:
            for waiter in self.restore.waiters.values():
                waiter.restore_waiting = False
            self.restore.waiters.clear()
            self.store.cancel(self.restore.ticket)
        deadline = time.monotonic() + timeout
        self.store.close(timeout=0)  # signal only; undrained tickets retain locks
        while True:
            self.poll()
            remaining = max(0.0, deadline - time.monotonic())
            if not self.outstanding:
                return self.store.close(timeout=remaining)
            if remaining <= 0:
                return False
            time.sleep(min(0.001, remaining))  # shutdown/control only
