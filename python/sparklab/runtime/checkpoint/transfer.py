"""Bounded staged GPU copies for immutable prefix records.

Callbacks run only on the prefix I/O thread. Its stream/events may wait there;
the model worker only polls readiness and retains pool ownership. Every source
read is protected by an exact-endpoint lease, and every restore remains private.
"""

from __future__ import annotations

import math
import threading

import torch

from .record import RecordError


class _Transfer:
    def __init__(self, adapter, chunk_bytes):
        self.adapter = adapter
        self.device = adapter.kv.device
        if self.device.type != "cuda":
            raise ValueError("staged native transfer requires CUDA")
        if chunk_bytes < 4096 or chunk_bytes % 4096:
            raise ValueError("staging windows must use whole 4 KiB units")
        self.chunk_bytes = chunk_bytes
        self._stream = None
        self._buffer = None
        self._event = None
        self._io_done = False
        self._actor_thread = threading.get_ident()
        self.transferred_bytes = 0

    def _prepare(self):
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)
            self._buffer = torch.empty(
                self.chunk_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            self._event = torch.cuda.Event()
            self._wait_producer()

    def _wait_producer(self):
        pass

    def io_finished(self):
        # Called by the worker's finally block, including on cancellation/error.
        # No pool mutation or logical ownership is released on this thread.
        self._io_done = True

    def transfer_complete(self):
        return self._io_done and (self._stream is None or self._stream.query())

    def _wait_copy(self):
        self._event.record(self._stream)
        self._event.synchronize()  # I/O worker only; protects staging reuse

    @staticmethod
    def _blocks(tensor):
        if tensor.is_contiguous():
            yield tensor.reshape(-1)
        else:
            # Slot selection makes the layer axis strided; each layer's state
            # remains contiguous. Never materialize the entire recurrent pool.
            for layer in range(tensor.shape[0]):
                part = tensor[layer]
                if not part.is_contiguous():
                    raise RuntimeError("unqualified noncontiguous checkpoint layout")
                yield part.reshape(-1)


class GPUStateSource(_Transfer):
    def __init__(self, cache, endpoint, compatibility, *, chunk_bytes):
        super().__init__(cache.adapter, chunk_bytes)
        self.cache = cache
        self.endpoint = endpoint
        self.compatibility = compatibility
        self.tokens = endpoint.tokens
        self.layout = self.adapter.layout(endpoint.state_slot, len(endpoint.tokens))
        self._parts = self.adapter.components(endpoint.state_slot)
        self._kv = self.adapter.kv._kv_buffer
        self._mapping = None
        self._released = False
        endpoint.readers += 1

    def _wait_producer(self):
        if self.endpoint.ready is not None:
            self._stream.wait_event(self.endpoint.ready)

    def _stage(self, tensor):
        size = tensor.numel() * tensor.element_size()
        if size > self.chunk_bytes:
            raise AssertionError("GPU staging window exceeded")
        self._buffer[:size].copy_(
            tensor.contiguous().view(torch.uint8).reshape(-1), non_blocking=True
        )
        self._wait_copy()
        self.transferred_bytes += size
        return memoryview(self._buffer[:size].numpy())

    @torch.inference_mode()
    def chunks(self, name, chunk_bytes):
        if chunk_bytes != self.chunk_bytes:
            raise RecordError("source/store staging bounds differ")
        self._prepare()
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            if name == "kv":
                if self._mapping is None:
                    pieces = [self.endpoint.anchor.kv_indices]
                    if self.endpoint.tail is not None:
                        pieces.append(
                            self.endpoint.tail[
                                : len(self.tokens) % self.cache.manager.page_size
                            ]
                        )
                    self._mapping = torch.cat(pieces)
                row_bytes = math.prod(self._kv.shape[4:]) * self._kv.element_size()
                rows = self.chunk_bytes // row_bytes
                if not rows:
                    raise RecordError("staging window cannot hold one KV row")
                for kind in range(2):
                    for layer in range(self._kv.shape[1]):
                        plane = self._kv[kind, layer].flatten(0, 1)
                        for start in range(0, len(self.tokens), rows):
                            selected = self._mapping[start : start + rows].long()
                            block = plane.index_select(0, selected)
                            yield self._stage(block)
            else:
                for block in self._blocks(self._parts[name]):
                    elements = self.chunk_bytes // block.element_size()
                    for start in range(0, block.numel(), elements):
                        yield self._stage(block[start : start + elements])

    def discard_unsubmitted(self):
        if self._stream is not None or self._released:
            raise AssertionError("cannot discard a dispatched transfer")
        self.cache.release(self.endpoint)
        self._released = True

    def release_source(self):
        if threading.get_ident() != self._actor_thread:
            raise AssertionError("only the model worker releases pool ownership")
        if not self.transfer_complete() or self._released:
            raise RuntimeError("source transfer still owns prefix resources")
        self.cache.release(self.endpoint)
        self._released = True


class GPUStateSink(_Transfer):
    def __init__(
        self,
        adapter,
        compatibility,
        tokens,
        indices,
        state_slot,
        *,
        chunk_bytes,
        allocation_ready,
    ):
        super().__init__(adapter, chunk_bytes)
        self.compatibility = compatibility
        self.tokens = tuple(tokens)
        self.indices = indices[: len(self.tokens)]
        self.state_slot = state_slot
        self.allocation_ready = allocation_ready
        self.layout = adapter.layout(state_slot, len(self.tokens))
        self._parts = adapter.components(state_slot)
        self._kv = adapter.kv._kv_buffer
        self.validated = False
        self._begun = False

    def _wait_producer(self):
        if self.allocation_ready is not None:
            self._stream.wait_event(self.allocation_ready)

    def begin(self, header, layout):
        if (
            header["tokens"] != list(self.tokens)
            or header["compatibility"] != self.compatibility.fingerprint
            or header["scope"] != self.compatibility.scope
            or layout != self.layout
        ):
            raise RecordError("restore identity or complete state layout mismatch")
        self._begun = True

    def validated_record(self, header, layout):
        if not self._begun or layout != self.layout:
            raise AssertionError("validated restore lost its layout contract")
        self.validated = True

    def preferred_size(self, name, offset, maximum):
        if name == "kv":
            row_bytes = math.prod(self._kv.shape[4:]) * self._kv.element_size()
            plane_bytes = len(self.tokens) * row_bytes
            remaining_plane = plane_bytes - offset % plane_bytes
            size = min(maximum // row_bytes * row_bytes, remaining_plane)
            if not size:
                raise RecordError("staging window cannot hold one KV row")
            return size
        tensor = self._parts[name]
        element_size = tensor.element_size()
        maximum = maximum // element_size * element_size
        if not tensor.is_contiguous():
            layer_bytes = math.prod(tensor.shape[1:]) * element_size
            return min(maximum, layer_bytes - offset % layer_bytes)
        return maximum

    @torch.inference_mode()
    def chunk(self, name, offset, data):
        if not self._begun or not 0 < len(data) <= self.chunk_bytes:
            raise RecordError("invalid restore chunk")
        self._prepare()
        host = torch.frombuffer(data, dtype=torch.uint8)
        self._buffer[: len(data)].copy_(host)
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            if name == "kv":
                row_bytes = math.prod(self._kv.shape[4:]) * self._kv.element_size()
                plane_bytes = len(self.tokens) * row_bytes
                if offset % row_bytes or len(data) % row_bytes:
                    raise RecordError("unaligned logical KV row")
                plane = offset // plane_bytes
                row = offset % plane_bytes // row_bytes
                rows = len(data) // row_bytes
                if row + rows > len(self.tokens):
                    raise RecordError("restore crosses a KV plane")
                kind, layer = divmod(plane, self._kv.shape[1])
                payload = (
                    self._buffer[: len(data)]
                    .view(self._kv.dtype)
                    .reshape(rows, *self._kv.shape[4:])
                    .to(self.device, non_blocking=True)
                )
                destination = self._kv[kind, layer].flatten(0, 1)
                destination.index_copy_(
                    0, self.indices[row : row + rows].long(), payload
                )
            else:
                tensor = self._parts[name]
                if tensor.is_contiguous():
                    destination = tensor.view(torch.uint8).reshape(-1)
                    local_offset = offset
                else:
                    layer_bytes = math.prod(tensor.shape[1:]) * tensor.element_size()
                    layer, local_offset = divmod(offset, layer_bytes)
                    destination = tensor[layer].view(torch.uint8).reshape(-1)
                if local_offset + len(data) > destination.numel():
                    raise RecordError("restore crosses a state component block")
                destination[local_offset : local_offset + len(data)].copy_(
                    self._buffer[: len(data)], non_blocking=True
                )
            self._wait_copy()
            self.transferred_bytes += len(data)

    def transfer_complete(self):
        return super().transfer_complete() and (
            self.allocation_ready is None or self.allocation_ready.query()
        )
