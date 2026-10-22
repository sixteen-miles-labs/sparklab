"""Exact consumed-prefix identity and complete MHA/GDN state copy contract.

This does not enable shared-cache reuse. Runtime ownership/publication integrates
separately, after continuation tests qualify the model's concrete state layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import torch


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Compatibility:
    fingerprint: str
    scope: str

    @classmethod
    def from_config(cls, config: Mapping):
        required = {
            "artifact",
            "precision",
            "model_config",
            "tokenizer",
            "template",
            "state_layout",
            "cache_geometry",
            "continuation",
            "scope",
        }
        if not required <= config.keys() or not all(
            config[k] is not None for k in required
        ):
            raise ValueError("incomplete checkpoint compatibility identity")
        if not isinstance(config["scope"], str) or not 1 <= len(config["scope"]) <= 256:
            raise ValueError("an explicit checkpoint isolation scope is required")
        return cls(canonical_digest(dict(config)), config["scope"])

    def prefix_key(self, tokens):
        return canonical_digest(
            {"compatibility": self.fingerprint, "tokens": list(tokens)}
        )

    def prefix_keys(self, tokens, lengths):
        """Compute bounded candidate digests in one pass over the consumed IDs."""
        targets = set(lengths)
        if any(type(n) is not int or not 0 < n <= len(tokens) for n in targets):
            raise ValueError("invalid prefix lookup endpoint")
        digest = hashlib.sha256(
            ('{"compatibility":"' + self.fingerprint + '","tokens":[').encode()
        )
        result = {}
        for i, token in enumerate(tokens, 1):
            if i > 1:
                digest.update(b",")
            digest.update(str(token).encode())
            if i in targets:
                finalized = digest.copy()
                finalized.update(b"]}")
                result[i] = finalized.hexdigest()
        return result


def eligible_endpoints(full_tokens, message_prefixes=(), terminal=None):
    """Rendered/tokenized candidates qualify only by exact prefix equality."""
    tokens = tuple(full_tokens)
    endpoints = {len(tokens)} if tokens else set()
    for candidate in message_prefixes:
        candidate = tuple(candidate)
        if (
            candidate
            and len(candidate) <= len(tokens)
            and tokens[: len(candidate)] == candidate
        ):
            endpoints.add(len(candidate))
    if terminal is not None:
        if not 0 <= terminal <= len(tokens):
            raise ValueError("terminal endpoint is outside the consumed sequence")
        if terminal:
            endpoints.add(terminal)
    return tuple(sorted(endpoints))


@dataclass(frozen=True)
class StateLayout:
    components: tuple[tuple[str, tuple[int, ...], str], ...]
    page_size: int
    layer_map: tuple[int, ...] | None

    @property
    def fingerprint(self):
        return canonical_digest(
            {
                "components": self.components,
                "page_size": self.page_size,
                "layer_map": self.layer_map,
            }
        )


class StateImage:
    """An owned immutable logical image. Restore copies never alias its storage.

    The payload is private to the adapter and serializer. External inspection
    returns clones. CUDA producer completion must be retained by a transfer
    ticket before physical source pages/slots may be reused.
    """

    def __init__(self, compatibility, tokens, layout, tensors, ready=None):
        self.compatibility = compatibility
        self.tokens = tuple(tokens)
        self.layout = layout
        self._tensors = tensors
        self.ready = ready
        self.key = compatibility.prefix_key(self.tokens)

    @property
    def consumed(self):
        return len(self.tokens)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self._tensors.values())

    def inspect(self):
        return {k: v.clone() for k, v in self._tensors.items()}

    @property
    def supports_full_prompt_resume(self):
        # No saved resume logits/hidden vector. Integration must recompute a safe
        # suffix from an earlier endpoint rather than sample an old request's token.
        return False


class MHARecurrentAdapter:
    """Portable copy layout for MHA KV plus every registered GDN slot component.

    SWA, QSA/index/compressor pools and speculative continuation require their
    own complete contracts. Reject them here instead of copying only their KV.
    """

    def __init__(self, kv_pool, linear_pool=None, *, continuation="target-only"):
        from sparklab.runtime.kvcache.mha_pool import MHAKVCache

        if type(kv_pool) is not MHAKVCache:
            raise ValueError(
                "complete checkpoint adapter requires the exact MHA pool contract"
            )
        if continuation != "target-only":
            raise ValueError(
                "draft continuation is not yet qualified for this state contract"
            )
        self.kv = kv_pool
        self.linear = linear_pool
        self.page_size = int(kv_pool._kv_buffer.shape[3])

    def components(self, slot):
        if self.linear is None:
            return {}
        self.linear.materialize_verify_state()
        if type(slot) is not int or not 0 < slot < self.linear.num_slots:
            raise ValueError("checkpoint must use a private live recurrent slot")
        return {
            "conv": self.linear.conv_states[:, slot],
            "recurrent": self.linear.recurrent_states[:, slot],
            **{"aux." + k: v[slot] for k, v in self.linear._aux_states.items()},
        }

    def layout(self, slot, consumed):
        buf = self.kv._kv_buffer
        shape = (2, buf.shape[1], consumed, *buf.shape[4:])
        parts = {"kv": (shape, str(buf.dtype))}
        parts.update(
            {
                k: (tuple(v.shape), str(v.dtype))
                for k, v in self.components(slot).items()
            }
        )
        return StateLayout(
            tuple(
                sorted((k, tuple(shape), dtype) for k, (shape, dtype) in parts.items())
            ),
            self.page_size,
            tuple(self.kv._layer_map) if self.kv._layer_map is not None else None,
        )

    def required_bytes(self, slot, consumed):
        buf = self.kv._kv_buffer
        per_token = 2 * buf.shape[1] * buf.shape[4] * buf.shape[5] * buf.element_size()
        return consumed * per_token + sum(
            t.numel() * t.element_size() for t in self.components(slot).values()
        )

    def _indices(self, indices, consumed):
        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim != 1
            or indices.numel() != consumed
        ):
            raise ValueError("physical mapping must match the exact consumed endpoint")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("physical mapping must contain integer indices")
        # CPU validation has no GPU synchronization. Device mappings are produced
        # by the cache owner; bounds remain enforced by index_select/index_copy.
        if indices.device.type == "cpu" and consumed:
            if (
                indices.min() < 0
                or indices.max() >= self.kv._kv_buffer.shape[2] * self.page_size
            ):
                raise ValueError("physical mapping is out of bounds")
        return indices.to(device=self.kv.device, dtype=torch.long)

    def snapshot(self, compatibility, tokens, indices, slot=None, *, max_bytes):
        tokens = tuple(tokens)
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("checkpoint needs nonempty consumed token IDs")
        required = self.required_bytes(slot, len(tokens))
        if required > max_bytes:
            return None  # bounded capacity pressure skips publication
        index = self._indices(indices, len(tokens))
        buf = self.kv._kv_buffer.flatten(2, 3)
        tensors = {"kv": buf.index_select(2, index)}
        tensors.update({k: t.clone() for k, t in self.components(slot).items()})
        ready = None
        if self.kv.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.kv.device))
        return StateImage(
            compatibility, tokens, self.layout(slot, len(tokens)), tensors, ready
        )

    def restore(self, image, compatibility, indices, slot=None):
        # Validate ALL components before mutating a live destination.
        if (
            image.compatibility != compatibility
            or image.key != compatibility.prefix_key(image.tokens)
        ):
            raise ValueError("incompatible checkpoint identity")
        layout = self.layout(slot, image.consumed)
        if layout != image.layout:
            raise ValueError("incompatible checkpoint state layout")
        if set(image._tensors) != {name for name, _, _ in layout.components}:
            raise ValueError("incomplete checkpoint state components")
        for name, shape, dtype in layout.components:
            tensor = image._tensors[name]
            if tuple(tensor.shape) != shape or str(tensor.dtype) != dtype:
                raise ValueError("checkpoint component geometry mismatch")
        index = self._indices(indices, image.consumed)
        if image.ready is not None:
            if self.kv.device.type != "cuda":
                raise ValueError(
                    "CUDA checkpoint requires an explicit staging transfer"
                )
            torch.cuda.current_stream(self.kv.device).wait_event(image.ready)
        self.kv._kv_buffer.flatten(2, 3).index_copy_(2, index, image._tensors["kv"])
        for name, dst in self.components(slot).items():
            dst.copy_(image._tensors[name])
