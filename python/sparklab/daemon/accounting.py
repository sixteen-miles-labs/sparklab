"""Durable final-accounting receipts for engine lifecycle operations.

Each receipt is one independent JSON file.  A stop is allowed to signal the engine only after
its receipt has reached this outbox, so a daemon/Desktop crash cannot create an unrecorded gap.
The module is deliberately stdlib-only: the daemon must remain torch/CUDA free.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Callable
from typing import Any


class AccountingError(RuntimeError):
    """Base class for fail-closed accounting lifecycle errors."""


class AccountingPrepareError(AccountingError):
    """The engine's prepare-stop endpoint failed or returned a non-sealed snapshot."""


class AccountingOutboxError(AccountingError):
    """A receipt could not be durably persisted or read."""


class PrepareStopUnavailable(AccountingError):
    """The engine predates the prepare-stop endpoint (HTTP 404/405)."""


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")


def _validate_receipt_id(receipt_id: str) -> str:
    if not isinstance(receipt_id, str) or _SAFE_ID.fullmatch(receipt_id) is None:
        raise ValueError("invalid accounting receiptId")
    return receipt_id


def stable_receipt_id(identity: str) -> str:
    """Return the same receipt ID when a stop is retried for the same engine identity."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sparklab-accounting:{identity}"))


class AccountingOutbox:
    """Crash-safe receipt outbox backed by one atomically-replaced JSON file per receipt."""

    def __init__(
        self,
        path: str,
        *,
        replace_fn: Callable[[str, str], None] = os.replace,
    ) -> None:
        self.path = path
        self._replace = replace_fn
        self._lock = threading.RLock()

    def persist(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Durably store ``receipt`` and return the canonical persisted document.

        Persistence is idempotent by ``receiptId``.  If a prior daemon wrote the receipt and
        crashed before signalling the engine, a retry reuses that exact document rather than
        creating a second accounting event.
        """
        receipt_id = _validate_receipt_id(receipt.get("receiptId"))
        final_path = self._receipt_path(receipt_id)
        with self._lock:
            existing = self._read_one(final_path, missing_ok=True)
            if existing is not None:
                return existing

            os.makedirs(self.path, exist_ok=True)
            tmp_path = os.path.join(self.path, f".{receipt_id}.{uuid.uuid4().hex}.tmp")
            try:
                with open(tmp_path, "x", encoding="utf-8") as fh:
                    json.dump(
                        receipt, fh, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    fh.write("\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                self._replace(tmp_path, final_path)
                self._fsync_dir()
            except (OSError, TypeError, ValueError) as exc:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise AccountingOutboxError(f"failed to persist accounting receipt: {exc}") from exc
            return dict(receipt)

    def pending(self) -> list[dict[str, Any]]:
        """Return all unacknowledged receipts in deterministic creation order."""
        with self._lock:
            try:
                names = sorted(name for name in os.listdir(self.path) if name.endswith(".json"))
            except FileNotFoundError:
                return []
            except OSError as exc:
                raise AccountingOutboxError(f"failed to list accounting outbox: {exc}") from exc

            receipts = [self._read_one(os.path.join(self.path, name)) for name in names]
            return sorted(receipts, key=lambda doc: (doc.get("createdAt", 0), doc["receiptId"]))

    def ack(self, receipt_id: str) -> dict[str, Any]:
        """Idempotently acknowledge and remove one receipt."""
        receipt_id = _validate_receipt_id(receipt_id)
        path = self._receipt_path(receipt_id)
        with self._lock:
            try:
                os.remove(path)
            except FileNotFoundError:
                return {"acked": True, "already": True, "receiptId": receipt_id}
            except OSError as exc:
                raise AccountingOutboxError(
                    f"failed to acknowledge accounting receipt: {exc}"
                ) from exc
            self._fsync_dir()
            return {"acked": True, "already": False, "receiptId": receipt_id}

    def _receipt_path(self, receipt_id: str) -> str:
        return os.path.join(self.path, f"{receipt_id}.json")

    def _read_one(self, path: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise AccountingOutboxError(f"accounting receipt vanished while reading: {path}")
        except (OSError, ValueError) as exc:
            raise AccountingOutboxError(f"failed to read accounting receipt {path}: {exc}") from exc
        if not isinstance(doc, dict):
            raise AccountingOutboxError(f"accounting receipt is not an object: {path}")
        try:
            receipt_id = _validate_receipt_id(doc.get("receiptId"))
        except ValueError as exc:
            raise AccountingOutboxError(f"invalid accounting receipt {path}: {exc}") from exc
        if os.path.basename(path) != f"{receipt_id}.json":
            raise AccountingOutboxError(f"accounting receipt filename/id mismatch: {path}")
        return doc

    def _fsync_dir(self) -> None:
        """Best-effort directory fsync; unsupported on Windows but required on Linux durability."""
        flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
        try:
            fd = os.open(self.path, flags)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
