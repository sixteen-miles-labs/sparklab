"""Internal, opt-in resource reservations for native checkpoint experiments."""

from dataclasses import dataclass
import os


def exact_metadata_bytes():
    value = int(os.getenv("SPARKLAB_EXACT_PREFIX_METADATA_MIB", "16"))
    if not 1 <= value <= 64:
        raise ValueError("exact prefix metadata budget must be 1..64 MiB")
    return value << 20


@dataclass(frozen=True)
class DiskSettings:
    root: str
    quota_bytes: int
    max_records: int
    tickets: int
    staging_bytes: int
    policy: str
    timeout_seconds: float
    scope: str

    @classmethod
    def from_env(cls):
        root = os.getenv("SPARKLAB_PREFIX_STORE", "")
        if not root:
            return None
        if os.getenv("SPARKLAB_EXACT_PREFIX", "0") != "1":
            raise ValueError("disk prefix retention requires SPARKLAB_EXACT_PREFIX=1")
        quota = int(os.getenv("SPARKLAB_PREFIX_DISK_GIB", "8"))
        records = int(os.getenv("SPARKLAB_PREFIX_DISK_RECORDS", "128"))
        tickets = int(os.getenv("SPARKLAB_PREFIX_IO_TICKETS", "4"))
        staging = int(os.getenv("SPARKLAB_PREFIX_STAGING_MIB", "8"))
        policy = os.getenv("SPARKLAB_PREFIX_RESTORE_POLICY", "measured")
        timeout = float(os.getenv("SPARKLAB_PREFIX_RESTORE_TIMEOUT", "30"))
        scope = os.getenv("SPARKLAB_PREFIX_SCOPE", "single-instance")
        if not (
            1 <= quota <= 1024
            and 1 <= records <= 4096
            and 1 <= tickets <= 8
            and 1 <= staging <= 64
        ):
            raise ValueError("invalid prefix disk/index/queue/staging bounds")
        if policy not in {"measured", "calibrate", "always"}:
            raise ValueError("restore policy must be measured, calibrate or always")
        if not 0 < timeout <= 300 or not 1 <= len(scope) <= 256:
            raise ValueError("invalid prefix timeout or scope")
        return cls(
            root, quota << 30, records, tickets, staging << 20, policy, timeout, scope
        )

    @property
    def reserve_bytes(self):
        # All admitted/undrained staging windows, active copy/read temporaries,
        # bounded decoded headers and token tuples, and one serial JSON parser.
        # Reserve host and device storage from the same physical GB10 budget.
        estimate = (
            (self.tickets + 5) * self.staging_bytes
            + (64 << 20)
            + self.tickets * (12 << 20)
            + (8 << 20)
            + self.max_records * 512
        )
        quantum = 64 << 20
        return (estimate + quantum - 1) // quantum * quantum


def runtime_reserve_bytes():
    disk = DiskSettings.from_env()
    exact = (
        3 * exact_metadata_bytes()
        if os.getenv("SPARKLAB_EXACT_PREFIX", "0") == "1"
        else 0
    )
    adaptive = 0
    if os.getenv("SPARKLAB_DFLASH_POLICY", "fixed") == "cost":
        adaptive = 2 << 30  # bounded prepared verification graph reservation
    return exact + (disk.reserve_bytes if disk is not None else 0) + adaptive
