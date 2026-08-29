from __future__ import annotations

import importlib.util
from pathlib import Path


_PATH = Path(__file__).parents[1] / "benchmarks" / "validate_conversion_promotion.py"
_SPEC = importlib.util.spec_from_file_location("validate_conversion_promotion", _PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate = _MODULE.validate


def _record() -> dict:
    return {
        "status": "completed",
        "returncode": 0,
        "termination_reason": None,
        "vmstat": {"delta": {"oom_kill": 0, "pswpout": 0}},
        "swap": {"growth_bytes": 0},
        "memory": {"available_min_bytes": 1, "process_group_peak_rss_bytes": 1},
        "disk": {"available_min_bytes": 1},
    }


def test_safe_conversion_passes():
    assert validate(_record()) == []


def test_oom_swap_or_guard_failure_blocks_runtime():
    record = _record()
    record["status"] = "guard_terminated"
    record["returncode"] = -15
    record["termination_reason"] = "MemAvailable below 12 GiB"
    record["vmstat"]["delta"] = {"oom_kill": 1, "pswpout": 2}
    record["swap"]["growth_bytes"] = 4096
    errors = validate(record)
    for marker in ("status", "returncode", "termination_reason", "oom_kill", "pswpout", "growth_bytes"):
        assert any(marker in error for error in errors)


def test_missing_safety_telemetry_blocks_runtime():
    record = _record()
    record["vmstat"] = {"delta": {}}
    record["swap"] = {}
    record["memory"] = {}
    record["disk"] = {}
    assert len(validate(record)) == 6
