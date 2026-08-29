from __future__ import annotations

import importlib.util
from pathlib import Path


_PATH = Path(__file__).parents[1] / "benchmarks" / "validate_decode_promotion.py"
_SPEC = importlib.util.spec_from_file_location("validate_decode_promotion", _PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate_row = _MODULE.validate_row


def _valid_row(tokens: int = 16) -> dict:
    telemetry = {
        "memory_guard_triggered": False,
        "mem_available_gib_min": 20.0,
        "oom_kill_delta": 0,
        "swap_out_pages_delta": 0,
        "swap_growth_bytes": 0,
    }
    return {
        "checkpoint": {"format": "ftw", "fingerprint": "abc123", "bytes": 1234},
        "completion_tokens": tokens,
        "events": tokens,
        "output_sha1": "0123456789ab",
        "output_text": "We need solve this carefully.",
        "oom_markers": 0,
        "lifecycle_telemetry": telemetry,
        "gpu_telemetry": {k: telemetry[k] for k in (
            "oom_kill_delta", "swap_out_pages_delta", "swap_growth_bytes"
        )},
    }


def test_valid_promotion_row_passes():
    assert validate_row(_valid_row(), 16) == []


def test_unsafe_or_incomplete_row_fails_every_gate():
    row = _valid_row()
    row["checkpoint"]["format"] = "unknown"
    row["completion_tokens"] = 15
    row["output_text"] = "\ufffd"
    row["oom_markers"] = 1
    row["lifecycle_telemetry"]["memory_guard_triggered"] = True
    row["lifecycle_telemetry"]["oom_kill_delta"] = 1
    row["lifecycle_telemetry"]["swap_out_pages_delta"] = 2
    row["lifecycle_telemetry"]["swap_growth_bytes"] = 4096
    errors = validate_row(row, 16)
    assert any("checkpoint.format" in error for error in errors)
    assert any("completion_tokens" in error for error in errors)
    assert any("output_text" in error for error in errors)
    assert any("oom_markers" in error for error in errors)
    assert any("memory_guard_triggered" in error for error in errors)
    assert any("oom_kill_delta" in error for error in errors)
    assert any("swap_out_pages_delta" in error for error in errors)
    assert any("swap_growth_bytes" in error for error in errors)


def test_missing_telemetry_is_not_treated_as_zero():
    row = _valid_row()
    row["gpu_telemetry"] = {}
    errors = validate_row(row, 16)
    assert errors == [
        "gpu_telemetry.oom_kill_delta is missing",
        "gpu_telemetry.swap_out_pages_delta is missing",
        "gpu_telemetry.swap_growth_bytes is missing",
    ]
