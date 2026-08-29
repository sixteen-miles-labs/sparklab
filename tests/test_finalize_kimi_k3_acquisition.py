from __future__ import annotations

import json
import sys

from benchmarks import finalize_kimi_k3_acquisition as finalize


def test_uses_validated_manifest_shape_and_physical_bytes(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for shard in range(1, 97):
        (model / f"model-{shard:05d}-of-000096.safetensors").write_bytes(b"x")

    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "started_at": "2026-08-29T01:00:00+00:00",
                "total_bytes": 120,
                "oom_kill_baseline": 10,
                "pswpout_baseline": 20,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "completed_at": "2026-08-29T01:01:00+00:00",
                "revision": "pinned",
                "artifacts": {
                    "source": {
                        "format": "safetensors",
                        "validation": {"physical_bytes": 96},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "final.json"
    monkeypatch.setattr(finalize, "_vmstat", lambda: {"oom_kill": 10, "pswpout": 21})
    monkeypatch.setattr(finalize, "_mem_available", lambda: 123)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_kimi_k3_acquisition.py",
            "--progress",
            str(progress),
            "--model",
            str(model),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--revision",
            "pinned",
        ],
    )

    assert finalize.main() == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["manifest_error"] is None
    assert record["committed_bytes"] == record["expected_bytes"] == 96
    assert record["declared_bytes"] == 120
    assert record["ended_at"] == "2026-08-29T01:01:00+00:00"
    assert record["duration_s"] == 60
    assert record["swap_out_pages_delta"] == 1
