"""No-sync, bounded scheduler snapshots and actual output/codec handoff."""

from types import SimpleNamespace
import json

import torch

from sparklab.core import Batch
from sparklab.message import BaseTokenizerMsg, BaseFrontendMsg, UserReply
from sparklab.runtime.scheduler.scheduler import Scheduler
from sparklab.runtime.scheduler.telemetry import RuntimeTelemetry, Shape
from sparklab.serving.stats import StatsTracker, build_stats
from sparklab.tokenizer.server import _detokenized_replies
from tests.runtime.scheduler.test_abort_inflight_prefill import _setup, _launch_req


def shell():
    def forbidden():
        raise AssertionError("periodic snapshot read device counters")

    return SimpleNamespace(
        _runtime_timing=True,
        runtime_metrics=RuntimeTelemetry(trace_capacity=1000),
        cache_manager=SimpleNamespace(exact_endpoints=None),
        prefix_retention=SimpleNamespace(
            snapshot=lambda: {
                "unreleased_tickets": 2,
                "worker_observation": {"observed_at": 1.0, "counters": {"writes": 3}},
            }
        ),
        engine=SimpleNamespace(
            moe_offload_cache=SimpleNamespace(
                decode_miss_stats=forbidden,
                disk_source=SimpleNamespace(
                    stats=lambda: {
                        "read_ops": 7,
                        "physical_bytes": 1234,
                        "read_seconds": 0.5,
                        "cache_hits": torch.tensor(999),
                        "path": "must-not-be-exported",
                    }
                ),
            )
        ),
    )


def test_periodic_export_uses_cached_expert_observation_and_fixed_host_fields():
    owner = shell()
    first = Scheduler._runtime_snapshot(
        owner,
        2.0,
        {"kv_used_pages": 4},
        {"missing": 5, "active": 20, "routing": ["not in scalar export"]},
    )
    assert first["expert_decode"] == {
        "observed_at": 2.0,
        "counters": {"missing": 5, "active": 20},
    }
    assert first["expert_disk"]["counters"] == {
        "read_ops": 7,
        "physical_bytes": 1234,
        "read_seconds": 0.5,
    }
    assert first["prefix_store"]["worker_observation"]["observed_at"] == 1.0
    # An existing terminal observation may refresh between rate-limited exports.
    assert Scheduler._runtime_snapshot(owner, 2.5, {}, {"missing": 8}) is None
    second = Scheduler._runtime_snapshot(owner, 3.0, {"kv_used_pages": 6})
    assert second["expert_decode"] == {"observed_at": 2.5, "counters": {"missing": 8}}
    assert first["expert_decode"]["counters"]["missing"] == 5
    assert Scheduler._runtime_snapshot(owner, 3.999, {}) is None
    owner._runtime_timing = False
    owner.runtime_metrics = None
    assert Scheduler._runtime_snapshot(owner, 100.0, {}) is None


def test_shape_and_trace_pressure_cannot_expand_the_ipc_snapshot():
    owner = shell()
    owner.prefix_retention = SimpleNamespace(
        snapshot=lambda: {
            "records": 128,
            "unreleased_tickets": 8,
            "restore_waiters": 128,
            "service_costs": [
                {
                    "kind": "read" if i % 2 else "write",
                    "tokens": 1 << (i % 17),
                    "samples": 32,
                    "p90_seconds": 0.25,
                }
                for i in range(32)
            ],
        }
    )
    for i in range(10000):
        owner.runtime_metrics.gpu_completed(
            Shape.of("prefill", i % 128 + 1, i + 1), 0.01
        )
        owner.runtime_metrics.trace.append({"prompt": "private data" * 1000})
    result = Scheduler._runtime_snapshot(owner, 1.0, {})
    assert result["scheduler"]["trace"] == []
    assert len(result["scheduler"]["gpu_costs"]) == 64
    assert all(
        item["samples"] <= 128 and item["p90_seconds"] == 0.01
        for item in result["scheduler"]["gpu_costs"]
    )
    encoded = json.dumps(result)
    assert "private data" not in encoded and len(encoded) < 32768


def test_named_workspace_bytes_use_tensor_metadata_and_keep_missing_buffers_explicit():
    owner = shell()
    first = Scheduler._runtime_snapshot(owner, 1.0, {})
    assert first["retained_workspaces"]["verification_transactions"] is None
    owner.engine.linear_state_pool = SimpleNamespace(
        verify_recurrent_states=torch.empty(2, 3, dtype=torch.float32),
        verify_conv_inputs=torch.empty(4, dtype=torch.float16),
        verify_state_indices=torch.empty(3, dtype=torch.int64),
        _verify_aux_inputs={"unexported-name": torch.empty(7, dtype=torch.int64)},
    )
    owner.engine._prefix_reserved_bytes = 4096
    owner.engine.adaptive_graph_bytes = 8192
    second = Scheduler._runtime_snapshot(owner, 2.0, {})
    workspace = second["retained_workspaces"]
    assert workspace["verification_transactions"] == {
        "verify_recurrent_states": 24,
        "verify_conv_inputs": 8,
        "verify_state_indices": 24,
    }
    assert workspace["verification_auxiliary_bytes"] == 56
    assert workspace["prefix_reserved_bytes"] == 4096
    assert workspace["adaptive_graph_reserved_bytes"] == 8192
    assert "unexported-name" not in json.dumps(second)


def test_actual_burst_drain_attaches_one_snapshot_and_preserves_usage(monkeypatch):
    pool, cm, tm, _, _, sent, owner = _setup()
    owner._runtime_timing = True
    owner.runtime_metrics = RuntimeTelemetry()
    owner._runtime_events = []
    req = _launch_req(
        pool, cm, tm, torch.tensor([1, 2], dtype=torch.int32), output_len=6
    )
    batch = Batch(reqs=[req], phase="decode")
    output = SimpleNamespace(
        next_tokens_cpu=torch.tensor([20, 21, 22], dtype=torch.int32),
        token_counts=(3,),
        copy_done_event=SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(
        "sparklab.runtime.scheduler.scheduler.time",
        SimpleNamespace(monotonic=lambda: 10.0),
    )
    Scheduler._process_last_data(owner, (SimpleNamespace(batch=batch), output))
    assert len(sent) == 3
    assert (
        sent[0].runtime_snapshot["scheduler"]["counters"]["committed_output_tokens"]
        == 3
    )
    assert sent[1].runtime_snapshot is sent[2].runtime_snapshot is None
    # The real internal codec, not a parallel JSON interpretation.
    reply = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(sent[0]))
    snapshot = reply.runtime_snapshot
    detokenized = _detokenized_replies(
        SimpleNamespace(detokenize=lambda messages: ["text"]), [reply]
    )[0]
    user = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(detokenized))
    stats = StatsTracker()
    stats.observe(user, now=11.0)
    assert stats.native_runtime == snapshot
    assert stats.completion_tokens_total == 1
    assert stats.native_runtime["resources"]["kv_used_pages"] == sent[0].kv_used_pages


def test_frontend_retains_one_snapshot_and_reports_age_without_changing_totals(
    monkeypatch,
):
    stats = StatsTracker()
    config = SimpleNamespace(
        served_model_name="test",
        max_seq_len=4096,
        model_config=SimpleNamespace(
            has_linear_attention=False, has_swa_attention=False, is_moe=False
        ),
    )
    state = SimpleNamespace(stats=stats, config=config, ready_at=None)
    assert "native_runtime" not in build_stats(state, 0, 0)
    for i in range(1000):
        stats.observe(
            UserReply(1, "", False, runtime_snapshot={"observed_at": float(i)}), now=i
        )
    monkeypatch.setattr(
        "sparklab.serving.stats.time", SimpleNamespace(monotonic=lambda: 1004.0)
    )
    assert build_stats(state, 0, 0)["native_runtime"]["age_seconds"] == 5.0
    assert stats.prompt_tokens_total == stats.completion_tokens_total == 0
    assert stats.native_runtime == {"observed_at": 999.0}


def test_real_expert_disk_and_prefix_counters_export_together_without_actor_reads(
    tmp_path, monkeypatch
):
    from sparklab.checkpoint.ftw import FTWReader, FTWDiskExpertSource
    from tests.checkpoint.test_ftw_expert_rows import _write_ftw
    from tests.runtime.checkpoint.test_service import seeded

    values = torch.arange(6 * 513, dtype=torch.int16).view(6, 513)
    _write_ftw(tmp_path / "experts", [("weight", values)])
    reader = FTWReader(str(tmp_path / "experts"))
    source = FTWDiskExpertSource(
        reader,
        reader.expert_row_descriptors(num_layers=2),
        {"weight": SimpleNamespace(tensor=torch.empty(3, 513, dtype=torch.int16))},
        {"weight": SimpleNamespace(tensor=torch.empty(2, 513, dtype=torch.int16))},
    )
    service, *_ = seeded(tmp_path / "prefixes", monkeypatch)
    try:
        source.stage(0, [0, 1])
        source.stage(0, [0])
        owner = shell()
        owner.prefix_retention = service
        owner.engine.moe_offload_cache.disk_source = source

        def forbidden(*args, **kwargs):
            raise AssertionError("periodic actor observation touched the filesystem")

        with monkeypatch.context() as guards:
            guards.setattr("os.open", forbidden)
            guards.setattr("os.scandir", forbidden)
            result = Scheduler._runtime_snapshot(owner, 1.0, {})
        expert = result["expert_disk"]["counters"]
        assert expert["read_ops"] == 2
        assert expert["logical_bytes"] == 2 * values[0].numel() * values.element_size()
        assert expert["physical_bytes"] >= expert["logical_bytes"]
        assert expert["cache_hits"] == 1 and expert["cache_misses"] == 2
        assert result["prefix_store"]["worker_observation"]["records"] == 1
        assert (
            result["prefix_store"]["worker_observation"]["counters"]["recovered"] == 1
        )
    finally:
        assert service.close()
        source.close()
        reader.close()
