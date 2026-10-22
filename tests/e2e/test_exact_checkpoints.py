"""Complete real-model continuation with identical consumed execution partitions.

SPARKLAB_EXACT_TEST_MODEL points to a complete supported hybrid MHA checkpoint.
This is separate from cross-partition greedy/quality qualification: quantized
projection grouping can change tied logits even when stored bytes are exact.
"""

import os
import hashlib
import json
import time
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch

from tests.e2e.runtime_profiles import model_options

pytestmark = pytest.mark.needs_weights


def test_real_hybrid_exact_resume_matches_cold_state_and_two_branches(
    monkeypatch, tmp_path
):
    path = os.getenv("SPARKLAB_EXACT_TEST_MODEL")
    if not path or not Path(path).is_dir():
        pytest.skip("set SPARKLAB_EXACT_TEST_MODEL")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and complete checkpoint")
    # Keep the Engine process CUDA-fresh; query memory in a short-lived child.
    import math
    import subprocess
    import sys

    guard = float(os.getenv("SPARKLAB_EXACT_TEST_MIN_FREE_GIB", "30"))
    if not math.isfinite(guard) or guard < 30:
        raise ValueError("complete exact-state validation requires at least 30 GiB")
    def require_memory():
        free = float(subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.cuda.mem_get_info()[0]/2**30)"],
            text=True, timeout=30,
        ))
        if free < guard:
            pytest.skip("complete exact-state profile needs 30 GiB free physical GPU memory")
        return free

    require_memory()
    from sparklab.core import SamplingParams
    from sparklab.llm import LLM
    from sparklab.message import PromptAdmittedMsg
    from sparklab.runtime.checkpoint.state import Compatibility

    lengths = tuple(
        int(n)
        for n in os.getenv(
            "SPARKLAB_EXACT_TEST_LENGTHS", "1,15,16,17,31,63,64,65"
        ).split(",")
    )
    if not lengths or not all(0 < n <= 32768 for n in lengths):
        raise ValueError("exact test endpoints must be 1..32768")
    streamed = os.getenv("SPARKLAB_EXACT_TEST_STREAM_HASH", "0") == "1"
    context = max(4096, max(lengths) + 16)
    cache_tokens = int(
        os.getenv(
            "SPARKLAB_EXACT_TEST_CACHE_TOKENS",
            str(8192 if context == 4096 else context + 8192),
        )
    )
    monkeypatch.setenv("SPARKLAB_EXACT_PREFIX", "1")
    monkeypatch.setenv("SPARKLAB_SCHEDULER_POLICY", "baseline")
    monkeypatch.setenv("SPARKLAB_DFLASH_POLICY", "fixed")
    monkeypatch.setenv("SPARKLAB_PREFIX_STORE", str(tmp_path / "prefixes"))
    monkeypatch.setenv("SPARKLAB_PREFIX_RESTORE_POLICY", "always")
    monkeypatch.setenv("SPARKLAB_PREFIX_STAGING_MIB", "1")
    identity = Compatibility.from_config(
        dict(
            artifact=str(Path(path).resolve()),
            precision="native",
            model_config="unchanged-loaded-model",
            tokenizer="same-loaded-tokenizer",
            template="raw-text",
            state_layout="mha-gdn-v1",
            cache_geometry={"page_size": 16},
            continuation="target-only",
            scope="test-instance",
        )
    )

    def byte_digest(adapter, mapping, slot, consumed):
        # Independent test observation: read the real logical mapping directly,
        # not through the production record encoder/decoder or staging transport.
        # Bounded 8 MiB windows avoid retaining two multi-GiB state copies.
        window = 8 << 20
        kv = adapter.kv._kv_buffer
        per_row = kv.shape[4] * kv.shape[5] * kv.element_size()
        rows = max(1, window // per_row)
        checksums = {}
        digest = hashlib.sha256()
        for kind in range(2):
            for layer in range(kv.shape[1]):
                plane = kv[kind, layer].flatten(0, 1)
                for start in range(0, consumed, rows):
                    block = plane.index_select(
                        0, mapping[start : start + rows].long()
                    ).cpu()
                    digest.update(
                        block.contiguous().view(torch.uint8).numpy().tobytes()
                    )
        checksums["kv"] = digest.hexdigest()
        for name, tensor in adapter.components(slot).items():
            digest = hashlib.sha256()
            parts = (
                [tensor]
                if tensor.is_contiguous()
                else (tensor[layer] for layer in range(tensor.shape[0]))
            )
            for part in parts:
                assert part.is_contiguous()
                flat = part.view(-1)
                count = max(1, window // part.element_size())
                for start in range(0, flat.numel(), count):
                    data = (
                        flat[start : start + count].cpu().contiguous().view(torch.uint8)
                    )
                    digest.update(data.numpy().tobytes())
            checksums[name] = digest.hexdigest()
        return SimpleNamespace(layout=adapter.layout(slot, consumed), hashes=checksums)

    class MeasuredLLM(LLM):
        force_endpoints = ()
        capture_length = None
        captured = None
        admissions = []

        def offline_receive_msg(self, blocking=False):
            messages = super().offline_receive_msg(blocking)
            for message in messages:
                message.checkpoint_endpoints = self.force_endpoints
            return messages

        def offline_send_result(self, replies):
            self.admissions.extend(
                (m.prompt_tokens, m.cached_tokens)
                for m in replies
                if isinstance(m, PromptAdmittedMsg)
            )
            return super().offline_send_result(replies)

        def _forward(self, forward_input):
            result = super()._forward(forward_input)
            if forward_input.batch.is_prefill:
                for req in forward_input.batch.reqs:
                    if req.cached_len == self.capture_length:
                        cm = self.cache_manager
                        if streamed:
                            self.captured = byte_digest(
                                cm.exact_endpoints.adapter,
                                cm.page_table[req.table_idx, : req.cached_len],
                                req.linear_slot_idx,
                                req.cached_len,
                            )
                        else:
                            self.captured = cm.exact_endpoints.adapter.snapshot(
                                identity,
                                req.input_ids[: req.cached_len].tolist(),
                                cm.page_table[req.table_idx, : req.cached_len],
                                req.linear_slot_idx,
                                max_bytes=256 << 20,
                            )
            return result

        def clear_owned_cache(self):
            assert (
                not self.decode_manager.runnable and not self.prefill_manager.runnable
            )
            torch.cuda.synchronize(self.device)
            self.cache_manager.check_integrity()
            cm = self.cache_manager
            states = (
                cm.prefix_cache.mamba_evictable_size + cm.prefix_cache.mamba_protected
            )
            assert (
                cm.linear_state_pool.num_free_slots
                + states
                + len(cm.exact_endpoints.entries)
                == cm.linear_state_pool.num_slots - 1
            )
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            self.cache_manager.check_integrity()

    profile_name = os.getenv("SPARKLAB_EXACT_TEST_MODEL_PROFILE", "default")
    profile = dict(
        attention_backend="triton",
        nvfp4_backend="triton",
        max_running_req=2,
        cuda_graph_max_bs=2,
        page_size=16,
        max_extend_tokens=512,
        num_token_override=cache_tokens,
        max_seq_len_override=context,
        memory_ratio=0.4,
    )
    profile.update(model_options(path, profile_name))
    free_before_boot = require_memory()
    llm = MeasuredLLM(path, **profile)
    report = {
        "artifact": json.loads((Path(path) / "freetoken_weight.json").read_text())[
            "fingerprint"
        ],
        "model_profile": profile_name,
        "free_gpu_gib_before_boot": free_before_boot,
        "minimum_free_gpu_gib": guard,
        "cache_tokens": cache_tokens,
        "context": context,
        "chunk": 512,
        "streamed_byte_hash": streamed,
        "instrumented_not_performance": True,
        "cases": [],
    }
    report_path = os.getenv("SPARKLAB_EXACT_TEST_EVIDENCE")
    try:
        for disk in (False, True):
            for n in lengths:
                prefix = " navigation" * n
                prefix_ids = llm.tokenizer.encode(prefix, add_special_tokens=False)
                assert len(prefix_ids) == n
                for suffix in (" map", " compass"):
                    started = time.monotonic()
                    if report_path:
                        print(
                            f"EXACT case source={'disk' if disk else 'memory'} endpoint={n} suffix={suffix.strip()}",
                            flush=True,
                        )
                    llm.clear_owned_cache()
                    # Seed cold, regardless of earlier persistent records. Keep
                    # copy-out active, but bypass read admission for this control.
                    llm.prefill_manager.prefix_retention = None
                    llm.capture_length = None
                    llm.force_endpoints = ()
                    llm.generate(
                        [prefix], SamplingParams(max_tokens=1, ignore_eos=True)
                    )
                    torch.cuda.synchronize(
                        llm.device
                    )  # observe completed publication before requiring a deepest hit
                    if disk:
                        assert llm.prefix_retention.outstanding == 0
                        llm.clear_owned_cache()
                        llm.prefill_manager.prefix_retention = llm.prefix_retention
                        before_restores = llm.prefix_retention.stats[
                            "restores_completed"
                        ]
                    llm.capture_length = n + 1
                    restored = llm.generate(
                        [prefix + suffix], SamplingParams(max_tokens=4, ignore_eos=True)
                    )
                    restored_state = llm.captured
                    assert llm.admissions[-1][1] == n, (
                        disk,
                        n,
                        suffix,
                        dict(llm.prefix_retention.stats),
                        llm.prefix_retention.store.failure,
                        [
                            (e.consumed, e.key[:8])
                            for e in llm.prefix_retention.store.descriptors()
                        ],
                        dict(llm.prefix_retention.store.store.stats),
                    )
                    if disk:
                        assert (
                            llm.prefix_retention.stats["restores_completed"]
                            > before_restores
                        )
                    llm.clear_owned_cache()
                    llm.prefill_manager.prefix_retention = None
                    # The independent cold run computes the same partitions from zero.
                    # Explicit token endpoints are internal test inputs, not a public API.
                    llm.force_endpoints = tuple(e for e in (n - 1, n) if e > 0)
                    cold = llm.generate(
                        [prefix + suffix], SamplingParams(max_tokens=4, ignore_eos=True)
                    )
                    cold_state = llm.captured
                    assert llm.admissions[-1][1] == 0
                    torch.cuda.synchronize(llm.device)
                    assert restored == cold, (n, suffix, restored, cold)
                    assert restored_state.layout == cold_state.layout
                    if streamed:
                        assert restored_state.hashes == cold_state.hashes, (n, suffix)
                    else:
                        for name, tensor in restored_state._tensors.items():
                            assert torch.equal(
                                tensor.view(torch.uint8),
                                cold_state._tensors[name].view(torch.uint8),
                            ), (n, suffix, name)
                    report["cases"].append(
                        {
                            "source": "disk" if disk else "memory",
                            "endpoint": n,
                            "suffix": suffix.strip(),
                            "prefix_hit": n,
                            "state_equal": True,
                            "output_equal": True,
                            "instrumented_seconds": time.monotonic() - started,
                        }
                    )
                    del restored_state, cold_state
                    llm.captured = None
    finally:
        try:
            if report_path:
                destination = Path(report_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(report, indent=2))
        finally:
            llm.shutdown()
