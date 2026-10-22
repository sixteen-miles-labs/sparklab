"""Complete local Qwen/DFlash model gate, including depth-zero transitions.

Explicit artifact paths are required. State/copy oracles are separate from greedy
cross-shape parity; any numerical/quality difference fails that latter gate.
"""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

pytestmark = pytest.mark.needs_weights


def _free_gpu_bytes_before_engine():
    # Engine.__init__ requires CUDA to be uninitialized in its own process.
    # Query the same visible device in a short-lived child instead of poisoning
    # the offline LLM's startup precondition with cuda.mem_get_info here.
    return int(
        subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.cuda.mem_get_info()[0])"],
            text=True,
            timeout=30,
        ).strip()
    )


def test_real_dflash_prepared_depth_transitions_and_greedy_parity(monkeypatch):
    model = os.getenv("SPARKLAB_DFLASH_TEST_MODEL")
    draft = os.getenv("SPARKLAB_DFLASH_TEST_DRAFT")
    if not model or not draft or not Path(model).is_dir() or not Path(draft).is_dir():
        pytest.skip("set SPARKLAB_DFLASH_TEST_MODEL and SPARKLAB_DFLASH_TEST_DRAFT")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and complete target/draft checkpoints")
    minimum_free = float(os.getenv("SPARKLAB_DFLASH_TEST_MIN_FREE_GIB", "30"))
    if not math.isfinite(minimum_free) or minimum_free < 30:
        raise ValueError(
            "DFlash validation requires a free-memory floor of at least 30 GiB"
        )
    if _free_gpu_bytes_before_engine() / (1 << 30) < minimum_free:
        pytest.skip("insufficient free physical GPU memory for complete target/draft")
    from sparklab.core import SamplingParams
    from sparklab.llm import LLM
    from benchmarks.runtime_quality import GreedyLogitTrace, first_divergence

    monkeypatch.setenv("SPARKLAB_DFLASH_POLICY", "cost")
    monkeypatch.setenv("SPARKLAB_RUNTIME_TIMING", "1")
    monkeypatch.setenv("SPARKLAB_DFLASH2_VERIFY_GRAPH", "1")
    monkeypatch.setenv("SPARKLAB_EXACT_PREFIX", "0")
    monkeypatch.delenv("SPARKLAB_PREFIX_STORE", raising=False)
    monkeypatch.delenv("SPARKLAB_DFLASH_FIXED_DEPTH", raising=False)
    monkeypatch.delenv("SPARKLAB_DFLASH_DEPTHS", raising=False)

    class ObservedLLM(LLM):
        rounds = None
        transitions = None
        transition_index = 0
        draft_context_checks = 0
        state_checks = 0
        logit_trace = None
        observing_logits = False
        logit_phase = None

        def _schedule_decode_batch(self, service_budget=None):
            if self.transitions is not None:
                depth = self.transitions[self.transition_index % len(self.transitions)]
                self.transition_index += 1
                self.engine.adaptive_dflash = replace(
                    self.engine.adaptive_dflash, override=depth
                )
            # The proposal must never change any consumed target/draft KV rows.
            reqs = list(self.decode_manager.running_reqs)
            before = None
            if reqs:
                req = reqs[0]
                mapping = self.cache_manager.page_table[
                    req.table_idx, : req.cached_len
                ].long()
                pools = self.engine.kv_cache._groups
                before = [
                    pool._kv_buffer.flatten(2, 3).index_select(2, mapping).clone()
                    for pool in pools
                ]
            result = super()._schedule_decode_batch(service_budget)
            if before is not None:
                for pool, prior in zip(pools, before):
                    after = pool._kv_buffer.flatten(2, 3).index_select(2, mapping)
                    assert torch.equal(prior.view(torch.uint8), after.view(torch.uint8))
                self.draft_context_checks += 1
            return result

        def _forward(self, forward_input):
            batch = forward_input.batch
            pool = self.engine.linear_state_pool
            old_conv = old_aux = None
            if batch.is_verify:
                req = batch.reqs[0]
                old_conv = pool.conv_states[:, req.linear_slot_idx].clone()
                # Inactive prefix lanes must never contaminate a shorter graph.
                rows = batch.positions.numel()
                pool.verify_recurrent_states[:, 0, rows:].fill_(float("nan"))
                pool.verify_conv_inputs[:, rows:].fill_(float("nan"))
                old_aux = {
                    k: v[req.linear_slot_idx].clone()
                    for k, v in pool._aux_states.items()
                }
            from sparklab.runtime.scheduler.prefill import ChunkedReq

            self.observing_logits = not isinstance(batch.reqs[0], ChunkedReq)
            self.logit_phase = batch.phase
            try:
                result = super()._forward(forward_input)
            finally:
                self.observing_logits = False
            if batch.is_verify:
                req = batch.reqs[0]
                length = req.adaptive_round.accepted + 1
                history = old_conv.shape[-1]
                expected = torch.cat(
                    (old_conv, pool.verify_conv_inputs[:, :length].transpose(1, 2)), -1
                )[..., -history:]
                assert torch.equal(pool.conv_states[:, req.linear_slot_idx], expected)
                assert torch.equal(
                    pool.recurrent_states[:, req.linear_slot_idx],
                    pool.verify_recurrent_states[:, 0, length - 1],
                )
                for name, previous in old_aux.items():
                    raw = pool._verify_aux_inputs[name][:length].T
                    expected = torch.cat((previous, raw), -1)[
                        ..., -previous.shape[-1] :
                    ]
                    assert torch.equal(
                        pool._aux_states[name][req.linear_slot_idx], expected
                    )
                self.state_checks += 1
            return result

        def _process_last_data(self, last_data):
            completed = []
            if last_data is not None:
                for req in last_data[0].batch.reqs:
                    current = req.adaptive_round
                    if current is not None:
                        completed.append(
                            (
                                current,
                                {
                                    "depth": current.depth,
                                    "accepted": current.accepted,
                                    "consumed": req.cached_len,
                                },
                            )
                        )
            result = super()._process_last_data(last_data)
            for current, record in completed:
                timings = current.gpu_timings()
                assert all(
                    name in timings
                    for name in (
                        "proposal_gpu_seconds",
                        "verification_gpu_seconds",
                        "commit_gpu_seconds",
                        "handoff_gpu_seconds",
                        "round_gpu_seconds",
                    )
                )
                record.update(timings)
                self.rounds.append(record)
            return result

        def cold(self):
            assert (
                not self.decode_manager.runnable and not self.prefill_manager.runnable
            )
            torch.cuda.synchronize(self.device)
            self.cache_manager.check_integrity()
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)

    llm = ObservedLLM(
        model,
        attention_backend="triton",
        nvfp4_backend="triton",
        nvfp4_prefill_backend="w4a16",
        speculative_method="dflash2",
        speculative_tokens=12,
        speculative_draft_model=draft,
        max_running_req=1,
        cuda_graph_max_bs=1,
        page_size=16,
        max_extend_tokens=512,
        num_token_override=8192,
        max_seq_len_override=4096,
        memory_ratio=0.4,
    )
    evidence = {
        "prepared_depths": llm.engine.adaptive_dflash.depths,
        "graph_bytes": llm.engine.adaptive_graph_bytes,
        "cases": [],
        "logit_diagnostics": {
            "maximum_tokens_per_case": 64,
            "top_k": 8,
            "synchronous_cpu_observations": True,
            "performance_qualification": False,
        },
    }
    parity_errors = []
    forced_prefixes = []
    import sparklab.runtime.engine.engine as engine_module

    original_sample = llm.engine.sampler.sample
    original_target_verify = engine_module._verify_speculative_tokens

    def observe_logits(logits, chosen):
        if llm.observing_logits and llm.logit_trace is not None:
            llm.logit_trace.observe(logits, chosen, phase=llm.logit_phase)

    def observed_sample(logits, args):
        chosen = original_sample(logits, args)
        observe_logits(logits, chosen)
        return chosen

    def observed_verify(logits, drafts, probabilities, sampling):
        accepted, chosen = original_target_verify(
            logits, drafts, probabilities, sampling
        )
        observe_logits(logits, chosen)
        return accepted, chosen

    monkeypatch.setattr(llm.engine.sampler, "sample", observed_sample)
    monkeypatch.setattr(engine_module, "_verify_speculative_tokens", observed_verify)
    strata = {
        "prose": "Explain how a compass helps a hiker find a campsite, in three short sentences.",
        "code": "Write a Python function that returns the first n Fibonacci numbers. Include only code.",
        "math": "A box contains 12 red and 8 blue balls. What fraction are blue? Explain briefly.",
        "json": "Return only JSON with keys city and country for the capital of France.",
        "multilingual": "用中文简短解释为什么天空是蓝色的。",
    }
    try:
        for stratum, prompt in strata.items():
            control = None
            control_trace = None
            for mode in (0, "repeat0", 3, 7, 11, "transitions", "adaptive"):
                llm.cold()
                llm.rounds = []
                llm.transitions = (0, 3, 7, 11) if mode == "transitions" else None
                llm.transition_index = 0
                llm.logit_trace = GreedyLogitTrace(limit=64)
                llm.engine.adaptive_dflash = replace(
                    llm.engine.adaptive_dflash,
                    override=0
                    if mode == "repeat0"
                    else mode
                    if isinstance(mode, int)
                    else None,
                )
                started = time.monotonic()
                output = llm.generate(
                    [prompt],
                    SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
                )
                record = {
                    "stratum": stratum,
                    "mode": mode,
                    "seconds": time.monotonic() - started,
                    "output": output,
                    "rounds": llm.rounds,
                    "greedy_logits": llm.logit_trace.tokens,
                }
                evidence["cases"].append(record)
                llm.logit_trace.validate_output(output[0]["token_ids"])
                if control is None:
                    control = output
                    control_trace = llm.logit_trace
                if output != control:
                    candidate_ids, reference_ids = (
                        output[0]["token_ids"],
                        control[0]["token_ids"],
                    )
                    first = next(
                        (
                            i
                            for i, pair in enumerate(zip(candidate_ids, reference_ids))
                            if pair[0] != pair[1]
                        ),
                        min(len(candidate_ids), len(reference_ids)),
                    )
                    parity_errors.append(
                        {
                            "stratum": stratum,
                            "mode": mode,
                            "first_token_difference": first,
                            "first_divergence_logits": first_divergence(
                                control_trace, llm.logit_trace
                            ),
                        }
                    )
                if mode == "transitions":
                    assert {r["depth"] for r in llm.rounds} >= {0, 3, 7, 11}
        assert llm.draft_context_checks > 0 and llm.state_checks > 0
        llm.logit_trace = None
        # Exercise every commit endpoint even when the draft naturally rejects
        # earlier. This test-only verifier supplies an exact consumed token chain;
        # it is not a sampling/quality or performance result.
        import sparklab.runtime.engine.engine as engine_module

        original_verify = engine_module._verify_speculative_tokens
        for depth in (3, 7, 11):
            for accepted in range(depth + 1):
                fired = []

                def forced(logits, drafts, probabilities, sampling):
                    if not fired and drafts.numel() == depth:
                        fired.append(True)
                        correction = (
                            torch.argmax(logits[accepted]).reshape(1).to(torch.int32)
                        )
                        return accepted, torch.cat((drafts[:accepted], correction))
                    return original_verify(logits, drafts, probabilities, sampling)

                llm.cold()
                llm.rounds = []
                llm.transitions = None
                llm.engine.adaptive_dflash = replace(
                    llm.engine.adaptive_dflash, override=depth
                )
                with monkeypatch.context() as verification:
                    verification.setattr(
                        engine_module, "_verify_speculative_tokens", forced
                    )
                    llm.generate(
                        ["Continue the list: north, south, east, west."],
                        SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
                    )
                assert fired, (depth, accepted)
                forced_prefixes.append({"depth": depth, "accepted": accepted})
        assert len(forced_prefixes) == 24
        assert not parity_errors, parity_errors
    finally:
        evidence["forced_prefixes"] = forced_prefixes
        evidence["quality_differences"] = parity_errors
        evidence["draft_context_checks"] = llm.draft_context_checks
        evidence["state_checks"] = llm.state_checks
        destination = os.getenv("SPARKLAB_DFLASH_TEST_OUTPUT")
        try:
            if destination:
                result_path = Path(destination)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(evidence, ensure_ascii=False, indent=2)
                )
        finally:
            llm.shutdown()
