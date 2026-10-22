#!/usr/bin/env python3
"""Offline fixed-workspace inventory on a complete native target/draft artifact.

No optimizer is enabled by this script. Cache capacities stay explicit. Run on a
permitted GB10; output is an allocation inventory, not a throughput measurement.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def tensor_bytes(value):
    return value.numel() * value.element_size() if value is not None else 0


def inventory(engine):
    import torch

    torch.cuda.synchronize(engine.device)
    torch.cuda.empty_cache()
    snapshots = torch.cuda.memory_snapshot()
    segments = []
    for segment in snapshots:
        segments.append(
            {key: value for key, value in segment.items() if key != "blocks"}
        )
    pool = engine.linear_state_pool
    transaction = {}
    if pool is not None:
        for name in (
            "verify_recurrent_states",
            "verify_conv_inputs",
            "verify_state_indices",
        ):
            transaction[name] = tensor_bytes(getattr(pool, name, None))
        transaction["verify_auxiliary_inputs"] = sum(
            tensor_bytes(t) for t in pool._verify_aux_inputs.values()
        )
    graph_rows = {}
    verification = getattr(engine.mtp_graph_runner, "runners", {})
    for rows, runner in verification.items():
        graph_rows[str(rows)] = {
            "pool": list(runner.graph.pool()),
            "input_logits_bytes": sum(
                tensor_bytes(v)
                for v in vars(runner.buffer).values()
                if isinstance(v, torch.Tensor)
            ),
        }
    return {
        "allocated_bytes": torch.cuda.memory_allocated(engine.device),
        "reserved_bytes": torch.cuda.memory_reserved(engine.device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(engine.device),
        "all_graph_reserved_delta": getattr(engine, "adaptive_graph_bytes", None),
        "verification_transactions": transaction,
        "verification_graphs": graph_rows,
        "decode_graph_pools": {
            str(bs): list(graph.pool())
            for bs, graph in engine.graph_runner.graph_map.items()
        },
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--operator-profile", action="store_true")
    parser.add_argument("--adaptive-output-tokens", type=int, default=0)
    options = parser.parse_args()
    os.environ["SPARKLAB_DFLASH_POLICY"] = "cost"
    os.environ["SPARKLAB_DFLASH2_VERIFY_GRAPH"] = "1"
    os.environ["SPARKLAB_EXACT_PREFIX"] = "0"
    os.environ.pop("SPARKLAB_PREFIX_STORE", None)
    from sparklab.core import SamplingParams
    from sparklab.llm import LLM
    import torch

    started = time.monotonic()
    llm = LLM(
        options.model,
        attention_backend="triton",
        nvfp4_backend="triton",
        nvfp4_prefill_backend="w4a16",
        speculative_method="dflash2",
        speculative_tokens=12,
        speculative_draft_model=options.draft,
        max_running_req=1,
        cuda_graph_max_bs=1,
        page_size=16,
        max_extend_tokens=512,
        num_token_override=options.cache_tokens,
        max_seq_len_override=options.context,
        memory_ratio=0.4,
    )

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    result = {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "artifact": json.loads(
            (Path(options.model) / "freetoken_weight.json").read_text()
        )["fingerprint"],
        "configuration": {
            "cache_tokens": options.cache_tokens,
            "context": options.context,
            "draft_block": 12,
            "depths": llm.engine.adaptive_dflash.depths,
            "precision": "NVFP4 weights, W4A16 prefill, BF16 compute/KV, FP32 recurrent",
            "attention": "triton",
            "page_size": 16,
            "concurrency": 1,
        },
        "startup_seconds": time.monotonic() - started,
    }
    try:
        result["after_startup"] = inventory(llm.engine)
        prompt = "List three practical uses for a map and a compass."
        for depth in llm.engine.adaptive_dflash.depths:
            from dataclasses import replace

            llm.engine.adaptive_dflash = replace(
                llm.engine.adaptive_dflash, override=depth
            )
            llm.generate(
                [prompt],
                SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
            )
        result["after_all_depths"] = inventory(llm.engine)
        if options.adaptive_output_tokens:
            if not 1 <= options.adaptive_output_tokens <= 512:
                raise ValueError("adaptive smoke output must be 1..512")
            llm.engine.adaptive_dflash = replace(
                llm.engine.adaptive_dflash, override=None
            )
            llm.generate(
                ["用中文简短解释为什么天空是蓝色的。"],
                SamplingParams(
                    temperature=0.0,
                    max_tokens=options.adaptive_output_tokens,
                    ignore_eos=True,
                ),
            )
            request = next(iter(llm.finished_reqs))
            result["adaptive_policy_samples"] = {
                str(depth): len(values)
                for depth, values in request.adaptive_policy.costs.items()
            }
            result["adaptive_policy_rounds"] = request.adaptive_policy.rounds
            result["adaptive_smoke_outputs"] = options.adaptive_output_tokens

        long_prompt = " navigation" * 512
        result["prefill_tokens"] = len(llm.tokenizer.encode(long_prompt))
        if options.operator_profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
            ) as measured:
                llm.generate(
                    [long_prompt],
                    SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
                )
            operators = []
            for event in measured.key_averages(group_by_input_shape=True):
                operators.append(
                    {
                        "operator": event.key,
                        "input_shapes": event.input_shapes,
                        "calls": event.count,
                        "self_device_allocated_bytes": event.self_device_memory_usage,
                        "self_cpu_allocated_bytes": event.self_cpu_memory_usage,
                    }
                )
            result["operators"] = sorted(
                operators, key=lambda e: e["self_device_allocated_bytes"], reverse=True
            )[:64]
        else:
            llm.generate(
                [long_prompt],
                SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            )
        result["after_prefill_512"] = inventory(llm.engine)
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(result, indent=2))
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
