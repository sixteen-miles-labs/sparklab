from __future__ import annotations

import json
import os
import time
from collections import deque

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from sparklab.attention.linear import build_fla_metadata
from sparklab.core import Batch, Req
from sparklab.env import ENV
from sparklab.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from sparklab.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager
from .policy import SchedulingPolicy
from .telemetry import RuntimeTelemetry, Shape, host_scalars, workspace_snapshot, allocator_snapshot, ExpertCounterSampler, EXPERT_DECODE_FIELDS, EXPERT_DISK_FIELDS

if TYPE_CHECKING:
    from sparklab.runtime.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from sparklab.runtime.engine import Engine

        self.scheduling_policy = SchedulingPolicy(
            os.getenv("SPARKLAB_SCHEDULER_POLICY", "baseline"),
            delivery_budget=float(os.getenv("SPARKLAB_SCHEDULER_DELIVERY_MS", "100"))/1000,
            prefill_service_bound=float(os.getenv("SPARKLAB_SCHEDULER_PREFILL_SERVICE_MS", "1000"))/1000,
        )
        self.runtime_metrics = RuntimeTelemetry(
            trace_capacity=int(os.getenv("SPARKLAB_RUNTIME_TRACE_CAPACITY", "0"))
        )
        self._runtime_timing = (
            os.getenv("SPARKLAB_RUNTIME_TIMING", "0") == "1"
            or self.scheduling_policy.mode == "latency"
            or bool(os.getenv("SPARKLAB_PREFIX_STORE", ""))
        )
        self._runtime_events = deque()
        self.engine = Engine(config)
        expert_cache = self.engine.moe_offload_cache
        self._runtime_expert_sampler = (
            ExpertCounterSampler(expert_cache)
            if self._runtime_timing and expert_cache is not None and expert_cache.collect_stats
            else None
        )

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        if os.getenv("SPARKLAB_EXACT_PREFIX", "0") == "1":
            from sparklab.runtime.checkpoint.exact import ExactEndpointCache
            if (config.speculative_tokens or config.tp_info.size != 1
                    or not getattr(self.engine.model, "supports_exact_hybrid_checkpoint", False)):
                raise ValueError("exact prefix checkpoints require a qualified target-only TP=1 state contract")
            self.cache_manager.exact_endpoints = ExactEndpointCache(
                self.cache_manager, self.engine.kv_cache,
                max_records=int(os.getenv("SPARKLAB_EXACT_PREFIX_RECORDS", "8")),
            )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.prefix_retention = None
        if os.getenv("SPARKLAB_PREFIX_STORE", ""):
            from sparklab.runtime.checkpoint.config import DiskSettings
            from sparklab.runtime.checkpoint.identity import runtime_identity
            from sparklab.runtime.checkpoint.service import PrefixRetention

            settings = DiskSettings.from_env()
            exact = self.cache_manager.exact_endpoints
            compatibility = runtime_identity(
                config.model_path, exact.adapter, config, scope=settings.scope,
            )
            self.prefix_retention = PrefixRetention(
                exact, compatibility, settings, prefill_cost=self._prefix_prefill_cost,
                max_waiters=min(128, max(2, config.max_running_req * 2)),
            )
            self.prefill_manager.prefix_retention = self.prefix_retention

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from sparklab.serving.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()
        self._poll_runtime_timings()
        if os.getenv("SPARKLAB_RUNTIME_LOG", "0") == "1":
            logger.info_rank0("Runtime measurements: " + json.dumps(self.runtime_metrics.snapshot()))
            retention = getattr(self, "prefix_retention", None)
            if retention is not None:
                logger.info_rank0("Prefix measurements: " + json.dumps(retention.snapshot()))

    def _prefix_prefill_cost(self, tokens):
        """Observed same-path costs only; unknown shapes never cost zero."""
        total = 0.
        while tokens:
            count = min(tokens, self.prefill_budget)
            seconds = self.runtime_metrics.estimate(Shape.of("prefill", 1, count), None)
            if seconds is None:
                return None
            total += seconds
            tokens -= count
        return total

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        retention = getattr(self, "prefix_retention", None)
        assert retention is None or not retention.outstanding, "rebuild requires drained prefix I/O"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        retention = getattr(self, "prefix_retention", None)
        if retention is not None:
            retention.poll()
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
            or (retention is not None and retention.outstanding)
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()
        if retention is not None:
            retention.poll()
            if forward_input is None and retention.outstanding:
                # Bounded polling backoff, no filesystem or transfer wait on the
                # model worker. Incoming messages are checked on the next pass.
                time.sleep(.001)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        mtp_overlap = (
            getattr(self.config, "speculative_method", None) == "mtp"
            and getattr(self.engine.model, "supports_mtp_prefix_recovery", False)
            and os.getenv("SPARKLAB_MTP_OVERLAP", "0") == "1"
        )
        if (ENV.DISABLE_OVERLAP_SCHEDULING
                or self.cache_manager.exact_endpoints is not None
                or (self.config.speculative_tokens and not mtp_overlap)):
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self, *, failed: bool = False) -> None:
        try:
            retention = getattr(self, "prefix_retention", None)
            if retention is not None and not retention.close(timeout=5.):
                raise RuntimeError("prefix transfers still own runtime pools at shutdown")
            if failed:
                # Preserve pools through the transfer drain. A failed rank cannot
                # require healthy peers to join a shutdown barrier, or assume a
                # usable CUDA context for graph/engine teardown. Process exit
                # owns final device cleanup after this local I/O drain.
                return
            torch.cuda.synchronize(self.device)
            self._poll_runtime_timings()
            self.sync_all_ranks()
            self.engine.shutdown()
        finally:
            SchedulerIOMixin.close_io(self)

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, output = last_data[0].batch, last_data[1]
        if hasattr(output, "next_tokens_cpu"):
            next_tokens_cpu = output.next_tokens_cpu
            copy_done = output.copy_done_event
            token_counts = output.token_counts
        else:
            # Compatibility with simple ForwardOutput-shaped tuples used by
            # direct scheduler callers and older in-process integrations.
            _, next_tokens_cpu, copy_done = output
            token_counts = tuple(1 for _ in batch.reqs)
        copy_done.synchronize()
        generated_at = time.monotonic()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            token_offset = 0
            for i, req in enumerate(batch.reqs):
                count = token_counts[i]
                req_tokens = next_tokens_cpu[token_offset : token_offset + count]
                token_offset += count
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    elif (self.cache_manager.exact_endpoints is not None
                          and req.cached_len in req.checkpoint_endpoints):
                        self.cache_manager.release_exact_source(req)
                        self.cache_manager.exact_endpoints.publish(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                finished = False
                for token_index, next_tensor in enumerate(req_tokens):
                    req.append_host(next_tensor.unsqueeze(0))
                    next_token = int(next_tensor.item())
                    # Multi-token verification can finish at any accepted token.
                    # Host length is the exact per-token boundary; device_len is
                    # already at the post-verification boundary.
                    hit_length = req.input_ids.numel() >= req.max_device_len
                    hit_eos = (
                        not req.sampling_params.ignore_eos
                        and next_token in self.eos_token_ids
                    )
                    matched_stop = (
                        self._match_stop_str(req)
                        if not hit_eos and req.sampling_params.stop_strs
                        else None
                    )
                    finished = hit_length or hit_eos or matched_stop is not None
                    finish_reason = (
                        ("stop" if (hit_eos or matched_stop is not None) else "length")
                        if finished else None
                    )
                    if (
                        next_token == self.toolcall_anchor_id
                        and req.toolcall_anchor_len is None
                        and not finished
                    ):
                        req.toolcall_anchor_len = req.input_ids.numel()
                    request_timing = None
                    if req.timing is not None:
                        request_timing = dict(req.timing, first_generated=generated_at)
                        req.timing = None  # one first-output observation, including bursts
                    reply.append(DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=matched_stop,
                        stop_strs=req.sampling_params.stop_strs or None,
                        timing=request_timing,
                    ))
                    if finished:
                        # A following speculative block may already have run
                        # under overlap. Never donate recurrent state past the
                        # client-visible terminal token, even when EOS was the
                        # last token of the block being drained.
                        if req.cached_len > req.input_ids.numel():
                            req.state_overadvanced = True
                        req.cached_len = min(req.cached_len, req.input_ids.numel())
                        break

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        metrics = getattr(self, "runtime_metrics", None)
        if metrics is not None:
            now = time.monotonic()
            delivered = {m.uid for m in reply}
            for req in batch.reqs:
                if req.uid in delivered:
                    if req.last_output_ready_at is not None:
                        metrics.observe("output_ready_gap", max(0., now-req.last_output_ready_at))
                    req.last_output_ready_at = now
            metrics.counters["committed_output_tokens"] += len(reply)
            Scheduler._poll_runtime_timings(self)
        self.finished_reqs = new_finished_reqs
        speculative_tokens = int(getattr(self.config, "speculative_tokens", 0) or 0)
        if new_finished_reqs and speculative_tokens:
            stats = getattr(batch, "speculative_stats_snapshot", self.engine.mtp_stats)
            drafted = stats["drafted"]
            rate = stats["accepted"] / drafted if drafted else 0.0
            logger.info_rank0(
                "MTP summary: steps=%d, accepted=%d/%d (%.1f%%), "
                "outputs=%d, target_forwards=%d, outputs/target=%.2f, "
                "replays=%d/%d, fast_commits=%d",
                speculative_tokens,
                stats["accepted"], drafted, 100.0 * rate,
                stats["outputs"], stats["target_forwards"],
                stats["outputs"] / max(stats["target_forwards"], 1),
                stats.get("replay_calls", 0), stats.get("replay_tokens", 0),
                stats.get("fast_carry_commits", 0),
            )
            speculative_stats = getattr(self.engine.model, "speculative_stats", None)
            if speculative_stats is not None and (details := speculative_stats()):
                logger.info_rank0(
                    "DSpark stats: " + json.dumps(details, separators=(",", ":"))
                )
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            moe_stats = None
            # Lightweight scheduler harnesses and non-engine embedders may not
            # provide an engine object. MoE stats are optional response metadata;
            # absence must not break terminal-token delivery.
            engine = getattr(self, "engine", None)
            moe_cache = getattr(engine, "moe_offload_cache", None)
            if moe_cache is not None and moe_cache.collect_stats and any(m.finished for m in reply):
                moe_stats = moe_cache.decode_miss_stats()
                moe_stats["per_layer"] = moe_cache.decode_miss_stats_per_layer()["per_layer"]
                routing = moe_cache.decode_routing_stats()
                if routing:
                    moe_stats["routing"] = routing
                if moe_cache.disk_source is not None:
                    moe_stats["disk"] = moe_cache.disk_source.stats()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
                if m.finished:
                    m.moe_stats = moe_stats
            snapshot = Scheduler._runtime_snapshot(self, time.monotonic(), {
                "kv_used_pages": used, "kv_total_pages": total,
                "mamba_used_slots": mamba_used, "mamba_total_slots": mamba_total,
                "swa_used_tokens": swa_used, "swa_total_tokens": swa_total,
                "gpu_reserved_bytes": mem,
                "running_requests": len(self.decode_manager.running_reqs),
                "pending_requests": len(self.prefill_manager.pending_list),
            }, moe_stats)
            if snapshot is not None:
                reply[0].runtime_snapshot = snapshot
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        self.send_result(reply)
        # Feedback belongs to the completed forward, after its output fence and
        # host handoff. It must run even when runtime telemetry is disabled.
        for req in batch.reqs:
            round_data = getattr(req, "adaptive_round", None)
            if round_data is not None:
                req.adaptive_round = None
                if not req.aborted:
                    seconds = round_data.finish()
                    gpu_timings = round_data.gpu_timings()
                    if metrics is not None:
                        if gpu_timings is not None:
                            metrics.observe("speculative_round_gpu", gpu_timings["round_gpu_seconds"])
                            for phase in ("proposal", "verification", "commit", "handoff"):
                                value = gpu_timings.get(phase + "_gpu_seconds")
                                if value is not None:
                                    metrics.observe("speculative_" + phase + "_gpu", value)
                        metrics.speculative_round(round_data.depth, round_data.accepted)
                        if metrics.trace.maxlen:
                            metrics.trace.append({"time": time.monotonic(), "phase": "adaptive",
                                                  "depth": round_data.depth,
                                                  "accepted": round_data.accepted,
                                                  "round_seconds": seconds,
                                                  **(gpu_timings or {})})

    def _runtime_snapshot(self, now, resources, expert_terminal=None):
        """At most one bounded piggyback per second; no file I/O or device wait.

        Idle snapshots deliberately remain last-observed, with an explicit age.
        Detailed rings stay in their existing diagnostic owners, outside IPC.
        """
        if not getattr(self, "_runtime_timing", False):
            return None
        cache = getattr(getattr(self, "engine", None), "moe_offload_cache", None)
        sampler = getattr(self, "_runtime_expert_sampler", None)
        if sampler is not None:
            observed = sampler.poll(cache, self.engine.stream, now)
            previous_expert = getattr(self, "_runtime_expert_decode", None)
            if observed is not None and (
                previous_expert is None or observed["observed_at"] > previous_expert["observed_at"]
            ):
                self._runtime_expert_decode = observed
        if expert_terminal is not None:
            self._runtime_expert_decode = {
                "observed_at": now,
                "counters": host_scalars(expert_terminal, EXPERT_DECODE_FIELDS),
            }
        previous = getattr(self, "_runtime_last_export", None)
        if previous is not None and now - previous < 1.:
            return None
        self._runtime_last_export = now
        # Query host allocator metadata once per exported observation, not once
        # per token. A single read keeps allocated/reserved from the same sample.
        resources = {**resources, **allocator_snapshot(getattr(self, "device", None))}
        retention = getattr(self, "prefix_retention", None)
        exact = self.cache_manager.exact_endpoints
        source = getattr(cache, "disk_source", None)
        return {
            "schema_version": 1,
            "observed_at": now,
            "scheduler": self.runtime_metrics.snapshot(include_trace=False),
            "resources": resources,
            "retained_workspaces": workspace_snapshot(getattr(self, "engine", None), sampler),
            "exact_prefix": ({
                "records": len(exact.entries), "metadata_bytes": exact.metadata_bytes,
                "published": exact.published, "skipped": exact.skipped, "hits": exact.hits,
                "readers": sum(entry.readers for entry in exact.entries.values()),
                "private_restore_pages": exact.restoring_pages,
                "private_restore_slots": exact.restoring_states,
            } if exact is not None else None),
            "prefix_store": retention.snapshot() if retention is not None else None,
            "expert_disk": ({"observed_at": now,
                             "counters": host_scalars(source.stats(), EXPERT_DISK_FIELDS)}
                            if source is not None else None),
            "expert_decode": getattr(self, "_runtime_expert_decode", None),
        }

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix. Byte-fallback tokens can split one Unicode character
        across multiple tokens, so bound lookback by UTF-8 bytes plus one context
        token, rather than by character count."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_bytes = max(len(s.encode("utf-8", errors="surrogatepass")) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_bytes + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            if msg.timing is not None:
                msg.timing["scheduler_received"] = time.monotonic()
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            if msg.mm_inputs is not None:
                try:
                    tower = getattr(self.engine.model, "_vision", None)
                    if tower is None:
                        raise ValueError("Image inputs require a loaded vision tower")
                    if input_len > self.prefill_budget:
                        raise ValueError("Image prompt exceeds single-prefill budget; shorten it or increase --max-extend-tokens")
                    payload = msg.mm_inputs
                    pixels = payload['pixels'].reshape(payload['pixel_shape']).to(self.device, dtype=torch.bfloat16)
                    grids = payload['grids'].reshape(-1, 3).to(self.device)
                    msg.mm_embeds = tower(pixels, grid_thw=grids, return_dict=True).pooler_output
                    msg.mm_positions = payload['positions'].reshape(3, -1).to(self.device)
                    msg.mm_delta = payload['delta']
                    msg.mm_inputs = None
                except Exception as exc:
                    self.send_result([ErrorReplyMsg(uid=msg.uid, error=f"could not encode images: {exc}")])
                    return
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif (self.prefill_manager.runnable or self.decode_manager.runnable
                  or (getattr(self, "prefix_retention", None) is not None
                      and self.prefix_retention.outstanding)):
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Cancellation can discard a verified block before its tokens reach the
        # host history. Never donate the later recurrent state under that shorter
        # prefix key: radix insertion would own fewer pages than cached_len and
        # strand the difference. Keep allocated_len intact so cleanup returns all
        # speculative lookahead pages, including any page-alignment padding.
        if req.cached_len > req.input_ids.numel():
            req.state_overadvanced = True
            req.cached_len = req.input_ids.numel()
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from sparklab.runtime.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from sparklab.runtime.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        # Native MTP writes its own KV while proposing the next draft block,
        # after this target forward has returned. Reserve those physical rows
        # now; verification will fill the corresponding token_pool entries on
        # the next scheduler iteration. This is required for page_size=1 and
        # also removes page-boundary-dependent draft widths.
        lookahead = (
            int(getattr(self.config, "speculative_tokens", 0) or 0)
            if getattr(self.config, "speculative_method", "none") in {"mtp", "dflash2"}
            else 0
        )
        if lookahead:
            actual_lens = [req.device_len for req in batch.reqs]
            for req in batch.reqs:
                req.device_len = min(req.max_device_len, req.device_len + lookahead)
            try:
                self.cache_manager.allocate_paged(batch.reqs)
            finally:
                for req, actual in zip(batch.reqs, actual_lens):
                    req.device_len = actual
        else:
            self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        if any(req.mm_positions is not None for req in batch.reqs):
            parts = []
            for req in batch.reqs:
                start, end = req.cached_len, req.device_len
                if req.mm_positions is None:
                    parts.append(torch.arange(start, end, device=self.device).expand(3, -1))
                elif batch.is_prefill:
                    parts.append(req.mm_positions[:, start:end])
                else:
                    parts.append((torch.arange(start, end, device=self.device) + req.mm_delta).expand(3, -1))
            batch.mm_positions = torch.cat(parts, dim=1)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate per-request vision soft tokens (in request order) for a prefill
        batch so the model can scatter them at image-token positions. ``req.mm_embeds``
        is kept (not cleared) so the cache manager can recognize multimodal requests and
        keep them out of the shared prefix cache (image placeholders share a token id but
        carry per-image content)."""
        parts = [req.mm_embeds for req in batch.reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_decode_batch(self, service_budget=None):
        engine = getattr(self, "engine", None)
        if getattr(engine, "adaptive_dflash", None) is not None:
            running = self.decode_manager.running_reqs
            if len(running) > 1:
                raise AssertionError("adaptive DFlash is qualified only for batch one")
            if running:
                req = next(iter(running))
                anchor = self.token_pool[req.table_idx, req.cached_len].reshape(1)
                engine.prepare_adaptive_round(req, anchor, service_budget=service_budget,
                                              phase_timing=getattr(self, "_runtime_timing", False))
        return self.decode_manager.schedule_next_batch()

    def _schedule_next_batch(self) -> ForwardInput | None:
        policy = getattr(self, "scheduling_policy", None)
        decision = None
        service_budget = None
        queued_before, selection_at = 0, None
        observed_output_age, observed_queue_age = None, None
        prefill_unavailable = False
        if policy is not None:
            self._poll_runtime_timings()
            now = time.monotonic()
            pending = self.prefill_manager.pending_list
            running = self.decode_manager.running_reqs
            queued_before, selection_at = len(pending), now
            observed_output_age = max((max(0., now-(r.last_output_ready_at
                if r.last_output_ready_at is not None else r.first_dispatch_at))
                for r in running if r.last_output_ready_at is not None or r.first_dispatch_at is not None), default=None)
            observed_queue_age = max((max(0., now-r.queued_at) for r in pending), default=None)
            output_age = observed_output_age if observed_output_age is not None else 0.
            alignment = max(self.config.page_size, 64 if self.cache_manager.is_hybrid else 1)
            shape = Shape.of("prefill", min(len(pending), self.config.max_running_req), self.prefill_budget)
            decode_shape = Shape.of("decode", len(running), len(running))
            decision, service_budget = policy.choose_coordinated(
                tp_info=getattr(self.config, "tp_info", None),
                cpu_group=getattr(self, "tp_cpu_group", None),
                now=now, pending=bool(pending), decoders=bool(running),
                oldest_queue_age=observed_queue_age if observed_queue_age is not None else 0.,
                output_age=output_age, prefill_budget=self.prefill_budget, alignment=alignment,
                prefill_seconds=self.runtime_metrics.estimate(shape, .25),
                decode_seconds=self.runtime_metrics.estimate(decode_shape, .05),
                atomic_prompt=any(r.mm_embeds is not None for r in pending),
            )
        if decision is not None and decision.phase == "decode":
            batch = self._schedule_decode_batch(service_budget)
        else:
            budget = decision.token_budget if decision is not None else self.prefill_budget
            batch = self.prefill_manager.schedule_next_batch(budget)
            if batch is None:
                prefill_unavailable = queued_before > 0
                batch = self._schedule_decode_batch(service_budget)
        if batch is None:
            return None
        if getattr(batch, "phase", None) == "decode" and batch.size == 1:
            req = batch.reqs[0]
            drafts = req.speculative_drafts
            req.speculative_drafts = None
            if drafts is not None:
                count = min(int(drafts.numel()), max(0, req.remain_len - 1))
                if count:
                    start = req.device_len
                    self.token_pool[req.table_idx, start : start + count].copy_(
                        drafts[:count]
                    )
                    batch.verify_cached_lens = (req.cached_len,)
                    req.device_len += count
                    batch.phase = "verify"
                    batch.return_all_logits = True
                    batch.disable_state_tracking = True
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        if policy is not None:
            now = time.monotonic()
            policy.dispatched(batch.phase, now)
            for req in batch.reqs:
                if req.first_dispatch_at is None:
                    req.first_dispatch_at = now
                    if req.queued_at is not None:
                        self.runtime_metrics.observe("queue_wait", max(0.,now-req.queued_at))
            self.runtime_metrics.dispatch(
                phase=batch.phase, requests=batch.padded_size,
                tokens=sum(r.extend_len for r in batch.reqs),
                cached_tokens=batch.log_cached_tokens, queued=len(self.prefill_manager.pending_list),
                reason="prefill_unavailable" if prefill_unavailable else decision.reason,
                policy_reason=decision.reason, queued_before=queued_before,
                selection_at=selection_at, output_age=observed_output_age, queue_age=observed_queue_age,
                token_budget=decision.token_budget, active_requests=batch.size,
            )
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        timings = {}
        for req in getattr(batch, "reqs", ()):
            if req.timing is not None and "admitted" not in req.timing:
                req.timing["admitted"] = time.monotonic()
                timings[req.uid] = dict(req.timing)
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens,
                                  timing=timings.get(uid))
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _poll_runtime_timings(self) -> None:
        events = getattr(self, "_runtime_events", ())
        while events and events[0][2].query():
            shape, start, end = events.popleft()
            self.runtime_metrics.gpu_completed(shape, start.elapsed_time(end)/1000.)

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        started = time.monotonic()
        batch, sample_args, input_mapping, output_mapping = forward_input
        for req in batch.reqs:
            if req.timing is not None:
                req.timing.setdefault("first_dispatch", started)
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        timing = None
        if getattr(self, "_runtime_timing", False):
            if len(self._runtime_events) < 64:
                timing = (Shape.of(batch.phase, batch.padded_size, sum(r.extend_len for r in batch.reqs)),
                          torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                timing[1].record(self.engine.stream)
            else:
                self.runtime_metrics.counters["timing_skipped"] += 1
        forward_output = self.engine.forward_batch(batch, sample_args)
        if forward_output.managed_writes:
            table_idx, position, token = batch.speculative_write
            self.token_pool[table_idx, position] = token
        else:
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        adaptive_events = [req.adaptive_round.gpu_end for req in batch.reqs
                           if getattr(req, "adaptive_round", None) is not None
                           and req.adaptive_round.gpu_end is not None]
        if adaptive_events:
            for end in adaptive_events:
                end.record(self.engine.stream)
            # Reuse the existing drain fence after the scheduler-owned token
            # write. No extra wait; the round covers every committed GPU write.
            forward_output.copy_done_event.record(self.engine.stream)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        metrics = getattr(self, "runtime_metrics", None)
        if metrics is not None:
            metrics.observe("host_submit", time.monotonic()-started)
            if timing is not None:
                timing[2].record(self.engine.stream)
                self._runtime_events.append(timing)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
