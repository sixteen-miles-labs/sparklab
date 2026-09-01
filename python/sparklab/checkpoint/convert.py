"""Convert an HF safetensors checkpoint into a SparkLab Weight (FTW) checkpoint.

Model-agnostic: it drives the *existing* per-model loaders once and stores their output, so
no per-model conversion code is needed.

* dense weights = exactly what ``load_weight(include_moe_experts=...)`` yields (post
  fusion/TP-shard) -> ``kind="weight"``; at load they feed ``model.load_state_dict``.
* offload experts = exactly what ``load_expert_banks()`` produces (post
  backend-repack pinned banks + alpha scale vectors) -> ``kind="experts_bank"`` (alphas are
  told apart at load by their reserved names, so they need no separate kind).

The output directory is a self-contained checkpoint (config + tokenizer copied), so you can
point ``--model`` straight at it; the load path auto-detects the FTW and reads it (FTW).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import threading

import torch

from .ftw import DEFAULT_SHARD_LIMIT, FTWWriter, layer_bank_entry_name

# Machine-readable convert progress for a supervising process (e.g. a GUI frontend parses
# these `SPARKLAB_CONVERT <phase> <done> <total>` stdout lines to drive its convert bar). Gated by
# SPARKLAB_CONVERT_PROGRESS=1 so plain CLI use isn't spammed; the human tqdm bars stay on
# stderr. Phases: `dense` (indeterminate, done=total=0), `experts` (byte totals), `finalize`.
_EMIT_PROGRESS = os.environ.get("SPARKLAB_CONVERT_PROGRESS") == "1"
_GLM_KDA_QUANT_ENV = "_SPARKLAB_CONVERT_GLM5_KDA_QUANT"


def _progress(phase: str, done: int = 0, total: int = 0) -> None:
    if _EMIT_PROGRESS:
        print(f"SPARKLAB_CONVERT {phase} {done} {total}", flush=True)


def _source_fingerprint(
    model_path: str, model_config, *, device, expert_quantization: str | None = None
) -> str:
    """Identity of (checkpoint + quant + GPU capability), stored in the FTW index so
    it's clear what an FTW was built from. Cheap (stat only)."""
    h = hashlib.sha256()
    h.update(f"quant={getattr(model_config, 'expert_quant', None)}|".encode())
    # Model-specific resident-weight transformations must participate in artifact
    # identity too.  GLM-5.3's optimized FTW keeps the source NVFP4 experts but stores
    # its bandwidth-dominant KDA projections as FP8; without this discriminator it
    # would collide with the all-BF16-resident artifact built from the same source.
    model_args = getattr(model_config, "glm5_next_args", None)
    kda_quant = getattr(model_args, "kda_quant", "none")
    if kda_quant != "none":
        h.update(f"kda_quant={kda_quant}|".encode())
    if expert_quantization is not None:
        h.update(f"target_expert_quant={expert_quantization}|".encode())
    h.update(f"arch={getattr(model_config, 'architectures', None)}|".encode())
    try:  # nvfp4 marlin/b12x layout depends on compute capability
        h.update(f"cc={torch.cuda.get_device_capability(device)}|".encode())
    except Exception:
        pass
    files = sorted(
        glob.glob(os.path.join(model_path, "*.safetensors")) + glob.glob(os.path.join(model_path, "*.gguf"))
    )
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}|".encode())
    return h.hexdigest()[:16]

# Checkpoint metadata to carry over so the FTW dir is a usable checkpoint on its own.
# (Weight shards + the safetensors index are intentionally NOT copied.)
# Everything that is NOT a weight shard is metadata we carry over verbatim. A whitelist
# misses model-specific layouts (e.g. DSV4's inference/config.json + encoding/ live in
# subdirs), so we copy every non-weight file preserving its relative path instead.
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".ftw")  # .ftw: a nested FTW, not source
_SKIP_NAMES = ("model.safetensors.index.json",)  # indexes shards the FTW replaces
# .sparklab_expert_cache: the legacy per-bank cache (can be tens of GB of stale .bin)
_SKIP_DIRS = (".git", ".cache", ".sparklab_expert_cache")


def _copy_metadata(model_path: str, out_dir: str) -> list[str]:
    """Copy all non-weight files (config, tokenizer, remote-code, nested model configs)
    preserving directory structure, so the FTW dir is a self-contained checkpoint."""
    if os.path.isfile(model_path):
        # A single-file source has no sibling metadata to walk: a .gguf carries its config
        # AND tokenizer in its own KV section. Emit a metadata-only copy (header + KV, no
        # weight data) the FTW dir resolves those from. We deliberately do NOT sweep the
        # file's parent dir -- an HF gguf snapshot dir can hold unrelated blobs.
        from sparklab.models.gguf.reader import (
            FTW_METADATA_GGUF,
            is_gguf_path,
            write_metadata_gguf,
        )

        if is_gguf_path(model_path):
            os.makedirs(out_dir, exist_ok=True)
            write_metadata_gguf(model_path, os.path.join(out_dir, FTW_METADATA_GGUF))
            return [FTW_METADATA_GGUF]
        return []

    out_abs = os.path.abspath(out_dir)
    copied = []
    for root, dirs, files in os.walk(model_path):
        dirs[:] = [d for d in dirs
                   if d not in _SKIP_DIRS and os.path.abspath(os.path.join(root, d)) != out_abs]
        for name in files:
            if name.endswith(_WEIGHT_SUFFIXES) or name in _SKIP_NAMES:
                continue
            src = os.path.join(root, name)
            rel = os.path.relpath(src, model_path)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    return copied


class _ConvertSink:
    """Layer-completion sink for ``load_expert_banks(layer_sink=...)``: writes each
    completed layer's banks as their own FTW entries immediately (name
    ``f"{bank_name}#L{layer_id:05d}"``, kind ``"experts_bank"``) and releases them, so
    conversion RAM peaks at ~in-flight layers instead of the whole bank set.

    Only engaged for streamable formats -- ``ExpertBanks.streamed`` reports whether this
    actually fired; if not, the caller falls back to the materialize-and-write path
    instead. The progress bar is created lazily on the first call, so a format that never
    streams never shows one.

    ``FTWWriter`` buffers file/shard state and is not thread-safe; completion callbacks
    can fire from the loader's own reader threads, so the write+release is serialized
    under one lock (disk-bound anyway).
    """

    def __init__(self, writer: FTWWriter, desc: str = "Converting expert banks") -> None:
        self._writer = writer
        self._desc = desc
        self._bar = None
        self._lock = threading.Lock()
        self._seen: set[int] = set()
        self.n_written = 0
        self.n_bytes = 0

    def __call__(self, layer_id: int, banks: dict) -> None:
        with self._lock:
            assert layer_id not in self._seen, f"layer {layer_id} streamed to the sink twice"
            self._seen.add(layer_id)
            if self._bar is None:
                from sparklab.utils.progress import byte_bar

                self._bar = byte_bar(0, self._desc)  # total unknown up front (streamed)
            nbytes = 0
            for bank_name, bank in banks.items():
                self._writer.add_tensor(
                    layer_bank_entry_name(bank_name, layer_id), bank.tensor, kind="experts_bank"
                )
                nbytes += bank.nbytes
                bank.release()
                self.n_written += 1
            self.n_bytes += nbytes
            self._bar.update(nbytes)
            # Cumulative BYTES (not the bank count): the supervisor maps this against the
            # known expert-pool size for a smooth phase-budgeted bar. Total stays 0 (unknown
            # up front while streaming); the materialize path below emits a real total.
            _progress("experts", self.n_bytes, 0)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    @property
    def num_layers(self) -> int:
        return len(self._seen)


class _OwnedTensorBank:
    """Minimal releasable bank passed from a conversion transform to ``_ConvertSink``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.nbytes = tensor.numel() * tensor.element_size()

    def release(self) -> None:
        self.tensor = torch.empty(0, dtype=self.tensor.dtype)


def _quantize_nvfp4_bank(
    tensor: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one ``[E, O, K]`` BF16 expert bank to native ModelOpt NVFP4.

    FlashInfer produces canonical low-nibble-first E2M1 weights and linear E4M3
    block scales.  A separate per-expert global keeps the full FP4 dynamic range
    without coupling experts with different magnitudes.  The output is exactly
    the native six-bank layout consumed by SparkLab's inline-dequant kernels.
    """
    if tensor.ndim != 3 or tensor.dtype != torch.bfloat16:
        raise ValueError(
            f"NVFP4 conversion expects [E, O, K] BF16 banks, got "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
        )
    experts, out_features, in_features = tensor.shape
    if in_features % 16:
        raise ValueError(f"NVFP4 input width must be divisible by 16, got {in_features}")

    import flashinfer
    from flashinfer.quantization import SfLayout

    packed = torch.empty(
        (experts, out_features, in_features // 2), dtype=torch.uint8
    )
    scales = torch.empty(
        (experts, out_features, in_features // 16), dtype=torch.float8_e4m3fn
    )
    globals_ = torch.empty((experts, out_features), dtype=torch.float16)
    for expert in range(experts):
        source = tensor[expert].to(device)
        maximum = source.float().abs().nan_to_num().max()
        # E4M3 max (448) times E2M1 max (6). A zero tensor uses unit
        # scaling and remains exactly zero.
        quant_scale = torch.where(
            maximum > 0,
            maximum.new_tensor(448.0 * 6.0) / maximum,
            maximum.new_tensor(1.0),
        ).reshape(1)
        quantized, block_scales = flashinfer.nvfp4_quantize(
            source,
            quant_scale,
            sfLayout=SfLayout.layout_linear,
        )
        packed[expert].copy_(quantized.view(torch.uint8).cpu())
        scales[expert].copy_(
            block_scales.view(torch.float8_e4m3fn)
            .reshape(out_features, in_features // 16)
            .cpu()
        )
        globals_[expert].fill_(float(quant_scale.reciprocal().item()))
    return packed, scales, globals_


class _Nvfp4QuantizeSink:
    """Stream BF16 expert layers through NVFP4 quantization into the FTW sink."""

    def __init__(self, outer: _ConvertSink, device: torch.device) -> None:
        self._outer = outer
        self._device = device

    def __call__(self, layer_id: int, banks: dict) -> None:
        expected = {"gate_up", "down"}
        if set(banks) != expected:
            raise ValueError(
                f"BF16-to-NVFP4 transform expected banks {sorted(expected)}, "
                f"found {sorted(banks)}"
            )
        transformed = {}
        try:
            for source_name, prefix in (("gate_up", "gate_up"), ("down", "down")):
                packed, scales, globals_ = _quantize_nvfp4_bank(
                    banks[source_name].tensor, self._device
                )
                transformed[f"{prefix}_packed"] = _OwnedTensorBank(packed)
                transformed[f"{prefix}_scale"] = _OwnedTensorBank(scales)
                transformed[f"{prefix}_global"] = _OwnedTensorBank(globals_)
            self._outer(layer_id, transformed)
        finally:
            for bank in banks.values():
                bank.release()
            torch.cuda.empty_cache()


def _write_config_override(out_dir: str, name: str, value: str) -> None:
    """Record a conversion-owned format choice in the self-contained HF config."""
    path = os.path.join(out_dir, "config.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    config[name] = value
    if isinstance(config.get("text_config"), dict):
        config["text_config"][name] = value
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _convert_checkpoint(
    model_path: str,
    out_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    moe_backend: str = "offload",
    nvfp4_backend: str = "triton",
    expert_quantization: str | None = None,
    shard_limit: int = DEFAULT_SHARD_LIMIT,
    device: str | None = None,
) -> dict:
    """Write ``model_path`` as an FTW checkpoint at ``out_dir``. Returns the index dict.

    The FTW format is TP-agnostic and conversion runs single-process, so the resulting
    checkpoint records no TP layout and loads independently of the runtime TP setting."""
    from sparklab.runtime.distributed import DistributedInfo, set_tp_info, try_get_tp_info
    from sparklab.runtime.engine.config import EngineConfig
    from sparklab.models.weight import load_weight
    from sparklab.moe.expert_banks import load_expert_banks
    from .ftw import is_ftw_checkpoint

    if is_ftw_checkpoint(model_path):
        raise SystemExit(f"{model_path} is already an FTW checkpoint")
    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=1)
        tp = try_get_tp_info()
    elif tp.size != 1:
        raise SystemExit(
            f"FTW conversion runs single-process and the format records no TP layout, "
            f"but TP is already set to size={tp.size}"
        )
    dev = torch.device(device or "cuda:0")
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)  # init CUDA context (needed by nvfp4 backend pick / pinning)

    cfg = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(tp.rank, tp.size),
        dtype=dtype,
        moe_backend=moe_backend,
        nvfp4_backend=nvfp4_backend,
    )
    mc = cfg.model_config
    # A fused DSV4 checkpoint carries three DSpark MoE layers under ``mtp.*``.
    # Convert them into the same expert-bank artifact even though target-only
    # serving does not instantiate the draft. FTW readers can expose either the
    # 43-layer prefix or the complete 46-layer bank at runtime.
    dsv4_args = getattr(mc, "dsv4_args", None)
    if dsv4_args is not None and (
        int(getattr(dsv4_args, "n_mtp_layers", 0) or 0) > 0
        and tuple(getattr(dsv4_args, "dspark_target_layer_ids", ()) or ())
        and int(getattr(dsv4_args, "dspark_markov_rank", 0) or 0) > 0
    ):
        from dataclasses import replace

        object.__setattr__(mc, "dsv4_args", replace(dsv4_args, dspark_enabled=True))
        object.__setattr__(mc, "speculative_method", "dspark")
    # Conversion bypasses Engine._adjust_config(), which normally copies this runtime
    # choice onto the parsed ModelConfig before expert-bank construction. Do it here so
    # FTW is written in the requested backend-owned layout (for example SM12x b12x), not
    # silently in ModelConfig's portable Triton default.
    object.__setattr__(mc, "nvfp4_backend", cfg.nvfp4_backend)
    offload = moe_backend == "offload" and getattr(mc, "is_moe", False)
    include_moe_experts = not offload
    if expert_quantization not in {None, "nvfp4"}:
        raise ValueError(
            f"unsupported target expert quantization: {expert_quantization!r}"
        )
    if expert_quantization is not None and (
        not offload or getattr(mc, "expert_quant", "none") != "none"
    ):
        raise ValueError(
            "target expert quantization requires an unquantized offload-MoE source"
        )

    from sparklab.utils.progress import byte_bar, count_bar

    writer = FTWWriter(out_dir, shard_limit=shard_limit)
    n_weight = n_speculative = n_bank = n_alpha = 0

    # 1) dense weights (host tensors; load straight to CPU to avoid GPU pressure)
    _progress("dense", 0, 0)  # phase start; per-tensor cumulative bytes follow (total unknown)
    dense_bytes = 0
    for name, tensor in count_bar(load_weight(model_path, torch.device("cpu"),
                                              include_moe_experts=include_moe_experts),
                                  "Converting dense weights"):
        writer.add_tensor(name, tensor, kind="weight")
        n_weight += 1
        dense_bytes += tensor.numel() * tensor.element_size()
        _progress("dense", dense_bytes, 0)

    # Optional checkpoint-native draft weights are a separate kind so ordinary
    # target serving never reads or allocates them.
    from sparklab.models.register import _load_attr, get_model_spec

    spec = get_model_spec(mc.architectures[0])
    try:
        iter_speculative = _load_attr(spec.module, "iter_speculative_weights")
    except AttributeError:
        iter_speculative = None
    if iter_speculative is not None:
        for name, tensor in count_bar(
            iter_speculative(model_path, torch.device("cpu")),
            "Converting speculative weights",
        ):
            writer.add_tensor(name, tensor, kind="speculative_weight")
            n_speculative += 1

    # 2) offload expert banks (post-repack) + alpha scales (slow path auto-picks parallel/serial)
    quant_format = None
    num_layers = None
    if offload:
        # Streamable formats (bf16, ds_fp4, every nvfp4 backend, gpt-oss mxfp4, q4_0,
        # qwen3_5 fp8/bf16-dequant) write each layer to its own FTW entry as it completes.
        # Marlin/b12x use a wrapper sink that repacks one completed native NVFP4 layer before
        # forwarding it, so those large layouts remain bounded too. Which path a provider
        # actually took is reported by ExpertBanks.streamed rather than guessed here.
        sink = _ConvertSink(writer)
        layer_sink = (
            _Nvfp4QuantizeSink(sink, dev)
            if expert_quantization == "nvfp4"
            else sink
        )
        banks = load_expert_banks(
            model_path, mc, device=dev, dtype=dtype, layer_sink=layer_sink
        )
        quant_format = expert_quantization or banks.quant_format
        if banks.streamed:
            sink.close()
            num_layers = sink.num_layers  # however many distinct layers the sink actually saw
            n_bank = sink.n_written
            assert num_layers > 0, (
                "provider reported streamed=True but the sink never fired -- the FTW "
                "would silently have no expert banks"
            )
            # Formats that fold their global scales (nvfp4 marlin/b12x) stream the weight
            # banks per layer but keep the alphas as flat [L*E] GPU vectors; write those as
            # flat reserved-name entries (same kind + names the materialize branch uses, so
            # the reader's reserved-name path reconstructs them identically).
            for an in ("gate_up_alpha", "down_alpha"):
                alpha = getattr(banks, an, None)
                if alpha is not None:
                    writer.add_tensor(an, alpha, kind="experts_bank")
                    n_alpha += 1
        else:
            # The on-disk format keeps one contiguous region per bank and the writer only
            # has whole-tensor add_tensor, so the per-layer sources reassemble into one
            # flat tensor (a per-bank host RAM spike during conversion).
            items = []
            for name, per_layer in banks.sources.items():
                if num_layers is None:
                    num_layers = len(per_layer)
                else:
                    assert len(per_layer) == num_layers, (name, len(per_layer), num_layers)
                items.append((name, torch.cat(per_layer, dim=0) if len(per_layer) > 1 else per_layer[0]))
            for an in ("gate_up_alpha", "down_alpha"):
                if getattr(banks, an, None) is not None:
                    items.append((an, getattr(banks, an)))
            total_bytes = sum(t.numel() * t.element_size() for _, t in items)
            bar = byte_bar(total_bytes, "Converting expert banks")
            done_bytes = 0
            _progress("experts", 0, total_bytes)
            for name, tensor in items:
                writer.add_tensor(name, tensor, kind="experts_bank")
                nbytes = tensor.numel() * tensor.element_size()
                bar.update(nbytes)
                done_bytes += nbytes
                _progress("experts", done_bytes, total_bytes)
                n_bank += name not in ("gate_up_alpha", "down_alpha")
                n_alpha += name in ("gate_up_alpha", "down_alpha")
            bar.close()

    _progress("finalize")  # writing shard index + copying config/tokenizer
    copied = _copy_metadata(model_path, out_dir)
    if expert_quantization is not None:
        _write_config_override(out_dir, "sparklab_expert_quant", expert_quantization)
    if kda_quantization := os.environ.get(_GLM_KDA_QUANT_ENV):
        _write_config_override(out_dir, "sparklab_kda_quant", kda_quantization)

    # Models with very large non-parameter runtime stores can stream a self-contained
    # side artifact beside FTW (Qwen4 PLE's 95 GiB random-row n-gram table). The hook
    # runs before finalize, so a failed extraction never publishes a valid FTW index.
    try:
        external_hook = _load_attr(spec.module, "copy_external_artifacts")
    except AttributeError:
        external_hook = None
    external_artifacts = (
        external_hook(model_path, out_dir, mc) if external_hook is not None else []
    )

    try:
        fingerprint = _source_fingerprint(
            model_path, mc, device=dev, expert_quantization=expert_quantization
        )
    except Exception:
        fingerprint = None

    index = writer.finalize({
        "source_model_path": os.path.abspath(model_path),
        "fingerprint": fingerprint,
        # quant_format records the actual on-disk bank layout (e.g. nvfp4_marlin vs
        # nvfp4_b12x): the suffix is a runtime backend pick (GPU capability / env), NOT in
        # config, and the stored bytes are physically repacked into it -- so it's kept and
        # read back at load (ftw.load_ftw_banks). dtype/moe_backend were dropped: each
        # tensor already carries its own dtype, and nothing reads a model-level backend.
        "quant_format": quant_format,
        "expert_quantization": expert_quantization,
        # The reader takes num_layers from the model config (copied into this
        # checkpoint); recording it here too gives load_ftw_banks a cross-check that
        # the banks match the config they ship with. None for non-offload checkpoints.
        "expert_bank_num_layers": num_layers,
        "counts": {
            kind: count
            for kind, count in (
                ("weight", n_weight),
                ("speculative_weight", n_speculative),
                ("experts_bank", n_bank + n_alpha),
            )
            if count
        },
        "copied_metadata": copied,
        "external_artifacts": external_artifacts,
    })
    return index


def convert_checkpoint(
    model_path: str,
    out_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    moe_backend: str = "offload",
    nvfp4_backend: str = "triton",
    expert_quantization: str | None = None,
    kda_quantization: str | None = None,
    shard_limit: int = DEFAULT_SHARD_LIMIT,
    device: str | None = None,
) -> dict:
    """Write a self-contained FTW checkpoint, including requested format transforms."""
    if kda_quantization not in {None, "fp8_pertensor"}:
        raise ValueError(f"unsupported KDA quantization: {kda_quantization!r}")
    previous = os.environ.get(_GLM_KDA_QUANT_ENV)
    try:
        if kda_quantization is None:
            os.environ.pop(_GLM_KDA_QUANT_ENV, None)
        else:
            os.environ[_GLM_KDA_QUANT_ENV] = kda_quantization
        return _convert_checkpoint(
            model_path,
            out_dir,
            dtype=dtype,
            moe_backend=moe_backend,
            nvfp4_backend=nvfp4_backend,
            expert_quantization=expert_quantization,
            shard_limit=shard_limit,
            device=device,
        )
    finally:
        if previous is None:
            os.environ.pop(_GLM_KDA_QUANT_ENV, None)
        else:
            os.environ[_GLM_KDA_QUANT_ENV] = previous


__all__ = ["convert_checkpoint"]
