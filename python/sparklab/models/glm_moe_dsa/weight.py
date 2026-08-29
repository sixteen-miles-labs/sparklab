"""Weight loading for GLM-5.2 (``glm_moe_dsa``).

Resident (non routed-expert) weights are bf16 in the checkpoint. What this loader
yields follows the quant modes RESOLVED IN ``parse_config`` (``ModelConfig.attn_quant``
/ ``dense_quant`` / ``lm_head_quant``, from the SPARKLAB_GLM_*_FP8 switches, default
on): in the default fp8 mode the big projections are requantized at load to W8A16
fp8-e4m3 with per-output-row scales (an extra ``*.weight_scale`` tensor per
projection); with the switches off everything streams through verbatim as bf16. The
router selection bias is remapped ``mlp.gate.e_score_correction_bias ->
mlp.e_score_correction_bias``; the DSA indexer tensors load bf16 on "full" indexer
layers (serving runs faithful DSA top-k sparse attention; see attention.py); only the
trailing MTP layer is skipped. Routed experts are NVFP4
and go to the offload cache via the shared glm4_moe loader (identical key layout).

FTW caveat: an FTW checkpoint stores whatever iter_weights yielded at CONVERSION time,
and the model is built from the env at SERVE time -- the two must agree (a mismatch
fails loudly in load_state_dict on the ``*.weight_scale`` keys). The active modes are
logged at load so conversion logs record the choice.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import safetensors
import torch
from sparklab.runtime.distributed import get_tp_info
from sparklab.models.glm4_moe.weight import (
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
)
from sparklab.models.loader import drop_page_cache
from sparklab.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config

# fp8-e4m3 dynamic range for the per-row W8A16 quantization of the big MLA projections.
_FP8_MAX = 448.0


def _quant_fp8_per_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row fp8-e4m3 quantization: ``w ~= weight_fp8 * scale[:, None]``."""
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / _FP8_MAX).clamp(min=1e-12)
    q = (wf / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best effort
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.2 stores routed experts as NVFP4 and only supports the offload backend; "
        "experts are loaded into the offload cache via load_nvfp4_expert_sources()."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    dense = config.first_k_dense_replace
    attn_fp8 = config.attn_quant == "fp8_pertensor"
    mlp_fp8 = config.dense_quant == "fp8_pertensor"
    head_fp8 = config.lm_head_quant == "fp8_pertensor"
    if primary:
        from sparklab.utils import init_logger

        init_logger(__name__).info(
            f"GLM-5.2 resident quant: attn={config.attn_quant} dense={config.dense_quant} "
            f"lm_head={config.lm_head_quant} (SPARKLAB_GLM_ATTN_FP8/SPARKLAB_GLM_MLP_FP8; "
            "an FTW conversion records these choices implicitly -- serve with the same flags)"
        )
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.2 dense weights",
            disable=not primary,
        ):
            a = f"model.layers.{layer}.self_attn"
            fp8_projs = (
                ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj") if attn_fp8 else ()
            )
            for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                w = reader.get(f"{a}.{proj}.weight")
                if proj in fp8_projs:
                    q, scale = _quant_fp8_per_row(w)
                    yield f"{a}.{proj}.weight", q
                    yield f"{a}.{proj}.weight_scale", scale
                else:
                    yield f"{a}.{proj}.weight", w
            for norm in ("q_a_layernorm", "kv_a_layernorm"):
                yield f"{a}.{norm}.weight", reader.get(f"{a}.{norm}.weight")
            # DSA lightning indexer ("full" layers only; "shared" layers reuse their
            # group leader's selection and ship no indexer tensors). Always bf16.
            idx_types = config.glm_dsa_args.indexer_types
            if idx_types and idx_types[layer] == "full":
                for proj in ("wq_b", "wk", "weights_proj"):
                    yield f"{a}.indexer.{proj}.weight", reader.get(f"{a}.indexer.{proj}.weight")
                yield f"{a}.indexer.k_norm.weight", reader.get(f"{a}.indexer.k_norm.weight")
                yield f"{a}.indexer.k_norm.bias", reader.get(f"{a}.indexer.k_norm.bias")
            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield (
                    f"model.layers.{layer}.{norm}.weight",
                    reader.get(f"model.layers.{layer}.{norm}.weight"),
                )

            m = f"model.layers.{layer}.mlp"

            def _mlp_weight(key: str):
                w = reader.get(f"{key}.weight")
                if mlp_fp8:
                    q, scale = _quant_fp8_per_row(w)
                    yield f"{key}.weight", q
                    yield f"{key}.weight_scale", scale
                else:
                    yield f"{key}.weight", w

            if layer < dense:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _mlp_weight(f"{m}.{proj}")
            else:
                yield f"{m}.gate.weight", reader.get(f"{m}.gate.weight")
                yield (
                    f"{m}.e_score_correction_bias",
                    reader.get(f"{m}.gate.e_score_correction_bias").to(torch.bfloat16),
                )
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _mlp_weight(f"{m}.shared_experts.{proj}")

        yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight")
        yield "model.norm.weight", reader.get("model.norm.weight")
        head = reader.get("lm_head.weight")
        if head_fp8 and not config.tie_word_embeddings:
            q, scale = _quant_fp8_per_row(head)
            yield "lm_head.weight", q
            yield "lm_head.weight_scale", scale
        else:
            yield "lm_head.weight", head
    finally:
        reader.close()


__all__ = ["iter_weights", "load_nvfp4_expert_sources", "load_nvfp4_expert_sources_parallel"]
