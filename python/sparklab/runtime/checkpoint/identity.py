"""Portable compatibility identity from the loaded artifact and complete layout."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

from sparklab.version import __version__
from .state import Compatibility


def file_digest(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity(model_path, adapter, config, *, scope):
    """Startup-only file hashing, before request dispatch. Never hash live pages."""
    root = Path(model_path)
    index_path = root / "freetoken_weight.json"
    index = json.loads(index_path.read_text())
    fingerprint = index.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(
            "persistent prefixes require an identified immutable FTW artifact"
        )
    model_files = ("config.json", "hf_quant_config.json", "freetoken_weight.json")
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    template_files = [root / "chat_template.jinja"]
    # Transformers' current local layout is chat_templates/. Retain the older
    # spelling in the identity too. Bound discovery before sorting/materializing
    # all filenames, rather than rejecting only after an unbounded glob list.
    for directory in ("chat_templates", "additional_chat_templates"):
        extra = root / directory
        if extra.is_dir():
            for candidate in extra.glob("*.jinja"):
                if len(template_files) >= 64:
                    raise ValueError(
                        "too many template files for bounded checkpoint identity"
                    )
                template_files.append(candidate)
    template_files.sort()
    kv = adapter.kv._kv_buffer
    pool = adapter.linear
    layout = {
        "version": "mha-gdn-1",
        "runtime_version": __version__,
        "kv_layers": kv.shape[1],
        "kv_heads_dim": tuple(kv.shape[4:]),
        "kv_dtype": str(kv.dtype),
        "layer_map": adapter.kv._layer_map,
    }
    if pool is not None:
        layout.update(
            linear_layers=sorted(pool._local_index),
            conv_shape=(pool.conv_states.shape[0], *pool.conv_states.shape[2:]),
            conv_dtype=str(pool.conv_states.dtype),
            recurrent_shape=(
                pool.recurrent_states.shape[0],
                *pool.recurrent_states.shape[2:],
            ),
            recurrent_dtype=str(pool.recurrent_states.dtype),
            auxiliary={
                k: (shape, str(dtype), fill)
                for k, (shape, dtype, fill) in pool._aux_specs.items()
            },
        )
    rotary = getattr(config.model_config, "rotary_config", None)
    return Compatibility.from_config(
        {
            "artifact": {
                "fingerprint": fingerprint,
                "index_sha256": file_digest(index_path),
            },
            "precision": {
                "compute": str(config.dtype),
                "attention": getattr(config.model_config, "attn_quant", None),
                "expert": getattr(config.model_config, "expert_quant", None),
                "nvfp4_backend": getattr(config, "nvfp4_backend", None),
                "nvfp4_prefill_backend": getattr(config, "nvfp4_prefill_backend", None),
            },
            "model_config": {
                "files": {name: file_digest(root / name) for name in model_files},
                "rotary": asdict(rotary) if is_dataclass(rotary) else rotary,
            },
            "tokenizer": {name: file_digest(root / name) for name in tokenizer_files},
            "template": {
                str(path.relative_to(root)): file_digest(path)
                for path in template_files
            },
            "state_layout": layout,
            "cache_geometry": {
                "page_size": adapter.page_size,
                "attention_backend": config.attention_backend,
            },
            "continuation": "target-only",
            "scope": scope,
        }
    )
