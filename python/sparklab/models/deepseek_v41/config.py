"""Native, single-request DeepSeek V4.1 research configuration."""
import json
from dataclasses import fields
from pathlib import Path

from sparklab.models.config import ModelConfig, RotaryConfig
from .args import ModelArgs

MAX_CONTEXT = 2048
CACHE_BYTES = 24 << 30


def load_args(path):
    raw = json.loads((Path(path) / "inference/config.json").read_text())
    required = {"dim", "n_layers", "compress_ratios", "kv_source_layers",
                "index_source_layers", "engram_layer_ids", "engram_num_embeddings"}
    if not required <= raw.keys():
        raise ValueError(f"incomplete DeepSeek V4.1 config: missing {sorted(required - raw.keys())}")
    args = ModelArgs(**{key: value for key, value in raw.items()
                        if key in {f.name for f in fields(ModelArgs)}})
    args.max_seq_len = MAX_CONTEXT
    args.max_batch_size = 1
    args.vision_n_layers = 0
    if len(args.compress_ratios) < args.n_layers or any(
        ratio not in (0, 1, 2) for ratio in args.compress_ratios[:args.n_layers]
    ):
        raise ValueError("DeepSeek V4.1 requires per-layer compression ratios 0, 1 or 2")
    if not set(args.kv_source_layers) <= set(args.index_source_layers):
        raise ValueError("every compressed KV source must produce index keys")
    for layer in (*args.kv_source_layers, *args.index_source_layers):
        if not 0 <= layer < args.n_layers or not args.compress_ratios[layer]:
            raise ValueError("invalid DeepSeek V4.1 sparse source layer")
    return args


def parse_config(hf_config):
    path = getattr(hf_config, "_name_or_path", "")
    args = load_args(path)
    # The generic pool provides scheduler token admission. The bounded research
    # decoder owns its window, compressed and n-gram state, and disables prefix
    # reuse; it does not route V4.1 through the incompatible V4 paged pool.
    return ModelConfig(
        num_layers=args.n_layers, num_qo_heads=args.n_heads, num_kv_heads=1,
        head_dim=args.head_dim, hidden_size=args.dim, vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim, rms_norm_eps=args.norm_eps,
        rotary_config=RotaryConfig(head_dim=args.head_dim, rotary_dim=args.rope_head_dim,
                                   max_position=MAX_CONTEXT, base=args.rope_theta, scaling=None),
        hidden_act="silu", tie_word_embeddings=False,
        num_experts=args.n_routed_experts, num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim, norm_topk_prob=args.norm_topk_prob,
        model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"],
        moe_enabled=True, expert_quant="ds_fp4", single_stream_only=True,
        dsv41_args=args, speculative_method="none", speculative_tokens=0,
    )
