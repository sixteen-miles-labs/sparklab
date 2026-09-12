"""DeepSeek V4.1 reference configuration (MIT; see LICENSE.deepseek)."""
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelArgs:
    """Field names are exactly the config JSON keys. The defaults are a small model that
    `python model.py` can run, not the released shapes -- though the scale-independent
    values (norm_eps, score_func, hc_*, engram_*) do match it."""

    # runtime limits rather than model shape: they size the KV caches
    max_batch_size: int = 4
    max_seq_len: int = 4096
    temperature: float = 1
    dtype: Literal["bf16", "fp8"] = "fp8"
    expert_dtype: Literal["fp4"] | None = "fp4"
    vocab_size: int = 129280
    dim: int = 1024
    moe_inter_dim: int = 1024
    n_layers: int = 5
    n_mtp_layers: int = 1  # extra draft layers appended after the backbone, indices n_layers..
    n_heads: int = 16
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    # attention: latent q/kv projections, plus a LoRA-factorised output projection over o_groups
    q_lora_rank: int = 256
    head_dim: int = 128
    rope_head_dim: int = 32
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 256
    # sparse attention: every layer attends over a sliding window, and may add compressed KV on top
    window_size: int = 128
    # one entry per layer, MTP layers included: 0 = sliding window only, r = KV compressed r-to-1
    compress_ratios: tuple[int, ...] = (0, 2, 2, 1, 1, 0)
    # layers sharing a ratio also share one compressed KV and one indexer, produced by the first
    kv_source_layers: tuple[int, ...] = (1, 3)
    index_source_layers: tuple[int, ...] = (1, 3)
    # rope, with YaRN extrapolation when original_seq_len > 0. Compressed KV rotates at its own
    # theta because one latent stands for compress_ratio tokens, so its positions are further apart.
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # the indexer: a small extra attention that scores compressed positions, so each query can keep
    # just `index_topk` of them. Names match DeepSeek-V3.2-Exp, where this mechanism first appeared.
    index_n_heads: int = 16
    index_head_dim: int = 64
    index_topk: int = 64
    # candidate pre-filtering: candidate_source_layer < 0 turns it off and the other two are unused
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # hyper-connections: the residual stream is carried as hc_mult parallel copies
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # engram: n-gram hash lookups added into the residual stream at a few layers
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()  # unpadded table rows; each rank allocates ceil(rows / world_size)
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0  # bucket size each (n-gram size, head) starts searching primes from
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2  # token that fills n-gram slots with no history; matches training
    # size of the compressed tokenizer vocab; every hash multiplier is derived from it
    engram_compressed_vocab_size: int = 0
    # vision (VL); vision_n_layers == 0 disables the vision path
    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 544 * 544
    vision_max_wh_ratio: int | None = None
    # raw id of <｜deepseek_image｜>; every position of an image span carries this id in input_ids
    image_token_id: int = 129264
    # DSpark draft head stored under the mtp.* checkpoint namespace.
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 0
    dspark_n_activated_experts: int = 0

    @property
    def vision_enabled(self) -> bool:
        return self.vision_n_layers > 0

    def get_moe_config(self, layer_id: int) -> tuple[int, int]:
        """Return the routed/activated expert counts for a given layer."""
        if layer_id < self.n_layers:
            return self.n_routed_experts, self.n_activated_experts
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_n_activated_experts or self.n_activated_experts,
        )
