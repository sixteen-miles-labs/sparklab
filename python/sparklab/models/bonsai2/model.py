"""Reuse the native Qwen decoder with packed ternary projections."""

import torch

from sparklab.layers import BaseOP, LinearReplicated
from sparklab.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata
from sparklab.models.qwen3_5_moe.model import Qwen3_5MoEForCausalLM

from .weights import mapping, validate_rotation


class TernaryLinear(BaseOP):
    def __init__(self, width, rows, kind):
        self.in_features, self.out_features = width, rows
        self._kind = kind
        self.qweight = torch.empty(
            rows, width // 128 * (34 if kind == 142 else 28), dtype=torch.uint8
        )
        self.signs = torch.empty(width, dtype=torch.float32)

    def forward(self, x):
        from sparklab.kernels.triton.bonsai import linear, rotate

        return linear(rotate(x, self.signs), self.qweight, self._kind)


class TernaryEmbedding(TernaryLinear):
    def __init__(self, width, rows, kind):
        super().__init__(width, rows, kind)
        self.embedding_dim = width

    def forward(self, ids):
        from sparklab.kernels.triton.bonsai import embedding, rotate

        x = embedding(ids, self.qweight, self._kind, self.in_features)
        return rotate(x, self.signs, inverse=True)


class TernaryHead(TernaryLinear):
    def forward(self, x):
        from sparklab.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill and not batch.return_all_logits:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return super().forward(x)


class InputProjection(BaseOP):
    def forward(self, x):
        return torch.cat(
            [p.forward(x) for p in (self.qkv, self.z, self.b, self.a)], dim=-1
        )


class Qwen3_5BonsaiForCausalLM(Qwen3_5MoEForCausalLM):
    def prepare_for_weight_load(self, model_path, *, dummy=False):
        from sparklab.runtime.distributed import get_tp_info

        if get_tp_info().size != 1:
            raise ValueError("Bonsai GGUF requires TP=1")
        from sparklab.models.gguf.reader import is_gguf_path

        if not is_gguf_path(model_path):
            raise ValueError(
                "Bonsai currently loads a direct .gguf file; FTW conversion is not supported"
            )
        if self.model.norm.weight.dtype != torch.bfloat16:
            raise ValueError("Bonsai currently requires --dtype bfloat16")
        m = load_gguf_metadata(model_path)
        validate_rotation(m)
        self._bonsai_metadata = m
        ts = {t.name: t for t in iter_gguf_tensors(model_path)}
        with torch.device("meta"):
            for layer in self.model.layers.op_list:
                if hasattr(layer, "linear_attn"):
                    layer.linear_attn.in_proj = InputProjection()
            for dest, sources in mapping(m).items():
                items = [ts[s] for s in sources]
                width, rows, kind = (
                    items[0].shape[-1],
                    sum(t.shape[0] for t in items),
                    items[0].ggml_type,
                )
                if kind in (142, 143):
                    cls = (
                        TernaryEmbedding
                        if dest == "model.embed_tokens"
                        else TernaryHead
                        if dest == "lm_head"
                        else TernaryLinear
                    )
                    op = cls(width, rows, kind)
                else:
                    op = LinearReplicated(width, rows, has_bias=False)
                    op.weight = op.weight.to(self.model.norm.weight.dtype)
                owner = self
                parts = dest.split(".")
                for p in parts[:-1]:
                    owner = owner.op_list[int(p)] if p.isdigit() else getattr(owner, p)
                setattr(owner, parts[-1], op)
