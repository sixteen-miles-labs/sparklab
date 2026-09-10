from types import SimpleNamespace

import pytest
import torch

from sparklab.models.qwen3_5_moe.vision import MultimodalRotary, image_positions, request_positions


def test_image_features_are_inserted_before_residual_stream_replication(monkeypatch):
    from sparklab.models.qwen4_exp.model import Qwen4ExpModel

    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    ids = torch.tensor([1, 99, 99, 2])
    image = torch.tensor([[10., 20.], [30., 40.]])
    model.embed_tokens = SimpleNamespace(forward=lambda ids: ids[:, None].float().expand(-1, 2))
    model.layers = SimpleNamespace(op_list=[])
    model.hyper_connection_mixer = SimpleNamespace(forward=lambda hidden: hidden[:, :2])
    model._hc_count, model._image_token_id = 4, 99
    monkeypatch.setattr('sparklab.models.qwen4_exp.model.get_global_ctx',
                        lambda: SimpleNamespace(batch=SimpleNamespace(mm_embeds=image)))
    sample, streams = model.forward(ids, return_multi=True)
    torch.testing.assert_close(sample[1:3], image)
    for stream in range(4):
        torch.testing.assert_close(streams[:, stream * 2:(stream + 1) * 2], sample)
    assert ids.tolist() == [1, 99, 99, 2]  # PLE retains the original placeholder history.


def test_flash_next_positions_and_index_rotations_match_publisher():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpModel, Qwen4ExpTextRotaryEmbedding, apply_rotary_pos_emb,
    )
    cfg = Qwen4ExpTextConfig(hidden_size=256, num_attention_heads=1, head_dim=256,
        rope_parameters={'rope_type':'default','rope_theta':10000000.,'partial_rotary_factor':.25,
                         'mrope_section':[11,11,10],'mrope_interleaved':True})
    ids = torch.tensor([1] + [99] * 16 + [2, 3])
    grids = torch.tensor([[1, 8, 8]])
    p, delta = image_positions(ids, grids, 99, 2)
    shim = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    shim.get_vision_position_ids = Qwen4ExpModel.get_vision_position_ids.__get__(shim)
    expected, shift = Qwen4ExpModel.get_rope_index(shim, ids[None], (ids == 99).long()[None], grids)
    torch.testing.assert_close(p, expected[:, 0])
    assert delta == shift.item()
    # Four-token pooled-key starts can land inside images or after the prompt.
    starts = torch.tensor([0, 4, 8, 12, 16, 20, 24])
    req = SimpleNamespace(mm_positions=p, mm_delta=delta)
    positions = request_positions(req, starts)
    full = torch.cat((expected[:, 0], (torch.arange(len(ids), 25) + delta).expand(3, -1)), 1)
    torch.testing.assert_close(positions, full[:, starts])
    q, k = torch.randn(len(starts), 4 * 128), torch.randn(len(starts), 128)
    rotary = MultimodalRotary(None, 128, 64, 10000000., [11, 11, 10])
    aq, ak = rotary.forward(positions, q, k)
    cos, sin = Qwen4ExpTextRotaryEmbedding(cfg)(q, positions[:, None])
    eq, ek = apply_rotary_pos_emb(q.view(-1, 4, 128).transpose(0, 1)[None],
                                k[:, None, :].transpose(0, 1)[None], cos, sin)
    torch.testing.assert_close(aq, eq[0].transpose(0, 1).reshape_as(q))
    torch.testing.assert_close(ak, ek[0].transpose(0, 1).reshape_as(k))


def test_qsa_pool_uses_spatial_group_starts_and_decode_delta(monkeypatch):
    from sparklab.attention.qsa import QSAAttnBackend

    backend = QSAAttnBackend.__new__(QSAAttnBackend)
    backend.args = SimpleNamespace(index_compress_ratio=4)
    backend.config = SimpleNamespace(rms_norm_eps=1e-6)
    backend.device = torch.device('cpu')
    cache = torch.randn(32, 8)
    backend.kvcache = SimpleNamespace(index_k_cache=lambda slot: cache)
    monkeypatch.setattr('sparklab.attention.qsa.get_global_ctx',
                        lambda: SimpleNamespace(page_table=torch.arange(32)[None]))
    p, delta = image_positions(torch.tensor([1]+[99]*4+[2]), torch.tensor([[1,4,4]]), 99, 2)
    req = SimpleNamespace(cached_len=0, extend_len=12, table_idx=0, mm_positions=p, mm_delta=delta)
    seen = []
    rotary = SimpleNamespace(forward=lambda positions, q, k: (seen.append(positions.clone()) or q, k))
    backend._pool_completed_keys(0, [req], torch.zeros(8), rotary)
    torch.testing.assert_close(seen[0], request_positions(req, torch.tensor([0,4,8])))
    assert seen[0].shape == (3,3)


def test_text_rotary_still_uses_original_kernel():
    q,k,p = torch.randn(3,8),torch.randn(3,8),torch.arange(3)
    base = SimpleNamespace(forward=lambda pos,q,k:(q+1,k+2))
    aq,ak = MultimodalRotary(base,8,4,10000.,[1,1,0]).forward(p,q,k)
    torch.testing.assert_close(aq,q+1)
    torch.testing.assert_close(ak,k+2)
