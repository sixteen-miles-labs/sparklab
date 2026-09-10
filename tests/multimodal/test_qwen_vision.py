from types import SimpleNamespace

import pytest
import torch

from sparklab.models.qwen3_5_moe.vision import apply_mrope, image_positions
from sparklab.message import UserMsg
from sparklab.core import SamplingParams


def test_image_positions_match_publisher_with_two_different_grids():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    ids = torch.tensor([2, 3] + [99] * 6 + [4, 5] + [99] * 4 + [6])
    grids = torch.tensor([[1, 4, 6], [1, 4, 4]])
    positions, delta = image_positions(ids, grids, 99, 2)
    shim = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    shim.get_vision_position_ids = Qwen3_5Model.get_vision_position_ids.__get__(shim)
    expected, shift = Qwen3_5Model.get_rope_index(shim, ids[None], (ids == 99).long()[None], grids)
    torch.testing.assert_close(positions, expected[:, 0])
    assert delta == shift.item()
    assert positions[:, -1].tolist() == [9, 9, 9]


def test_interleaved_partial_rope_matches_publisher():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb
    cfg = Qwen3_5TextConfig(hidden_size=256, num_attention_heads=2, head_dim=128,
        rope_parameters={'rope_type':'default','rope_theta':10000000.,'partial_rotary_factor':.5,
                         'mrope_section':[11,11,10],'mrope_interleaved':True})
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    p = torch.tensor([[0,1,1,2],[0,1,2,3],[0,2,1,3]])
    q, k = torch.randn(4,256), torch.randn(4,128)
    actual_q, actual_k = apply_mrope(q,k,p,128,64,10000000.,[11,11,10])
    cos,sin = rope(q,p[:,None])
    eq,ek = apply_rotary_pos_emb(q.view(4,2,128).transpose(0,1)[None],k.view(4,1,128).transpose(0,1)[None],cos,sin)
    torch.testing.assert_close(actual_q,eq[0].transpose(0,1).reshape(4,-1))
    torch.testing.assert_close(actual_k,ek[0].transpose(0,1).reshape(4,-1))


def test_image_payload_survives_backend_serialization():
    payload={'pixels':torch.arange(12,dtype=torch.float32),'pixel_shape':[2,6],
             'grids':torch.tensor([1,2,4]),'positions':torch.arange(9),'delta':-2}
    msg=UserMsg(1,torch.tensor([1,2,3],dtype=torch.int32),SamplingParams(),mm_inputs=payload)
    restored=UserMsg.decoder(msg.encoder())
    for name in ('pixels','grids','positions'):
        torch.testing.assert_close(restored.mm_inputs[name],payload[name])
    assert restored.mm_inputs['pixel_shape']==[2,6]


def test_bad_image_placeholder_counts_fail():
    with pytest.raises(ValueError,match='placeholder count'):
        image_positions(torch.tensor([1,99,2]),torch.tensor([[1,4,4]]),99,2)


def test_text_only_manager_rejects_images_before_loading_processor():
    from sparklab.tokenizer.tokenize import TokenizeManager
    manager = TokenizeManager.__new__(TokenizeManager)
    manager._vision_source = None
    with pytest.raises(ValueError, match='--vision-model'):
        manager._prepare_images(SimpleNamespace())


def test_vision_messages_are_opt_in_and_preserve_image_order():
    from sparklab.serving.generation import render_messages
    parts = [{'type':'image_url','image_url':{'url':'data:image/png;base64,AA=='}},
             {'type':'text','text':'Compare these'},
             {'type':'image_url','image_url':{'url':'data:image/png;base64,AQ=='}}]
    messages = [{'role':'user','content':parts}]
    with pytest.raises(ValueError,match='text-only'):
        render_messages(messages)
    assert render_messages(messages,allow_images=True)[0]['content']==parts


def test_image_decode_shift_preserves_text_continuation():
    ids = torch.tensor([2] + [99] * 16 + [3, 4])
    positions,delta = image_positions(ids,torch.tensor([[1,8,8]]),99,2)
    continuation = torch.arange(len(ids),len(ids)+3)+delta
    assert continuation.tolist()==[int(positions.max())+i for i in (1,2,3)]


def test_remote_images_are_rejected_without_network_access():
    pytest.importorskip('PIL')
    from sparklab.multimodal.qwen import QwenImageProcessor
    processor = QwenImageProcessor.__new__(QwenImageProcessor)
    with pytest.raises(ValueError,match='base64 data:image'):
        processor.prepare([{'role':'user','content':[
            {'type':'image_url','image_url':{'url':'https://example.com/image.png'}}]}],None,{})
