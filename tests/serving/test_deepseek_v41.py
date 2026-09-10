import json
from types import SimpleNamespace

import pytest

from sparklab.serving.function_call_parser import FunctionCallParser
from sparklab.tokenizer.tokenize import TokenizeManager, _load_dsv4_encoder_if_needed

TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
}}}]
TEXT = ('<｜DSML｜ calls><｜DSML｜ invoke name="read">'
        '<｜DSML｜ parameter name="path" string="true">a"b.txt</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="limit" string="false">3</｜DSML｜ parameter>'
        '</｜DSML｜ invoke></｜DSML｜ calls>')


def test_v41_tool_call_nonstream_and_every_character_boundary():
    parser = FunctionCallParser(TOOLS, tool_call_parser="deepseekv41")
    parsed = parser.parse_non_stream(TEXT)
    assert parsed.calls[0].name == "read"
    assert json.loads(parsed.calls[0].parameters) == {"path": 'a"b.txt', "limit": 3}
    parser = FunctionCallParser(TOOLS, tool_call_parser="deepseekv41")
    fragments = []
    normal = ""
    for char in TEXT:
        text, calls = parser.parse_stream_chunk(char)
        normal += text
        fragments.extend(call.parameters for call in calls)
    assert normal == ""
    assert json.loads("".join(fragments)) == {"path": 'a"b.txt', "limit": 3}


def test_v41_encoder_filename_and_numeric_effort(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}')
    (tmp_path / "encoding").mkdir()
    (tmp_path / "encoding/encoding.py").write_text(
        'def encode_messages(messages, thinking_mode, reasoning_effort=None):\n'
        '    return str(reasoning_effort)\n'
    )
    tokenizer = SimpleNamespace(name_or_path=str(tmp_path), chat_template=None)
    encoder = _load_dsv4_encoder_if_needed(tokenizer)
    assert encoder.__name__ == "encoding_dsv41"
    manager = TokenizeManager(tokenizer)
    assert manager._sanitize_effort({"reasoning_effort": 37}) == {"reasoning_effort": 37}
    for value in (0, 101, True):
        with pytest.raises(ValueError, match="integer"):
            manager._sanitize_effort({"reasoning_effort": value})
