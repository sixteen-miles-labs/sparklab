"""Muse Glimmer ATEM reasoning parser (to=self / to=user / tool channels) and
tool-call detector (<atem:function_calls> invoke/parameter XML) -- one-shot and
streaming."""

from __future__ import annotations

import json

import pytest

from sparklab.serving.function_call_parser import (
    Function,
    FunctionCallParser,
    MuseGlimmerDetector,
    Tool,
)
from sparklab.serving.reasoning_parser import (
    ATEM_START,
    MuseGlimmerReasoningParser,
    ReasoningParser,
)


def _tools():
    return [
        Tool(function=Function(name="weather.get", parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "units": {"type": "object"},
            },
        })),
        Tool(function=Function(name="fs.write", parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        })),
    ]


def _atem(name: str, params: dict[str, str]) -> str:
    body = "".join(
        f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in params.items()
    )
    return (
        f'<atem:function_calls>\n<atem:invoke name="{name}">\n{body}'
        f"</atem:invoke>\n</atem:function_calls>"
    )


def _tool_channel(name: str, params: dict[str, str], *, closer: str = "<|eot|>") -> str:
    return f"<|start|>assistant to={name}<|message|>{_atem(name, params)}{closer}"


def _stream(parser, text: str, step: int = 3):
    reasoning, content = [], []
    for i in range(0, len(text), step):
        r = parser.parse_streaming_increment(text[i : i + step])
        reasoning.append(r.reasoning_text)
        content.append(r.normal_text)
    r = parser.flush()
    reasoning.append(r.reasoning_text)
    content.append(r.normal_text)
    return "".join(reasoning), "".join(content)


def _stream_detect(det, text: str, tools, step: int = 3):
    normal, calls = [], []
    for i in range(0, len(text), step):
        r = det.parse_streaming_increment(text[i : i + step], tools)
        normal.append(r.normal_text)
        calls.extend(r.calls)
    while True:
        r = det.parse_streaming_increment("", tools)
        if not r.normal_text and not r.calls:
            break
        normal.append(r.normal_text)
        calls.extend(r.calls)
    normal.append(det.finish_streaming())
    return "".join(normal), calls


def _assemble(calls):
    """(name, joined-args-json) per streamed call, in order."""
    out: list[list] = []
    for c in calls:
        if c.name is not None:
            out.append([c.name, ""])
        if c.parameters:
            out[-1][1] += c.parameters
    return [(name, json.loads(args or "{}")) for name, args in out]


# ---------------------------------------------------------------------------
# Reasoning parser
# ---------------------------------------------------------------------------
def test_reasoning_registered():
    assert ReasoningParser.ReasoningParserEnum["muse_glimmer"] is MuseGlimmerReasoningParser


def test_reasoning_bare_first_segment_self_then_user():
    text = (
        " to=self<|message|>Let me think about this.<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
    )
    r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "Let me think about this."
    assert r.normal_text == "The answer is 42."


def test_reasoning_user_only_turn():
    r = MuseGlimmerReasoningParser().detect_and_parse(" to=user<|message|>Hi!<|eot|>")
    assert r.reasoning_text == "" and r.normal_text == "Hi!"


def test_reasoning_recipientless_header_is_content():
    r = MuseGlimmerReasoningParser().detect_and_parse("assistant<|message|>Plain.<|eot|>")
    assert r.reasoning_text == "" and r.normal_text == "Plain."


def test_reasoning_plain_text_passthrough():
    # Raw, non-templated output carries no ATEM markers and must pass through.
    text = "Just a plain answer, no channels."
    r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "" and r.normal_text == text
    p = MuseGlimmerReasoningParser()
    reasoning, content = _stream(p, text)
    assert reasoning == "" and content == text


def test_reasoning_tool_channel_preserved_verbatim():
    text = (
        " to=self<|message|>check weather<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        + _atem("weather.get", {"city": "Paris"})
        + "<|eot|>"
    )
    r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "check weather"
    assert r.normal_text.startswith("<|start|>assistant to=weather.get<|message|>")
    assert "<atem:function_calls>" in r.normal_text and r.normal_text.endswith("<|eot|>")

    # A tool call as the FIRST (bare) segment: the header-open parser completes
    # the bare header with its synthetic <|start|> seed, so the detector always
    # receives a fully delimited channel -- no bare-header guessing downstream.
    bare = " to=weather.get<|message|>" + _atem("weather.get", {"city": "Rome"}) + "<|eot|>"
    r2 = MuseGlimmerReasoningParser().detect_and_parse(bare)
    assert r2.normal_text.startswith("<|start|> to=weather.get<|message|>")
    assert r2.normal_text.endswith("<|eot|>")


def test_reasoning_streaming_matches_one_shot():
    text = (
        " to=self<|message|>step one\nstep two<|eom|>"
        "<|start|>assistant to=user<|message|>Done: 42.<|eot|>"
    )
    one = MuseGlimmerReasoningParser().detect_and_parse(text)
    reasoning, content = _stream(MuseGlimmerReasoningParser(), text)
    assert reasoning == one.reasoning_text == "step one\nstep two"
    assert content == one.normal_text == "Done: 42."


def test_reasoning_streaming_holds_partial_markers():
    p = MuseGlimmerReasoningParser()
    out = p.parse_streaming_increment(" to=user<|message|>Hello <|eo")
    assert out.normal_text == "Hello "  # the partial closer is held, not leaked
    out = p.parse_streaming_increment("t|>")
    assert out.normal_text == ""
    r = p.flush()
    assert r.normal_text == "" and r.reasoning_text == ""


def test_reasoning_undecided_header_delivered_at_flush():
    # Held while it could still become a bare header; at end-of-stream it is a
    # (short) reply and must be delivered, not dropped.
    p = MuseGlimmerReasoningParser()
    assert p.parse_streaming_increment(" to=se").normal_text == ""
    r = p.flush()
    assert r.normal_text == " to=se" and r.reasoning_text == ""


def test_reasoning_multiple_self_segments_concatenate():
    text = (
        " to=self<|message|>alpha<|eom|>"
        "<|start|>assistant to=self<|message|>beta<|eom|>"
        "<|start|>assistant to=user<|message|>done<|eot|>"
    )
    r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "alphabeta"
    assert r.normal_text == "done"


# ---------------------------------------------------------------------------
# Tool-call detector
# ---------------------------------------------------------------------------
def test_detect_and_parse_typed_args():
    det = MuseGlimmerDetector()
    text = "One sec. " + _tool_channel(
        "weather.get", {"city": "Paris", "days": "3", "units": '{"temp": "C"}'}
    )
    assert det.has_tool_call(text)
    res = det.detect_and_parse(text, _tools())
    assert res.normal_text == "One sec."
    assert len(res.calls) == 1
    call = res.calls[0]
    assert call.name == "weather.get"
    args = json.loads(call.parameters)
    assert args == {"city": "Paris", "days": 3, "units": {"temp": "C"}}


def test_detect_and_parse_multiple_invokes_in_one_block():
    det = MuseGlimmerDetector()
    block = (
        "<|start|>assistant to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        "</atem:invoke>\n"
        '<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Tokyo</atem:parameter>\n'
        "</atem:invoke>\n"
        "</atem:function_calls><|eot|>"
    )
    res = det.detect_and_parse(block, _tools())
    assert [json.loads(c.parameters)["city"] for c in res.calls] == ["Paris", "Tokyo"]
    assert [c.tool_index for c in res.calls] == [0, 1]


def test_string_values_kept_verbatim():
    # The template's contract: "spaces for string values are not stripped".
    det = MuseGlimmerDetector()
    content = "line1\nline2\n"
    text = _tool_channel("fs.write", {"path": " /tmp/x ", "content": content})
    res = det.detect_and_parse(text, _tools())
    assert json.loads(res.calls[0].parameters) == {"path": " /tmp/x ", "content": content}

    normal, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assert _assemble(calls) == [("fs.write", {"path": " /tmp/x ", "content": content})]
    assert normal.strip() == ""


def test_streaming_matches_one_shot_with_channel_markup():
    text = "Checking. " + _tool_channel("weather.get", {"city": "Paris", "days": "3"})
    one = MuseGlimmerDetector().detect_and_parse(text, _tools())

    det = MuseGlimmerDetector()
    normal, calls = _stream_detect(det, text, _tools())
    assert normal.strip() == one.normal_text.strip() == "Checking."
    assert _assemble(calls) == [
        (c.name, json.loads(c.parameters)) for c in one.calls
    ] == [("weather.get", {"city": "Paris", "days": 3})]
    # ledgers the serving layer reads at stream end
    assert det.prev_tool_call_arr[0]["name"] == "weather.get"
    assert det.prev_tool_call_arr[0]["arguments"] == {"city": "Paris", "days": 3}


def test_bare_block_outside_tool_channel_is_text_not_execution():
    # Echo-becomes-execution guard (vLLM's rule): ATEM markup is executed only
    # inside a tool-recipient channel. A block in plain text / a to=user body --
    # e.g. the system prompt's own ATEM example echoed back -- renders as text.
    text = "Look: " + _atem("weather.get", {"city": "Oslo"})
    normal, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assert calls == []
    assert normal == text
    res = MuseGlimmerDetector().detect_and_parse(text, _tools())
    assert res.calls == [] and res.normal_text == text

    quoted = (
        " to=user<|message|>Call tools like this:\n"
        + _atem("weather.get", {"city": "Echo"})
        + "\nGot it?<|eot|>"
    )
    normal, calls = _stream_detect(MuseGlimmerDetector(), quoted, _tools())
    assert calls == []
    assert "Call tools like this:" in normal and "Got it?" in normal
    assert "<atem:function_calls>" in normal  # the quote stays visible text


def test_streaming_bare_first_segment_tool_channel():
    # No reasoning parser upstream: generation starts with the header continuation.
    text = " to=weather.get<|message|>" + _atem("weather.get", {"city": "Lima"}) + "<|eot|>"
    normal, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assert normal.strip() == ""
    assert _assemble(calls) == [("weather.get", {"city": "Lima"})]


def test_streaming_self_channel_swallowed_without_reasoning_parser():
    text = (
        " to=self<|message|>secret chain of thought<|eom|>"
        "<|start|>assistant to=user<|message|>Visible.<|eot|>"
    )
    normal, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assert calls == []
    assert "secret" not in normal
    assert normal.strip() == "Visible."


def test_streaming_two_tool_channels_sequential_indices():
    text = (
        _tool_channel("weather.get", {"city": "Paris"}, closer="<|eom|>")
        + _tool_channel("weather.get", {"city": "Rome"})
    )
    _, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assembled = _assemble(calls)
    assert assembled == [
        ("weather.get", {"city": "Paris"}),
        ("weather.get", {"city": "Rome"}),
    ]
    named = [c.tool_index for c in calls if c.name is not None]
    assert named == [0, 1]


def test_streaming_trailing_text_defers_until_after_call():
    det = MuseGlimmerDetector()
    chunk = "Pre. " + _tool_channel("weather.get", {"city": "Paris"}) + " Post."
    r1 = det.parse_streaming_increment(chunk, _tools())
    assert r1.normal_text == "Pre. "
    assert [c.name for c in r1.calls if c.name] == ["weather.get"]
    r2 = det.parse_streaming_increment("", _tools())
    assert r2.normal_text == " Post." and r2.calls == []


def test_streaming_plain_text_never_held():
    det = MuseGlimmerDetector()
    r = det.parse_streaming_increment("Just a normal answer, no tools.", _tools())
    assert r.normal_text == "Just a normal answer, no tools."
    assert r.calls == []


def test_truncated_call_suppressed_and_recovered():
    # Generation cut mid-invoke (max_tokens). The call already started streaming
    # (its Start was emitted at the invoke open), so the serving layer closes it
    # from unstreamed_arguments (the detector's partial-parse ledger), and
    # finish_streaming must not leak the raw markup.
    parser = FunctionCallParser(_tools(), "muse_glimmer")
    det = parser.detector
    truncated = (
        "Ordering. <|start|>assistant to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        '<atem:parameter name="days">4'
    )
    normal, calls = "", []
    for i in range(0, len(truncated), 11):
        r = det.parse_streaming_increment(truncated[i : i + 11], _tools())
        normal += r.normal_text
        calls.extend(r.calls)
    assert normal == "Ordering. "
    assert [c.name for c in calls if c.name] == ["weather.get"]
    # the completed parameter survives in the ledger; the truncated one is dropped
    assert json.loads(parser.unstreamed_arguments(0)) == {"city": "Paris"}
    assert det.finish_streaming() == ""


def test_unknown_tool_follows_forwarding_policy():
    import sparklab.serving.function_call_parser as fcp

    text = _tool_channel("nope.call", {"a": "1"})
    res = MuseGlimmerDetector().detect_and_parse(text, _tools())
    assert [c.name for c in res.calls] == ["nope.call"]  # default: forwarded

    orig = fcp.FORWARD_UNKNOWN_TOOLS
    try:
        fcp.FORWARD_UNKNOWN_TOOLS = False
        res = MuseGlimmerDetector().detect_and_parse(text, _tools())
        assert res.calls == []
    finally:
        fcp.FORWARD_UNKNOWN_TOOLS = orig


def test_empty_parameters_invoke():
    det = MuseGlimmerDetector()
    block = (
        "<|start|>assistant to=weather.get<|message|>"
        '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    res = det.detect_and_parse(block, _tools())
    assert json.loads(res.calls[0].parameters) == {}
    _, calls = _stream_detect(MuseGlimmerDetector(), block, _tools())
    assert _assemble(calls) == [("weather.get", {})]


# ---------------------------------------------------------------------------
# Review regressions (PR #4)
# ---------------------------------------------------------------------------
def test_channel_close_mid_invoke_does_not_corrupt_next_call():
    """HIGH-1: a channel closer arriving mid-invoke (literal <|eom|> inside a
    parameter value, or a truncated invoke) must finalize the open call --
    completed params kept, JSON closed, ordinal advanced -- so the NEXT
    channel's call never merges into it."""
    text = (
        "<|start|>assistant to=fs.write<|message|><atem:function_calls>\n"
        '<atem:invoke name="fs.write">\n'
        '<atem:parameter name="path">/tmp/t</atem:parameter>\n'
        '<atem:parameter name="content">stop token is <|eom|>'
        "<|start|>assistant to=weather.get<|message|>"
        + _atem("weather.get", {"city": "Paris"})
        + "<|eot|>"
    )
    for step in (3, 7, len(text)):  # chunked and single-chunk
        det = MuseGlimmerDetector()
        _, calls = _stream_detect(det, text, _tools(), step=step)
        assembled = _assemble(calls)
        assert [a[0] for a in assembled] == ["fs.write", "weather.get"], assembled
        # MED-2: the truncated call's streamed fragments still concatenate to
        # VALID JSON; the completed parameter survives, the cut one is closed.
        first_args = assembled[0][1]
        assert first_args["path"] == "/tmp/t"
        assert assembled[1][1] == {"city": "Paris"}
        named = [c.tool_index for c in calls if c.name is not None]
        assert named == [0, 1]  # distinct ordinals: no merge
        assert det.prev_tool_call_arr[0]["arguments"]["path"] == "/tmp/t"
        assert det.prev_tool_call_arr[1]["arguments"] == {"city": "Paris"}


def test_headerless_channel_switch_executes_the_call():
    """HIGH-2: the model may leave to=self without <|eom|>, writing
    to=<tool><|message|> directly; a complete match is a segment boundary
    (vLLM's rule) in both parsers."""
    text = (
        " to=self<|message|>I should check the weather. to=weather.get<|message|>"
        + _atem("weather.get", {"city": "Paris"})
        + "<|eot|>"
    )
    # reasoning parser: the switch ends reasoning and preserves the tool slice
    one = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert one.reasoning_text == "I should check the weather."
    assert one.normal_text.startswith("to=weather.get<|message|>")
    reasoning, content = _stream(MuseGlimmerReasoningParser(), text)
    assert reasoning.rstrip() == "I should check the weather."
    assert content.startswith("to=weather.get<|message|>")

    # detector on the preserved slice
    det = MuseGlimmerDetector()
    normal, calls = _stream_detect(det, content, _tools())
    assert _assemble(calls) == [("weather.get", {"city": "Paris"})]
    assert normal.strip() == ""

    # and end-to-end without a reasoning parser upstream
    det2 = MuseGlimmerDetector()
    normal2, calls2 = _stream_detect(det2, text, _tools())
    assert _assemble(calls2) == [("weather.get", {"city": "Paris"})]
    assert "I should check" not in normal2  # self body swallowed


def test_headerless_switch_to_user_streams_content():
    text = " to=self<|message|>quick thought to=user<|message|>Here you go.<|eot|>"
    r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "quick thought"
    assert r.normal_text == "Here you go."
    reasoning, content = _stream(MuseGlimmerReasoningParser(), text)
    assert reasoning.rstrip() == "quick thought"
    assert content == "Here you go."


def test_doubled_tool_name_collapses_to_registered_head():
    """MED-1: the template renders a bare-name tool's namespace as name.*, so the
    model emits get_weather.get_weather; collapse iff the head is registered."""
    tools = [
        Tool(function=Function(name="get_weather", parameters={
            "type": "object", "properties": {"city": {"type": "string"}},
        }))
    ]
    text = (
        "<|start|>assistant to=get_weather.get_weather<|message|>"
        + _atem("get_weather.get_weather", {"city": "Paris"})
        + "<|eot|>"
    )
    det = MuseGlimmerDetector()
    _, calls = _stream_detect(det, text, tools)
    assert _assemble(calls) == [("get_weather", {"city": "Paris"})]
    res = MuseGlimmerDetector().detect_and_parse(text, tools)
    assert [c.name for c in res.calls] == ["get_weather"]

    # a genuinely namespaced name ("weather.get") is never collapsed, and a
    # doubled form that IS registered stays as-is
    res2 = MuseGlimmerDetector().detect_and_parse(
        _tool_channel("weather.get", {"city": "Rome"}), _tools()
    )
    assert [c.name for c in res2.calls] == ["weather.get"]
    tools_doubled = tools + [
        Tool(function=Function(name="get_weather.get_weather", parameters={}))
    ]
    res3 = MuseGlimmerDetector().detect_and_parse(text, tools_doubled)
    assert [c.name for c in res3.calls] == ["get_weather.get_weather"]


def test_text_after_tool_channel_still_streams():
    """MED-3 companion: after a tool channel closes, following text streams; the
    detector must not stay latched in tool mode."""
    text = _tool_channel("weather.get", {"city": "Paris"}) + " All done."
    det = MuseGlimmerDetector()
    normal, calls = _stream_detect(det, text, _tools())
    assert _assemble(calls) == [("weather.get", {"city": "Paris"})]
    assert normal.strip() == "All done."


def test_one_shot_classifies_channels_without_reasoning_parser():
    """MED-5: with --reasoning-parser off, non-streaming parsing must still run
    the channel classification: to=self never leaks, to=user is kept."""
    text = (
        " to=self<|message|>hidden chain of thought<|eom|>"
        "<|start|>assistant to=user<|message|>Let me check.<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        + _atem("weather.get", {"city": "Paris"})
        + "<|eot|>"
    )
    res = MuseGlimmerDetector().detect_and_parse(text, _tools())
    assert "hidden chain of thought" not in res.normal_text
    assert res.normal_text == "Let me check."
    assert [c.name for c in res.calls] == ["weather.get"]
    assert json.loads(res.calls[0].parameters) == {"city": "Paris"}


def test_content_starting_with_header_lookalikes_streams():
    """MED-4: committing to "header" requires a full to=/assistant +
    <|message|> match; lookalike content must stream as text."""
    for text in (
        "assistant is a role name, not a header.",
        "to=whom it may concern: hello.",
    ):
        det = MuseGlimmerDetector()
        normal, calls = _stream_detect(det, text, _tools())
        assert calls == []
        assert normal == text
        p = MuseGlimmerReasoningParser()
        reasoning, content = _stream(p, text)
        assert reasoning == "" and content == text


def test_tool_slice_holds_partial_markers_while_streaming():
    """LOW-1: the verbatim tool-channel slice must hold back a partial marker;
    emitted content never shrinks when "<|start|" turns out to open the NEXT
    segment."""
    p = MuseGlimmerReasoningParser()
    block = _tool_channel("weather.get", {"city": "Paris"}, closer="")
    emitted = []
    out = p.parse_streaming_increment(block + "<|star")
    emitted.append(out.normal_text)
    assert not "".join(emitted).endswith("<|star")  # partial opener held back
    out = p.parse_streaming_increment("t|>assistant to=user<|message|>done<|eot|>")
    emitted.append(out.reasoning_text or "")
    content = "".join(e for e in emitted if e) + out.normal_text + "".join(p.flush().normal_text)
    assert "<|star" + "t|>" not in content.replace("<|start|>", "")  # no split debris


def test_truncated_tool_channel_warns(caplog):
    """LOW-5: a truncated tool channel is logged, not silently dropped."""
    import logging

    text = (
        "<|start|>assistant to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n<atem:parameter name="city">Par<|eot|>'
    )
    det = MuseGlimmerDetector()
    with caplog.at_level(logging.WARNING):
        _stream_detect(det, text, _tools())
    assert any("mid-invoke" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Review regressions (PR #4, round 2)
# ---------------------------------------------------------------------------
def _pipe(text, tools, step=3):
    """The serving pipeline shape: reasoning parser first, its content feeds the
    detector; returns (reasoning, detector_normal, assembled_calls)."""
    rp = MuseGlimmerReasoningParser()
    det = MuseGlimmerDetector()
    reasoning, normal, calls = "", [], []

    def feed(piece):
        if not piece:
            return
        r = det.parse_streaming_increment(piece, tools)
        normal.append(r.normal_text)
        calls.extend(r.calls)

    for i in range(0, len(text), step):
        r = rp.parse_streaming_increment(text[i : i + step])
        reasoning += r.reasoning_text
        feed(r.normal_text)
    r = rp.flush()
    reasoning += r.reasoning_text
    feed(r.normal_text)
    while True:
        r = det.parse_streaming_increment("", tools)
        if not r.normal_text and not r.calls:
            break
        normal.append(r.normal_text)
        calls.extend(r.calls)
    calls.extend(det.finalize_streaming())
    normal.append(det.finish_streaming())
    return reasoning, "".join(normal), calls


@pytest.mark.parametrize(
    "switch",
    [
        "<|start|>assistant to=user<|message|>",  # abutting header, no <|eom|>
        "to=user<|message|>",  # headerless switch
    ],
)
def test_tool_channel_without_terminator_keeps_rest_of_turn(switch):
    """R2 HIGH-1: a tool channel ending via an abutting header (no closer) must
    reach the detector as a delimited block -- the rest of the turn survives."""
    text = (
        " to=weather.get<|message|>"
        + _atem("weather.get", {"city": "Paris"})
        + switch
        + "visible answer<|eot|>"
    )
    reasoning, normal, calls = _pipe(text, _tools())
    assert _assemble(calls) == [("weather.get", {"city": "Paris"})]
    assert normal.strip() == "visible answer"
    assert reasoning == ""


def test_eos_mid_invoke_finalizes_with_valid_fragments():
    """R2 HIGH-2: max_tokens mid-invoke -- finalize_streaming emits the closing
    fragments so the client's concatenated argument JSON ends valid, and the
    parse ledger serves unstreamed_arguments (even for an empty-dict call)."""
    parser = FunctionCallParser(_tools(), "muse_glimmer")
    det = parser.detector
    truncated = (
        "<|start|>assistant to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n<atem:parameter name="city">Par'
    )
    frags = ""
    for i in range(0, len(truncated), 7):
        r = det.parse_streaming_increment(truncated[i : i + 7], _tools())
        frags += "".join(c.parameters or "" for c in r.calls)
    closing = parser.finalize_stream()
    frags += "".join(c.parameters or "" for c in closing)
    assert json.loads(frags) == {"city": "Par"}  # valid, truncated value closed
    assert json.loads(parser.unstreamed_arguments(0)) == {"city": "Par"}
    assert det.finish_streaming() == ""

    # empty-dict ledger entries serve too ({} used to be falsy-skipped)
    parser2 = FunctionCallParser(_tools(), "muse_glimmer")
    det2 = parser2.detector
    trunc2 = (
        "<|start|>assistant to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n'
    )
    for i in range(0, len(trunc2), 7):
        det2.parse_streaming_increment(trunc2[i : i + 7], _tools())
    frags2 = "".join(c.parameters or "" for c in parser2.finalize_stream())
    assert json.loads(frags2) == {}
    assert parser2.unstreamed_arguments(0) == "{}"


def test_one_shot_keeps_second_invoke_block_in_one_channel():
    """R2 HIGH-3: two <atem:function_calls> blocks inside ONE channel -- the
    one-shot path (a streaming replay over an undrained buffer) must parse both,
    not discard the second as markup debris at the channel close."""
    text = (
        "<|start|>assistant to=weather.get<|message|>"
        + _atem("weather.get", {"city": "Paris"})
        + "\n"
        + _atem("weather.get", {"city": "Rome"})
        + "<|eot|>"
    )
    res = MuseGlimmerDetector().detect_and_parse(text, _tools())
    assert [json.loads(c.parameters)["city"] for c in res.calls] == ["Paris", "Rome"]
    assert [c.tool_index for c in res.calls] == [0, 1]
    # streaming agrees
    _, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assert [a[1]["city"] for a in _assemble(calls)] == ["Paris", "Rome"]


def test_reasoning_strength_kwarg_does_not_swallow_the_thinking_toggle():
    """R2 HIGH-4 (post effort-unification): a muse-style reasoning_strength kwarg
    riding along must not disable the broadcast thinking toggle."""
    from sparklab.serving.model_meta import effort_toggle_kwargs

    out = effort_toggle_kwargs("high", {"reasoning_strength": "high"})
    assert out.get("enable_thinking") is True  # the toggle broadcast still maps
    assert out.get("reasoning_strength") == "high"  # the explicit kwarg rides along
    assert out.get("reasoning_effort") == "high"


def test_reasoning_streaming_buffer_stays_trimmed():
    """R2 MED-5: streaming must consume the buffer as it emits (the full-buffer
    re-scan per chunk was quadratic and stalled the serving loop on long CoT)."""
    p = MuseGlimmerReasoningParser()
    p.parse_streaming_increment(" to=self<|message|>")
    for _ in range(2000):
        p.parse_streaming_increment("think " * 4)
    assert len(p._buffer) < 256  # consumed, not accumulated


def test_flush_delivers_stray_closer_tail():
    """R2 MED-6(b): prose stranded after a literal closer is delivered at
    end-of-stream instead of dropped."""
    text = "The token <|eot|> ends a turn."
    p = MuseGlimmerReasoningParser()
    content = ""
    for i in range(0, len(text), 5):
        content += p.parse_streaming_increment(text[i : i + 5]).normal_text
    content += p.flush().normal_text
    assert "The token " in content and " ends a turn." in content


def test_flush_delivers_header_lookalike_reply():
    """R2 MED-6(a): a complete reply that merely looks like a bare-header prefix
    is content at end-of-stream, in both parsers."""
    for reply in ("assistant", "to=me@example.com"):
        p = MuseGlimmerReasoningParser()
        out = p.parse_streaming_increment(reply)
        got = out.normal_text + p.flush().normal_text
        assert got == reply
        det = MuseGlimmerDetector()
        r = det.parse_streaming_increment(reply, _tools())
        assert (r.normal_text + det.finish_streaming()) == reply


def test_channel_prose_drop_warns(caplog):
    """R2 small: discarded tool-channel prose leaves a log line."""
    import logging

    text = (
        "<|start|>assistant to=weather.get<|message|>calling now: "
        + _atem("weather.get", {"city": "Paris"})
        + "<|eot|>"
    )
    det = MuseGlimmerDetector()
    with caplog.at_level(logging.WARNING):
        _, calls = _stream_detect(det, text, _tools())
    assert _assemble(calls) == [("weather.get", {"city": "Paris"})]
    prose_warnings = [r for r in caplog.records if "inside a tool channel" in r.message]
    assert len(prose_warnings) == 1  # accumulated: ONE warning, not one per fragment


# ---------------------------------------------------------------------------
# Review regressions (PR #4, round 3)
# ---------------------------------------------------------------------------
def test_seek_mode_streams_eagerly_and_stays_trimmed():
    """R3 MED-1: text after a closer (a headerless continuation, e.g. a
    repetition loop) must stream live with a bounded buffer, not accumulate
    until flush."""
    p = MuseGlimmerReasoningParser()
    p.parse_streaming_increment(" to=user<|message|>answer<|eom|>")
    streamed = ""
    for _ in range(500):
        streamed += p.parse_streaming_increment("loop ").normal_text
    assert len(p._buffer) < 256  # consumed, not held until EOS
    assert len(streamed) > 2000  # and the client actually saw it live
    assert "loop loop" in streamed


def test_seek_mode_eos_around_incomplete_header():
    """R3 MED-2: at EOS, text before a complete <|start|> whose <|message|>
    never arrived is delivered; the discard is capped at a plausible header
    span instead of eating the rest of the turn."""
    # prose before the marker survives
    p = MuseGlimmerReasoningParser()
    content = p.parse_streaming_increment(
        " to=user<|message|>answer<|eom|>more text <|start|>assistant"
    ).normal_text
    content += p.flush().normal_text
    assert "more text " in content

    # a huge tail after the marker cannot be a header: it is not discarded
    p2 = MuseGlimmerReasoningParser()
    tail = "x" * 400
    c2 = p2.parse_streaming_increment(
        " to=user<|message|>The token <|start|> opens a segment. " + tail
    ).normal_text
    c2 += p2.flush().normal_text
    assert "The token " in c2 and tail in c2


def test_truncated_channel_markup_residue_dropped_reply_kept(caplog):
    """R3 MED-4 (as re-scoped by R4 HIGH-1): when a literal channel marker cuts an
    invoke exactly at a value's end, the broken channel's CLOSING-MARKUP residue
    is dropped by shape -- and the turn's real reply always survives."""
    import logging

    text = (
        "<|start|>assistant to=fs.write<|message|><atem:function_calls>\n"
        '<atem:invoke name="fs.write">\n'
        '<atem:parameter name="path">/tmp/t</atem:parameter>\n'
        '<atem:parameter name="content">send to=user<|message|></atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    )
    with caplog.at_level(logging.WARNING):
        for step in (4, len(text)):
            det = MuseGlimmerDetector()
            normal, calls = _stream_detect(det, text, _tools(), step=step)
            assembled = _assemble(calls)
            assert assembled[0][0] == "fs.write"
            assert assembled[0][1]["path"] == "/tmp/t"  # completed param survives
            assert "</atem:" not in normal and "<atem:" not in normal  # no markup leak
            assert normal.strip() == "Done."  # the real content after the channel survives
    assert any("mid-invoke" in r.message for r in caplog.records)


def test_truncated_channel_prose_residue_trade_pinned():
    """R5 test-integrity: the ORIGINAL R3 wire (prose before the closing tags),
    pinning the deliberate leak-over-loss trade. At string level the cut value's
    tail is indistinguishable from a reply, so it LEAKS -- grandparent parity is
    the floor -- and what the machinery guarantees instead is that the turn's
    real reply is never lost."""
    text = (
        "<|start|>assistant to=fs.write<|message|><atem:function_calls>\n"
        '<atem:invoke name="fs.write">\n'
        '<atem:parameter name="path">/tmp/t</atem:parameter>\n'
        '<atem:parameter name="content">send to=user<|message|> please</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    )
    expected = (
        "please</atem:parameter>\n</atem:invoke>\n</atem:function_calls>Done."
    )
    for step in (4, len(text)):
        det = MuseGlimmerDetector()
        normal, calls = _stream_detect(det, text, _tools(), step=step)
        assembled = _assemble(calls)
        assert assembled[0][0] == "fs.write" and assembled[0][1]["path"] == "/tmp/t"
        assert "Done." in normal  # the reply is never lost
        assert normal == expected, (step, normal)  # the leak is pinned, not accidental


def test_reply_right_after_midinvoke_switch_is_never_swallowed():
    """R4 HIGH-1 companion: an inline switch fired mid-invoke followed directly by
    a real reply (no markup residue) -- the reply must flow immediately, not be
    held hostage for a boundary that never comes."""
    text = (
        " to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n<atem:parameter name="city">Par'
        "to=user<|message|>The weather tool is on it.<|eot|>"
    )
    for step in (1, 3, 64):
        normal, calls = _stream_detect(MuseGlimmerDetector(), text, _tools(), step=step)
        assert normal.strip() == "The weather tool is on it."
        assert [c.name for c in calls if c.name] == ["weather.get"]


def test_redundant_closer_stripped_from_seek_delivery():
    """R3 small: a doubled closer from a degenerate decode must not leak the
    second token into content (the detector already filters; the reasoning
    parser's seek path now does too)."""
    text = " to=user<|message|>The answer is 42.<|eot|><|eot|>"
    p = MuseGlimmerReasoningParser()
    content = ""
    for i in range(0, len(text), 5):
        content += p.parse_streaming_increment(text[i : i + 5]).normal_text
    content += p.flush().normal_text
    assert content == "The answer is 42."
    one = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert one.normal_text == "The answer is 42."


def test_stray_closer_one_shot_matches_streaming():
    """R3 small: the one-shot passthrough shortcut must route closer-containing
    text through the replay, matching the streamed result."""
    text = "The token <|eot|> ends a turn."
    one = MuseGlimmerReasoningParser().detect_and_parse(text)
    p = MuseGlimmerReasoningParser()
    streamed = ""
    for i in range(0, len(text), 5):
        streamed += p.parse_streaming_increment(text[i : i + 5]).normal_text
    streamed += p.flush().normal_text
    assert one.normal_text == streamed.strip()
    assert "<|eot|>" not in one.normal_text


def test_mixed_quant_groups_rejected_in_any_order():
    """R3 small: a {nvfp4, mxfp4} mixed checkpoint must raise regardless of the
    groups' key order (the first-group short-circuit accepted one order)."""
    import tests.models.test_muse_glimmer as m

    for order in (("group_0", "group_1"), ("group_1", "group_0")):
        hf = m._hf_config(quantized=True)
        groups = hf.quantization_config["config_groups"]
        nvfp4 = groups["group_0"]
        mxfp4 = {"weights": {"num_bits": 4, "type": "float", "group_size": 32, "strategy": "group"}}
        hf.quantization_config["config_groups"] = {
            order[0]: nvfp4 if order[0] == "group_0" else mxfp4,
            order[1]: mxfp4 if order[1] == "group_1" else nvfp4,
        }
        from sparklab.models.muse_glimmer.config import parse_config as pc

        with pytest.raises(ValueError, match="unsupported compressed-tensors"):
            pc(hf)


# ---------------------------------------------------------------------------
# Review regressions (PR #4, round 4)
# ---------------------------------------------------------------------------
def test_pipeline_delivers_reply_after_truncated_invoke():
    """R4 HIGH-1, PRODUCTION ORDER (reasoning parser first, its content feeds the
    detector -- the shape standalone-detector tests missed twice): a truncated
    invoke must never swallow the rest of the turn's reply."""
    # reviewer wire 1: literal marker cuts the value, closing tags follow
    wire1 = (
        "<|start|>assistant to=fs.write<|message|><atem:function_calls>\n"
        '<atem:invoke name="fs.write">\n'
        '<atem:parameter name="path">/tmp/t</atem:parameter>\n'
        '<atem:parameter name="content">send to=user<|message|></atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    )
    # reviewer wire 2: invoke left unclosed before an abutting user header
    wire2 = (
        " to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n<atem:parameter name="city">Par'
        "<|start|>assistant to=user<|message|>The weather tool is on it.<|eot|>"
    )
    for wire, reply, first_call in (
        (wire1, "Done.", "fs.write"),
        (wire2, "The weather tool is on it.", "weather.get"),
    ):
        for step in (1, 3, 64):
            _, normal, calls = _pipe(wire, _tools(), step=step)
            assert reply in normal, (wire[:40], step, normal)
            assert "<atem:" not in normal and "</atem:" not in normal
            named = [c.name for c in calls if c.name]
            assert named and named[0] == first_call


def test_stray_start_before_a_real_header_does_not_form_a_giant_header():
    """R4 MED-2: the span bound applies even when a <|message|> is already in the
    buffer -- a literal <|start|> + junk + the NEXT segment's header must not be
    parsed as one giant header (which also let a to= inside the junk hijack the
    recipient), and streaming must agree with one-shot."""
    junk = "P" * 300
    text = (
        " to=user<|message|>ok<|eom|><|start|>" + junk
        + "<|start|>assistant to=user<|message|>more<|eot|>"
    )
    p = MuseGlimmerReasoningParser()
    streamed_r, streamed_c = "", ""
    for i in range(0, len(text), 5):
        r = p.parse_streaming_increment(text[i : i + 5])
        streamed_r += r.reasoning_text
        streamed_c += r.normal_text
    fl = p.flush()
    streamed_r += fl.reasoning_text
    streamed_c += fl.normal_text
    one = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert "ok" in streamed_c and "more" in streamed_c
    assert junk in streamed_c  # delivered, not silently swallowed
    assert one.normal_text == streamed_c.strip()

    # a to=self inside the junk must not hijack the following user reply
    hijack = (
        " to=user<|message|>first.<|eom|><|start|>x to=self y" + "Q" * 250
        + "<|start|>assistant to=user<|message|>the actual answer<|eot|>"
    )
    one = MuseGlimmerReasoningParser().detect_and_parse(hijack)
    assert "the actual answer" in one.normal_text
    assert "the actual answer" not in one.reasoning_text


@pytest.mark.parametrize("name_len", [64, 100, 110])
def test_protocol_legal_long_tool_names_survive_streaming(name_len):
    """R4 MED-3: headers stay parseable up to the span bound even when their
    <|message|> is still mid-arrival; the tool call AND the trailing reply
    survive at every step size, matching one-shot."""
    name = "t" * name_len
    tools = [Tool(function=Function(name=name, parameters={
        "type": "object", "properties": {"city": {"type": "string"}},
    }))]
    text = (
        f" to=self<|message|>go<|eom|><|start|>assistant to={name}<|message|>"
        + _atem(name, {"city": "Paris"})
        + "<|eot|><|start|>assistant to=user<|message|>done<|eot|>"
    )
    one_r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert "done" in one_r.normal_text
    for step in (1, 5, 64):
        reasoning, normal, calls = _pipe(text, tools, step=step)
        assert reasoning.strip() == "go"
        assert normal.strip() == "done", (name_len, step, normal)
        named = [c.name for c in calls if c.name]
        assert named and named[0].startswith("t" * 64)


def test_eos_start_candidate_delivered_across_layers():
    """R4 item-4 / R5 root-cause fix: at EOS a <|start|> that never received its
    <|message|> is NOT a header -- BOTH layers deliver the tail and drop only
    the marker, so the raw-bytes vs closer-stripped span mismatch between the
    layers has nothing left to disagree about."""
    # R5 MED-2 PoC (ordinary terminators only): the pipeline must deliver
    # everything the parser alone delivers.
    prose = "p" * 119
    wire = " to=user<|message|>The wire format is: <|start|>" + prose + "<|eot|><|end_of_text|>"
    parser_only = ""
    p = MuseGlimmerReasoningParser()
    for i in range(0, len(wire), 7):
        parser_only += p.parse_streaming_increment(wire[i : i + 7]).normal_text
    parser_only += p.flush().normal_text
    assert prose in parser_only
    for step in (1, 7, 4096):
        _, piped, _ = _pipe(wire, _tools(), step=step)
        assert "The wire format is: " in piped
        assert prose in piped, (step, piped)  # pipeline == parser, chars not swallowed

    # R4 reviewer wire: the released marker's tail survives the pipeline too
    wire2 = " to=user<|message|>answer<|eot|>prose <|start|>" + "x" * 50 + "<|eom|>" * 15
    for step in (1, 7):
        _, piped2, _ = _pipe(wire2, _tools(), step=step)
        assert "answer" in piped2 and "prose " in piped2
        assert "x" * 50 in piped2, (step, piped2)


@pytest.mark.parametrize("junk_len", [103, 115, 127, 140])
def test_reply_after_stray_start_junk_segment_survives(junk_len):
    """R5 HIGH-1: a real to=user reply following a stray <|start|>+junk segment
    must never be dropped -- previously the released marker + junk + reply sat
    under the detector's hold threshold and died in finish_streaming."""
    wire = (
        " to=user<|message|>ok<|eom|><|start|>" + "J" * junk_len
        + "<|start|>assistant to=user<|message|>Here you go.<|eot|>"
    )
    for step in (1, 7, 4096):
        _, piped, _ = _pipe(wire, _tools(), step=step)
        assert "Here you go." in piped, (junk_len, step, piped)
        assert "ok" in piped
    # one-shot agrees
    rp = MuseGlimmerReasoningParser().detect_and_parse(wire)
    one = MuseGlimmerDetector().detect_and_parse(rp.normal_text, _tools())
    assert "Here you go." in one.normal_text


def test_detector_giant_header_bound_pinned():
    """R5 test-integrity: the detector's own msg-too-far bound. A stray
    <|start|> + junk ahead of a real tool channel in ONE detector buffer must
    not merge into a giant header. NOTE which assertion is load-bearing: with
    the bound deleted the tool call still executes correctly -- what breaks is
    the ~300 chars of user content silently vanishing into the giant header, so
    it is the junk/content assertion below that pins the bound, not the calls
    assertion."""
    junk = "J" * 300
    text = (
        "ok<|eom|><|start|>" + junk
        + _tool_channel("weather.get", {"city": "P"})
        + "<|start|>assistant to=user<|message|>done<|eot|>"
    )
    for step in (7, len(text)):
        det = MuseGlimmerDetector()
        normal, calls = _stream_detect(det, text, _tools(), step=step)
        assert _assemble(calls) == [("weather.get", {"city": "P"})], (step,)
        assert "done" in normal and junk in normal


# ---------------------------------------------------------------------------
# Review regressions (PR #4, round 6): turn-start state is read from the
# prompt (header-open), not guessed from the first bytes. The bare-header
# lookalike family -- shapes the old prefix-guessing blanked to an empty turn.
# ---------------------------------------------------------------------------
def test_junk_recipient_past_cap_yields_empty_channel_not_empty_turn():
    """``to=`` + 70 chars + <|message|>: the recipient (capped at 64) is a junk
    tool name, so the channel body is not user-visible -- but the turn is NOT
    blanked: a following real reply flows."""
    wire = (
        "to=" + "j" * 70 + "<|message|>hidden<|eot|>"
        "<|start|>assistant to=user<|message|>Real reply.<|eot|>"
    )
    for step in (1, 7, 4096):
        _, piped, calls = _pipe(wire, _tools(), step=step)
        assert "Real reply." in piped, (step, piped)
        assert "hidden" not in piped  # the junk channel is what the model said
    rp = MuseGlimmerReasoningParser().detect_and_parse(wire)
    one = MuseGlimmerDetector().detect_and_parse(rp.normal_text, _tools())
    assert "Real reply." in one.normal_text and "hidden" not in one.normal_text


def test_glued_assistant_recipient_header_parses():
    """``assistantto=x<|message|>``: the recipient regex still finds to=x, so
    the segment routes as a (junk) tool channel and the next reply flows."""
    wire = (
        "assistantto=x<|message|>hello<|eot|>"
        "<|start|>assistant to=user<|message|>Hi.<|eot|>"
    )
    for step in (1, 7, 4096):
        _, piped, _ = _pipe(wire, _tools(), step=step)
        assert "Hi." in piped, (step, piped)
        assert "hello" not in piped


def test_deep_leading_whitespace_header_delivers_reply():
    """>8 leading newlines before ``to=user<|message|>``: the old bare-header
    regex allowed at most 8 whitespace chars while the undecided hold lstripped
    unbounded, blanking the turn; the full-header machinery has no such split."""
    wire = "\n" * 9 + "to=user<|message|>Hello!<|eot|>"
    for step in (1, 7, 4096):
        _, piped, _ = _pipe(wire, _tools(), step=step)
        assert "Hello!" in piped, (step, piped)
    rp = MuseGlimmerReasoningParser().detect_and_parse(wire)
    assert "Hello!" in rp.normal_text


def test_prose_before_first_full_header_not_swallowed_into_seed():
    """The header-open seed must not merge turn-start prose with the FIRST real
    header into one giant header: a control token inside a header candidate
    voids the candidate (headers never contain markers). Without the parser's
    marker-inside rule, 'Pre. ' is swallowed into the seed's header -- here the
    header routes to=self, so the swallowed prefix vanishes entirely (a tool
    channel would carry it verbatim and mask the loss downstream)."""
    wire = (
        "Pre. <|start|>assistant to=self<|message|>think<|eom|>"
        "<|start|>assistant to=user<|message|>Post.<|eot|>"
    )
    for step in (1, 7, 4096):
        reasoning, piped, _ = _pipe(wire, _tools(), step=step)
        assert "Pre. " in piped and "Post." in piped, (step, piped)
        assert "think" in reasoning and "Pre" not in reasoning
    r = MuseGlimmerReasoningParser().detect_and_parse(wire)
    assert "Pre." in r.normal_text and "think" in r.reasoning_text
    # and with a tool channel: the call still parses, the prose still flows
    wire2 = (
        "Pre. " + _tool_channel("weather.get", {"city": "P"})
        + "<|start|>assistant to=user<|message|>Post.<|eot|>"
    )
    for step in (1, 7, 4096):
        _, piped2, calls2 = _pipe(wire2, _tools(), step=step)
        assert _assemble(calls2) == [("weather.get", {"city": "P"})], (step,)
        assert "Pre. " in piped2 and "Post." in piped2, (step, piped2)


def test_headerless_turn_delivers_and_seed_never_leaks():
    """Degenerate output that never emits <|message|>: the header-open seed is
    ruled a non-header and only the model's own bytes are delivered -- the
    synthetic <|start|> must not leak into content at any length."""
    for text in ("Just prose, no protocol.", "x" * 200):
        for step in (1, 7, 4096):
            _, piped, _ = _pipe(text, _tools(), step=step)
            assert piped == text, (len(text), step, piped)
            assert ATEM_START not in piped
        rp = MuseGlimmerReasoningParser().detect_and_parse(text)
        assert rp.normal_text == text


_CORPUS_REPLY = "All good."
# Historical degenerate shapes x chunkings: streaming, one-shot and the production
# pipeline must agree on the reply and the calls (the regression class that bit
# this PR twice was "standalone green, pipeline broken").
_WIRE_CORPUS = [
    # clean turn
    " to=self<|message|>think<|eom|><|start|>assistant to=user<|message|>All good.<|eot|>",
    # tool channel then reply
    " to=self<|message|>t<|eom|>" + _tool_channel("weather.get", {"city": "P"}, closer="<|eom|>")
    + "<|start|>assistant to=user<|message|>All good.<|eot|>",
    # headerless switches everywhere
    " to=self<|message|>t to=weather.get<|message|>"
    + _atem("weather.get", {"city": "P"}) + "to=user<|message|>All good.<|eot|>",
    # truncated invoke then abutting user header
    " to=weather.get<|message|><atem:function_calls>\n"
    '<atem:invoke name="weather.get">\n<atem:parameter name="city">P'
    "<|start|>assistant to=user<|message|>All good.<|eot|>",
    # stray literal closer inside the reply (redundant <|eom|> mid-sentence)
    " to=user<|message|>All <|eom|>good.<|eot|>",
    # plain clean reply
    " to=user<|message|>All good.<|eot|>",
    # bare first segment is the reply
    " to=user<|message|>All good.<|eom|>",
    # stray <|start|>+junk segment ahead of the real reply (R5 HIGH-1 shape)
    " to=user<|message|>prelude<|eom|><|start|>" + "J" * 115
    + "<|start|>assistant to=user<|message|>All good.<|eot|>",
    # stream dies inside an unfinished header candidate (R5 EOS-family shape)
    " to=user<|message|>All good.<|eot|><|start|>" + "p" * 119 + "<|end_of_text|>",
    # truncated invoke leaving real closing tags before the reply (exercises
    # the </atem: leak clause below)
    " to=fs.write<|message|><atem:function_calls>\n"
    '<atem:invoke name="fs.write">\n'
    '<atem:parameter name="path">/tmp/t</atem:parameter>\n'
    '<atem:parameter name="content">send to=user<|message|></atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eot|>"
    "<|start|>assistant to=user<|message|>All good.<|eot|>",
]


@pytest.mark.parametrize("wire_idx", range(len(_WIRE_CORPUS)))
@pytest.mark.parametrize("step", [1, 7, 4096])
def test_wire_corpus_pipeline_consistency(wire_idx, step):
    wire = _WIRE_CORPUS[wire_idx]
    reasoning, normal, calls = _pipe(wire, _tools(), step=step)
    assert _CORPUS_REPLY in normal, (wire_idx, step, normal)
    assert "<atem:" not in normal and "</atem:" not in normal
    assert "<|" not in normal.replace("<|start|>", "")
    # one-shot detector agrees on calls AND arguments for the same classified content
    rp = MuseGlimmerReasoningParser().detect_and_parse(wire)
    one = MuseGlimmerDetector().detect_and_parse(rp.normal_text, _tools())
    assert [(c.name, json.loads(c.parameters)) for c in one.calls] == _assemble(calls)
    # the one-shot DETECTOR's content must carry the reply itself (the parser's
    # content always does, so an or-check here would never fail)
    assert _CORPUS_REPLY in one.normal_text, (wire_idx, one.normal_text)


# ---------------------------------------------------------------------------
# Wiring: auto-selection and thinking gears
# ---------------------------------------------------------------------------
def test_auto_selection_picks_muse_glimmer():
    from unittest.mock import patch

    from sparklab.serving.args import parse_args

    class _Config:
        def to_dict(self):
            return {
                "architectures": ["MuseGlimmerForConditionalGeneration"],
                "model_type": "muse_glimmer",
                "text_config": {"model_type": "muse_glimmer_text"},
            }

    with patch("sparklab.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", "/models/anon"])
    assert args.tool_call_parser == "muse_glimmer"
    assert args.reasoning_parser == "muse_glimmer"


class _MuseLikeTokenizer:
    """The muse template's effort surface: grades ``reasoning_strength`` (default
    high), validates nothing, ignores the thinking-toggle spellings."""

    chat_template = "muse"  # non-empty: the dsv4 encoder probe must not engage
    name_or_path = "muse-like"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        strength = kwargs.get("reasoning_strength") or "high"
        return f"<|begin_of_text|>system: Reasoning strength: {strength}. user: ping"

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        import torch

        assert add_special_tokens is False  # the template owns the bos
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


def test_effort_broadcast_reaches_reasoning_strength():
    """The render layer broadcasts reasoning_effort in every spelling the
    ecosystem's templates read (the thinking-toggle rule); muse's template picks
    up ``reasoning_strength``. An explicit caller spelling wins."""
    from sparklab.tokenizer.tokenize import TokenizeManager

    manager = TokenizeManager(_MuseLikeTokenizer())
    prompt = manager._render([{"role": "user", "content": "hi"}], None, {"reasoning_effort": "low"})
    assert "Reasoning strength: low." in prompt
    prompt = manager._render(
        [{"role": "user", "content": "hi"}],
        None,
        {"reasoning_effort": "low", "reasoning_strength": "xhigh"},
    )
    assert "Reasoning strength: xhigh." in prompt


def test_probed_muse_gears_and_effort_vocabulary():
    """The checkpoint-probing pipeline derives muse's gears through the broadcast:
    the template grades effort without validating it, so it is served the graded
    ladder (the model card's trained levels, xhigh included) with the template's
    own default (high); never-advertised dialects quantize onto the ladder
    instead of reaching the model verbatim."""
    from sparklab.serving.model_meta import derive_think_gears
    from sparklab.tokenizer.effort import effective_efforts, quantize_effort
    from sparklab.tokenizer.tokenize import TokenizeManager

    manager = TokenizeManager(_MuseLikeTokenizer())
    profile = manager.thinking_profile()
    assert profile.efforts.consumes_effort and not profile.efforts.validates
    assert profile.efforts.default == "high"
    assert not profile.toggleable  # the template reads no on/off spelling

    gears, default, kwargs = derive_think_gears(profile, parser_configured=True)
    assert gears == ("low", "medium", "high", "xhigh") and default == "high"
    assert kwargs["low"] == {"reasoning_effort": "low"}
    assert effective_efforts(profile.efforts) == frozenset({"low", "medium", "high", "xhigh"})
    # the native vocabulary passes through untouched...
    assert quantize_effort("xhigh", profile.efforts) == "xhigh"
    # ...while off-ladder dialects quantize instead of interpolating verbatim
    assert quantize_effort("minimal", profile.efforts) == "low"
    assert quantize_effort("max", profile.efforts) == "xhigh"
    assert quantize_effort("none", profile.efforts) is None  # template default applies
