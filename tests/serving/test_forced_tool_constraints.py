"""Forced constraints must be rejected before admission or opening SSE."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sparklab.serving import openai_api as O, responses_api as R, anthropic_api as A
from tests.serving.test_openai_api import FakeState as OS
from tests.serving.test_responses_api import FakeState as RS
from tests.serving.test_anthropic_api import FakeState as AS

TOOL = {"type":"function", "function":{"name":"lookup_fact", "description":"Lookup a fact",
    "parameters":{"type":"object", "properties":{"fact":{"type":"string"}}, "required":["fact"]}}}
CASES = [
    ("openai", "required"),
    ("openai", {"type":"function", "function":{"name":"lookup_fact"}}),
    ("responses", "required"),
    ("responses", {"type":"function", "name":"lookup_fact"}),
    ("responses", {"type":"allowed_tools", "mode":"required", "tools":[{"type":"function", "name":"lookup_fact"}]}),
    ("anthropic", {"type":"any"}),
    ("anthropic", {"type":"tool", "name":"lookup_fact"}),
]


def setup(api, choice, stream=False):
    payload = {"model":"test", "tool_choice":choice, "stream":stream}
    if api=="openai":
        payload.update(messages=[{"role":"user","content":"lookup fact k0"}], tools=[TOOL], max_tokens=32)
        return O, OS([]), O.ChatCompletionRequest, O.chat_request_to_genspec, O.register_openai_routes, "/v1/chat/completions", payload
    if api=="responses":
        payload.update(input="lookup fact k0", tools=[{"type":"function", **TOOL["function"]}],max_output_tokens=32)
        return R, RS([]), R.ResponsesRequest, R.convert_responses_to_genspec, R.register_responses_routes, "/v1/responses", payload
    payload.update(messages=[{"role":"user","content":"lookup fact k0"}],tools=[{
        "name":"lookup_fact", "description":"Lookup a fact", "input_schema":TOOL["function"]["parameters"]}],max_tokens=32)
    return A, AS([]), A.AnthropicMessagesRequest, A.convert_anthropic_to_genspec, A.register_anthropic_routes, "/v1/messages", payload


@pytest.mark.parametrize("api,choice", CASES)
@pytest.mark.parametrize("stream", [False,True])
def test_forced_request_rejected_before_admission_and_sse(api,choice,stream):
    module,state,model,convert,register,path,payload=setup(api,choice,stream)
    admitted=[]
    def fail():
        admitted.append(True)
        raise AssertionError("request admitted before rejecting unsupported constraint")
    state.new_user=fail
    app=FastAPI()
    register(app,lambda:state,lambda:{})
    with TestClient(app) as client:
        response=client.post(path,json=payload)
    assert response.status_code==400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"]=="invalid_request_error"
    assert "tool_choice" in response.json()["error"]["message"]
    assert not admitted
    assert "data:" not in response.text


@pytest.mark.parametrize("api,choice", CASES)
def test_direct_conversion_cannot_bypass_constraint_check(api,choice):
    _,_,model,convert,_,_,payload=setup(api,choice)
    with pytest.raises(ValueError,match="tool_choice"):
        convert(model.model_validate(payload),{})


@pytest.mark.parametrize("api", ["openai","responses","anthropic"])
@pytest.mark.parametrize("choice", [None,"auto","none"])
def test_auto_none_and_omitted_still_convert(api,choice):
    if api=="anthropic" and choice is not None:
        choice={"type":choice}
    _,_,model,convert,_,_,payload=setup(api,choice)
    spec=convert(model.model_validate(payload),{})
    assert bool(spec.template_tools)==(choice!="none" and choice!={"type":"none"})
