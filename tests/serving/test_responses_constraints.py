"""Unsupported constraints must fail before any request is admitted."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sparklab.serving import responses_api as RP
from tests.serving.test_responses_api import FakeState

BAD_TEXT = [
    {"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}}},
    {"format": {"type": "json_object"}},
    {"format": {"type": "regex", "pattern": "[0-9]+"}},
    {"format": "json_schema"},
    {"format": []},
    {"format": {"schema": {"type": "object"}}},
    "json_schema",
    [],
]
GOOD_TEXT = [None, {}, {"verbosity": "low"}, {"format": {"type": "text"}}]


def client_for(state):
    app = FastAPI()
    RP.register_responses_routes(app, lambda: state, lambda: {})
    return TestClient(app)


@pytest.mark.parametrize("text", BAD_TEXT)
@pytest.mark.parametrize("stream", [False, True])
def test_route_rejects_constraint_before_admission(text, stream):
    state = FakeState([("unconstrained", True, 5, 1)])
    with client_for(state) as client:
        response = client.post("/v1/responses", json={"model": "test", "input": "answer", "text": text, "stream": stream})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error" and "text" in error["message"]
    assert state._uid == 0 and state.last_sent is None
    assert "response.created" not in response.text


@pytest.mark.parametrize("text", BAD_TEXT)
def test_conversion_rejects_constraint(text):
    request = RP.ResponsesRequest(model="test", input="answer", text=text)
    with pytest.raises(ValueError, match="text"):
        RP.convert_responses_to_genspec(request, {})


@pytest.mark.parametrize("text", GOOD_TEXT)
@pytest.mark.parametrize("stream", [False, True])
def test_plain_text_still_admits(text, stream):
    state = FakeState([("plain answer", True, 5, 1)])
    with client_for(state) as client:
        response = client.post("/v1/responses", json={"model": "test", "input": "answer", "text": text, "stream": stream})
    assert response.status_code == 200
    assert state._uid == 1 and state.last_sent is not None
    assert "plain answer" in response.text
