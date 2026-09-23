from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from sparklab.backends import BackendError, RuntimeRequest, get_backend
from sparklab.catalog import get_recipe


def test_container_plan_and_port_validation(tmp_path, monkeypatch):
    recipe = get_recipe("djev")
    backend = get_backend("djev")
    monkeypatch.setattr(backend, "validate_artifact", lambda *_: None)
    request = RuntimeRequest(recipe.slug, recipe.recipe_version, recipe.model, tmp_path, recipe.deployment)
    plan = backend.build_launch_plan(replace(request, extra_args=("--port", "9011")))
    assert "127.0.0.1:9011:8011" in plan.command
    assert "--network" not in plan.command
    assert "--memory-swap" in plan.command and "64g" in plan.command
    assert f"{tmp_path}:/model:ro" in plan.command
    assert plan.capabilities == ("jev-systemone", "structured-chat")
    for extra in (("--host", "0.0.0.0"), ("--port", "0"), ("--port", "abc"),
                  ("--model", "other"), ("--trust-remote-code",)):
        with pytest.raises(BackendError):
            backend.build_launch_plan(replace(request, extra_args=extra))


@pytest.mark.parametrize("key,value", [("canvas_length", 17), ("max_num_seqs", True),
                                     ("image", "other:latest"), ("memory_gib", 0)])
def test_invalid_runtime_settings_rejected(key, value):
    deployment = get_recipe("djev").deployment
    with pytest.raises(BackendError):
        get_backend("djev").validate_deployment(replace(
            deployment, backend_options={**deployment.backend_options, key: value}))


def test_checkpoint_rejects_wrong_model_and_missing_shards(tmp_path, monkeypatch):
    from sparklab.backends import djev
    backend = get_backend("djev")
    deployment = get_recipe("djev").deployment
    (tmp_path / "config.json").write_text('{}')
    assert not backend.accepts_artifact(tmp_path, deployment)
    monkeypatch.setattr(djev, "CONFIG_SHA256", hashlib.sha256(b'{}').hexdigest())
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"):
        (tmp_path / name).write_text('{}')
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "missing.safetensors"}}))
    with pytest.raises(BackendError, match="missing shard"):
        backend.validate_artifact(tmp_path, deployment)


def test_supervisor_flags_and_failed_startup():
    from sparklab.external.djev.supervisor import commands, wait_ready
    args = SimpleNamespace(model="/model", served_model_name="test", canvas_length=32,
                           max_num_seqs=32, max_model_len=8192, kv_cache_memory_bytes=2 << 30)
    engine, gateway = commands(args)
    assert engine[engine.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert "--async-scheduling" in engine
    assert "--kv-cache-memory-bytes" in engine
    assert "sparklab.external.djev.gateway" in gateway
    with pytest.raises(RuntimeError, match="exited"):
        wait_ready(SimpleNamespace(poll=lambda: 1, returncode=1), "http://unused")


def test_supervisor_stops_child_process_group():
    from sparklab.external.djev.supervisor import stop_processes
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    stop_processes([child])
    assert child.poll() is not None


@pytest.fixture
def gateway(monkeypatch):
    from sparklab.external.djev import upstream
    from sparklab.external.djev.gateway import Handler
    monkeypatch.setattr(upstream, "ARGS", SimpleNamespace(upstream="http://127.0.0.1:1", model="test"))
    monkeypatch.setattr(upstream, "API_KEY", "")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def test_gateway_health_follows_upstream_and_lists_model(gateway):
    assert request(gateway + "/health")[0] == 503
    assert request(gateway + "/v1/models")[1]["data"][0]["id"] == "test"


@pytest.mark.parametrize("body,status", [([], 400), ({"seed": "bad"}, 400),
    ({"model": "wrong"}, 400), ({"questions": {}}, 422),
    ({"questions": {"q": {"type": "choice", "criteria": ["bad"]}}}, 422)])
def test_bad_requests_return_json_errors(gateway, body, status):
    code, result = request(gateway + "/v1/systemone", body)
    assert code == status
    assert "error" in result


def test_concurrent_gateway_requests_keep_state(gateway, monkeypatch):
    from sparklab.external.djev import upstream
    def decide(schema, state, seed):
        qid = schema["questions"][0]["id"]
        return ({"answers": {qid: {"type": "noul", "noul": float(seed % 2), "label": "yes"}},
                 "diagnostics": {"prompt_tokens": 1, "timing": {"reads": 1, "total_ms": 1}}}, 1)
    monkeypatch.setattr(upstream, "decide", decide)
    def run(i):
        code, body = request(gateway + "/v1/systemone", {"seed": i, "state": {"id": i},
            "questions": {f"q{i}": {"type": "noul"}}})
        assert code == 200
        assert body["answers"] == {f"q{i}": {"type": "noul", "noul": float(i % 2)}}
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(32)))


def test_upstream_source_is_exactly_pinned():
    from sparklab.external.djev import upstream
    assert hashlib.sha256(Path(upstream.__file__).read_text().replace(
        'import base64  # SparkLab: standard library; no optional pybase64 dependency.',
        'import pybase64 as base64').encode()).hexdigest() == (
        "7cd9aa0081090c064eaac28db0f54f812749eeb3ae787d7f7653d2e35d8a938f")


def test_template_requires_single_token_and_pins_fixed_positions(monkeypatch):
    from sparklab.external.djev import upstream
    monkeypatch.setattr(upstream, "CANVAS_LEN", 32)
    monkeypatch.setattr(upstream, "CANVAS_STEP", 16)
    # A character tokenizer deliberately makes 'yes'/'no' invalid as one slot.
    monkeypatch.setattr(upstream, "TOK", SimpleNamespace(encode=lambda s, **_: list(s.encode())))
    with pytest.raises(upstream.SchemaError, match="single token"):
        upstream.resolve_template([{"id": "x", "labels": ["yes", "no"]}], [], "", "lines")
    template, slots = upstream.resolve_template([{"id": "x", "labels": ["A", "B"]}], [], "", "lines")
    pinned = upstream.pin_xargs(template, slots, 2)["diffusion_pinned"]
    assert slots[0]["pos"] not in pinned
    assert len(pinned) == 15
    assert upstream.build_canvas(template, slots, 1) != upstream.build_canvas(template, slots, 2)


def test_dependency_cycles_and_invalid_alternatives_rejected():
    from sparklab.external.djev import upstream
    for body in ({"questions": {"x": {"type": "choice", "criteria": {"only": "one"}}}},
                 {"questions": {"x": {"type": "noul", "depends_on": ["y"]},
                                "y": {"type": "noul", "depends_on": ["x"]}}}):
        with pytest.raises(upstream.SchemaError):
            upstream.jev_schema(body)


def test_images_preserve_order_and_missing_state_is_rejected():
    from sparklab.external.djev import upstream
    images = upstream.jev_images(["data:image/png;base64,YQ==", "data:image/png;base64,Yg=="])
    content = upstream.jev_state({"state": {"ticket": "inspect"}}, images)
    assert content[:2] == images
    assert content[2] == {"type": "text", "text": '{"ticket": "inspect"}'}
    with pytest.raises(upstream.SchemaError, match="state"):
        upstream.jev_state({})
    with pytest.raises(upstream.SchemaError, match="images"):
        upstream.jev_images(["https://example.com/image.png"])


def test_djev_acquisition_uses_pinned_source_and_rejects_prepare(tmp_path):
    from sparklab.acquire import AcquisitionError, acquire_recipe
    recipe = get_recipe("djev")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        p = Path(kwargs["local_dir"])
        (p / "config.json").write_text('{}')
        return str(p)
    result = acquire_recipe(recipe, root=str(tmp_path), downloader=download)
    assert calls[0]["repo_id"] == recipe.model
    assert calls[0]["revision"] == recipe.revision
    assert result["manifest"]["deployment"]["backend"] == "djev"
    with pytest.raises(AcquisitionError, match="omit --prepare"):
        acquire_recipe(recipe, root=str(tmp_path), prepare=True, downloader=download)


def test_health_recovers_when_upstream_is_ready(gateway, monkeypatch):
    from http.server import BaseHTTPRequestHandler
    from sparklab.external.djev import upstream
    class Ready(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Ready)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(upstream.ARGS, 'upstream', f'http://127.0.0.1:{server.server_port}')
    try:
        code, body = request(gateway + '/health')
        assert code == 200 and body['status'] == 'ok' and body['model'] == 'test'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert request(gateway + '/health')[0] == 503


def test_large_requests_rejected_before_body_read(gateway):
    req = urllib.request.Request(gateway + '/v1/systemone', data=b'{}',
        headers={'content-type':'application/json', 'content-length': str(17 * 1024 * 1024)})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req, timeout=5)
    assert error.value.code == 413
