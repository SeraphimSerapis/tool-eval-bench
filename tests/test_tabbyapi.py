"""TabbyAPI identity wins over the llama.cpp discovery format (issue #214)."""

from __future__ import annotations

import httpx
import pytest

from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.legacy_parser import _make_parser
from tool_eval_bench.cli.server import detect_backend_from_response
from tool_eval_bench.schema import ARGS_SCHEMA
from tool_eval_bench.utils import metadata

# Shapes from theroyallab/tabbyAPI's endpoints/core/router.py and types/model.py.
MODELS = {"data": [{"id": "gpt-3.5-turbo", "owned_by": "tabbyAPI"}]}
SERVICE = {"version": 0.1, "software": {"name": "TabbyAPI"}}
MODEL = {
    "id": "loaded-model-exl3",
    "owned_by": "tabbyAPI",
    "parameters": {"max_seq_len": 32768, "max_batch_size": 4, "cache_mode": "Q8"},
}
PROPS = {"default_generation_settings": {"n_ctx": 16384}, "total_slots": 2}


def install_transport(monkeypatch, respond):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw),
    )


@pytest.mark.parametrize("base_url", ["http://test", "http://test/v1"])
@pytest.mark.parametrize("backend", ["tabbyapi", "unknown", "llamacpp"])
async def test_owner_precedes_compatible_props(monkeypatch, base_url, backend):
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        body = {"/v1/models": MODELS, "/v1/model": MODEL, "/props": PROPS}.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    install_transport(monkeypatch, respond)
    assert await metadata.probe_backend_hint(base_url, "test-key") == ("tabbyapi", "TabbyAPI")
    context = await metadata.collect_run_context(
        model="gpt-3.5-turbo", backend=backend, base_url=base_url, api_key="test-key"
    )
    assert context.engine_name == "TabbyAPI"
    assert context.engine_version is None
    assert context.server_model_id == "loaded-model-exl3"
    assert context.quantization == "EXL3"
    assert context.max_model_len == 32768
    assert context.slot_count == 4
    assert context.gpu_count is None
    assert context.spec_decoding is None
    assert all(request.url.path != "/props" for request in requests)


@pytest.mark.parametrize("models_status", [200, 401, 404])
@pytest.mark.parametrize("base_url", ["http://test", "http://test/v1"])
async def test_serviceinfo_fallback_precedes_props(monkeypatch, models_status, base_url):
    def respond(request):
        if request.url.path == "/v1/models":
            return httpx.Response(models_status, json={"data": []})
        body = {"/.well-known/serviceinfo": SERVICE, "/props": PROPS}.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    install_transport(monkeypatch, respond)
    assert await metadata.probe_backend_hint(base_url) == ("tabbyapi", "TabbyAPI")
    info = await metadata._probe_engine(base_url, None, "unknown")
    assert info["engine_name"] == "TabbyAPI"
    assert info["max_model_len"] == 16384
    assert info["slot_count"] == 2
    assert "engine_version" not in info  # 0.1 is the service-info schema version.


@pytest.mark.parametrize("owner", ["tabbyAPI", "TabbyAPI", "TABBYAPI"])
def test_local_discovery(owner):
    resp = httpx.Response(200, json={"data": [{"id": "m", "owned_by": owner}]})
    assert detect_backend_from_response(resp, 5000) == ("tabbyapi", "TabbyAPI")


@pytest.mark.parametrize(
    "body",
    [
        {"software": {"name": "tabbyapi-proxy"}},
        {"software": {"name": None}},
        {"software": "TabbyAPI"},
        {"data": [{"id": "TabbyAPI", "owned_by": "user"}]},
        {"data": [{"id": "m", "owned_by": "tabbyapi-proxy"}]},
        {"version": 0.1},
        [],
    ],
)
def test_no_identity_from_names_or_malformed_serviceinfo(body):
    assert metadata.backend_from_response(httpx.Response(200, json=body)) is None


@pytest.mark.parametrize("value", [True, 0, -1, "32768", None])
async def test_invalid_capacity_is_unknown(monkeypatch, value):
    def respond(request):
        body = {
            "/v1/model": {"parameters": {"max_seq_len": value, "max_batch_size": value}},
            "/props": {"default_generation_settings": {"n_ctx": value}, "total_slots": value},
        }.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    install_transport(monkeypatch, respond)
    assert await metadata._probe_engine("http://test", None, "tabbyapi") == {
        "engine_name": "TabbyAPI"
    }


@pytest.mark.parametrize("status,body", [(403, {}), (503, {}), (200, []), (200, "not JSON")])
async def test_unavailable_metadata_keeps_explicit_identity(monkeypatch, status, body):
    def respond(request):
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    install_transport(monkeypatch, respond)
    assert await metadata._probe_engine("http://test", None, "tabbyapi") == {
        "engine_name": "TabbyAPI"
    }


def test_explicit_backend_uses_existing_adapter_and_schema():
    args = _make_parser().parse_args(["--backend", "tabbyapi"])
    service = BenchmarkService(repo=None, reporter=None)
    assert isinstance(service._adapter_for(args.backend, "http://test"), OpenAICompatibleAdapter)
    choices = next(item["choices"] for item in ARGS_SCHEMA if item["name"] == "backend")
    assert "tabbyapi" in choices
