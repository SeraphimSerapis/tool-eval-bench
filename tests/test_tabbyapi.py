"""TabbyAPI names itself, then serves a llama-server-shaped /props (issue #214).

Payloads follow theroyallab/tabbyAPI: ``endpoints/core/router.py`` and
``endpoints/core/types/model.py``. ``/health`` and ``/.well-known/serviceinfo``
need no key; ``/v1/models`` and ``/props`` do when auth is enabled.
"""

from __future__ import annotations

import httpx
import pytest

from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.legacy_parser import _make_parser
from tool_eval_bench.utils import metadata

KEY = "test-key"
MODEL_ID = "Qwen3-8B-exl3-4.0bpw"
MODELS = {
    "object": "list",
    "data": [
        {
            "id": MODEL_ID,
            "object": "model",
            "owned_by": "tabbyAPI",
            "parameters": None,
            "meta": {"n_ctx_train": 40960, "n_ctx": 16384, "n_vocab": 0, "n_embd": 0},
        }
    ],
}
PROPS = {
    "total_slots": 4,
    "model_path": f"/models/{MODEL_ID}",
    "chat_template": "{{ messages }}",
    "default_generation_settings": {"n_ctx": 16384},
    "modalities": {"vision": False},
}
SERVICEINFO = {
    "version": 0.1,
    "software": {"name": "TabbyAPI", "repository": "https://github.com/theroyallab/tabbyAPI"},
    "api": {"openai": {"name": "OpenAI API", "relative_url": "/v1", "version": 1}},
}
HEALTH = {"status": "healthy", "issues": []}


def install_tabby(monkeypatch, *, auth: bool, models=MODELS, serviceinfo=None, props=None):
    """Serve a TabbyAPI-shaped endpoint and return the list of requests it saw."""
    requests: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json=HEALTH, headers={"server": "uvicorn"})
        if path == "/.well-known/serviceinfo":
            if serviceinfo is not None:
                return serviceinfo()
            return httpx.Response(200, json=SERVICEINFO, headers={"server": "uvicorn"})
        protected = {"/v1/models": models, "/props": PROPS}
        if path not in protected:
            return httpx.Response(404, json={"detail": "Not Found"})
        if auth and request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"detail": "Please provide an API key"})
        if path == "/props" and props is not None:
            return props()
        return httpx.Response(200, json=protected[path], headers={"server": "uvicorn"})

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw),
    )
    return requests


def _bearer(api_key: str | None) -> str | None:
    return f"Bearer {api_key}" if api_key else None


@pytest.mark.parametrize(("auth", "api_key"), [(False, None), (True, KEY)])
@pytest.mark.parametrize("backend", ["tabbyapi", "unknown"])
async def test_identity_and_metadata(monkeypatch, auth, api_key, backend):
    requests = install_tabby(monkeypatch, auth=auth)

    assert await metadata.probe_backend_hint("http://test/v1", api_key) == ("tabbyapi", "TabbyAPI")
    context = await metadata.collect_run_context(
        model=MODEL_ID, backend=backend, base_url="http://test/v1", api_key=api_key
    )
    assert context.engine_name == "TabbyAPI"
    assert context.engine_version is None
    assert context.max_model_len == 16384
    assert context.slot_count == 4
    assert context.quantization == "EXL3"
    props_requests = [r for r in requests if r.url.path == "/props"]
    assert props_requests, "metadata probe never read /props"
    assert all(r.headers.get("authorization") == _bearer(api_key) for r in props_requests)


async def test_keyed_server_without_a_key_is_identified_by_serviceinfo(monkeypatch):
    requests = install_tabby(monkeypatch, auth=True)

    assert await metadata.probe_backend_hint("http://test", None) == ("tabbyapi", "TabbyAPI")
    assert any(r.url.path == "/v1/models" for r in requests), "owned_by was not tried first"
    assert await metadata._probe_engine("http://test", None, "tabbyapi") == {
        "engine_name": "TabbyAPI"
    }
    info = await metadata._probe_engine("http://test", None, "unknown")
    assert info["engine_name"] == "TabbyAPI"


async def test_serviceinfo_beats_a_readable_llama_props(monkeypatch):
    # A proxy that drops owned_by leaves /props readable and llama-shaped; the
    # unauthenticated serviceinfo document must still win over that shape.
    install_tabby(monkeypatch, auth=True, models={"object": "list", "data": [{"id": MODEL_ID}]})

    assert await metadata.probe_backend_hint("http://test", KEY) == ("tabbyapi", "TabbyAPI")
    info = await metadata._probe_engine("http://test", KEY, "unknown")
    assert info["engine_name"] == "TabbyAPI"
    assert info["slot_count"] == 4


@pytest.mark.parametrize(
    "props",
    [
        # TabbyAPI answers /props with an error until a model is loaded.
        lambda: httpx.Response(503, json={"detail": "No models are currently loaded."}),
        lambda: httpx.Response(400, json={"detail": "No models are currently loaded."}),
        lambda: httpx.Response(200, text="<html>proxy error</html>"),
    ],
)
@pytest.mark.parametrize("backend", ["tabbyapi", "unknown"])
async def test_keyed_server_without_a_loaded_model_records_only_its_name(
    monkeypatch, props, backend
):
    install_tabby(monkeypatch, auth=True, models={"object": "list", "data": []}, props=props)

    assert await metadata.probe_backend_hint("http://test", KEY) == ("tabbyapi", "TabbyAPI")
    assert await metadata._probe_engine("http://test", KEY, backend) == {"engine_name": "TabbyAPI"}


@pytest.mark.parametrize("owner", ["tabbyAPI", "TABBYAPI", "tabbyapi", " tabbyAPI "])
def test_owned_by_matches_case_insensitively(owner):
    body = {"data": [{"id": "m", "owned_by": owner}]}
    assert metadata.backend_from_models(body) == ("tabbyapi", "TabbyAPI")


@pytest.mark.parametrize("owner", ["tabby", "tabbyapi2", "tabbyapi-proxy", "", None, 7])
def test_owned_by_requires_the_exact_name(owner):
    assert metadata.backend_from_models({"data": [{"id": "m", "owned_by": owner}]}) is None


def _timeout() -> httpx.Response:
    raise httpx.ReadTimeout("serviceinfo stalled")


@pytest.mark.parametrize(
    "serviceinfo",
    [
        lambda: httpx.Response(404),
        lambda: httpx.Response(503, json=SERVICEINFO),
        lambda: httpx.Response(200, text="TabbyAPI"),
        lambda: httpx.Response(200, json=[SERVICEINFO]),
        lambda: httpx.Response(200, json={"software": "TabbyAPI"}),
        lambda: httpx.Response(200, json={"software": ["TabbyAPI"]}),
        lambda: httpx.Response(200, json={"software": {"name": 7}}),
        lambda: httpx.Response(200, json={"software": {"name": ""}}),
        lambda: httpx.Response(200, json={"software": {"name": "   "}}),
        lambda: httpx.Response(200, json={"software": {"name": "Tabby"}}),
        lambda: httpx.Response(200, json={"name": "TabbyAPI"}),
        _timeout,
    ],
)
async def test_unusable_serviceinfo_falls_through(monkeypatch, serviceinfo):
    # No key, so /v1/models and /props answer 401 and serviceinfo is the only hope.
    install_tabby(monkeypatch, auth=True, serviceinfo=serviceinfo)

    assert await metadata.probe_backend_hint("http://test", None) is None
    assert "engine_name" not in await metadata._probe_engine("http://test", None, "unknown")


def test_backend_label_is_accepted_and_uses_existing_adapter():
    args = _make_parser().parse_args(["--backend", "tabbyapi"])
    service = BenchmarkService(repo=None, reporter=None)
    assert isinstance(service._adapter_for(args.backend, "http://test"), OpenAICompatibleAdapter)
