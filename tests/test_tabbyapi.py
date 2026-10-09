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
# /v1/model returns the loaded model's full card; the list entries carry no parameters.
LOADED = {
    "id": MODEL_ID,
    "object": "model",
    "owned_by": "tabbyAPI",
    "parameters": {"max_seq_len": 16384, "max_batch_size": 4, "cache_mode": "FP16"},
}


def install_tabby(
    monkeypatch, *, auth: bool, models=MODELS, serviceinfo=None, props=None, loaded=None
):
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
        protected = {"/v1/models": models, "/v1/model": LOADED, "/props": PROPS}
        if path not in protected:
            return httpx.Response(404, json={"detail": "Not Found"})
        if auth and request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"detail": "Please provide an API key"})
        if path == "/props" and props is not None:
            return props()
        if path == "/v1/model" and loaded is not None:
            return loaded()
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
    assert context.server_model_id == MODEL_ID
    for path in ("/props", "/v1/model"):
        seen = [r for r in requests if r.url.path == path]
        assert seen, f"metadata probe never read {path}"
        assert all(r.headers.get("authorization") == _bearer(api_key) for r in seen)


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
    install_tabby(
        monkeypatch,
        auth=True,
        models={"object": "list", "data": []},
        props=props,
        loaded=props,  # /v1/model has the same check_model_container dependency
    )

    assert await metadata.probe_backend_hint("http://test", KEY) == ("tabbyapi", "TabbyAPI")
    assert await metadata._probe_engine("http://test", KEY, backend) == {"engine_name": "TabbyAPI"}


# What /v1/models returns for an admin key, or any request with auth disabled: the
# model directory, in directory order, with the loaded model anywhere in it.
CATALOG = {
    "object": "list",
    "data": [
        {"id": "Llama-3.1-70B-Instruct-AWQ", "object": "model", "owned_by": "tabbyAPI"},
        {"id": MODEL_ID, "object": "model", "owned_by": "tabbyAPI"},
    ],
}
# use_dummy_models puts these aliases ahead of every other entry.
DUMMY_FIRST = {
    "object": "list",
    "data": [{"id": "gpt-3.5-turbo", "object": "model", "owned_by": "tabbyAPI"}, *CATALOG["data"]],
}


@pytest.mark.parametrize("models", [CATALOG, DUMMY_FIRST], ids=["admin-catalog", "dummy-first"])
@pytest.mark.parametrize("backend", ["tabbyapi", "unknown"])
@pytest.mark.parametrize(("auth", "api_key"), [(False, None), (True, KEY)], ids=["no-auth", "key"])
async def test_loaded_model_beats_the_first_list_entry(monkeypatch, models, backend, auth, api_key):
    install_tabby(monkeypatch, auth=auth, models=models)

    context = await metadata.collect_run_context(
        model="anything", backend=backend, base_url="http://test/v1", api_key=api_key
    )
    assert context.server_model_id == MODEL_ID
    # Quantization follows the loaded model, not the AWQ checkpoint listed first.
    assert context.quantization == "EXL3"


async def test_loaded_model_drops_the_root_of_the_replaced_entry(monkeypatch):
    # TabbyAPI sends no root, but a rewriting proxy might; the quantization guess
    # prefers root, so the replaced entry's root must not survive.
    first = {"id": "Llama-3.1-70B-Instruct-AWQ", "root": "meta/Llama-3.1-70B-Instruct-AWQ"}
    install_tabby(monkeypatch, auth=True, models={"object": "list", "data": [first]})

    context = await metadata.collect_run_context(
        model="anything", backend="tabbyapi", base_url="http://test/v1", api_key=KEY
    )
    assert context.server_model_id == MODEL_ID
    assert context.server_model_root is None
    assert context.quantization == "EXL3"


async def test_loaded_model_id_is_recorded_verbatim(monkeypatch):
    # The id is a directory name that chat requests must match exactly; do not strip it.
    padded = {**LOADED, "id": f" {MODEL_ID} "}
    install_tabby(monkeypatch, auth=True, loaded=lambda: httpx.Response(200, json=padded))

    info = await metadata._probe_engine("http://test", KEY, "tabbyapi")
    assert info["server_model_id"] == f" {MODEL_ID} "


@pytest.mark.parametrize(
    "loaded",
    [
        lambda: httpx.Response(404, json={"detail": "Not Found"}),
        lambda: httpx.Response(200, text="<html>proxy error</html>"),
        lambda: httpx.Response(200, json=[LOADED]),
        lambda: httpx.Response(200, json={"object": "model"}),
        lambda: httpx.Response(200, json={"id": None}),
        lambda: httpx.Response(200, json={"id": 7}),
        lambda: httpx.Response(200, json={"id": ""}),
        lambda: httpx.Response(200, json={"id": "   "}),
    ],
    ids=["route-absent", "html", "list-body", "no-id", "null-id", "int-id", "empty", "blank"],
)
async def test_unusable_loaded_card_keeps_the_list_entry(monkeypatch, loaded):
    install_tabby(monkeypatch, auth=True, models=CATALOG, loaded=loaded)

    # The entry matching the requested model, not the AWQ checkpoint listed first.
    info = await metadata._probe_engine("http://test", KEY, "tabbyapi", model=MODEL_ID)
    assert info["server_model_id"] == MODEL_ID
    assert info["slot_count"] == 4  # /props is still read


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
