"""Engine identity is not implied by a health response, port, or compatible metrics."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.server import DISCOVERY_PORTS, detect_backend_from_response
from tool_eval_bench.utils import metadata


@pytest.mark.parametrize("body", [{}, {"status": "ok"}, {"version": "0.16.2"}])
async def test_generic_props_and_health_do_not_identify_llamacpp(monkeypatch, body):
    real_client = httpx.AsyncClient

    def respond(request):
        if request.url.path in ("/props", "/health"):
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    assert await metadata.probe_backend_hint("http://test") is None
    assert "engine_name" not in await metadata._probe_engine("http://test", None, "unknown")


@pytest.mark.parametrize(
    "path,body,headers",
    [
        ("/v1/models", {"data": [{"owned_by": "llamacpp"}]}, {}),
        ("/v1/models", {"data": []}, {"server": "llama.cpp"}),
        ("/props", {"build_info": "b11418-9871df591", "total_slots": 2}, {}),
        ("/props", {"default_generation_settings": {}, "total_slots": 2}, {}),
    ],
)
async def test_llamacpp_positive_identifiers(monkeypatch, path, body, headers):
    real_client = httpx.AsyncClient

    def respond(request):
        if request.url.path == path:
            return httpx.Response(200, json=body, headers=headers)
        return httpx.Response(404)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    assert await metadata.probe_backend_hint("http://test/v1") == ("llamacpp", "llama.cpp")
    info = await metadata._probe_engine("http://test/v1", None, "unknown")
    assert info["engine_name"] == "llama.cpp"


LLAMA_PROPS = {
    "build_info": "b11418-9871df591",
    "default_generation_settings": {"n_ctx": 8192},
    "total_slots": 2,
}


def _install(monkeypatch, respond):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )


async def test_real_llamacpp_is_identified_by_owner_and_reports_its_build(monkeypatch):
    def respond(request):
        body = {
            "/v1/models": {"data": [{"id": "m", "owned_by": "llamacpp"}]},
            "/props": LLAMA_PROPS,
            "/health": {"status": "ok"},
        }.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    _install(monkeypatch, respond)
    assert await metadata.probe_backend_hint("http://test") == ("llamacpp", "llama.cpp")
    info = await metadata._probe_engine("http://test", None, "unknown")
    assert info["engine_name"] == "llama.cpp"
    assert info["engine_version"] == "b11418-9871df591"
    assert info["slot_count"] == 2


@pytest.mark.parametrize("api_key", ["test-key", None])
async def test_keyed_llamacpp_gets_the_key_only_when_one_is_given(monkeypatch, api_key):
    # llama-server --api-key protects everything except /health.
    seen: list[httpx.Request] = []

    def respond(request):
        seen.append(request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.headers.get("authorization") != "Bearer test-key":
            return httpx.Response(401)
        body = {"/v1/models": {"data": [{"id": "m"}]}, "/props": LLAMA_PROPS}.get(path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    _install(monkeypatch, respond)
    hint = await metadata.probe_backend_hint("http://test", api_key)
    info = await metadata._probe_engine("http://test", api_key, "llamacpp")

    if api_key:
        assert hint == ("llamacpp", "llama.cpp")
        assert info["engine_version"] == "b11418-9871df591"
        props = [r for r in seen if r.url.path == "/props"]
        assert props and all(r.headers["authorization"] == "Bearer test-key" for r in props)
    else:
        assert hint is None
        assert "engine_name" not in info
        assert all("authorization" not in r.headers for r in seen)


@pytest.mark.parametrize(
    ("props_extra", "headers", "is_llamacpp"),
    [
        # Another server named on the response: the llama shape is not trusted.
        ({}, {"server": "litellm"}, False),
        ({"build_info": "Strata 0.1.40.3"}, {}, False),
        # llama.cpp named, or nothing named: the shape decides.
        ({}, {"server": "llama.cpp"}, True),
        ({}, {}, True),
    ],
)
async def test_props_naming_another_server_is_not_llamacpp(
    monkeypatch, props_extra, headers, is_llamacpp
):
    def respond(request):
        if request.url.path == "/props":
            return httpx.Response(200, json={**LLAMA_PROPS, **props_extra}, headers=headers)
        return httpx.Response(404)

    _install(monkeypatch, respond)
    info = await metadata._probe_engine("http://test", None, "llamacpp")
    assert (info.get("engine_name") == "llama.cpp") is is_llamacpp


async def test_server_header_on_props_is_a_declared_identity(monkeypatch):
    def respond(request):
        if request.url.path == "/props":
            return httpx.Response(200, json=LLAMA_PROPS, headers={"server": "litellm"})
        return httpx.Response(404)

    _install(monkeypatch, respond)
    assert await metadata.probe_backend_hint("http://test") == ("litellm", "LiteLLM")


@pytest.mark.parametrize(
    ("path", "server", "hint"),
    [
        ("/health", "litellm", ("litellm", "LiteLLM")),
        ("/props", "vllm/0.6.3", ("vllm", "vLLM")),
    ],
)
async def test_unlabelled_engine_name_agrees_with_a_declared_only_hint(
    monkeypatch, path, server, hint
):
    # The Server header is the only evidence, so the vLLM and LiteLLM metadata
    # probes find nothing. The engine name must still match the hint.
    def respond(request):
        if request.url.path == path:
            return httpx.Response(200, json={"status": "ok"}, headers={"server": server})
        return httpx.Response(404)

    _install(monkeypatch, respond)
    assert await metadata.probe_backend_hint("http://test") == hint
    info = await metadata._probe_engine("http://test", None, "unknown")
    assert info == {"engine_name": hint[1]}


@pytest.mark.parametrize(
    ("path", "response", "expected"),
    [
        (
            "/version",
            httpx.Response(200, json={"version": "0.11.0"}),
            {"engine_name": "vLLM", "engine_version": "0.11.0"},
        ),
        (
            "/health",
            httpx.Response(200, json={}, headers={"x-litellm-version": "1.77.0"}),
            {"engine_name": "LiteLLM", "engine_version": "1.77.0"},
        ),
    ],
)
async def test_unlabelled_vllm_and_litellm_report_their_versions(
    monkeypatch, path, response, expected
):
    def respond(request):
        return response if request.url.path == path else httpx.Response(404)

    _install(monkeypatch, respond)
    assert await metadata._probe_engine("http://test", None, "unknown") == expected


async def test_owner_declared_first_wins_over_a_props_naming_strata(monkeypatch):
    # /v1/models is asked before /props. Its owner labels the server, but a
    # /props that names Strata is still not read as a llama.cpp build.
    def respond(request):
        body = {
            "/v1/models": {"data": [{"id": "m", "owned_by": "llamacpp"}]},
            "/props": {**LLAMA_PROPS, "build_info": "Strata 0.1.40.3"},
        }.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    _install(monkeypatch, respond)
    assert await metadata.probe_backend_hint("http://test") == ("llamacpp", "llama.cpp")
    info = await metadata._probe_engine("http://test", None, "unknown")
    assert info["engine_name"] == "llama.cpp"
    assert "engine_version" not in info
    assert "slot_count" not in info


@pytest.mark.parametrize(
    ("backend", "engine_name"), [("strata", "Strata"), ("tabbyapi", "TabbyAPI")]
)
async def test_explicit_label_on_llamacpp_is_trusted(monkeypatch, backend, engine_name):
    def respond(request):
        body = {
            "/v1/models": {"data": [{"id": "m", "owned_by": "llamacpp"}]},
            "/props": LLAMA_PROPS,
        }.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    _install(monkeypatch, respond)
    info = await metadata._probe_engine("http://test", None, backend)
    assert info["engine_name"] == engine_name
    assert info["max_model_len"] == 8192
    assert info["slot_count"] == 2
    assert "engine_version" not in info


async def test_explicit_llamacpp_label_on_versioned_strata_records_no_engine(monkeypatch):
    def respond(request):
        body = {
            "/v1/models": {"data": [{"id": "m"}]},
            "/props": {**LLAMA_PROPS, "build_info": "Strata 0.1.40.3"},
            "/health": {"status": "ok", "service": "strata"},
        }.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    _install(monkeypatch, respond)
    info = await metadata._probe_engine("http://test", None, "llamacpp")
    assert "engine_name" not in info
    assert "engine_version" not in info
    assert "slot_count" not in info


@pytest.mark.parametrize("reverse", [False, True])
def test_halogen_identity_overrides_borrowed_metrics(reverse):
    # Halogen 0.16.2 documents both namespaces for llama-swap compatibility.
    lines = ["llamacpp:prompt_tokens_total 0", "halogen:requests_total 0"]
    if reverse:
        lines.reverse()
    assert metadata.detect_backend_from_metrics("\n".join(lines)) == ("halogen", "Halogen Flash")
    assert metadata.detect_backend_from_metrics("# halogen:requests_total 0") is None


async def test_halogen_metrics_identify_backend_and_existing_adapter(monkeypatch):
    real_client = httpx.AsyncClient

    def respond(request):
        if request.url.path == "/metrics":
            return httpx.Response(
                200, text="llamacpp:prompt_tokens_total 0\nhalogen:requests_total 0\n"
            )
        return httpx.Response(404)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    assert await metadata.probe_backend_hint("http://test") == ("halogen", "Halogen Flash")
    context = await metadata.collect_run_context(
        model="m", backend="halogen", base_url="http://test"
    )
    assert context.engine_name == "Halogen Flash"
    assert context.engine_version is None
    service = BenchmarkService(repo=None, reporter=None)
    assert isinstance(service._adapter_for("halogen"), OpenAICompatibleAdapter)
    assert isinstance(service._adapter_for("unknown"), OpenAICompatibleAdapter)


@pytest.mark.parametrize("port", [p for p, _, _ in DISCOVERY_PORTS] + [9999])
@pytest.mark.parametrize("server", ["", "uvicorn", "ollama"])
def test_ports_and_unrelated_headers_do_not_identify_an_engine(port, server):
    response = httpx.Response(200, json={"data": []}, headers={"server": server})
    assert detect_backend_from_response(response, port) == ("unknown", "inference server")


async def test_public_api_defaults_to_unknown_without_changing_adapter(monkeypatch):
    from tool_eval_bench.api import run_benchmark

    run = AsyncMock(return_value={"scores": {}})
    monkeypatch.setattr(BenchmarkService, "run_benchmark", run)
    _install(monkeypatch, lambda request: httpx.Response(404))
    await run_benchmark(model="m", base_url="http://test", persist=False, scenarios=[])
    assert run.call_args.kwargs["backend"] == "unknown"
    assert run.call_args.kwargs["wire_format"] is None
