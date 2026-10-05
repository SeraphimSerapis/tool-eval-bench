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
    await run_benchmark(model="m", base_url="http://test", persist=False, scenarios=[])
    assert run.call_args.kwargs["backend"] == "unknown"
    assert run.call_args.kwargs["wire_format"] is None
