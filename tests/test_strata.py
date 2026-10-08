"""Strata speaks the OpenAI API and exposes llama.cpp-compatible props."""

from __future__ import annotations

import httpx
import pytest

from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.legacy_parser import _make_parser
from tool_eval_bench.utils import metadata

MODELS = {"data": [{"id": "qwen3.8-flash-next-iq2_xs", "meta": {"n_ctx": 32768}}]}
PROPS = {
    "build_info": "Strata 0.1.40.3",
    "default_generation_settings": {"n_ctx": 32768, "params": {}},
    "total_slots": 1,
}
HEALTH = {"status": "ok", "service": "strata", "max_context": 32768, "loaded": True}


def install_transport(monkeypatch, respond):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw),
    )


@pytest.mark.parametrize("base_url", ["http://test", "http://test/v1"])
@pytest.mark.parametrize("backend", ["strata", "unknown"])
async def test_identity_and_authenticated_metadata(monkeypatch, base_url, backend):
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        body = {"/v1/models": MODELS, "/props": PROPS, "/health": HEALTH}.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    install_transport(monkeypatch, respond)
    assert await metadata.probe_backend_hint(base_url, "test-key") == ("strata", "Strata")
    context = await metadata.collect_run_context(
        model="qwen3.8-flash-next-iq2_xs", backend=backend, base_url=base_url, api_key="test-key"
    )
    assert context.engine_name == "Strata"
    assert context.engine_version == "0.1.40.3"
    assert context.max_model_len == 32768
    assert context.slot_count == 1
    assert context.quantization == "IQ2_XS"
    assert context.gpu_count is None
    assert context.spec_decoding is None
    assert context.server_model_root is None
    assert all(not request.url.path.startswith("/v1/props") for request in requests)


@pytest.mark.parametrize("props_status", [403, 404])
async def test_health_identity_without_props(monkeypatch, props_status):
    def respond(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=HEALTH)
        return httpx.Response(props_status)

    install_transport(monkeypatch, respond)
    assert await metadata.probe_backend_hint("http://test") == ("strata", "Strata")
    info = await metadata._probe_engine("http://test", None, "strata")
    assert info == {"engine_name": "Strata", "max_model_len": 32768}


@pytest.mark.parametrize("window", [True, 0, -1, "32768", None])
async def test_invalid_capacity_is_not_inferred(monkeypatch, window):
    def respond(request):
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    **PROPS,
                    "default_generation_settings": {"n_ctx": window},
                    "total_slots": window,
                },
            )
        return httpx.Response(404)

    install_transport(monkeypatch, respond)
    info = await metadata._probe_engine("http://test", None, "strata")
    assert info == {"engine_name": "Strata", "engine_version": "0.1.40.3"}


@pytest.mark.parametrize(
    "body",
    [
        {"status": "ok", "max_context": 32768},
        {"service": "stratagem"},
        {"build_info": "Stratagem 1.0"},
        {"data": [{"id": "strata", "owned_by": "user"}]},
        {"build_info": None},
        [],
    ],
)
def test_unrelated_responses_do_not_identify_strata(body):
    assert metadata.backend_from_response(httpx.Response(200, json=body)) is None


def test_non_json_response_does_not_identify_strata():
    assert metadata.backend_from_response(httpx.Response(200, text="Strata")) is None


def test_explicit_backend_uses_existing_adapter():
    args = _make_parser().parse_args(["--backend", "strata"])
    service = BenchmarkService(repo=None, reporter=None)
    assert isinstance(service._adapter_for(args.backend, "http://test"), OpenAICompatibleAdapter)


@pytest.mark.parametrize("quant", ["Q2_0", "Q8_0", "IQ2_XS", "IQ3_XXS", "TQ1_0", "TQ2_0"])
def test_native_gguf_quantization_names(quant):
    assert metadata._guess_quantization(f"model-{quant}-00001-of-00002.gguf") == quant
    assert metadata._guess_quantization(f"model-{quant.lower()}") == quant


@pytest.mark.parametrize("name", ["model-AQ8_0", "model-IQ2_XSextra"])
def test_quantization_requires_token_boundaries(name):
    assert metadata._guess_quantization(name) is None
