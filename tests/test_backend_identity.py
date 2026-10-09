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
    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

    run = AsyncMock(return_value={"scores": {}})
    monkeypatch.setattr(BenchmarkService, "run_benchmark", run)
    _install(monkeypatch, lambda request: httpx.Response(404))
    await run_benchmark(
        model="m", base_url="http://test", persist=False, scenarios=ALL_SCENARIOS[:1]
    )
    assert run.call_args.kwargs["backend"] == "unknown"
    assert run.call_args.kwargs["wire_format"] is None


# Strata's /metrics in the Prometheus text format, from serve/prometheus.py
# render() at Niko1221/Strata@fb58e0d for an idle server. Trimmed to fewer
# families and buckets; every kept line is verbatim and in upstream order.
STRATA_PROMETHEUS = """\
# HELP vllm:num_requests_running Requests reading their prompt or generating.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen3-8B"} 0
# HELP vllm:prompt_tokens_total Prompt tokens of the finished requests.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="Qwen3-8B"} 1200
# HELP vllm:spec_decode_num_accepted_tokens_total MTP draft tokens accepted.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{model_name="Qwen3-8B"} 0
# HELP vllm:time_to_first_token_seconds Time to the first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="Qwen3-8B",le="0.001"} 0
vllm:time_to_first_token_seconds_bucket{model_name="Qwen3-8B",le="+Inf"} 0
vllm:time_to_first_token_seconds_sum{model_name="Qwen3-8B"} 0.0
vllm:time_to_first_token_seconds_count{model_name="Qwen3-8B"} 0
# HELP strata:live_state 1 for what the engine is doing (live.state).
# TYPE strata:live_state gauge
strata:live_state{model_name="Qwen3-8B",state="unloaded"} 0
strata:live_state{model_name="Qwen3-8B",state="idle"} 1
strata:live_state{model_name="Qwen3-8B",state="reading"} 0
strata:live_state{model_name="Qwen3-8B",state="generating"} 0
# HELP strata:engine_max_context The engine's context (engine.max_context).
# TYPE strata:engine_max_context gauge
strata:engine_max_context{model_name="Qwen3-8B"} 32768
"""

# The default JSON /metrics, which Service.metrics() returns, compacted as
# Strata's _json() sends it.
STRATA_METRICS_JSON = {
    "engine": {"model": "Qwen3-8B", "max_context": 32768, "images": False},
    "live": {"state": "idle", "queued": 0},
    "requests": [],
    "totals": {"requests": 3, "prompt_tokens": 1200, "output_tokens": 300},
}


@pytest.mark.parametrize("reverse", [False, True])
def test_strata_prometheus_metrics_name_strata_not_vllm(reverse):
    # Strata exports vLLM's metric names so vLLM dashboards read it unchanged.
    lines = STRATA_PROMETHEUS.splitlines()
    if reverse:
        lines.reverse()
    assert metadata.detect_backend_from_metrics("\n".join(lines)) == ("strata", "Strata")


def test_vllm_metrics_without_a_strata_namespace_stay_vllm():
    vllm_only = "\n".join(line for line in STRATA_PROMETHEUS.splitlines() if "strata:" not in line)
    assert metadata.detect_backend_from_metrics(vllm_only) == ("vllm", "vLLM")
    assert metadata.detect_backend_from_metrics("# HELP strata:live_state x\n") is None


def test_strata_default_json_metrics_name_no_engine():
    text = httpx.Response(200, json=STRATA_METRICS_JSON).text
    assert metadata.detect_backend_from_metrics(text) is None


def _strata(*, prometheus: bool):
    """A Strata server: no /version, a declared /health service, versioned /props.

    With *prometheus*, /metrics serves the text format, as Strata does for
    ``Accept: text/plain`` or ``?format=prometheus``. Otherwise it applies
    Strata's own negotiation, which gives the probe's ``Accept: */*`` JSON.
    """

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/metrics":
            accept = request.headers.get("accept", "")
            negotiated = "text/plain" in accept or "openmetrics" in accept
            if prometheus or negotiated or "format=prometheus" in str(request.url.query):
                return httpx.Response(200, text=STRATA_PROMETHEUS)
            return httpx.Response(200, json=STRATA_METRICS_JSON)
        body = {
            "/v1/models": {"object": "list", "data": [{"id": "m", "object": "model"}]},
            "/health": {"status": "ok", "max_context": 32768, "service": "strata"},
            "/props": {
                "default_generation_settings": {"n_ctx": 32768, "params": {}},
                "total_slots": 2,
                "build_info": "Strata 0.1.40.3",
            },
        }.get(path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    return respond


@pytest.mark.parametrize("prometheus", [False, True])
async def test_strata_is_identified_whichever_metrics_format_it_serves(monkeypatch, prometheus):
    _install(monkeypatch, _strata(prometheus=prometheus))

    hint = await metadata.probe_backend_hint("http://test")
    context = await metadata.collect_run_context(
        model="m", backend=hint[0] if hint else "unknown", base_url="http://test"
    )

    assert hint == ("strata", "Strata")
    assert context.engine_name == "Strata"
    assert context.engine_version == "0.1.40.3"
    assert context.max_model_len == 32768
    assert context.slot_count == 2
