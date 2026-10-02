"""TensorFold protocol fixtures from upstream 56e2e3e, without a live model."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest

from tests.conftest import MeasurementTestClient
from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.server import detect_backend_from_response
from tool_eval_bench.runner import speculative, throughput
from tool_eval_bench.runner.context_pressure import detect_context_size
from tool_eval_bench.runner.spec_live import _parse_snapshot, compute_delta
from tool_eval_bench.utils import metadata

MODELS = {"data": [{"id": "local-model", "object": "model", "owned_by": "tensorfold"}]}
METRICS = """\
# HELP tensorfold:mtp_drafted_total Draft tokens verified on finished requests.
tensorfold:mtp_drafted_total 100
tensorfold:mtp_accepted_total 80
tensorfold:spec_decode_num_draft_tokens_total 100
tensorfold:spec_decode_num_accepted_tokens_total 80
tensorfold:prompt_tokens_total 200
tensorfold:generation_tokens_total 90
tensorfold:num_requests_running 2
tensorfold:num_requests_waiting 1
tensorfold:kv_cache_usage_perc{stream="0"} 0.25
"""


def _sse(*chunks: dict) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


@pytest.mark.parametrize("base_url", ["http://test", "http://test/v1"])
@pytest.mark.parametrize(
    "health",
    [
        {"status": "ok", "model": "local-model", "max_batch_size": 8},
        {"ok": True, "backend": "tensorfold", "context_length": 32768, "streams": {"max": 2}},
    ],
)
async def test_detection_and_deployment_metadata(monkeypatch, base_url, health):
    real_client = httpx.AsyncClient

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=MODELS)
        if request.url.path == "/health":
            return httpx.Response(200, json=health)
        return httpx.Response(404)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    assert await metadata.probe_backend_hint(base_url) == ("tensorfold", "TensorFold")
    context = await metadata.collect_run_context(
        model="local-model", backend="tensorfold", base_url=base_url
    )
    assert context.engine_name == "TensorFold"
    assert context.engine_version is None
    assert context.max_model_len == health.get("context_length")
    assert context.slot_count == health.get("streams", {}).get("max")
    # A batch width is not a server slot count; an alias is not a checkpoint.
    assert context.server_model_root is None
    assert context.spec_decoding is None
    assert context.quantization is None


@pytest.mark.parametrize("body", [MODELS, {"data": []}, {"data": [None]}, [], None])
def test_initial_discovery_and_false_positives(body):
    response = httpx.Response(200, json=body)
    expected = ("tensorfold", "TensorFold") if body == MODELS else ("vllm", "inference server")
    assert detect_backend_from_response(response, 8080) == expected


def test_backend_uses_existing_adapter_and_metrics_identity():
    service = BenchmarkService(repo=None, reporter=None)
    assert isinstance(service._adapter_for("tensorfold", "http://test"), OpenAICompatibleAdapter)
    assert metadata.detect_backend_from_metrics(METRICS) == ("tensorfold", "TensorFold")
    assert metadata.detect_backend_from_metrics("# tensorfold:requests_running 1\n") is None


@pytest.mark.parametrize("window", [32768, 0, -1, True, "32768", None])
async def test_context_size_from_cuda_health(window):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=MODELS)
        assert request.url.path == "/health"
        return httpx.Response(200, json={"backend": "tensorfold", "context_length": window})

    def factory(**kwargs):
        return HTTPMeasurementClient(transport=httpx.MockTransport(handler), **kwargs)

    result = await detect_context_size(
        "http://test/v1", "local-model", "key", client_factory=factory
    )
    assert result == (32768 if type(window) is int and window > 0 else None)
    assert requests[-1].headers["Authorization"] == "Bearer key"


def test_spec_metrics_aliases_are_not_double_counted():
    counters = speculative.parse_prometheus_spec_metrics(METRICS)
    assert counters.draft_tokens == 100
    assert counters.accepted_tokens == 80
    assert counters.acceptance_rate == 0.8
    assert counters.acceptance_length is None
    previous = _parse_snapshot(METRICS.replace(" 100\n", " 0\n").replace(" 80\n", " 0\n"))
    current = _parse_snapshot(METRICS)
    assert current.spec_backend == "tensorfold"
    assert current.spec_method == "unknown"
    assert current.has_counter_spec_decode
    assert current.prompt_tokens_total == 200
    assert current.generation_tokens_total == 90
    assert current.running_reqs == 2
    assert current.waiting_reqs == 1
    assert current.kv_cache_usage == 0.25
    delta = compute_delta(replace(previous, timestamp=1), replace(current, timestamp=2))
    assert delta.spec_metrics_source == "tensorfold"
    assert delta.acceptance_rate == 0.8
    assert delta.cumulative_acceptance_rate == 0.8
    assert delta.acceptance_length is None
    assert delta.draft_window is None


@pytest.mark.parametrize("active", [True, False])
async def test_generic_mtp_metric_does_not_identify_proposer(active):
    metrics = METRICS if active else METRICS.replace(" 100\n", " 0\n").replace(" 80\n", " 0\n")
    async with MeasurementTestClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=metrics))
    ) as client:
        info = await speculative.detect_spec_decoding(client, "http://test")
    assert info.has_prometheus
    assert info.active is active
    assert info.method == "unknown"


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("drafted,accepted", [(10, 8), (0, 0)])
async def test_request_local_spec_stats_win_over_global_counters(
    monkeypatch, backend, drafted, accepted
):
    stats = {"drafted": drafted, "accepted": accepted, "rounds": 99}
    extension = {"speculative": stats} if backend == "mlx" else {"tensorfold": stats}
    body = _sse(
        {"choices": [{"delta": {"content": "Hello"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 9},
            **extension,
        },
    )
    monkeypatch.setattr(
        speculative,
        "_build_messages",
        AsyncMock(return_value=[{"role": "user", "content": "Hello"}]),
    )
    async with MeasurementTestClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                text=body if r.method == "POST" else METRICS,
                headers={
                    "Content-Type": "text/event-stream" if r.method == "POST" else "text/plain"
                },
            )
        )
    ) as client:
        sample = await speculative.measure_spec_single(
            client,
            "http://test",
            "local-model",
            spec_info=speculative.SpecDecodeInfo(has_prometheus=True),
        )
    assert sample.acceptance_source == "response"
    assert sample.draft_tokens_delta == drafted
    assert sample.accepted_tokens_delta == accepted
    assert sample.num_drafts_delta is None
    assert sample.acceptance_length is None
    assert sample.draft_window is None
    assert sample.acceptance_rate == (0.8 if drafted else None)


@pytest.mark.parametrize(
    "stats",
    [
        {},
        {"drafted": True, "accepted": 0},
        {"drafted": -1, "accepted": 0},
        {"drafted": 1, "accepted": 2},
        {"drafted": 1.0, "accepted": 1},
        {"drafted": 2, "accepted": "1"},
    ],
)
async def test_invalid_request_counters_are_not_reported(monkeypatch, stats):
    monkeypatch.setattr(speculative, "_build_messages", AsyncMock(return_value=[]))
    body = _sse({"choices": [{"delta": {"content": "Hello"}}], "tensorfold": stats})
    async with MeasurementTestClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
        )
    ) as client:
        sample = await speculative.measure_spec_single(client, "http://test", "local-model")
    assert sample.acceptance_source is None
    assert sample.acceptance_rate is None


@pytest.mark.parametrize(
    "error,status,infrastructure",
    [
        ({"type": "server_error", "message": "generation failed"}, 500, True),
        (
            {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "message": "context overflow",
            },
            400,
            True,
        ),
        ({"type": "invalid_request_error", "message": "bad arguments"}, 400, False),
        ("generation failed", 500, True),
    ],
)
async def test_stream_errors_cannot_be_scored_as_model_answers(error, status, infrastructure):
    body = _sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"city":'},
                            }
                        ]
                    }
                }
            ]
        },
        {"error": error},
    )
    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
        )
    )
    try:
        result = await adapter.chat_completion(
            model="local-model",
            messages=[{"role": "user", "content": "Hello"}],
            base_url="http://test",
            stream=True,
        )
    finally:
        await adapter.aclose()
    assert result.transport_error_status == status
    assert result.transport_error_is_infrastructure is infrastructure
    assert result.tool_calls == []
    async with MeasurementTestClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
        )
    ) as client:
        sample = await throughput._stream_one(
            client, "http://test", "local-model", [], 16, None, tok_cfg=throughput.TokenizerConfig()
        )
    assert sample.error and "server error" in sample.error
    assert sample.tg_tokens == 0


@pytest.mark.parametrize(
    "health", [None, [], {"context_length": True, "streams": {"max": -1}}, {"streams": 2}]
)
async def test_invalid_health_keeps_identity_without_inventing_capacity(monkeypatch, health):
    real_client = httpx.AsyncClient

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=MODELS)
        return httpx.Response(200, json=health)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    info = await metadata._probe_engine("http://test", None, "tensorfold")
    assert info["engine_name"] == "TensorFold"
    assert "max_model_len" not in info
    assert "slot_count" not in info


@pytest.mark.parametrize("status,body", [(404, None), (200, "not JSON")])
async def test_unavailable_health_does_not_erase_identity(monkeypatch, status, body):
    real_client = httpx.AsyncClient

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=MODELS)
        return httpx.Response(status, text=body or "")

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    info = await metadata._probe_engine("http://test/v1", None, "vllm")
    assert info["engine_name"] == "TensorFold"
    assert "max_model_len" not in info


async def test_models_context_field_does_not_depend_on_health():
    models = {"data": [{**MODELS["data"][0], "max_model_len": 16384}]}

    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json=models)

    def factory(**kwargs):
        return HTTPMeasurementClient(transport=httpx.MockTransport(handler), **kwargs)

    assert (
        await detect_context_size("http://test/v1", "local-model", client_factory=factory) == 16384
    )


@pytest.mark.parametrize(
    "model,quantization",
    [
        ("Vontra/Qwen3.8-27B-MLX-4bit", "MLX-4bit"),
        ("turboderp/Qwen3.8-27B-exl3", "EXL3"),
        ("nvidia/Qwen3.8-27B-NVFP4", "NVFP4"),
        ("local-model", None),
    ],
)
def test_quantization_name_hints(model, quantization):
    assert metadata._guess_quantization(model) == quantization


async def test_failed_spec_stream_has_no_acceptance_metrics(monkeypatch):
    monkeypatch.setattr(speculative, "_build_messages", AsyncMock(return_value=[]))
    body = _sse(
        {
            "choices": [{"delta": {"content": "partial"}}],
            "speculative": {"drafted": 10, "accepted": 8},
        },
        {"error": {"type": "server_error", "message": "generation failed"}},
    )
    async with MeasurementTestClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
        )
    ) as client:
        sample = await speculative.measure_spec_single(client, "http://test", "local-model")
    assert sample.error
    assert sample.acceptance_source is None
    assert sample.acceptance_rate is None


async def test_fragmented_parallel_calls_reasoning_and_usage_roundtrip():
    calls = [
        {
            "index": i,
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":'},
        }
        for i in range(2)
    ]
    body = _sse(
        {"choices": [{"delta": {"reasoning_content": "Check both cities."}}]},
        {"choices": [{"delta": {"tool_calls": calls}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": i, "function": {"arguments": json.dumps(city) + "}"}}
                            for i, city in enumerate(["Berlin", "Paris"])
                        ]
                    }
                }
            ]
        },
        {
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 20},
        },
    )
    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
        )
    )
    try:
        result = await adapter.chat_completion(
            model="local-model",
            messages=[{"role": "user", "content": "Weather?"}],
            base_url="http://test",
            stream=True,
        )
    finally:
        await adapter.aclose()
    from tool_eval_bench.runner.orchestrator import _assistant_message

    assert result.reasoning == "Check both cities."
    assert [c.arguments for c in result.tool_calls] == [{"city": "Berlin"}, {"city": "Paris"}]
    assert result.finish_reason == "tool_calls"
    assert result.completion_tokens == 20
    assert _assistant_message(result)["reasoning_content"] == result.reasoning
