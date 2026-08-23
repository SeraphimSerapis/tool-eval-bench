"""Tests for the low-level measurement transport seam."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
from tool_eval_bench.runner import context_pressure, speculative, throughput


class _FakeResponse:
    def __init__(self, body: dict[str, Any], content_type: str = "application/json") -> None:
        self.status_code = 200
        self.text = ""
        self.headers = {"content-type": content_type}
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        return None

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self) -> AsyncIterator[str]:
        if False:
            yield ""


class _FakeMeasurementPort:
    """Semantic fake used by runners, never an HTTP-shaped test double."""

    _measurement_port = True

    async def tokenize(self, *, model: str, text: str) -> _FakeResponse:
        return _FakeResponse({"count": len(text)})

    async def models(self) -> _FakeResponse:
        return _FakeResponse({"data": []})

    async def metrics(self, *, metrics_url: str | None = None) -> _FakeResponse:
        return _FakeResponse({})

    async def completion(self, payload: dict[str, Any]) -> _FakeResponse:
        return _FakeResponse({})

    @asynccontextmanager
    async def stream_completion(self, payload: dict[str, Any]) -> AsyncIterator[_FakeResponse]:
        yield _FakeResponse(
            {
                "choices": [{"message": {"content": "complete response"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2},
            }
        )


@pytest.mark.asyncio
async def test_http_measurement_client_preserves_sse_lines() -> None:
    """The adapter exposes raw SSE lines without parsing or coalescing them."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices": [{"delta": {"content": "one"}}]}\n\n'
            b'data:{"choices": [{"delta": {"content": "two"}}]}\n\n'
            b"data: [DONE]\n\n",
        )

    async with HTTPMeasurementClient(
        base_url="http://test",
        timeout=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        async with client.stream_completion({}) as response:
            assert [line async for line in response.aiter_lines() if line] == [
                'data: {"choices": [{"delta": {"content": "one"}}]}',
                'data:{"choices": [{"delta": {"content": "two"}}]}',
                "data: [DONE]",
            ]


@pytest.mark.asyncio
async def test_runner_counts_ordinary_json_when_streaming_is_ignored() -> None:
    """A 200 JSON completion is a valid fallback, not a zero-token stream."""
    sample = await throughput._stream_one(
        _FakeMeasurementPort(),
        "http://ignored",
        "model",
        [{"role": "user", "content": "respond"}],
        4,
        None,
    )

    assert sample.error is None
    assert sample.pp_tokens == 11
    assert sample.tg_tokens == 2
    assert sample.ttft_ms >= 0


@pytest.mark.asyncio
async def test_adapter_owns_routes_auth_and_cross_origin_metrics_auth() -> None:
    """The HTTP adapter, not a runner, decides routes and credential scope."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [], "count": 3})

    async with HTTPMeasurementClient(
        base_url="http://api.example:8080/v1",
        api_key="test-key",
        timeout=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.models()
        await client.tokenize(model="model", text="abc")
        await client.metrics(metrics_url="http://metrics.example:9090/metrics")

    assert [request.url.path for request in seen] == ["/v1/models", "/tokenize", "/metrics"]
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert seen[1].headers["authorization"] == "Bearer test-key"
    assert "authorization" not in seen[2].headers


@pytest.mark.asyncio
async def test_adapter_preserves_explicit_empty_completion_headers() -> None:
    """Custom wire formats can intentionally opt out of bearer auth headers."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    async with HTTPMeasurementClient(
        base_url="http://api.example/v1",
        api_key="test-key",
        completion_url="http://custom.example/generate",
        completion_headers={},
        timeout=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.completion({"prompt": "hello"})

    assert seen[0].url == "http://custom.example/generate"
    assert "authorization" not in seen[0].headers


def test_measurement_runners_do_not_import_concrete_http_client() -> None:
    """HTTP ownership stays in the adapter, not the timing-sensitive runners."""
    for runner in (throughput, speculative, context_pressure):
        source = inspect.getsource(runner)
        assert "import httpx" not in source
        assert "utils.urls import" not in source
        assert "def _headers" not in source
