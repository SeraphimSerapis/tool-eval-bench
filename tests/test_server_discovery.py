"""Behavioral coverage for inference-server discovery."""

from __future__ import annotations

import httpx
import pytest
from rich.console import Console

from tool_eval_bench.cli import server


class _FakeAsyncClient:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def get(self, url: str):
        result = next(self._responses)
        if isinstance(result, Exception):
            raise result
        return result


def _response(status: int, url: str, **kwargs) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", url), **kwargs)


@pytest.mark.asyncio
async def test_discovery_skips_connection_errors_and_uses_models_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_error = httpx.ConnectError(
        "refused",
        request=httpx.Request("GET", "http://localhost:8000/v1/models"),
    )
    responses = [
        connect_error,
        _response(404, "http://localhost:4000/v1/models"),
        _response(
            200,
            "http://localhost:4000/models",
            headers={"server": "sglang"},
            json={"data": []},
        ),
    ]
    monkeypatch.setattr(
        server, "DISCOVERY_PORTS", [(8000, "vllm", "vLLM"), (4000, "litellm", "LiteLLM")]
    )
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(responses))

    result = await server._discover_async()

    assert result == ("http://localhost:4000", "sglang", "SGLang", 4000)


def test_discover_server_renders_interactive_success(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(record=True, width=100)
    discovered = ("http://localhost:8080", "llamacpp", "llama.cpp", 8080)
    monkeypatch.setattr(
        server.asyncio,
        "run",
        lambda coro: (coro.close(), discovered)[1],
    )

    assert server.discover_server(console=console) == ("http://localhost:8080", "llamacpp")
    assert "Auto-discovered llama.cpp" in console.export_text()


def test_server_headless_error_delegates_structured_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        "tool_eval_bench.cli.helpers.emit_headless_error",
        lambda code, message, *, exit_code: calls.append((code, message, exit_code)),
    )

    server._headless_error("no_server", "nothing listening", exit_code=3)

    assert calls == [("no_server", "nothing listening", 3)]


def _ports(monkeypatch: pytest.MonkeyPatch, responses: list) -> None:
    monkeypatch.setattr(
        server, "DISCOVERY_PORTS", [(3000, "unknown", "web app"), (8000, "unknown", "server")]
    )
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(responses))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "imposter",
    [
        {"text": "<html>dev server</html>", "headers": {"Content-Type": "text/html"}},
        {"text": ""},
        {"json": ["not", "an", "object"]},
        {"json": {"status": "ok"}},
        {"json": {"data": "not a list"}},
    ],
    ids=["html", "empty", "json-array", "object-without-list", "data-not-a-list"],
)
async def test_a_200_that_is_not_a_model_listing_is_skipped(
    monkeypatch: pytest.MonkeyPatch, imposter: dict
) -> None:
    _ports(
        monkeypatch,
        [
            _response(200, "http://localhost:3000/v1/models", **imposter),
            _response(200, "http://localhost:8000/v1/models", json={"data": [{"id": "m"}]}),
        ],
    )

    result = await server._discover_async()

    assert result is not None and result[0] == "http://localhost:8000"


@pytest.mark.asyncio
async def test_no_server_when_every_port_answers_with_html(monkeypatch: pytest.MonkeyPatch) -> None:
    _ports(
        monkeypatch,
        [
            _response(200, "http://localhost:3000/v1/models", text="<html></html>"),
            _response(200, "http://localhost:8000/v1/models", text="<html></html>"),
        ],
    )

    assert await server._discover_async() is None


@pytest.mark.asyncio
async def test_a_port_that_does_not_speak_http_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("GET", "http://localhost:3000/v1/models")
    _ports(
        monkeypatch,
        [
            httpx.RemoteProtocolError("illegal status line", request=request),
            _response(200, "http://localhost:8000/v1/models", json={"models": []}),
        ],
    )

    result = await server._discover_async()

    assert result is not None and result[0] == "http://localhost:8000"
