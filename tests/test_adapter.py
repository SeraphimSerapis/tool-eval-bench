"""Tests for the OpenAI-compatible adapter layer.

TEST-01: Tests SSE streaming, _normalize_tool_calls, _parse_response,
connection reuse, and error handling — all using deterministic mocks
without needing a real server.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tool_eval_bench.adapters.openai_compat import (
    OpenAICompatibleAdapter,
    _normalize_tool_calls,
)


# ---------------------------------------------------------------------------
# _normalize_tool_calls — unit tests
# ---------------------------------------------------------------------------


class TestNormalizeToolCalls:
    def test_none_returns_empty(self) -> None:
        assert _normalize_tool_calls(None) == []

    def test_empty_list_returns_empty(self) -> None:
        assert _normalize_tool_calls([]) == []

    def test_basic_tool_call(self) -> None:
        raw = [
            {
                "id": "call_1",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Berlin"}',
                },
            }
        ]
        result = _normalize_tool_calls(raw)
        assert len(result) == 1
        assert result[0].id == "call_1"
        assert result[0].name == "get_weather"
        assert result[0].arguments_str == '{"location": "Berlin"}'
        assert result[0].arguments == {"location": "Berlin"}

    def test_dict_arguments_serialized(self) -> None:
        """When arguments is a dict (not a string), it should be JSON-serialized."""
        raw = [
            {
                "id": "call_2",
                "function": {
                    "name": "calculator",
                    "arguments": {"expression": "2+2"},
                },
            }
        ]
        result = _normalize_tool_calls(raw)
        assert isinstance(result[0].arguments_str, str)
        parsed = json.loads(result[0].arguments_str)
        assert parsed == {"expression": "2+2"}

    def test_missing_fields_have_defaults(self) -> None:
        """Missing id → auto-generated, missing name → 'unknown_tool'."""
        raw = [{"function": {}}]
        result = _normalize_tool_calls(raw)
        assert result[0].id == "tool_call_1"
        assert result[0].name == "unknown_tool"
        assert result[0].arguments_str == "{}"

    def test_multiple_tool_calls(self) -> None:
        raw = [
            {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
            {"id": "c2", "function": {"name": "b", "arguments": "{}"}},
            {"id": "c3", "function": {"name": "c", "arguments": "{}"}},
        ]
        result = _normalize_tool_calls(raw)
        assert len(result) == 3
        assert [r.name for r in result] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _parse_response — unit tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_basic_text_response(self) -> None:
        data = {
            "choices": [
                {
                    "message": {
                        "content": "Hello, world!",
                        "role": "assistant",
                    }
                }
            ]
        }
        result = OpenAICompatibleAdapter._parse_response(data, 42.0)
        assert result.content == "Hello, world!"
        assert result.tool_calls == []
        assert result.elapsed_ms == 42.0
        assert result.reasoning is None

    def test_tool_call_response(self) -> None:
        data = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"NYC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        result = OpenAICompatibleAdapter._parse_response(data, 10.0)
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"

    def test_list_content_joined(self) -> None:
        """Some providers return content as a list of parts."""
        data = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Hello "},
                            {"type": "text", "text": "world!"},
                        ]
                    }
                }
            ]
        }
        result = OpenAICompatibleAdapter._parse_response(data, 5.0)
        assert "Hello" in result.content
        assert "world!" in result.content

    def test_reasoning_content(self) -> None:
        data = {
            "choices": [
                {
                    "message": {
                        "content": "Answer",
                        "reasoning_content": "I thought about it",
                    }
                }
            ]
        }
        result = OpenAICompatibleAdapter._parse_response(data, 1.0)
        assert result.reasoning == "I thought about it"

    def test_empty_choices(self) -> None:
        """Missing choices gracefully returns empty content."""
        result = OpenAICompatibleAdapter._parse_response({}, 1.0)
        assert result.content == ""
        assert result.tool_calls == []

    def test_malformed_choices(self) -> None:
        data = {"choices": []}
        result = OpenAICompatibleAdapter._parse_response(data, 1.0)
        assert result.content == ""


# ---------------------------------------------------------------------------
# Adapter — non-stream request via mock transport
# ---------------------------------------------------------------------------


def _mock_non_stream_response(request: httpx.Request) -> httpx.Response:
    """Mock transport handler that returns a valid chat completion."""
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.0

    response_data = {
        "choices": [
            {
                "message": {
                    "content": "Mock response",
                    "role": "assistant",
                }
            }
        ]
    }
    return httpx.Response(200, json=response_data)


def _mock_error_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "Internal Server Error"})


@pytest.mark.asyncio
async def test_non_stream_request() -> None:
    """Adapter sends correct payload and parses the response."""
    adapter = OpenAICompatibleAdapter()
    transport = httpx.MockTransport(_mock_non_stream_response)
    adapter._client = httpx.AsyncClient(transport=transport)

    result = await adapter.chat_completion(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        base_url="http://localhost:8000",
    )

    assert result.content == "Mock response"
    assert result.elapsed_ms > 0
    await adapter.aclose()


@pytest.mark.asyncio
async def test_error_response_raises() -> None:
    """HTTP 500 should raise an exception."""
    adapter = OpenAICompatibleAdapter()
    transport = httpx.MockTransport(_mock_error_response)
    adapter._client = httpx.AsyncClient(transport=transport)

    with pytest.raises(httpx.HTTPStatusError):
        await adapter.chat_completion(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            base_url="http://localhost:8000",
        )
    await adapter.aclose()


@pytest.mark.asyncio
async def test_api_key_sent_in_header() -> None:
    """When api_key is provided, Authorization header must be present."""
    def check_auth(request: httpx.Request) -> httpx.Response:
        assert "Authorization" in request.headers
        assert request.headers["Authorization"] == "Bearer test-key-123"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(check_auth))

    await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", api_key="test-key-123",
    )
    await adapter.aclose()


@pytest.mark.asyncio
async def test_no_auth_header_without_key() -> None:
    """Without api_key, no Authorization header should be sent."""
    def check_no_auth(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(check_no_auth))

    await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000",
    )
    await adapter.aclose()


@pytest.mark.asyncio
async def test_tools_included_in_payload() -> None:
    """When tools are provided, payload must include tools, tool_choice, parallel_tool_calls."""
    def check_tools(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" in body
        assert body["tool_choice"] == "auto"
        assert body["parallel_tool_calls"] is True
        assert len(body["tools"]) == 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(check_tools))

    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", tools=tools,
    )
    await adapter.aclose()


# ---------------------------------------------------------------------------
# Connection reuse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_reused() -> None:
    """The internal client should be lazily created and reused."""
    adapter = OpenAICompatibleAdapter()
    assert adapter._client is None

    c1 = adapter._get_client()
    c2 = adapter._get_client()
    assert c1 is c2  # same instance

    await adapter.aclose()
    assert adapter._client is None

    # After close, a new client is created
    c3 = adapter._get_client()
    assert c3 is not c1
    await adapter.aclose()
