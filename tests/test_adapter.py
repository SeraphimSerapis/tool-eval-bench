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


# ---------------------------------------------------------------------------
# 4xx graceful error handling
# ---------------------------------------------------------------------------


def _mock_4xx_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(422, json={"error": "Unprocessable Entity"})


@pytest.mark.asyncio
async def test_4xx_returns_graceful_result() -> None:
    """HTTP 4xx should be handled gracefully, returning a ChatCompletionResult."""
    adapter = OpenAICompatibleAdapter()
    transport = httpx.MockTransport(_mock_4xx_response)
    adapter._client = httpx.AsyncClient(transport=transport)

    result = await adapter.chat_completion(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        base_url="http://localhost:8000",
    )

    assert "[server error 422]" in result.content
    assert result.tool_calls == []
    assert result.elapsed_ms > 0
    await adapter.aclose()


# ---------------------------------------------------------------------------
# response_format and extra_params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_format_included_in_payload() -> None:
    """When response_format is provided, it must appear in the payload."""
    def check_format(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "response_format" in body
        assert body["response_format"]["type"] == "json_object"
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(check_format))

    await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000",
        response_format={"type": "json_object"},
    )
    await adapter.aclose()


@pytest.mark.asyncio
async def test_extra_params_merged_into_payload() -> None:
    """Extra params should be merged into the request payload."""
    def check_extra(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["seed"] == 42
        assert body["top_p"] == 0.9
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(check_extra))

    await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000",
        extra_params={"seed": 42, "top_p": 0.9},
    )
    await adapter.aclose()


# ---------------------------------------------------------------------------
# SSE streaming tests
# ---------------------------------------------------------------------------


def _sse_lines(*events: str, done: bool = True) -> str:
    """Build an SSE response body from a list of JSON-encodable event strings."""
    lines = []
    for ev in events:
        lines.append(f"data: {ev}\n\n")
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines)


def _mock_stream_transport(sse_body: str, status: int = 200):
    """Create a mock transport that returns an SSE stream."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_stream_basic_content() -> None:
    """Streaming should accumulate content from multiple chunks."""
    chunks = [
        json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
        json.dumps({"choices": [{"delta": {"content": " world"}}]}),
        json.dumps({"choices": [{"delta": {"content": "!"}}]}),
    ]
    body = _sse_lines(*chunks)

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=_mock_stream_transport(body))

    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", stream=True,
    )

    assert result.content == "Hello world!"
    assert result.tool_calls == []
    assert result.elapsed_ms > 0
    assert result.ttft_ms is not None
    assert result.ttft_ms > 0
    await adapter.aclose()


@pytest.mark.asyncio
async def test_stream_tool_calls() -> None:
    """Streaming should accumulate tool calls from delta chunks."""
    chunks = [
        json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "tc_1", "function": {"name": "get_weather", "arguments": '{"loc'}}
        ]}}]}),
        json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'ation": "NYC"}'}}
        ]}}]}),
    ]
    body = _sse_lines(*chunks)

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=_mock_stream_transport(body))

    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", stream=True,
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments_str == '{"location": "NYC"}'
    assert result.tool_calls[0].id == "tc_1"
    assert result.ttft_ms is not None
    await adapter.aclose()


@pytest.mark.asyncio
async def test_stream_reasoning_content() -> None:
    """Streaming should capture reasoning_content from deltas."""
    chunks = [
        json.dumps({"choices": [{"delta": {"reasoning_content": "Let me think..."}}]}),
        json.dumps({"choices": [{"delta": {"content": "The answer is 42"}}]}),
    ]
    body = _sse_lines(*chunks)

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=_mock_stream_transport(body))

    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", stream=True,
    )

    assert result.content == "The answer is 42"
    assert result.reasoning == "Let me think..."
    await adapter.aclose()


@pytest.mark.asyncio
async def test_stream_usage_extraction() -> None:
    """Streaming should extract token usage from the final chunk."""
    chunks = [
        json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
        json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
    ]
    body = _sse_lines(*chunks)

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=_mock_stream_transport(body))

    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", stream=True,
    )

    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    await adapter.aclose()


@pytest.mark.asyncio
async def test_stream_malformed_json_skipped() -> None:
    """Malformed JSON chunks in a stream should be silently skipped."""
    body = _sse_lines(
        "not valid json",
        json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
    )

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=_mock_stream_transport(body))

    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", stream=True,
    )

    assert result.content == "ok"
    await adapter.aclose()


@pytest.mark.asyncio
async def test_stream_empty_choices_skipped() -> None:
    """Chunks with empty choices should be skipped."""
    body = _sse_lines(
        json.dumps({"choices": []}),
        json.dumps({"choices": [{"delta": {"content": "data"}}]}),
    )

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=_mock_stream_transport(body))

    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", stream=True,
    )

    assert result.content == "data"
    await adapter.aclose()


@pytest.mark.asyncio
async def test_stream_5xx_raises() -> None:
    """HTTP 500 on a stream should raise HTTPStatusError."""
    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(
        transport=_mock_stream_transport("", status=500)
    )

    with pytest.raises(httpx.HTTPStatusError):
        await adapter.chat_completion(
            model="m", messages=[{"role": "user", "content": "hi"}],
            base_url="http://localhost:8000", stream=True,
        )
    await adapter.aclose()


@pytest.mark.asyncio
async def test_stream_4xx_returns_graceful_result() -> None:
    """HTTP 4xx on a stream should be handled gracefully."""
    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(
        transport=_mock_stream_transport("", status=422)
    )

    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", stream=True,
    )

    assert "[server error 422]" in result.content
    assert result.tool_calls == []
    await adapter.aclose()


@pytest.mark.asyncio
async def test_stream_multiple_tool_calls() -> None:
    """Streaming should handle multiple concurrent tool calls via index."""
    chunks = [
        json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "tc_a", "function": {"name": "func_a", "arguments": "{}"}}
        ]}}]}),
        json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 1, "id": "tc_b", "function": {"name": "func_b", "arguments": "{}"}}
        ]}}]}),
    ]
    body = _sse_lines(*chunks)

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=_mock_stream_transport(body))

    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        base_url="http://localhost:8000", stream=True,
    )

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].name == "func_a"
    assert result.tool_calls[0].id == "tc_a"
    assert result.tool_calls[1].name == "func_b"
    assert result.tool_calls[1].id == "tc_b"
    await adapter.aclose()
