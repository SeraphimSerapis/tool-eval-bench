"""A body that is not valid JSON is flagged by every adapter and never graded.

Each adapter degrades an unparseable 200 to ``content="[malformed response]"``.
The ``malformed`` flag lets callers tell that placeholder from model text. The
accuracy plugins' side is covered in ``test_plugin_scoring.py``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters.anthropic import AnthropicAdapter
from tool_eval_bench.adapters.gemini import GeminiAdapter
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.plugin import TransportRejectedError, raise_for_transport_error
from tool_eval_bench.domain.scenarios import (
    Category,
    FailureKind,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioStatus,
)
from tool_eval_bench.runner.orchestrator import probe_tool_choice_required, run_scenario

_MALFORMED = ChatCompletionResult(content="[malformed response]", malformed=True)


def _proxy_page(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="<html>proxy</html>")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_type", "base_url"),
    [
        (OpenAICompatibleAdapter, "http://x:8000"),
        (AnthropicAdapter, "https://opencode.ai/zen/go/v1/messages"),
        (GeminiAdapter, "https://generativelanguage.googleapis.com/v1beta"),
    ],
)
async def test_adapter_flags_a_body_that_is_not_json(
    adapter_type: type[BackendAdapter], base_url: str
) -> None:
    adapter: Any = adapter_type()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(_proxy_page))

    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}], base_url=base_url, api_key="k"
    )
    await adapter.aclose()

    assert result.content == "[malformed response]"
    assert result.malformed is True
    assert result.transport_error_status is None


@pytest.mark.asyncio
async def test_a_parseable_body_is_not_flagged() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(ok))
    result = await adapter.chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}], base_url="http://x:8000"
    )
    await adapter.aclose()

    assert result.content == "hello"
    assert result.malformed is False


# ---------------------------------------------------------------------------
# Streamed bodies
# ---------------------------------------------------------------------------

# adapter, base URL, a valid SSE body saying "hi", a valid SSE body with no output
_STREAMS = [
    (
        OpenAICompatibleAdapter,
        "http://x:8000",
        b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n',
        b"data: [DONE]\n\n",
    ),
    (
        AnthropicAdapter,
        "https://opencode.ai/zen/go/v1/messages",
        b"event: content_block_delta\n"
        b'data: {"type": "content_block_delta", "index": 0,'
        b' "delta": {"type": "text_delta", "text": "hi"}}\n\n',
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ),
    (
        GeminiAdapter,
        "https://generativelanguage.googleapis.com/v1beta",
        b'data: {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}\n\n',
        b'data: {"candidates": []}\n\n',
    ),
]


async def _stream(
    adapter_type: type[BackendAdapter],
    base_url: str,
    body: bytes,
    headers: dict[str, str] | None = None,
) -> ChatCompletionResult:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers=headers)

    adapter: Any = adapter_type()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    result: ChatCompletionResult = await adapter.chat_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        base_url=base_url,
        api_key="k",
        stream=True,
    )
    await adapter.aclose()
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_type", "base_url", "sse", "empty"), _STREAMS)
@pytest.mark.parametrize("content_type", ["text/html", "text/event-stream"])
async def test_streamed_body_with_no_sse_event_is_flagged(
    adapter_type: type[BackendAdapter], base_url: str, sse: bytes, empty: bytes, content_type: str
) -> None:
    result = await _stream(
        adapter_type,
        base_url,
        b"<html>\n<body>proxy</body>\n</html>",
        {"content-type": content_type},
    )

    assert result.content == "[malformed response]"
    assert result.malformed is True
    assert result.transport_error_status is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_type", "base_url", "sse", "empty"), _STREAMS)
@pytest.mark.parametrize("headers", [None, {"content-type": "text/plain"}])
async def test_sse_without_an_event_stream_content_type_is_accepted(
    adapter_type: type[BackendAdapter],
    base_url: str,
    sse: bytes,
    empty: bytes,
    headers: dict[str, str] | None,
) -> None:
    result = await _stream(adapter_type, base_url, sse, headers)

    assert result.content == "hi"
    assert result.malformed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_type", "base_url", "sse", "empty"), _STREAMS)
async def test_an_empty_but_valid_stream_is_not_flagged(
    adapter_type: type[BackendAdapter], base_url: str, sse: bytes, empty: bytes
) -> None:
    result = await _stream(adapter_type, base_url, empty, {"content-type": "text/event-stream"})

    assert result.content == ""
    assert result.malformed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_type", "base_url", "sse", "empty"), _STREAMS)
@pytest.mark.parametrize("body", [b"", b"\n\n", b": keep-alive\n\n: keep-alive\n\n"])
async def test_a_stream_with_no_event_at_all_is_flagged(
    adapter_type: type[BackendAdapter], base_url: str, sse: bytes, empty: bytes, body: bytes
) -> None:
    """An empty 200 or a comment-only stream carried no model output to grade."""
    result = await _stream(adapter_type, base_url, body, {"content-type": "text/event-stream"})

    assert result.content == "[malformed response]"
    assert result.malformed is True


# adapter, base URL, a non-streamed JSON response saying "hi"
_JSON_BODIES = [
    (OpenAICompatibleAdapter, "http://x:8000", {"choices": [{"message": {"content": "hi"}}]}),
    (
        AnthropicAdapter,
        "https://opencode.ai/zen/go/v1/messages",
        {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"},
    ),
    (
        GeminiAdapter,
        "https://generativelanguage.googleapis.com/v1beta",
        {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_type", "base_url", "body"), _JSON_BODIES)
@pytest.mark.parametrize("indent", [None, 2])
async def test_a_json_completion_labelled_as_an_event_stream_is_parsed(
    adapter_type: type[BackendAdapter], base_url: str, body: dict[str, Any], indent: int | None
) -> None:
    """A server that ignored stream=true sent a whole completion, not a broken stream."""
    encoded = json.dumps(body, indent=indent).encode()

    result = await _stream(adapter_type, base_url, encoded, {"content-type": "text/event-stream"})

    assert result.content == "hi"
    assert result.malformed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_type", "base_url", "body"), _JSON_BODIES)
@pytest.mark.parametrize("content_type", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("stream", [True, False])
async def test_a_json_body_that_is_not_an_object_is_flagged(
    adapter_type: type[BackendAdapter],
    base_url: str,
    body: dict[str, Any],
    content_type: str,
    stream: bool,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'"hi"', headers={"content-type": content_type})

    adapter: Any = adapter_type()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    result = await adapter.chat_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        base_url=base_url,
        api_key="k",
        stream=stream,
    )
    await adapter.aclose()

    assert result.content == "[malformed response]"
    assert result.malformed is True


# What streamGenerateContent returns without alt=sse: a JSON array of chunks.
_GEMINI_CHUNKS = [
    {"candidates": [{"content": {"parts": [{"text": "h"}]}}]},
    {
        "candidates": [
            {
                "content": {"parts": [{"functionCall": {"name": "f", "args": {"x": 1}}}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
    },
    {"candidates": [{"content": {"parts": [{"text": "i"}]}}]},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("stream", [True, False])
async def test_gemini_merges_a_json_array_of_chunks(content_type: str, stream: bool) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=json.dumps(_GEMINI_CHUNKS).encode(), headers={"content-type": content_type}
        )

    adapter = GeminiAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    result = await adapter.chat_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="k",
        stream=stream,
    )
    await adapter.aclose()

    assert result.malformed is False
    assert result.content == "hi"
    assert [(c.name, json.loads(c.arguments_str)) for c in result.tool_calls] == [("f", {"x": 1})]
    assert (result.prompt_tokens, result.completion_tokens) == (3, 2)
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_gemini_reports_an_error_chunk_in_a_json_array() -> None:
    chunks = [_GEMINI_CHUNKS[0], {"error": {"code": 503, "message": "overloaded"}}]

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chunks)

    adapter = GeminiAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    result = await adapter.chat_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="k",
        stream=True,
    )
    await adapter.aclose()

    assert result.transport_error_status == 503
    assert result.content == "[server error 503] overloaded"


def test_plugin_guard_raises_on_a_malformed_result() -> None:
    with pytest.raises(TransportRejectedError, match="not valid JSON"):
        raise_for_transport_error(_MALFORMED)
    ok = ChatCompletionResult(content="[malformed response]")
    assert raise_for_transport_error(ok) is ok


# ---------------------------------------------------------------------------
# Scored runner
# ---------------------------------------------------------------------------


class _Script(BackendAdapter):
    def __init__(self, results: list[ChatCompletionResult]) -> None:
        self.results = list(results)

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        return self.results.pop(0)


def _scenario_that_passes_anything() -> ScenarioDefinition:
    """Grades any answer as a PASS, so only the runner can turn the turn into a failure."""
    return ScenarioDefinition(
        id="MAL-01",
        title="malformed",
        category=Category.A,
        user_message="hi",
        description="",
        handle_tool_call=lambda state, call: {"ok": True},
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "graded"),
    )


async def _run(results: list[ChatCompletionResult]) -> Any:
    return await run_scenario(
        _Script(results),
        model="m",
        base_url="http://x",
        api_key=None,
        scenario=_scenario_that_passes_anything(),
    )


@pytest.mark.asyncio
async def test_runner_does_not_grade_a_malformed_first_turn() -> None:
    result = await _run([_MALFORMED])

    assert result.status == ScenarioStatus.FAIL
    assert result.failure_kind == FailureKind.SERVER_ERROR
    assert result.is_infrastructure_failure
    assert "transport_error=malformed_response" in result.raw_log


@pytest.mark.asyncio
async def test_runner_does_not_grade_a_malformed_turn_after_a_tool_call() -> None:
    # A 4xx after a model-authored tool call is graded as the model's fault;
    # an unparseable body is not, because the model cannot author the HTTP body.
    call = ProviderToolCall(id="c1", name="get_weather", arguments_str=json.dumps({"q": 1}))
    result = await _run([ChatCompletionResult(content="", tool_calls=[call]), _MALFORMED])

    assert result.failure_kind == FailureKind.SERVER_ERROR
    assert result.is_infrastructure_failure
    assert result.turn_count == 2


@pytest.mark.asyncio
async def test_runner_still_grades_ordinary_text() -> None:
    result = await _run([ChatCompletionResult(content="[malformed response]")])

    assert result.status == ScenarioStatus.PASS


@pytest.mark.asyncio
async def test_tool_choice_probe_reports_an_unreadable_response() -> None:
    probe = await probe_tool_choice_required(
        _Script([_MALFORMED]), model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )

    assert probe.enforced is False
    assert probe.detail == "probe response was not valid JSON"
