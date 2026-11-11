"""Regression tests for orchestrator, runner, and backend adapter state machines."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters.anthropic import AnthropicAdapter, _to_anthropic_messages
from tool_eval_bench.adapters.gemini import GeminiAdapter, _to_gemini_contents
from tool_eval_bench.adapters.http_retry import (
    RateLimitCoordinator,
    RetryingHTTPAdapter,
    _retry_after_seconds,
)
from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
from tool_eval_bench.adapters.openai_compat import (
    OpenAICompatibleAdapter,
    _normalize_tool_calls,
    _repair_streamed_tool_args,
)
from tool_eval_bench.adapters.systemone import SystemOneAdapter
from tool_eval_bench.domain.adapters import (
    BackendAdapter,
    ChatCompletionResult,
    ProviderToolCall,
)
from tool_eval_bench.domain.decision import YesNoQuestion
from tool_eval_bench.domain.models import ChatMessage
from tool_eval_bench.domain.scenarios import (
    Category,
    FailureKind,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.runner.context_pressure import (
    _RESERVED_FOR_OUTPUT,
    _RESERVED_FOR_SCENARIO,
    prepare_context_pressure,
)
from tool_eval_bench.runner.llama_benchy import _stat_mean
from tool_eval_bench.runner.orchestrator import (
    _assistant_message,
    _repair_json_str,
    run_all_scenarios,
    run_scenario,
)
from tool_eval_bench.runner.perf_presets import PerfPass, estimate_pass_seconds
from tool_eval_bench.runner.spec_detection import SpecDecodeInfo, parse_prometheus_spec_metrics
from tool_eval_bench.runner.spec_live import _parse_snapshot
from tool_eval_bench.runner.speculative import (
    SpecDecodeSample,
    measure_spec_single,
    pool_spec_samples,
)
from tool_eval_bench.runner.throughput import (
    TokenizerConfig,
    _stream_one,
    calibrate,
    measure_concurrent,
)


class _ByteChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _ScriptedAdapter(BackendAdapter):
    """Deterministic adapter returning scripted results or raising exceptions."""

    def __init__(self, steps: list[ChatCompletionResult | BaseException]) -> None:
        self._steps = list(steps)
        self.recorded_messages: list[list[ChatMessage]] = []

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: float = 120.0,
        api_key: str | None = None,
        base_url: str = "",
        extra_params: dict[str, Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = True,
        conversation_id: str | None = None,
    ) -> ChatCompletionResult:
        self.recorded_messages.append(json.loads(json.dumps(messages)))
        if not self._steps:
            return ChatCompletionResult(content="done", finish_reason="stop")
        step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def _make_scenario(
    scenario_id: str = "FIX-01",
    *,
    evaluate: Any = None,
    handle_tool_call: Any = None,
) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=scenario_id,
        title="Regression test scenario",
        description="Expected test tool workflow.",
        category=Category.A,
        user_message="Run the workflow.",
        handle_tool_call=handle_tool_call or (lambda state, call: {"ok": True, "id": call.id}),
        evaluate=evaluate or (lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok")),
    )


# 1. Orchestrator state machine and tool-call round-trip fixes


def test_repair_json_str_requires_json_object() -> None:
    assert _repair_json_str('{"city": "Berlin"}') == '{"city": "Berlin"}'
    assert json.loads(_repair_json_str('{"city": "Ber')) == {"city": "Ber"}

    for non_object in ("null", "[1, 2, 3]", "42", "true", '"plain"', "[1, 2"):
        assert _repair_json_str(non_object) == "{}"


@pytest.mark.asyncio
async def test_orchestrator_normalizes_non_object_tool_args_for_history_only() -> None:
    seen_records: list[ToolCallRecord] = []

    def handle(state: ScenarioState, call: ToolCallRecord) -> dict[str, Any]:
        seen_records.append(call)
        return {"status": "handled"}

    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content="",
                tool_calls=[ProviderToolCall(id="c1", name="get_weather", arguments_str="[1, 2]")],
                finish_reason="tool_calls",
            ),
            ChatCompletionResult(content="Done.", finish_reason="stop"),
        ]
    )
    result = await run_scenario(
        adapter,
        model="m",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=_make_scenario("FIX-ARGS", handle_tool_call=handle),
    )

    assert result.status == ScenarioStatus.PASS
    assert seen_records[0].raw_arguments == "[1, 2]"
    assert seen_records[0].arguments == {}
    turn_two_assistant = next(
        msg for msg in adapter.recorded_messages[1] if msg["role"] == "assistant"
    )
    assert turn_two_assistant["tool_calls"][0]["function"]["arguments"] == "{}"


@pytest.mark.asyncio
async def test_orchestrator_deduplicates_and_populates_tool_call_ids_across_turns() -> None:
    shared_fixture_call = ProviderToolCall(id="toolu_1", name="get_weather", arguments_str="{}")
    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content="",
                tool_calls=[
                    shared_fixture_call,
                    ProviderToolCall(id="toolu_1", name="get_weather", arguments_str="{}"),
                    ProviderToolCall(id="", name="get_weather", arguments_str="{}"),
                ],
                finish_reason="tool_calls",
            ),
            ChatCompletionResult(
                content="",
                tool_calls=[
                    ProviderToolCall(id="toolu_1", name="send_email", arguments_str="{}"),
                    ProviderToolCall(id="ctx_call_1", name="send_email", arguments_str="{}"),
                ],
                finish_reason="tool_calls",
            ),
            ChatCompletionResult(content="All set.", finish_reason="stop"),
        ]
    )
    pressure_messages: list[ChatMessage] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "ctx_call_1",
                    "type": "function",
                    "function": {"name": "search_files", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "ctx_call_1", "name": "search_files", "content": "{}"},
    ]

    await run_scenario(
        adapter,
        model="m",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=_make_scenario("FIX-IDS"),
        context_pressure_messages=pressure_messages,
    )

    # Original ProviderToolCall fixture must not be mutated in place.
    assert shared_fixture_call.id == "toolu_1"

    final_history = adapter.recorded_messages[2]
    all_assistant_ids = [
        tc["id"]
        for msg in final_history
        if msg["role"] == "assistant"
        for tc in (msg.get("tool_calls") or [])
    ]
    assert len(all_assistant_ids) == 6
    assert len(set(all_assistant_ids)) == 6
    assert all(bool(cid) for cid in all_assistant_ids)

    # Anthropic and Gemini wire converters must keep distinct IDs and names across turns.
    anthropic_msgs, _ = _to_anthropic_messages(final_history)
    tool_use_ids = [
        block["id"]
        for msg in anthropic_msgs
        if msg["role"] == "assistant"
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    assert len(tool_use_ids) == len(set(tool_use_ids)) == 6

    gemini_contents, _ = _to_gemini_contents(final_history)
    func_responses = [
        part["functionResponse"]["name"]
        for entry in gemini_contents
        for part in entry["parts"]
        if "functionResponse" in part
    ]
    assert func_responses == [
        "search_files",
        "get_weather",
        "get_weather",
        "get_weather",
        "send_email",
        "send_email",
    ]


@pytest.mark.asyncio
async def test_orchestrator_early_returns_preserve_prior_turn_metrics() -> None:
    first_turn = ChatCompletionResult(
        content="",
        tool_calls=[
            ProviderToolCall(id="c1", name="get_weather", arguments_str='{"city": "Paris"}')
        ],
        prompt_tokens=120,
        completion_tokens=30,
        ttft_ms=15.0,
        finish_reason="tool_calls",
    )

    # 1. Turn 2 malformed response
    res_malformed = await run_scenario(
        _ScriptedAdapter([first_turn, ChatCompletionResult(content="[malformed]", malformed=True)]),
        model="m",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=_make_scenario("FIX-METRIC-1"),
    )
    assert res_malformed.failure_kind == FailureKind.SERVER_ERROR
    assert res_malformed.prompt_tokens == 120
    assert res_malformed.completion_tokens == 30
    assert res_malformed.tool_call_arg_bytes == len(b'{"city": "Paris"}')
    assert res_malformed.tool_calls_made == ["get_weather(city=Paris)"]
    assert res_malformed.expected_behavior == "Expected test tool workflow."

    # 2. Turn 2 runtime timeout
    res_timeout = await run_scenario(
        _ScriptedAdapter([first_turn, httpx.ReadTimeout("timed out")]),
        model="m",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=_make_scenario("FIX-METRIC-2"),
    )
    assert res_timeout.failure_kind == FailureKind.TIMEOUT
    assert res_timeout.prompt_tokens == 120
    assert res_timeout.completion_tokens == 30
    assert res_timeout.tool_calls_made == ["get_weather(city=Paris)"]

    # 3. Evaluator exception after completed turns
    def broken_eval(state: ScenarioState) -> ScenarioEvaluation:
        raise RuntimeError("boom in evaluator")

    res_eval = await run_scenario(
        _ScriptedAdapter([first_turn, ChatCompletionResult(content="Sunny", finish_reason="stop")]),
        model="m",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=_make_scenario("FIX-METRIC-3", evaluate=broken_eval),
    )
    assert res_eval.failure_kind == FailureKind.EVALUATOR_ERROR
    assert res_eval.prompt_tokens == 120
    assert res_eval.completion_tokens == 30
    assert res_eval.tool_calls_made == ["get_weather(city=Paris)"]


@pytest.mark.asyncio
async def test_orchestrator_zero_max_turns_boundary() -> None:
    scenario = _make_scenario(
        "FIX-ZERO-TURNS",
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.FAIL, 0, "no turns"),
    )
    result = await run_scenario(
        _ScriptedAdapter([]),
        model="m",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=scenario,
        max_turns=0,
    )
    assert result.turn_count == 0
    assert result.turn_budget_exceeded is True
    assert result.failure_kind == FailureKind.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_run_all_scenarios_parallel_propagates_cancelled_error() -> None:
    adapter = _ScriptedAdapter([asyncio.CancelledError("user abort")])
    scenarios = [_make_scenario("FIX-CANCEL-1"), _make_scenario("FIX-CANCEL-2")]
    with pytest.raises(asyncio.CancelledError):
        await run_all_scenarios(
            adapter,
            model="m",
            base_url="http://localhost:8000",
            api_key=None,
            scenarios=scenarios,
            concurrency=2,
        )


# 2. OpenAI-compatible adapter fixes


def test_openai_normalize_tool_calls_and_parse_response_null_safety() -> None:
    calls = _normalize_tool_calls(
        [
            "not-a-dict",  # type: ignore[list-item]
            {"id": None, "function": None, "extra_content": "bad"},
            {"id": "", "function": {"name": None, "arguments": None}},
            {"id": "c3", "function": {"name": "calc", "arguments": [1, 2]}},
        ]
    )
    assert len(calls) == 3
    assert calls[0].id == "tool_call_2"
    assert calls[0].name == "unknown_tool"
    assert calls[0].arguments_str == "{}"
    assert calls[0].extra_content is None
    assert calls[1].id == "tool_call_3"
    assert calls[1].arguments_str == "{}"
    assert calls[2].arguments_str == "[1, 2]"
    assert calls[2].arguments == {}

    assert _repair_streamed_tool_args("[1, 2") == "[1, 2"

    for payload in (
        {"choices": None, "usage": "bad"},
        {"choices": [None]},
        {"choices": [{"message": None}]},
    ):
        parsed = OpenAICompatibleAdapter._parse_response(payload, 1.0)
        assert parsed.content == ""
        assert parsed.tool_calls == []

    err_parsed = OpenAICompatibleAdapter._parse_response(
        {"error": {"message": "Backend overloaded", "code": 503}},
        2.0,
    )
    assert err_parsed.transport_error_status == 500
    assert err_parsed.transport_error_is_infrastructure is True


@pytest.mark.asyncio
async def test_openai_stream_multiline_sse_crlf_and_non_dict_json() -> None:
    sse_bytes = (
        b"data: {broken\r\n"
        b"\r\n"
        b"data: {\r\n"
        b": keepalive\r\n"
        b'data:   "choices": [\r\n'
        b'data:     {"delta": {"content": "Hel", "tool_calls": ["skip"]}}\r\n'
        b"data:   ]\r\n"
        b"data: }\r\n\r\n"
        b"data: {\r\n"
        b'data:   "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],\r\n'
        b'data:   "usage": {"prompt_tokens": 9, "completion_tokens": 2}\r\n'
        b"data: }"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-mode") == "json-list":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=b"[1, 2, 3]",
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ByteChunks([sse_bytes]),
        )

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        ok = await adapter.chat_completion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            base_url="http://localhost:8000",
            stream=True,
        )
        assert ok.content == "Hello"
        assert ok.prompt_tokens == 9
        assert ok.completion_tokens == 2
        assert ok.finish_reason == "stop"

        bad_adapter = OpenAICompatibleAdapter(default_headers={"x-mode": "json-list"})
        bad_adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            malformed = await bad_adapter.chat_completion(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                base_url="http://localhost:8000",
                stream=True,
            )
            assert malformed.malformed is True
        finally:
            await bad_adapter.aclose()
    finally:
        await adapter.aclose()


# 3. Anthropic adapter fixes


@pytest.mark.asyncio
async def test_anthropic_stream_signature_concat_input_preservation_and_refusal() -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "plan"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig-A-"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig-B"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_pre",
                "name": "get_weather",
                "input": {"city": "Tokyo"},
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 8},
        },
    ]
    # Emit one event across multiple data: lines with CRLF to exercise multi-line SSE.
    first_pretty = "\r\n".join(
        f"data: {line}" for line in json.dumps(events[0], indent=2).splitlines()
    )
    rest_lines = "\r\n\r\n".join(f"data: {json.dumps(ev)}" for ev in events[1:])
    sse_payload = f"{first_pretty}\r\n\r\n{rest_lines}\r\n\r\n".encode()

    refusal_events = "\n\n".join(
        f"data: {json.dumps(ev)}"
        for ev in [
            {"type": "message_start", "message": {"usage": {"input_tokens": 4}}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "refusal", "stop_details": {"category": "cyber"}},
                "usage": {"output_tokens": 0},
            },
        ]
    ).encode()

    truncated_events = "\n\n".join(
        f"data: {json.dumps(ev)}"
        for ev in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_cut",
                    "name": "get_weather",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city": "San Fr'},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens"},
                "usage": {"output_tokens": 20},
            },
        ]
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        mode = request.headers.get("x-case", "normal")
        if mode == "refusal":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=refusal_events
            )
        if mode == "truncated":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=truncated_events
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_payload
        )

    for case, expected_check in (("normal", None), ("refusal", None), ("truncated", None)):
        _ = expected_check
        adapter = AnthropicAdapter(default_headers={"x-case": case})
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            if case == "normal":
                res = await adapter.chat_completion(
                    model="claude",
                    messages=[{"role": "user", "content": "hi"}],
                    base_url="https://api.anthropic.com",
                    stream=True,
                )
                assert res.tool_calls[0].arguments == {"city": "Tokyo"}
                blocks = res.message_extra_content["anthropic"]["thinking_blocks"]  # type: ignore[index]
                assert blocks[0]["signature"] == "sig-A-sig-B"
            elif case == "refusal":
                res = await adapter.chat_completion(
                    model="claude",
                    messages=[{"role": "user", "content": "hi"}],
                    base_url="https://api.anthropic.com",
                    stream=True,
                )
                assert res.content == "[no content: refusal (cyber)]"
            else:
                scen_res = await run_scenario(
                    adapter,
                    model="claude",
                    base_url="https://api.anthropic.com",
                    api_key="k",
                    scenario=_make_scenario(
                        "FIX-ANTH-TRUNC",
                        evaluate=lambda s: ScenarioEvaluation(ScenarioStatus.FAIL, 0, "wrong"),
                    ),
                )
                assert scen_res.failure_kind == FailureKind.REASONING_TRUNCATED
        finally:
            await adapter.aclose()

    err_unstreamed = AnthropicAdapter._parse_response(
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
        3.0,
    )
    assert err_unstreamed.transport_error_status == 529
    assert err_unstreamed.transport_error_is_infrastructure is True


# 4. Gemini adapter fixes


@pytest.mark.asyncio
async def test_gemini_thought_signature_round_trip_and_stream_block_reason() -> None:
    stream_chunks = [
        {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "", "thoughtSignature": "sig-final-chunk"}]},
                    "finishReason": "STOP",
                }
            ]
        },
    ]
    pretty_first = "\r\n".join(
        f"data: {line}" for line in json.dumps(stream_chunks[0], indent=2).splitlines()
    )
    sse_body = f"{pretty_first}\r\n\r\ndata: {json.dumps(stream_chunks[1])}\r\n\r\n".encode()

    blocked_body = f"data: {json.dumps({'promptFeedback': {'blockReason': 'SAFETY'}})}\n\n".encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-blocked") == "1":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=blocked_body
            )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_body)

    adapter = GeminiAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        res = await adapter.chat_completion(
            model="gemini-3-pro",
            messages=[{"role": "user", "content": "hi"}],
            base_url="https://generativelanguage.googleapis.com",
            stream=True,
        )
        assert res.content == "Hello"
        assert res.message_extra_content == {"thoughtSignature": "sig-final-chunk"}

        rebuilt = _assistant_message(res)
        contents, _ = _to_gemini_contents([{"role": "user", "content": "hi"}, rebuilt])
        assert contents[1]["parts"] == [{"text": "Hello", "thoughtSignature": "sig-final-chunk"}]
    finally:
        await adapter.aclose()

    blocked_adapter = GeminiAdapter(default_headers={"x-blocked": "1"})
    blocked_adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        blocked = await blocked_adapter.chat_completion(
            model="gemini-3-pro",
            messages=[{"role": "user", "content": "hi"}],
            base_url="https://generativelanguage.googleapis.com",
            stream=True,
        )
        assert blocked.content == "[no content: SAFETY]"
    finally:
        await blocked_adapter.aclose()

    # Message-level fallback signature on parallel tool calls only attaches to first call.
    parallel_contents, _ = _to_gemini_contents(
        [
            {
                "role": "assistant",
                "content": "",
                "extra_content": {"thoughtSignature": "sig-shared"},
                "tool_calls": [
                    {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "c2", "function": {"name": "b", "arguments": "{}"}},
                ],
            }
        ]
    )
    parts = parallel_contents[0]["parts"]
    assert parts[0].get("thoughtSignature") == "sig-shared"
    assert "thoughtSignature" not in parts[1]

    err_resp = GeminiAdapter._parse_response(
        {"error": {"code": 503, "message": "Service unavailable"}},
        1.5,
    )
    assert err_resp.transport_error_status == 503
    assert err_resp.transport_error_is_infrastructure is True


# 5. HTTP retry, rate-limit coordinator, and SystemOne fixes


@pytest.mark.asyncio
async def test_http_retry_nan_slot_rollback_and_stream_error_pacing() -> None:
    for header_val in ("nan", "inf", "-inf"):
        resp = httpx.Response(429, headers={"Retry-After": header_val})
        assert _retry_after_seconds(resp) is None

    body_inf = httpx.Response(
        429,
        json={
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "Infinitys",
                    }
                ]
            }
        },
    )
    assert _retry_after_seconds(body_inf) is None

    # Cancelled waiter inside RateLimitCoordinator.acquire rolls back its slot.
    coord = RateLimitCoordinator(step_seconds=5.0, max_interval_seconds=10.0)
    await coord.on_rate_limited(5.0)
    slot_before = coord._next_slot
    waiter = asyncio.create_task(coord.acquire())
    await asyncio.sleep(0)
    assert coord._next_slot > slot_before
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert coord._next_slot <= coord._pause_until

    # In-stream 500/429 soft errors must not count toward pacing decay.
    retrying = RetryingHTTPAdapter()
    await retrying._rate_limits.on_rate_limited(0.0)
    assert retrying._rate_limits.min_interval == 0.5
    for _ in range(10):
        await retrying._with_retries(
            lambda: asyncio.sleep(
                0,
                result=ChatCompletionResult(content="[err]", transport_error_status=503),
            ),
            url="http://localhost:8000/v1/chat/completions",
        )
    assert retrying._rate_limits.min_interval == 0.5

    for _ in range(8):
        await retrying._with_retries(
            lambda: asyncio.sleep(0, result=ChatCompletionResult(content="ok")),
            url="http://localhost:8000/v1/chat/completions",
        )
    assert retrying._rate_limits.min_interval == 0.0


@pytest.mark.asyncio
async def test_systemone_handles_null_input_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"answers": {"go": {"noul": 0.9}}, "usage": {"input_tokens": None}},
        )

    adapter = SystemOneAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        res = await adapter.decide(
            model="m",
            state="ready",
            questions={"go": YesNoQuestion(instructions="Proceed?")},
            base_url="http://localhost:8000",
        )
        assert res.answers["go"].probability_yes == pytest.approx(0.9)  # type: ignore[union-attr]
        assert res.input_tokens == 0
    finally:
        await adapter.aclose()


# 6. Runner benchmark edge cases (context_pressure, throughput, spec, presets)


@pytest.mark.asyncio
async def test_context_pressure_zero_available_and_negative_fill_tokens() -> None:
    exact_reserved = _RESERVED_FOR_OUTPUT + _RESERVED_FOR_SCENARIO

    cfg = await prepare_context_pressure(
        "http://localhost:8000",
        "m",
        None,
        0.5,
        context_size_override=exact_reserved,
        client_factory=HTTPMeasurementClient,
        fill_tokens=0,
    )
    assert cfg.ratio == 0.0
    assert cfg.fill_tokens == 0

    with pytest.raises(ValueError, match="non-negative"):
        await prepare_context_pressure(
            "http://localhost:8000",
            "m",
            None,
            0.5,
            context_size_override=32768,
            client_factory=HTTPMeasurementClient,
            fill_tokens=-5,
        )


@pytest.mark.asyncio
async def test_throughput_streamed_json_null_fields_and_cancellation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/tokenize"):
            return httpx.Response(404)
        mode = request.headers.get("x-tp-mode", "json-stream")
        if mode == "probe-null":
            return httpx.Response(200, json={"usage": {"prompt_tokens": None}})
        if mode == "json-stream":
            payload = json.dumps(
                {
                    "choices": [{"message": {"content": "hello world"}}],
                    "usage": {"prompt_tokens": 14, "completion_tokens": 2},
                }
            ).encode()
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=_ByteChunks([payload]),
            )
        sse_chunks = (
            b"data: 5\r\n\r\n"
            b'data: {"choices": [{"delta": null}]}\r\n\r\n'
            b"data: {\r\n"
            b'data:   "choices": [{"delta": {"content": "tok"}}],\r\n'
            b'data:   "usage": {"prompt_tokens": 10, "completion_tokens": 3}\r\n'
            b"data: }\r\n\r\n"
            b'data: {"choices": [], "usage": {"prompt_tokens": null, "completion_tokens": null}}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=_ByteChunks([sse_chunks]),
        )

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        completion_headers={"x-tp-mode": "probe-null"},
        transport=httpx.MockTransport(handler),
    ) as probe_client:
        tok_cfg = await calibrate(probe_client, "http://localhost:8000", "m")
        assert tok_cfg.calibration_confidence == "heuristic"

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        completion_headers={"x-tp-mode": "json-stream"},
        transport=httpx.MockTransport(handler),
    ) as json_client:
        sample = await _stream_one(
            json_client,
            "http://localhost:8000",
            "m",
            [{"role": "user", "content": "hi"}],
            4,
            None,
            TokenizerConfig(supports_return_token_ids=False),
        )
        assert sample.error is None
        assert sample.pp_tokens == 14
        assert sample.tg_tokens == 2

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        completion_headers={"x-tp-mode": "sse-nulls"},
        transport=httpx.MockTransport(handler),
    ) as sse_client:
        sample_sse = await _stream_one(
            sse_client,
            "http://localhost:8000",
            "m",
            [{"role": "user", "content": "hi"}],
            4,
            None,
            TokenizerConfig(supports_return_token_ids=False),
        )
        assert sample_sse.error is None
        assert sample_sse.pp_tokens == 10
        assert sample_sse.tg_tokens == 3

    def cancel_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            raise asyncio.CancelledError("abort concurrent sweep")
        return httpx.Response(404)

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        transport=httpx.MockTransport(cancel_handler),
    ) as cancel_client:
        with pytest.raises(asyncio.CancelledError):
            await measure_concurrent(
                cancel_client,
                "http://localhost:8000",
                "m",
                pp=32,
                tg=8,
                concurrency=2,
                tok_cfg=TokenizerConfig(supports_return_token_ids=False),
            )


@pytest.mark.asyncio
async def test_speculative_prometheus_reset_and_metric_parsers() -> None:
    empty_pooled = pool_spec_samples([])
    assert empty_pooled.error == "no samples"
    assert empty_pooled.runs == 0

    scrapes = [
        (
            "vllm:spec_decode_num_draft_tokens_total 100\n"
            "vllm:spec_decode_num_accepted_tokens_total 80\n"
            "vllm:spec_decode_num_drafts_total 25\n"
        ),
        (
            "vllm:spec_decode_num_draft_tokens_total 10\n"
            "vllm:spec_decode_num_accepted_tokens_total 8\n"
            "vllm:spec_decode_num_drafts_total 2\n"
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/metrics"):
            return httpx.Response(200, text=scrapes.pop(0))
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 16})
        sse = (
            'data: {"choices": [{"delta": {"content": "hello"}}], '
            '"usage": {"prompt_tokens": 16, "completion_tokens": 4}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse.encode(),
        )

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        spec_sample: SpecDecodeSample = await measure_spec_single(
            client,
            "http://localhost:8000",
            "m",
            pp=32,
            tg=4,
            tok_cfg=TokenizerConfig(
                has_tokenize_endpoint=True,
                supports_return_token_ids=False,
            ),
            spec_info=SpecDecodeInfo(method="eagle", has_prometheus=True),
        )
        assert spec_sample.error is None
        assert spec_sample.draft_tokens_delta is None
        assert spec_sample.accepted_tokens_delta is None

    prom_text = (
        "vllm:spec_decode_num_draft_tokens_total 1e400\n"
        "vllm:spec_decode_num_accepted_tokens_total 12\n"
        "vllm:spec_decode_num_drafts_total 1e400\n"
        'vllm:spec_decode_per_position_acceptance_rate{position="bad"} 0.8\n'
        'vllm:spec_decode_per_position_acceptance_rate{position="-1"} 0.7\n'
        'vllm:spec_decode_per_position_acceptance_rate{position="0"} 0.9\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="nope"} 5\n'
        'vllm:spec_decode_num_draft_tokens_per_pos_total{position="nope"} 5\n'
        "vllm:gpu_cache_usage_perc 1e400\n"
    )
    counters = parse_prometheus_spec_metrics(prom_text)
    assert counters.accepted_tokens == 12.0
    assert counters.draft_tokens == 0.0
    assert counters.has_num_drafts is False

    snap = _parse_snapshot(prom_text)
    assert snap.accepted_tokens == 12.0
    assert snap.draft_tokens == 0.0
    assert snap.gpu_cache_usage is None
    assert snap.per_position_rates == {0: 0.9}
    assert snap.per_position_counters == {}
    assert snap.per_position_drafted_counters == {}


def test_perf_presets_and_llama_benchy_stat_mean_guards() -> None:
    preset_pass = PerfPass(depths=(0, 4096), concurrency=(1, 2))
    assert (
        estimate_pass_seconds(preset_pass, pp=512, tg=128, runs=2, pp_tps=0.0, tg_tps=50.0) == 0.0
    )
    assert (
        estimate_pass_seconds(preset_pass, pp=512, tg=128, runs=2, pp_tps=500.0, tg_tps=0.0) == 0.0
    )

    assert _stat_mean({"mean": None}) == 0.0
    assert _stat_mean({"mean": "nan"}) == 0.0
    assert _stat_mean({"mean": "inf"}) == 0.0
    assert _stat_mean({"mean": "not-a-number"}) == 0.0
    assert _stat_mean({"mean": 42.5}) == 42.5
