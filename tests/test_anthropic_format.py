"""Anthropic Messages wire format: detection, adapter selection, and translation.

The benchmark keeps OpenAI-flavoured messages internally, so everything here
guards the boundary where those shapes become ``/v1/messages`` requests and
come back as ``ChatCompletionResult``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters.anthropic import (
    AnthropicAdapter,
    _apply_extra_params,
    _to_anthropic_messages,
    _to_anthropic_tools,
    _to_output_format,
    _to_tool_choice,
)
from tool_eval_bench.adapters.factory import build_adapter
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.adapters.requests import minimal_request
from tool_eval_bench.adapters.wire_format import (
    anthropic_messages_url,
    anthropic_models_url,
    detect_wire_format,
    resolve_wire_format,
)
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.domain.scenarios import (
    Category,
    FailureKind,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioStatus,
)
from tool_eval_bench.runner.orchestrator import _assistant_message, run_scenario
from tool_eval_bench.utils.headers import USER_AGENT
from tool_eval_bench.utils.openai_compat import sampling_retry_payload

ANTHROPIC_URL = "https://api.anthropic.com"
ZEN_URL = "https://opencode.ai/zen/go/v1/messages"


class TestWireFormatDetection:
    def test_anthropic_host_is_native(self) -> None:
        assert detect_wire_format(ANTHROPIC_URL) == "anthropic"
        assert detect_wire_format(f"{ANTHROPIC_URL}/v1") == "anthropic"

    def test_anthropic_compat_path_is_openai(self) -> None:
        assert detect_wire_format(f"{ANTHROPIC_URL}/v1/chat/completions") == "openai"

    def test_messages_path_on_any_host_is_native(self) -> None:
        assert detect_wire_format(ZEN_URL) == "anthropic"
        assert detect_wire_format("http://litellm:4000/v1/messages/") == "anthropic"

    def test_gateway_root_stays_openai(self) -> None:
        """Zen serves both formats; without a path hint the default wins."""
        assert detect_wire_format("https://opencode.ai/zen/v1") == "openai"

    def test_explicit_choice_beats_detection(self) -> None:
        assert resolve_wire_format("anthropic", "https://opencode.ai/zen/go") == "anthropic"
        assert resolve_wire_format("openai", ANTHROPIC_URL) == "openai"


class TestAnthropicUrls:
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://opencode.ai/zen/go",
            "https://opencode.ai/zen/go/",
            "https://opencode.ai/zen/go/v1",
            "https://opencode.ai/zen/go/v1/",
            ZEN_URL,
        ],
    )
    def test_every_spelling_lands_on_messages(self, base_url: str) -> None:
        assert anthropic_messages_url(base_url) == ZEN_URL

    def test_models_listing_url(self) -> None:
        assert anthropic_models_url(ZEN_URL) == "https://opencode.ai/zen/go/v1/models"
        assert anthropic_models_url(ANTHROPIC_URL) == f"{ANTHROPIC_URL}/v1/models"

    def test_query_parameters_stay_after_generated_endpoint(self) -> None:
        base_url = "https://opencode.ai/zen/go/v1?tenant=research"

        assert anthropic_messages_url(base_url) == (
            "https://opencode.ai/zen/go/v1/messages?tenant=research"
        )
        assert (
            anthropic_models_url(base_url) == "https://opencode.ai/zen/go/v1/models?tenant=research"
        )


class TestAdapterSelection:
    def test_native_url_builds_anthropic_adapter(self) -> None:
        assert isinstance(build_adapter(ZEN_URL), AnthropicAdapter)

    def test_explicit_format_overrides_the_url(self) -> None:
        assert isinstance(
            build_adapter("http://localhost:8000/v1", wire_format="anthropic"), AnthropicAdapter
        )
        assert isinstance(
            build_adapter(ANTHROPIC_URL, wire_format="openai"), OpenAICompatibleAdapter
        )

    def test_anthropic_is_a_supported_backend(self) -> None:
        service = BenchmarkService(repo=None, reporter=None)
        adapter = service._adapter_for("anthropic", ZEN_URL)  # noqa: SLF001
        assert isinstance(adapter, AnthropicAdapter)


class TestMessageTranslation:
    def test_system_messages_become_the_system_prompt(self) -> None:
        messages, system = _to_anthropic_messages(
            [
                {"role": "system", "content": "One."},
                {"role": "system", "content": "Two."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert system == "One.\n\nTwo."
        assert messages == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

    def test_tool_call_becomes_a_tool_use_block(self) -> None:
        messages, _ = _to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "Looking that up.",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'},
                        }
                    ],
                }
            ]
        )
        assert messages == [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking that up."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Berlin"},
                    },
                ],
            }
        ]

    def test_malformed_arguments_degrade_to_empty_input(self) -> None:
        messages, _ = _to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "t", "function": {"name": "f", "arguments": "{oops"}}],
                }
            ]
        )
        assert messages[0]["content"][0]["input"] == {}

    def test_tool_results_become_a_user_turn_and_merge(self) -> None:
        messages, _ = _to_anthropic_messages(
            [
                {"role": "tool", "tool_call_id": "a", "name": "f", "content": '{"ok": true}'},
                {"role": "tool", "tool_call_id": "b", "name": "g", "content": "plain"},
                {"role": "user", "content": "Thanks"},
            ]
        )
        assert [m["role"] for m in messages] == ["user"]
        assert messages[0]["content"] == [
            {"type": "tool_result", "tool_use_id": "a", "content": '{"ok": true}'},
            {"type": "tool_result", "tool_use_id": "b", "content": "plain"},
            {"type": "text", "text": "Thanks"},
        ]

    def test_thinking_blocks_replay_first_and_verbatim(self) -> None:
        """The runner copies message_extra_content back onto the assistant turn."""
        thinking = [{"type": "thinking", "thinking": "", "signature": "sig=="}]
        messages, _ = _to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "Done.",
                    "extra_content": {"anthropic": {"thinking_blocks": thinking}},
                    "tool_calls": [{"id": "t", "function": {"name": "f", "arguments": "{}"}}],
                }
            ]
        )
        assert messages[0]["content"][0] == thinking[0]
        assert [b["type"] for b in messages[0]["content"]] == ["thinking", "text", "tool_use"]

    def test_foreign_extra_content_is_ignored(self) -> None:
        messages, _ = _to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "Hi",
                    "extra_content": {"google": {"thoughtSignature": "x"}},
                }
            ]
        )
        assert messages[0]["content"] == [{"type": "text", "text": "Hi"}]


class TestToolTranslation:
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather lookup",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        },
        {"type": "function", "function": {"name": "ping", "parameters": {}}},
    ]

    def test_tools_keep_their_schema_under_input_schema(self) -> None:
        converted = _to_anthropic_tools(self.TOOLS)
        assert converted is not None
        assert converted[0] == {
            "name": "get_weather",
            "description": "Weather lookup",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }

    def test_no_arg_tool_gets_an_empty_object_schema(self) -> None:
        converted = _to_anthropic_tools(self.TOOLS)
        assert converted is not None
        assert converted[1]["input_schema"] == {"type": "object", "properties": {}}

    def test_no_tools_returns_none(self) -> None:
        assert _to_anthropic_tools(None) is None
        assert _to_anthropic_tools([]) is None

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            ("required", {"type": "any"}),
            ("any", {"type": "any"}),
            (None, None),
        ],
    )
    def test_tool_choice_modes(self, choice: str | None, expected: dict | None) -> None:
        assert _to_tool_choice(choice, True) == expected

    def test_named_tool_choice(self) -> None:
        assert _to_tool_choice({"type": "function", "function": {"name": "f"}}, True) == {
            "type": "tool",
            "name": "f",
        }

    def test_disabling_parallel_calls(self) -> None:
        assert _to_tool_choice("auto", False) == {"type": "auto", "disable_parallel_tool_use": True}
        assert _to_tool_choice(None, False) == {"type": "auto", "disable_parallel_tool_use": True}
        assert _to_tool_choice("none", False) == {"type": "none"}


class TestOutputFormat:
    def test_json_schema_maps_to_output_config_format(self) -> None:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "review",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "rating": {"type": "number", "minimum": 0, "maximum": 10},
                        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                    },
                    "required": ["rating"],
                    "additionalProperties": False,
                },
            },
        }
        assert _to_output_format(response_format) == {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "rating": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["rating"],
                "additionalProperties": False,
            },
        }

    def test_json_object_has_no_equivalent(self) -> None:
        assert _to_output_format({"type": "json_object"}) is None
        assert _to_output_format(None) is None


class TestExtraParams:
    def test_openai_aliases_are_renamed(self) -> None:
        payload: dict[str, Any] = {}
        headers: dict[str, str] = {}
        _apply_extra_params(payload, headers, {"top_p": 0.9, "stop": "END", "max_tokens": 12})
        assert payload == {"top_p": 0.9, "stop_sequences": ["END"], "max_tokens": 12}

    def test_only_stop_sequences_list_wraps_string_values(self) -> None:
        payload: dict[str, Any] = {}
        _apply_extra_params(
            payload,
            {},
            {"top_p": "0.9", "top_k": "40", "stop": "HALT", "stop_sequences": "DONE"},
        )
        assert payload["top_p"] == "0.9"
        assert payload["top_k"] == "40"
        assert payload["stop_sequences"] == ["DONE"]

        payload_list: dict[str, Any] = {}
        _apply_extra_params(payload_list, {}, {"stop": ["A", "B"]})
        assert payload_list["stop_sequences"] == ["A", "B"]

    def test_no_think_disables_thinking(self) -> None:
        payload: dict[str, Any] = {}
        _apply_extra_params(payload, {}, {"chat_template_kwargs": {"enable_thinking": False}})
        assert payload == {"thinking": {"type": "disabled"}}

    def test_native_keys_pass_through_and_betas_become_a_header(self) -> None:
        payload: dict[str, Any] = {"output_config": {"format": {"type": "json_schema"}}}
        headers: dict[str, str] = {}
        _apply_extra_params(
            payload,
            headers,
            {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "low"},
                "betas": ["fast-mode-2026-02-01"],
                "unknown": 1,
            },
        )
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"] == {"format": {"type": "json_schema"}, "effort": "low"}
        assert "unknown" not in payload
        assert headers["anthropic-beta"] == "fast-mode-2026-02-01"


class TestResponseParsing:
    def test_text_response(self) -> None:
        result = AnthropicAdapter._parse_response(  # noqa: SLF001
            {
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
            12.0,
        )
        assert result.content == "Hello"
        assert result.finish_reason == "stop"
        assert result.prompt_tokens == 7
        assert result.completion_tokens == 3
        assert result.message_extra_content is None

    def test_tool_use_and_thinking_are_split_out(self) -> None:
        thinking = {"type": "thinking", "thinking": "Let me check.", "signature": "sig=="}
        result = AnthropicAdapter._parse_response(  # noqa: SLF001
            {
                "content": [
                    thinking,
                    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"x": 1}},
                ],
                "stop_reason": "tool_use",
            },
            1.0,
        )
        assert result.content == ""
        assert result.reasoning == "Let me check."
        assert result.finish_reason == "tool_calls"
        assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
            ("toolu_1", "f", {"x": 1})
        ]
        assert result.message_extra_content == {"anthropic": {"thinking_blocks": [thinking]}}

    def test_thinking_round_trips_through_the_runner(self) -> None:
        thinking = {"type": "thinking", "thinking": "", "signature": "sig=="}
        redacted = {"type": "redacted_thinking", "data": "opaque"}
        result = AnthropicAdapter._parse_response(  # noqa: SLF001
            {
                "content": [
                    thinking,
                    redacted,
                    {"type": "tool_use", "id": "t", "name": "f", "input": {}},
                ],
                "stop_reason": "tool_use",
            },
            1.0,
        )
        replayed, _ = _to_anthropic_messages([_assistant_message(result)])
        assert replayed[0]["content"][:2] == [thinking, redacted]

    def test_refusal_surfaces_a_reason(self) -> None:
        result = AnthropicAdapter._parse_response(  # noqa: SLF001
            {
                "content": [],
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "cyber"},
            },
            1.0,
        )
        assert result.content == "[no content: refusal (cyber)]"
        assert result.finish_reason == "refusal"

    def test_max_tokens_maps_to_length(self) -> None:
        result = AnthropicAdapter._parse_response(  # noqa: SLF001
            {"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"}, 1.0
        )
        assert result.finish_reason == "length"


class _StreamResponse:
    headers = {"content-type": "text/event-stream"}

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StreamClient:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def stream(self, method: str, url: str, **kwargs: Any):
        client = self

        class _Ctx:
            async def __aenter__(self):
                return _StreamResponse(client._lines)

            async def __aexit__(self, *args: object) -> None:
                return None

        return _Ctx()


def _events(*events: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for event in events:
        lines.append(f"event: {event['type']}")
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    return lines


def _stream(lines: list[str]):
    return asyncio.run(
        AnthropicAdapter()._stream_request(  # type: ignore[arg-type]  # noqa: SLF001
            _StreamClient(lines), "http://x", {}, {}
        )
    )


class TestStreamParsing:
    def test_text_tool_use_and_usage_accumulate(self) -> None:
        result = _stream(
            _events(
                {"type": "message_start", "message": {"usage": {"input_tokens": 9}}},
                {"type": "ping"},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hel"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "lo"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "f",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"ci'},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": 'ty": "Berlin"}'},
                },
                {"type": "content_block_stop", "index": 1},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            )
        )
        assert result.content == "Hello"
        assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
            ("toolu_1", "f", {"city": "Berlin"})
        ]
        assert result.finish_reason == "tool_calls"
        assert result.prompt_tokens == 9
        assert result.completion_tokens == 4
        assert result.ttft_ms is not None

    def test_thinking_stream_starts_ttft_and_keeps_the_signature(self) -> None:
        result = _stream(
            _events(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "hmm"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig=="},
                },
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
            )
        )
        assert result.content == ""
        assert result.reasoning == "hmm"
        assert result.ttft_ms is not None
        assert result.message_extra_content == {
            "anthropic": {
                "thinking_blocks": [{"type": "thinking", "thinking": "hmm", "signature": "sig=="}]
            }
        }

    def test_unfinished_tool_input_is_still_parsed(self) -> None:
        """A stream cut before content_block_stop keeps what it has."""
        result = _stream(
            _events(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "t", "name": "f", "input": {}},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"x": 1}'},
                },
            )
        )
        assert result.tool_calls[0].arguments == {"x": 1}

    def test_error_event_is_a_transport_error(self) -> None:
        result = _stream(
            _events(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "partial"},
                },
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                },
            )
        )
        assert result.transport_error_status == 529
        assert result.transport_error_is_infrastructure is True
        assert result.tool_calls == []
        assert "overloaded_error" in result.content

    @pytest.mark.parametrize(
        ("error", "status", "infrastructure"),
        [
            ({"type": "rate_limit_error", "message": "per-minute limit"}, 429, True),
            ({"type": "api_error", "message": "boom"}, 500, True),
            ({"type": "invalid_request_error", "message": "bad"}, 400, False),
            ("gateway exploded", 500, True),
        ],
    )
    def test_error_event_status_classification(
        self, error: object, status: int, infrastructure: bool
    ) -> None:
        """A mid-stream 429 is infrastructure, as it is on the non-stream path."""
        result = _stream(_events({"type": "error", "error": error}))

        assert result.transport_error_status == status
        assert result.transport_error_is_infrastructure is infrastructure

    def test_malformed_lines_are_skipped(self) -> None:
        result = _stream(
            [
                "data: not-json",
                ": keepalive",
                'data:{"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"ok"}}',
            ]
        )
        assert result.content == "ok"

    def test_json_200_is_parsed_when_stream_is_ignored(self) -> None:
        class _JsonResponse:
            headers = {"content-type": "application/json"}

            def raise_for_status(self) -> None:
                return None

            async def aread(self) -> bytes:
                return b'{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}'

            async def aiter_lines(self):
                raise AssertionError("ordinary JSON must not be iterated as SSE")
                yield ""

        class _JsonClient:
            def stream(self, method: str, url: str, **kwargs: Any):
                class _Ctx:
                    async def __aenter__(self):
                        return _JsonResponse()

                    async def __aexit__(self, *args: object) -> None:
                        return None

                return _Ctx()

        result = asyncio.run(
            AnthropicAdapter()._stream_request(  # type: ignore[arg-type]  # noqa: SLF001
                _JsonClient(), "http://x", {}, {}
            )
        )
        assert result.content == "ok"


class TestRequestBuilding:
    """End-to-end through ``chat_completion`` against a mocked transport."""

    @staticmethod
    def _run(handler, **kwargs: Any) -> tuple[ChatCompletionResult, list[httpx.Request]]:
        seen: list[httpx.Request] = []

        def _handle(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request, len(seen))

        adapter = AnthropicAdapter()
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(_handle))  # noqa: SLF001
        result = asyncio.run(
            adapter.chat_completion(
                model="claude-opus-5",
                messages=[
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Weather in Berlin?"},
                ],
                tools=TestToolTranslation.TOOLS,
                base_url=ZEN_URL,
                api_key="secret",
                **kwargs,
            )
        )
        return result, seen

    def test_request_shape_and_headers(self) -> None:
        def handler(request: httpx.Request, _n: int) -> httpx.Response:
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "Sunny"}], "stop_reason": "end_turn"},
            )

        result, seen = self._run(handler, parallel_tool_calls=False)
        assert result.content == "Sunny"
        request = seen[0]
        assert str(request.url) == ZEN_URL
        assert request.headers["x-api-key"] == "secret"
        assert request.headers["authorization"] == "Bearer secret"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body["system"] == "Be brief."
        assert body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Weather in Berlin?"}]}
        ]
        assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert body["temperature"] == 0.0
        assert "stream" not in body

    def test_rejected_sampling_is_dropped_and_remembered(self) -> None:
        def handler(request: httpx.Request, n: int) -> httpx.Response:
            body = json.loads(request.content)
            if "temperature" in body:
                return httpx.Response(
                    400,
                    json={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "temperature: not supported on this model",
                        },
                    },
                )
            return httpx.Response(
                200, json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
            )

        result, seen = self._run(handler)
        assert result.content == "ok"
        assert len(seen) == 2
        assert "temperature" not in json.loads(seen[1].content)

    def test_other_400s_are_soft_errors(self) -> None:
        def handler(request: httpx.Request, _n: int) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "bad history"}})

        result, seen = self._run(handler)
        assert len(seen) == 1
        assert result.transport_error_status == 400
        assert "bad history" in result.content

    def test_overloaded_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tool_eval_bench.adapters import http_retry

        monkeypatch.setattr(http_retry, "_backoff_delay", lambda *args: 0.0)

        def handler(request: httpx.Request, n: int) -> httpx.Response:
            if n == 1:
                return httpx.Response(529, json={"error": {"type": "overloaded_error"}})
            return httpx.Response(
                200, json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
            )

        result, seen = self._run(handler)
        assert result.content == "ok"
        assert len(seen) == 2

    def test_response_format_maps_to_output_config_and_merges_extra_params(self) -> None:
        def handler(request: httpx.Request, _n: int) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": '{"ok": true}'}],
                    "stop_reason": "end_turn",
                },
            )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"count": {"type": "integer", "minimum": 1}},
                    "required": ["count"],
                },
            },
        }
        result, seen = self._run(
            handler,
            response_format=response_format,
            extra_params={"output_config": {"effort": "low"}},
        )
        assert result.content == '{"ok": true}'
        body = json.loads(seen[0].content)
        assert body["output_config"] == {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                },
            },
            "effort": "low",
        }

    def test_sampling_rejected_endpoint_cache_hit_skips_400_on_second_call(self) -> None:
        seen_bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen_bodies.append(body)
            if any(k in body for k in ("temperature", "top_p", "top_k")):
                return httpx.Response(
                    400,
                    json={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "temperature is not supported",
                        },
                    },
                )
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"},
            )

        adapter = AnthropicAdapter()
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: SLF001

        async def _call_twice() -> None:
            try:
                for _ in range(2):
                    res = await adapter.chat_completion(
                        model="claude-opus-5",
                        messages=[{"role": "user", "content": "hi"}],
                        base_url=ZEN_URL,
                        extra_params={"top_p": 0.9, "top_k": 20},
                    )
                    assert res.content == "ok"
            finally:
                await adapter.aclose()

        asyncio.run(_call_twice())
        # First call takes 2 requests (400 + retry); second call hits the cache and takes 1.
        assert len(seen_bodies) == 3
        assert "temperature" in seen_bodies[0]
        for body in seen_bodies[1:]:
            assert not ({"temperature", "top_p", "top_k"} & body.keys())


class TestSamplingRetryPayload:
    def test_only_named_keys_trigger(self) -> None:
        payload = {"model": "m", "temperature": 0.0, "top_p": 0.9}
        assert sampling_retry_payload(payload, 400, "temperature is not supported") == {
            "model": "m"
        }
        assert sampling_retry_payload(payload, 400, "unrelated error") is None
        assert sampling_retry_payload({"model": "m"}, 400, "temperature") is None
        assert sampling_retry_payload(payload, 500, "temperature") is None


class TestMinimalRequest:
    def test_warmup_speaks_messages(self) -> None:
        url, payload, headers = minimal_request(
            ZEN_URL,
            "claude-opus-5",
            "secret",
            wire_format="anthropic",
            max_tokens=4,
            extra_params={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
        )
        assert url == ZEN_URL
        assert payload == {
            "model": "claude-opus-5",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 4,
            "temperature": 0.0,
            "top_k": 1,
        }
        assert headers["x-api-key"] == "secret"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["User-Agent"] == USER_AGENT


class TestModelListing:
    def test_models_request_per_format(self) -> None:
        from tool_eval_bench.cli.model_probe import _models_request

        url, headers = _models_request(ZEN_URL, "k", "anthropic")
        assert url == "https://opencode.ai/zen/go/v1/models"
        assert headers == {
            "User-Agent": USER_AGENT,
            "anthropic-version": "2023-06-01",
            "x-api-key": "k",
            "Authorization": "Bearer k",
        }
        url, headers = _models_request("https://generativelanguage.googleapis.com", "k", "gemini")
        assert url.endswith("/v1beta/models")
        assert headers == {"User-Agent": USER_AGENT, "x-goog-api-key": "k"}
        url, headers = _models_request("http://localhost:8000/v1", None, "openai")
        assert (url, headers) == ("http://localhost:8000/v1/models", {"User-Agent": USER_AGENT})
        url, _ = _models_request(
            "https://proxy.example/inference/v1?tenant=research", None, "openai"
        )
        assert url == "https://proxy.example/inference/v1/models?tenant=research"


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

    scenario = ScenarioDefinition(
        id="FIX-ANTH-TRUNC",
        title="Truncation regression",
        description="Expected test tool workflow.",
        category=Category.A,
        user_message="Run the workflow.",
        handle_tool_call=lambda state, call: {"ok": True, "id": call.id},
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.FAIL, 0, "wrong"),
    )

    for case in ("normal", "refusal", "truncated"):
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
                    scenario=scenario,
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


@pytest.mark.asyncio
async def test_anthropic_stream_http_errors_and_sampling_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.adapters import http_retry

    monkeypatch.setattr(http_retry, "_backoff_delay", lambda *args: 0.0)
    seen_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        mode = request.headers.get("x-mode", "")
        body = json.loads(request.content)
        seen_bodies.append(body)
        if mode == "400":
            return httpx.Response(400, json={"error": {"message": "bad prompt"}})
        if mode == "sampling":
            if "temperature" in body:
                return httpx.Response(
                    400, json={"error": {"message": "temperature is not supported"}}
                )
            sse = "\n".join(
                _events(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "retried"},
                    },
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
                )
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)
        return httpx.Response(500, text="internal server error")

    for mode in ("400", "sampling", "500"):
        adapter = AnthropicAdapter(max_retries=1, default_headers={"x-mode": mode})
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: SLF001
        try:
            if mode == "400":
                res = await adapter.chat_completion(
                    model="claude",
                    messages=[{"role": "user", "content": "hi"}],
                    base_url=ZEN_URL,
                    stream=True,
                )
                assert res.transport_error_status == 400
                assert "bad prompt" in res.content
            elif mode == "sampling":
                res = await adapter.chat_completion(
                    model="claude",
                    messages=[{"role": "user", "content": "hi"}],
                    base_url=ZEN_URL,
                    stream=True,
                )
                assert res.content == "retried"
            else:
                with pytest.raises(httpx.HTTPStatusError):
                    await adapter.chat_completion(
                        model="claude",
                        messages=[{"role": "user", "content": "hi"}],
                        base_url=ZEN_URL,
                        stream=True,
                    )
        finally:
            await adapter.aclose()


@pytest.mark.asyncio
async def test_anthropic_stream_sse_edge_cases() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        mode = request.headers.get("x-mode", "")
        if mode == "non-dict-json":
            return httpx.Response(
                200, headers={"content-type": "application/json"}, content=b"[1, 2]"
            )
        if mode == "empty-sse":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=": keepalive\n\n"
            )
        if mode == "json-labeled-sse":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='{"content": [{"type": "text", "text": "plain-json"}], "stop_reason": "end_turn"}',
            )
        # Deltas arriving without prior content_block_start, plus non-dict tool JSON.
        sse = "\n".join(
            _events(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "thought"},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": "[1, 2]"},
                },
                {"type": "content_block_stop", "index": 1},
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "text_delta", "text": "answer"},
                },
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "unknown_delta", "x": 1},
                },
            )
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

    for mode in ("non-dict-json", "empty-sse", "json-labeled-sse", "orphan-deltas"):
        adapter = AnthropicAdapter(default_headers={"x-mode": mode})
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: SLF001
        try:
            res = await adapter.chat_completion(
                model="claude",
                messages=[{"role": "user", "content": "hi"}],
                base_url=ZEN_URL,
                stream=True,
            )
            if mode in ("non-dict-json", "empty-sse"):
                assert res.malformed is True
            elif mode == "json-labeled-sse":
                assert res.content == "plain-json"
                assert res.malformed is False
            else:
                assert res.reasoning == "thought"
                assert res.content == "answer"
                assert len(res.tool_calls) == 1
                assert res.tool_calls[0].arguments_str == "[1, 2]"
        finally:
            await adapter.aclose()
