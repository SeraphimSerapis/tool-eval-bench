"""Native Gemini wire format: detection, adapter selection, and translation.

The benchmark keeps OpenAI-flavoured messages internally, so everything here
guards the boundary where those shapes become ``:generateContent`` requests and
come back as ``ChatCompletionResult``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tool_eval_bench.adapters.factory import build_adapter
from tool_eval_bench.adapters.gemini import (
    GeminiAdapter,
    _generation_config,
    _to_gemini_contents,
    _to_gemini_schema,
    _to_gemini_tools,
    _to_tool_config,
)
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.adapters.wire_format import (
    detect_wire_format,
    gemini_generate_url,
    gemini_models_url,
    resolve_wire_format,
)
from tool_eval_bench.application.service import BenchmarkService

NATIVE_URL = "https://generativelanguage.googleapis.com"
COMPAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/v1"


class TestWireFormatDetection:
    def test_bare_google_host_is_native(self) -> None:
        assert detect_wire_format(NATIVE_URL) == "gemini"

    def test_versioned_google_host_is_native(self) -> None:
        assert detect_wire_format(f"{NATIVE_URL}/v1beta") == "gemini"

    def test_openai_compatibility_path_is_openai(self) -> None:
        assert detect_wire_format(COMPAT_URL) == "openai"

    def test_local_server_is_openai(self) -> None:
        assert detect_wire_format("http://localhost:8000/v1") == "openai"

    def test_empty_base_url_is_openai(self) -> None:
        assert detect_wire_format("") == "openai"

    def test_explicit_choice_beats_detection(self) -> None:
        assert resolve_wire_format("gemini", "http://localhost:8000/v1") == "gemini"
        assert resolve_wire_format("openai", NATIVE_URL) == "openai"

    def test_auto_and_none_detect(self) -> None:
        assert resolve_wire_format("auto", NATIVE_URL) == "gemini"
        assert resolve_wire_format(None, NATIVE_URL) == "gemini"

    def test_unknown_format_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown --format"):
            resolve_wire_format("anthropic", NATIVE_URL)


class TestGeminiUrls:
    def test_bare_host_gets_default_version(self) -> None:
        assert gemini_generate_url(NATIVE_URL, "gemini-3-flash") == (
            f"{NATIVE_URL}/v1beta/models/gemini-3-flash:generateContent"
        )

    def test_existing_version_is_kept(self) -> None:
        assert gemini_generate_url(f"{NATIVE_URL}/v1alpha", "m").startswith(
            f"{NATIVE_URL}/v1alpha/models/m:"
        )

    def test_trailing_slash_is_tolerated(self) -> None:
        assert gemini_generate_url(f"{NATIVE_URL}/", "m") == (
            f"{NATIVE_URL}/v1beta/models/m:generateContent"
        )

    def test_qualified_model_name_is_not_doubled(self) -> None:
        assert "models/models/" not in gemini_generate_url(NATIVE_URL, "models/gemini-3-flash")

    def test_streaming_uses_sse(self) -> None:
        assert gemini_generate_url(NATIVE_URL, "m", stream=True).endswith(
            ":streamGenerateContent?alt=sse"
        )

    def test_models_listing_url(self) -> None:
        assert gemini_models_url(NATIVE_URL) == f"{NATIVE_URL}/v1beta/models"


class TestAdapterSelection:
    def test_native_url_builds_gemini_adapter(self) -> None:
        assert isinstance(build_adapter(NATIVE_URL), GeminiAdapter)

    def test_compatibility_url_builds_openai_adapter(self) -> None:
        assert isinstance(build_adapter(COMPAT_URL), OpenAICompatibleAdapter)

    def test_explicit_format_overrides_the_url(self) -> None:
        assert isinstance(build_adapter(COMPAT_URL, wire_format="gemini"), GeminiAdapter)
        assert isinstance(build_adapter(NATIVE_URL, wire_format="openai"), OpenAICompatibleAdapter)

    def test_gemini_is_a_supported_backend(self) -> None:
        service = BenchmarkService(repo=None, reporter=None)
        adapter = service._adapter_for("gemini", NATIVE_URL)  # noqa: SLF001
        assert isinstance(adapter, GeminiAdapter)

    def test_backend_label_does_not_pick_the_format(self) -> None:
        """The endpoint decides the wire format; the label is only for reports."""
        service = BenchmarkService(repo=None, reporter=None)
        adapter = service._adapter_for("vllm", NATIVE_URL)  # noqa: SLF001
        assert isinstance(adapter, GeminiAdapter)

    def test_unknown_backend_still_rejected(self) -> None:
        service = BenchmarkService(repo=None, reporter=None)
        with pytest.raises(ValueError, match="gemini"):
            service._adapter_for("not-a-backend")  # noqa: SLF001


class TestMessageTranslation:
    def test_system_message_becomes_system_instruction(self) -> None:
        contents, system = _to_gemini_contents(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert system == {"parts": [{"text": "Be terse."}]}
        assert contents == [{"role": "user", "parts": [{"text": "Hi"}]}]

    def test_multiple_system_messages_are_joined(self) -> None:
        _, system = _to_gemini_contents(
            [
                {"role": "system", "content": "One."},
                {"role": "system", "content": "Two."},
            ]
        )
        assert system is not None
        assert system["parts"][0]["text"] == "One.\n\nTwo."

    def test_assistant_role_is_renamed_to_model(self) -> None:
        contents, _ = _to_gemini_contents([{"role": "assistant", "content": "Sure"}])
        assert contents[0]["role"] == "model"

    def test_tool_call_arguments_are_parsed_into_args(self) -> None:
        contents, _ = _to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'},
                        }
                    ],
                }
            ]
        )
        call = contents[0]["parts"][0]["functionCall"]
        assert call == {"name": "get_weather", "args": {"city": "Berlin"}, "id": "call_1"}

    def test_malformed_arguments_degrade_to_empty_args(self) -> None:
        contents, _ = _to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{oops"}}],
                }
            ]
        )
        assert contents[0]["parts"][0]["functionCall"]["args"] == {}

    def test_thought_signature_round_trips(self) -> None:
        contents, _ = _to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c",
                            "function": {"name": "f", "arguments": "{}"},
                            "extra_content": {"thoughtSignature": "sig-abc"},
                        }
                    ],
                }
            ]
        )
        assert contents[0]["parts"][0]["thoughtSignature"] == "sig-abc"

    def test_thought_signature_from_compat_layer_shape(self) -> None:
        """A trace recorded through the OpenAI layer nests it under a vendor key."""
        contents, _ = _to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c",
                            "function": {"name": "f", "arguments": "{}"},
                            "extra_content": {"google": {"thought_signature": "sig-xyz"}},
                        }
                    ],
                }
            ]
        )
        assert contents[0]["parts"][0]["thoughtSignature"] == "sig-xyz"

    def test_tool_result_becomes_a_user_function_response(self) -> None:
        contents, _ = _to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "get_weather", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 12}'},
            ]
        )
        response = contents[1]["parts"][0]["functionResponse"]
        assert contents[1]["role"] == "user"
        assert response["name"] == "get_weather"
        assert response["response"] == {"temp": 12}

    def test_non_json_tool_result_is_wrapped(self) -> None:
        contents, _ = _to_gemini_contents(
            [{"role": "tool", "name": "f", "tool_call_id": "c", "content": "plain text"}]
        )
        assert contents[0]["parts"][0]["functionResponse"]["response"] == {"result": "plain text"}

    def test_parallel_tool_results_merge_into_one_turn(self) -> None:
        contents, _ = _to_gemini_contents(
            [
                {"role": "tool", "name": "a", "tool_call_id": "c1", "content": "{}"},
                {"role": "tool", "name": "b", "tool_call_id": "c2", "content": "{}"},
            ]
        )
        assert len(contents) == 1
        assert len(contents[0]["parts"]) == 2


class TestToolTranslation:
    def _tool(self, parameters: dict[str, Any] | None) -> dict[str, Any]:
        function: dict[str, Any] = {"name": "get_weather", "description": "Weather"}
        if parameters is not None:
            function["parameters"] = parameters
        return {"type": "function", "function": function}

    def test_tools_become_function_declarations(self) -> None:
        tools = _to_gemini_tools(
            [self._tool({"type": "object", "properties": {"city": {"type": "string"}}})]
        )
        assert tools is not None
        declaration = tools[0]["functionDeclarations"][0]
        assert declaration["name"] == "get_weather"
        assert declaration["parameters"]["type"] == "OBJECT"
        assert declaration["parameters"]["properties"]["city"]["type"] == "STRING"

    def test_no_arg_tool_omits_parameters(self) -> None:
        """An empty object schema is rejected by the API, so it must be dropped."""
        tools = _to_gemini_tools([self._tool({"type": "object", "properties": {}})])
        assert tools is not None
        assert "parameters" not in tools[0]["functionDeclarations"][0]

    def test_no_tools_returns_none(self) -> None:
        assert _to_gemini_tools([]) is None
        assert _to_gemini_tools(None) is None

    def test_unsupported_schema_keys_are_dropped(self) -> None:
        converted = _to_gemini_schema(
            {
                "$schema": "https://json-schema.org/draft-07/schema#",
                "type": "object",
                "additionalProperties": False,
                "properties": {"n": {"type": "integer", "default": 1, "description": "count"}},
                "required": ["n"],
            }
        )
        assert "$schema" not in converted
        assert "additionalProperties" not in converted
        assert "default" not in converted["properties"]["n"]
        assert converted["properties"]["n"]["description"] == "count"
        assert converted["required"] == ["n"]

    def test_nested_array_items_are_converted(self) -> None:
        converted = _to_gemini_schema(
            {"type": "array", "items": {"type": "string", "additionalProperties": True}}
        )
        assert converted["items"] == {"type": "STRING"}

    @pytest.mark.parametrize(
        ("choice", "mode"),
        [("auto", "AUTO"), ("none", "NONE"), ("required", "ANY"), ("any", "ANY")],
    )
    def test_tool_choice_modes(self, choice: str, mode: str) -> None:
        config = _to_tool_config(choice)
        assert config == {"functionCallingConfig": {"mode": mode}}

    def test_named_tool_choice_restricts_the_call(self) -> None:
        config = _to_tool_config({"type": "function", "function": {"name": "get_weather"}})
        assert config == {
            "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}
        }


class TestGenerationConfig:
    def _config(self, **extra: Any) -> dict[str, Any]:
        return _generation_config(
            temperature=0.0,
            max_tokens=256,
            response_format=None,
            extra_params=extra or None,
        )

    def test_core_sampling_options(self) -> None:
        config = self._config()
        assert config == {"temperature": 0.0, "maxOutputTokens": 256}

    def test_openai_aliases_are_renamed(self) -> None:
        config = self._config(top_p=0.9, top_k=40, stop=["END"])
        assert config["topP"] == 0.9
        assert config["topK"] == 40
        assert config["stopSequences"] == ["END"]

    def test_no_think_maps_to_zero_thinking_budget(self) -> None:
        config = self._config(chat_template_kwargs={"enable_thinking": False})
        assert config["thinkingConfig"] == {"thinkingBudget": 0}

    def test_thinking_left_alone_when_enabled(self) -> None:
        config = self._config(chat_template_kwargs={"enable_thinking": True})
        assert "thinkingConfig" not in config

    def test_unknown_options_are_dropped(self) -> None:
        assert "guided_json" not in self._config(guided_json={"a": 1})

    def test_json_response_format(self) -> None:
        config = _generation_config(
            temperature=0.0,
            max_tokens=16,
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object", "properties": {}}},
            },
            extra_params=None,
        )
        assert config["responseMimeType"] == "application/json"
        assert config["responseSchema"]["type"] == "OBJECT"


class TestResponseParsing:
    def test_text_response(self) -> None:
        result = GeminiAdapter._parse_response(  # noqa: SLF001
            {
                "candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 2},
            },
            elapsed_ms=5.0,
        )
        assert result.content == "Hello"
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 2

    def test_thought_parts_become_reasoning(self) -> None:
        result = GeminiAdapter._parse_response(  # noqa: SLF001
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "thinking…", "thought": True}, {"text": "Answer"}]
                        }
                    }
                ]
            },
            elapsed_ms=1.0,
        )
        assert result.content == "Answer"
        assert result.reasoning == "thinking…"

    def test_function_call_is_converted(self) -> None:
        result = GeminiAdapter._parse_response(  # noqa: SLF001
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"city": "NYC"},
                                    },
                                    "thoughtSignature": "sig",
                                }
                            ]
                        }
                    }
                ]
            },
            elapsed_ms=1.0,
        )
        call = result.tool_calls[0]
        assert call.name == "get_weather"
        assert json.loads(call.arguments_str) == {"city": "NYC"}
        assert call.extra_content == {"thoughtSignature": "sig"}

    def test_missing_call_ids_are_assigned(self) -> None:
        result = GeminiAdapter._parse_response(  # noqa: SLF001
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "a", "args": {}}},
                                {"functionCall": {"name": "b", "args": {}}},
                            ]
                        }
                    }
                ]
            },
            elapsed_ms=1.0,
        )
        assert [c.id for c in result.tool_calls] == ["tool_call_1", "tool_call_2"]

    def test_blocked_prompt_surfaces_a_reason(self) -> None:
        result = GeminiAdapter._parse_response(  # noqa: SLF001
            {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}},
            elapsed_ms=1.0,
        )
        assert result.content == "[no content: SAFETY]"

    def test_truncated_candidate_surfaces_finish_reason(self) -> None:
        result = GeminiAdapter._parse_response(  # noqa: SLF001
            {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]},
            elapsed_ms=1.0,
        )
        assert result.content == "[no content: MAX_TOKENS]"


class _StreamResponse:
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
        self.request: dict[str, Any] | None = None

    def stream(self, method: str, url: str, **kwargs: Any):
        self.request = {"method": method, "url": url, **kwargs}
        client = self

        class _Ctx:
            async def __aenter__(self):
                return _StreamResponse(client._lines)

            async def __aexit__(self, *args: object) -> None:
                return None

        return _Ctx()


class TestStreamParsing:
    def test_sse_chunks_accumulate_text_calls_and_usage(self) -> None:
        chunks = [
            'data: {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}',
            'data: {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]}',
            'data: {"candidates": [{"content": {"parts": ['
            '{"functionCall": {"name": "f", "args": {"x": 1}}}]}}]}',
            'data: {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}',
            "data: [DONE]",
        ]
        adapter = GeminiAdapter()
        client = _StreamClient(chunks)

        result = asyncio.run(
            adapter._stream_request(client, "http://x", {}, {})  # type: ignore[arg-type]  # noqa: SLF001
        )

        assert result.content == "Hello"
        assert [c.name for c in result.tool_calls] == ["f"]
        assert result.prompt_tokens == 7
        assert result.completion_tokens == 3
        assert result.ttft_ms is not None

    def test_thought_only_stream_starts_ttft(self) -> None:
        """A thought chunk is generated output for TTFT purposes."""
        client = _StreamClient(
            [
                'data: {"candidates":[{"content":{"parts":[{"text":"thinking","thought":true}]}}]}',
                "data: [DONE]",
            ]
        )

        result = asyncio.run(
            GeminiAdapter()._stream_request(  # type: ignore[arg-type]  # noqa: SLF001
                client, "http://x", {}, {}
            )
        )

        assert result.content == ""
        assert result.reasoning == "thinking"
        assert result.ttft_ms is not None

    def test_malformed_chunks_are_skipped(self) -> None:
        adapter = GeminiAdapter()
        client = _StreamClient(
            [
                "data: not-json",
                ": keepalive",
                'data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}',
            ]
        )

        result = asyncio.run(
            adapter._stream_request(client, "http://x", {}, {})  # type: ignore[arg-type]  # noqa: SLF001
        )

        assert result.content == "ok"

    def test_sse_data_without_space_is_valid(self) -> None:
        adapter = GeminiAdapter()
        client = _StreamClient(
            [
                'data:{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}',
                "data:[DONE]",
            ]
        )

        result = asyncio.run(
            adapter._stream_request(client, "http://x", {}, {})  # type: ignore[arg-type]  # noqa: SLF001
        )

        assert result.content == "ok"

    def test_json_200_is_parsed_when_stream_is_ignored(self) -> None:
        class _JsonResponse:
            headers = {"content-type": "application/json"}

            def raise_for_status(self) -> None:
                return None

            async def aread(self) -> bytes:
                return b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}'

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
            GeminiAdapter()._stream_request(  # type: ignore[arg-type]  # noqa: SLF001
                _JsonClient(), "http://x", {}, {}
            )
        )

        assert result.content == "ok"
