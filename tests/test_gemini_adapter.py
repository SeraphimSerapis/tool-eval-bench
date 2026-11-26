"""Gap tests for the native Gemini adapter: streaming HTTP errors, thoughtSignature, and SSE."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters import http_retry
from tool_eval_bench.adapters.gemini import GeminiAdapter, _to_gemini_contents
from tool_eval_bench.adapters.requests import minimal_request

GEMINI_URL = "https://generativelanguage.googleapis.com"


@pytest.mark.asyncio
async def test_gemini_stream_http_4xx_returns_graceful_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"error": {"message": "invalid tool schema"}}')

    adapter = GeminiAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: SLF001
    try:
        res = await adapter.chat_completion(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            base_url=GEMINI_URL,
            stream=True,
        )
        assert res.transport_error_status == 400
        assert "invalid tool schema" in res.content
        assert res.tool_calls == []
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_gemini_stream_http_5xx_retries_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http_retry, "_backoff_delay", lambda *args: 0.0)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, text="service unavailable")

    adapter = GeminiAdapter(max_retries=1)
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: SLF001
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.chat_completion(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                base_url=GEMINI_URL,
                stream=True,
            )
        assert attempts["n"] == 2
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_gemini_stream_thought_signature_on_calls_and_non_sse_array() -> None:
    sse_lines = [
        "data: "
        + json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Checking weather...", "thought": True},
                                {
                                    "functionCall": {
                                        "id": "call_sig_1",
                                        "name": "get_weather",
                                        "args": {"city": "Berlin"},
                                    },
                                    "thoughtSignature": "sig-call-1",
                                },
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        ),
        "data: [DONE]",
    ]
    json_array = [
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "lo", "thoughtSignature": "sig-arr"}]},
                    "finishReason": "STOP",
                }
            ]
        },
    ]
    error_array = [
        {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]},
        {"error": {"code": 503, "message": "overloaded"}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        mode = request.headers.get("x-mode", "sse")
        if mode == "sse":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="\n\n".join(sse_lines) + "\n\n",
            )
        if mode == "json-array":
            return httpx.Response(
                200, headers={"content-type": "application/json"}, json=json_array
            )
        if mode == "error-array":
            return httpx.Response(
                200, headers={"content-type": "application/json"}, json=error_array
            )
        if mode == "bad-json":
            return httpx.Response(200, headers={"content-type": "application/json"}, content=b"42")
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=": comment only\n\n"
        )

    for mode in ("sse", "json-array", "error-array", "bad-json", "empty-sse"):
        adapter = GeminiAdapter(default_headers={"x-mode": mode})
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: SLF001
        try:
            res = await adapter.chat_completion(
                model="gemini-3-pro",
                messages=[{"role": "user", "content": "hi"}],
                base_url=GEMINI_URL,
                stream=True,
            )
            if mode == "sse":
                assert res.reasoning == "Checking weather..."
                assert len(res.tool_calls) == 1
                assert res.tool_calls[0].extra_content == {"thoughtSignature": "sig-call-1"}
            elif mode == "json-array":
                assert res.content == "Hello"
                assert res.message_extra_content == {"thoughtSignature": "sig-arr"}
            elif mode == "error-array":
                assert res.transport_error_status == 503
                assert res.transport_error_is_infrastructure is True
                assert "overloaded" in res.content
            else:
                assert res.malformed is True
        finally:
            await adapter.aclose()


def test_gemini_thought_signature_empty_text_and_vendor_nest() -> None:
    contents, _ = _to_gemini_contents(
        [
            {
                "role": "assistant",
                "content": "",
                "extra_content": {"gemini": {"thought_signature": "sig-empty-turn"}},
            }
        ]
    )
    assert contents == [
        {"role": "model", "parts": [{"text": "", "thoughtSignature": "sig-empty-turn"}]}
    ]


def test_minimal_request_gemini_without_api_key() -> None:
    url, payload, headers = minimal_request(
        GEMINI_URL,
        "gemini-2.5-flash",
        None,
        wire_format="gemini",
        extra_params={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert ":generateContent" in url
    assert "x-goog-api-key" not in headers
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert isinstance(payload["contents"], list)
    first_part: dict[str, Any] = payload["contents"][0]["parts"][0]
    assert first_part["text"] == "Say hello."
