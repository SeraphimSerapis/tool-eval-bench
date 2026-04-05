"""OpenAI-compatible adapter for any /v1/chat/completions endpoint.

Works with vLLM, LiteLLM, llama.cpp, and any other server exposing
the OpenAI chat completions API with tool-calling support.
When stream=True, uses SSE to measure time-to-first-token (TTFT).
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.utils.urls import chat_completions_url as _chat_completions_url


def _normalize_tool_calls(raw_calls: list[dict] | None) -> list[ProviderToolCall]:
    if not raw_calls:
        return []
    result = []
    for idx, call in enumerate(raw_calls):
        func = call.get("function", {})
        args = func.get("arguments", "{}")
        if isinstance(args, dict):
            args = json.dumps(args)
        result.append(
            ProviderToolCall(
                id=call.get("id", f"tool_call_{idx + 1}"),
                name=func.get("name", "unknown_tool"),
                arguments_str=args,
            )
        )
    return result


class OpenAICompatibleAdapter(BackendAdapter):
    """OpenAI-compatible adapter with connection reuse.

    Works with any server exposing /v1/chat/completions (vLLM, LiteLLM,
    llama.cpp, etc.).  The underlying httpx.AsyncClient is lazily created
    on first use and reused across all requests.
    Call ``aclose()`` when done (optional — Python GC handles cleanup).
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, creating it lazily on first access.

        The client is created WITHOUT a fixed timeout — callers pass
        per-request timeouts to avoid mismatch between warm-up (60s),
        throughput (180s), and scenario requests.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),  # generous default; overridden per-request
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client (releases connections)."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        base_url: str = "",
        extra_params: dict[str, Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = True,
    ) -> ChatCompletionResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = parallel_tool_calls

        if response_format:
            payload["response_format"] = response_format

        if extra_params:
            payload.update(extra_params)

        if stream:
            payload["stream"] = True

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = _chat_completions_url(base_url)
        client = self._get_client()
        req_timeout = httpx.Timeout(timeout_seconds)

        if stream:
            return await self._stream_request(client, url, payload, headers, req_timeout)
        return await self._non_stream_request(client, url, payload, headers, req_timeout)

    async def _non_stream_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict,
        headers: dict,
        timeout: httpx.Timeout | None = None,
    ) -> ChatCompletionResult:
        started = time.perf_counter()
        response = await client.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data = response.json()
        return self._parse_response(data, elapsed_ms)

    async def _stream_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict,
        headers: dict,
        timeout: httpx.Timeout | None = None,
    ) -> ChatCompletionResult:
        """Stream SSE response, measuring TTFT accurately."""
        started = time.perf_counter()
        ttft_ms: float | None = None
        content_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}  # index → {id, name, arguments}
        reasoning_parts: list[str] = []
        stream_usage: dict = {}  # usage from final chunk

        async with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                # Capture usage from final chunk (vLLM/OpenAI include this)
                if chunk.get("usage"):
                    stream_usage = chunk["usage"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                # Measure TTFT on first content or tool_call delta
                if ttft_ms is None and (delta.get("content") or delta.get("tool_calls")):
                    ttft_ms = (time.perf_counter() - started) * 1000

                # Accumulate content
                if delta.get("content"):
                    content_parts.append(delta["content"])

                # Accumulate reasoning
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    reasoning_parts.append(reasoning)

                # Accumulate tool calls
                for tc_delta in delta.get("tool_calls", []):
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.get("id", f"tool_call_{idx + 1}"),
                            "name": "",
                            "arguments": "",
                        }
                    func = tc_delta.get("function", {})
                    if func.get("name"):
                        tool_calls_map[idx]["name"] = func["name"]
                    if func.get("arguments"):
                        tool_calls_map[idx]["arguments"] += func["arguments"]
                    if tc_delta.get("id"):
                        tool_calls_map[idx]["id"] = tc_delta["id"]

        elapsed_ms = (time.perf_counter() - started) * 1000
        content = "".join(content_parts)
        reasoning_str = "".join(reasoning_parts) or None

        # Build tool calls
        tool_calls: list[ProviderToolCall] = []
        for idx in sorted(tool_calls_map.keys()):
            tc = tool_calls_map[idx]
            tool_calls.append(
                ProviderToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments_str=tc["arguments"],
                )
            )

        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_response={},
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            reasoning=reasoning_str,
            prompt_tokens=stream_usage.get("prompt_tokens"),
            completion_tokens=stream_usage.get("completion_tokens"),
        )

    @staticmethod
    def _parse_response(data: dict, elapsed_ms: float) -> ChatCompletionResult:
        message = {}
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError):
            pass

        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()

        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        reasoning = message.get("reasoning_content") or message.get("reasoning")

        # Extract token usage (most OpenAI-compatible servers include this)
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_response=data,
            elapsed_ms=elapsed_ms,
            reasoning=reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
