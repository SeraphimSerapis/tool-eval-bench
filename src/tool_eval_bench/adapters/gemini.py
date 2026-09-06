"""Native Gemini API adapter (https://ai.google.dev/api).

Speaks ``:generateContent`` / ``:streamGenerateContent`` directly rather than
going through Google's OpenAI compatibility layer.  The benchmark's internal
message and tool shapes stay OpenAI-flavoured — this adapter translates in both
directions so scenarios, evaluators, and reports are format-agnostic.

Translation notes that are not obvious from the shapes alone:

* System messages become ``systemInstruction``; Gemini's ``contents`` only
  accepts the ``user`` and ``model`` roles.
* Tool results go back as a ``user`` turn carrying ``functionResponse`` parts,
  matched to their call by name and id.
* Gemini 3 returns a ``thoughtSignature`` alongside a function call and requires
  it to be echoed back on the next request, so it round-trips through
  ``ProviderToolCall.extra_content``.
* JSON Schema is narrowed to the subset the API's ``Schema`` accepts; leaving
  ``additionalProperties`` or ``$schema`` in a declaration is rejected with a
  400 rather than ignored.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from tool_eval_bench.adapters.http_retry import RetryingHTTPAdapter
from tool_eval_bench.adapters.wire_format import gemini_generate_url
from tool_eval_bench.domain.adapters import (
    RETRYABLE_STATUS_CODES,
    BackendAdapter,
    ChatCompletionResult,
    ProviderToolCall,
)
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS, ChatMessage
from tool_eval_bench.utils.urls import redact_url as _redact_url

logger = logging.getLogger(__name__)


def _sse_payload(line: str) -> str | None:
    """Return the payload from an SSE data line.

    The optional space after ``data:`` is part of the SSE grammar, not a
    requirement.  Google and compatible gateways have emitted both
    ``data: {...}`` and ``data:{...}`` forms.
    """
    if not line.startswith("data:"):
        return None
    payload = line[5:]
    return payload[1:] if payload.startswith(" ") else payload


# Keys of the OpenAPI-derived ``Schema`` the API accepts in a function
# declaration.  Anything else (``additionalProperties``, ``$schema``,
# ``default``, ``exclusiveMinimum``, …) is a 400, so it is dropped.
_SCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "enum",
        "items",
        "properties",
        "required",
        "anyOf",
        "propertyOrdering",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
    }
)

# OpenAI sampling knobs that have a direct generationConfig equivalent.
_GENERATION_CONFIG_ALIASES: dict[str, str] = {
    "top_p": "topP",
    "topP": "topP",
    "top_k": "topK",
    "topK": "topK",
    "seed": "seed",
    "stop": "stopSequences",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
}

# Request-level keys a caller may set through --backend-kwargs; passed through
# untouched so Gemini-only features stay reachable.
_PASSTHROUGH_REQUEST_KEYS: frozenset[str] = frozenset(
    {"safetySettings", "cachedContent", "toolConfig", "labels"}
)

_TOOL_CHOICE_MODES: dict[str, str] = {
    "auto": "AUTO",
    "none": "NONE",
    "required": "ANY",
    "any": "ANY",
}


def _thought_signature(extra_content: Any) -> str | None:
    """Pull a thought signature out of whichever shape it round-tripped in.

    The native API nests it on the part; Google's OpenAI compatibility layer
    reports it under a provider extension.  Accept both so a trace recorded
    through one format can be replayed through the other.
    """
    if not isinstance(extra_content, dict):
        return None
    for container in (extra_content, extra_content.get("google"), extra_content.get("gemini")):
        if not isinstance(container, dict):
            continue
        for key in ("thoughtSignature", "thought_signature"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _to_gemini_schema(schema: Any) -> Any:
    """Narrow a JSON Schema to the subset accepted by a function declaration."""
    if isinstance(schema, list):
        return [_to_gemini_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "type" and isinstance(value, str):
            converted[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            converted[key] = {name: _to_gemini_schema(sub) for name, sub in value.items()}
        elif key in ("items", "anyOf"):
            converted[key] = _to_gemini_schema(value)
        else:
            converted[key] = value
    return converted


def _to_gemini_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool definitions into a single Gemini ``Tool`` entry."""
    if not tools:
        return None
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        func = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(func, dict) or not func.get("name"):
            continue
        declaration: dict[str, Any] = {"name": func["name"]}
        if func.get("description"):
            declaration["description"] = func["description"]
        parameters = func.get("parameters")
        # An empty object schema is rejected; omit parameters for no-arg tools.
        if isinstance(parameters, dict) and parameters.get("properties"):
            declaration["parameters"] = _to_gemini_schema(parameters)
        declarations.append(declaration)
    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def _to_tool_config(tool_choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Map an OpenAI ``tool_choice`` onto ``functionCallingConfig``."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        mode = _TOOL_CHOICE_MODES.get(tool_choice.lower())
        if mode is None:
            return None
        return {"functionCallingConfig": {"mode": mode}}
    func = tool_choice.get("function") if isinstance(tool_choice, dict) else None
    name = func.get("name") if isinstance(func, dict) else None
    if not name:
        return None
    return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}


def _tool_result_response(content: Any, name: str) -> dict[str, Any]:
    """Wrap a tool result in the object ``functionResponse.response`` requires."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {"result": content}
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}
    return {"result": content}


def _to_gemini_contents(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Split OpenAI-style messages into ``contents`` plus ``systemInstruction``."""
    contents: list[dict[str, Any]] = []
    system_texts: list[str] = []
    # Tool results reference a call id; Gemini matches on the function name, so
    # keep the mapping from the assistant turn that requested the call.
    call_names: dict[str, str] = {}

    def append(role: str, parts: list[dict[str, Any]]) -> None:
        if not parts:
            return
        # Consecutive same-role turns (parallel tool results) merge into one.
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
            return
        contents.append({"role": role, "parts": parts})

    for message in messages:
        role = message.get("role")
        raw_content = message.get("content")
        content = raw_content if isinstance(raw_content, str) else ""

        if role == "system":
            if content:
                system_texts.append(content)
            continue

        if role == "user":
            append("user", [{"text": content}] if content else [])
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"text": content})
            for call in message.get("tool_calls") or []:
                func = call.get("function") or {}
                name = func.get("name") or "unknown_tool"
                call_id = call.get("id")
                if call_id:
                    call_names[call_id] = name
                raw_args = func.get("arguments")
                if isinstance(raw_args, dict):
                    args = raw_args
                else:
                    try:
                        parsed = json.loads(raw_args or "{}")
                    except (json.JSONDecodeError, TypeError):
                        parsed = {}
                    args = parsed if isinstance(parsed, dict) else {}
                function_call: dict[str, Any] = {"name": name, "args": args}
                if call_id:
                    function_call["id"] = call_id
                part: dict[str, Any] = {"functionCall": function_call}
                signature = _thought_signature(call.get("extra_content")) or _thought_signature(
                    message.get("extra_content")
                )
                if signature:
                    part["thoughtSignature"] = signature
                parts.append(part)
            append("model", parts)
            continue

        if role in ("tool", "function"):
            call_id = message.get("tool_call_id")
            name = message.get("name") or call_names.get(call_id or "", "unknown_tool")
            response: dict[str, Any] = {
                "name": name,
                "response": _tool_result_response(raw_content, name),
            }
            if call_id:
                response["id"] = call_id
            # Function results are attributed to the user side of the turn.
            append("user", [{"functionResponse": response}])
            continue

        if content:
            append("user", [{"text": content}])

    system_instruction = {"parts": [{"text": "\n\n".join(system_texts)}]} if system_texts else None
    return contents, system_instruction


def _generation_config(
    *,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
    extra_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build ``generationConfig`` from the benchmark's sampling options."""
    config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }

    if response_format:
        kind = response_format.get("type")
        if kind in ("json_object", "json_schema"):
            config["responseMimeType"] = "application/json"
        schema = response_format.get("json_schema")
        if isinstance(schema, dict) and isinstance(schema.get("schema"), dict):
            config["responseSchema"] = _to_gemini_schema(schema["schema"])

    for key, value in (extra_params or {}).items():
        if key in _PASSTHROUGH_REQUEST_KEYS:
            continue
        if key == "generationConfig" and isinstance(value, dict):
            config.update(value)
        elif key == "thinkingConfig" and isinstance(value, dict):
            config["thinkingConfig"] = value
        elif key == "chat_template_kwargs" and isinstance(value, dict):
            # --no-think: the OpenAI-side switch maps onto a zero thinking budget.
            if value.get("enable_thinking") is False:
                config["thinkingConfig"] = {"thinkingBudget": 0}
        elif key in _GENERATION_CONFIG_ALIASES:
            config[_GENERATION_CONFIG_ALIASES[key]] = value
        elif key in ("max_tokens", "maxOutputTokens"):
            config["maxOutputTokens"] = value
        elif key == "temperature":
            config["temperature"] = value
        else:
            logger.debug("Dropping option %r: no native Gemini equivalent", key)

    return config


def _parse_parts(
    parts: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[ProviderToolCall]]:
    """Split candidate parts into text, reasoning, and tool calls."""
    texts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[ProviderToolCall] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        call = part.get("functionCall")
        if isinstance(call, dict):
            signature = part.get("thoughtSignature")
            tool_calls.append(
                ProviderToolCall(
                    # Ids are optional in the API; a stable one is assigned once
                    # the whole response (or stream) has been collected.
                    id=call.get("id") or "",
                    name=call.get("name", "unknown_tool"),
                    arguments_str=json.dumps(call.get("args") or {}),
                    extra_content=(
                        {"thoughtSignature": signature} if isinstance(signature, str) else None
                    ),
                )
            )
            continue
        text = part.get("text")
        if not isinstance(text, str) or not text:
            continue
        if part.get("thought"):
            thoughts.append(text)
        else:
            texts.append(text)
    return texts, thoughts, tool_calls


def _assign_call_ids(tool_calls: list[ProviderToolCall]) -> list[ProviderToolCall]:
    """Give every call an id, since the API only sends one for some models."""
    for index, call in enumerate(tool_calls):
        if not call.id:
            call.id = f"tool_call_{index + 1}"
    return tool_calls


class GeminiAdapter(RetryingHTTPAdapter, BackendAdapter):
    """Adapter for the native Gemini generateContent API.

    Shares the OpenAI adapter's retry, rate-limit, and connection handling; only
    the wire format differs.
    """

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        api_key: str | None = None,
        base_url: str = "",
        extra_params: dict[str, Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = True,
    ) -> ChatCompletionResult:
        contents, system_instruction = _to_gemini_contents(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": _generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_params=extra_params,
            ),
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        declarations = _to_gemini_tools(tools)
        if declarations:
            payload["tools"] = declarations
            tool_config = _to_tool_config(tool_choice)
            if tool_config:
                payload["toolConfig"] = tool_config

        for key in _PASSTHROUGH_REQUEST_KEYS:
            if extra_params and key in extra_params:
                payload[key] = extra_params[key]

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["x-goog-api-key"] = api_key

        url = gemini_generate_url(base_url, model, stream=stream)
        client = self._get_client()
        req_timeout = httpx.Timeout(timeout_seconds)

        async def attempt() -> ChatCompletionResult:
            await self._rate_limits.acquire()
            if stream:
                return await self._stream_request(client, url, payload, headers, req_timeout)
            return await self._non_stream_request(client, url, payload, headers, req_timeout)

        return await self._with_retries(attempt, url=url)

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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            graceful = self._graceful_error(exc, url, elapsed_ms, exc.response.text)
            if graceful is None:
                raise
            return graceful
        try:
            data = response.json()
        except Exception as exc:
            logger.warning("Malformed JSON in response from %s: %s", _redact_url(url), exc)
            return ChatCompletionResult(
                content="[malformed response]",
                tool_calls=[],
                raw_response={},
                elapsed_ms=elapsed_ms,
            )
        return self._parse_response(data, elapsed_ms)

    async def _stream_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict,
        headers: dict,
        timeout: httpx.Timeout | None = None,
    ) -> ChatCompletionResult:
        """Stream SSE chunks, measuring TTFT on the first content part."""
        started = time.perf_counter()
        ttft_ms: float | None = None
        texts: list[str] = []
        thoughts: list[str] = []
        tool_calls: list[ProviderToolCall] = []
        usage: dict[str, Any] = {}

        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=timeout
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                body = (await exc.response.aread()).decode("utf-8", errors="replace")
                graceful = self._graceful_error(exc, url, elapsed_ms, body)
                if graceful is None:
                    raise
                return graceful

            # A native endpoint or gateway may ignore the streaming request
            # and return a regular JSON completion with HTTP 200.  Parse it as
            # a valid non-streamed response instead of dropping the body.
            response_headers = getattr(response, "headers", None)
            content_type = str(
                response_headers.get("content-type", "text/event-stream")
                if response_headers is not None
                else "text/event-stream"
            )
            if "text/event-stream" not in content_type.lower():
                json_body = await response.aread()
                try:
                    data = json.loads(json_body)
                except (json.JSONDecodeError, TypeError):
                    pass
                else:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    return self._parse_response(data, elapsed_ms)

            async for line in response.aiter_lines():
                sse_data = _sse_payload(line)
                if sse_data is None:
                    continue
                chunk_str = sse_data.strip()
                if not chunk_str or chunk_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usageMetadata"):
                    usage = chunk["usageMetadata"]

                candidates = chunk.get("candidates") or []
                if not candidates:
                    continue
                parts = (candidates[0].get("content") or {}).get("parts") or []
                chunk_texts, chunk_thoughts, chunk_calls = _parse_parts(parts)
                # Native Gemini thinking chunks are generated output even
                # though they are not visible answer text.
                if ttft_ms is None and (chunk_texts or chunk_thoughts or chunk_calls):
                    ttft_ms = (time.perf_counter() - started) * 1000
                texts.extend(chunk_texts)
                thoughts.extend(chunk_thoughts)
                tool_calls.extend(chunk_calls)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return ChatCompletionResult(
            content="".join(texts),
            tool_calls=_assign_call_ids(tool_calls),
            raw_response={},
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            reasoning="".join(thoughts) or None,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )

    @staticmethod
    def _graceful_error(
        exc: httpx.HTTPStatusError,
        url: str,
        elapsed_ms: float,
        body: str,
    ) -> ChatCompletionResult | None:
        """Return a soft result for a 4xx, or None when the error must propagate.

        A 400 usually means the model produced arguments the API will not accept
        in the next request's history — the scenario should record that and move
        on.  5xx and rate-limit statuses are infrastructure and must be retried.
        """
        status = exc.response.status_code
        if status >= 500 or status in RETRYABLE_STATUS_CODES:
            return None
        snippet = body[:200].strip()
        logger.warning("Gemini API returned %d for %s: %s", status, _redact_url(url), snippet)
        return ChatCompletionResult(
            content=f"[server error {status}] {snippet}",
            tool_calls=[],
            raw_response={},
            elapsed_ms=elapsed_ms,
            transport_error_status=status,
        )

    @staticmethod
    def _parse_response(data: dict, elapsed_ms: float) -> ChatCompletionResult:
        candidates = data.get("candidates") or []
        parts: list[dict[str, Any]] = []
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
        texts, thoughts, tool_calls = _parse_parts(parts)

        content = "".join(texts)
        if not content and not tool_calls:
            # A blocked prompt or a candidate cut short returns no parts at all;
            # surface the reason instead of an empty, unexplained turn.
            reason = (candidates[0].get("finishReason") if candidates else None) or (
                data.get("promptFeedback") or {}
            ).get("blockReason")
            if reason and reason != "STOP":
                content = f"[no content: {reason}]"

        usage = data.get("usageMetadata") or {}
        return ChatCompletionResult(
            content=content,
            tool_calls=_assign_call_ids(tool_calls),
            raw_response=data,
            elapsed_ms=elapsed_ms,
            reasoning="".join(thoughts) or None,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )
