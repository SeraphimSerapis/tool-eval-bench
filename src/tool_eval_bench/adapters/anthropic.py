"""Native Anthropic Messages API adapter (https://platform.claude.com/docs/en/api/messages).

Speaks ``POST /v1/messages`` directly instead of Anthropic's OpenAI compatibility
layer.  That also covers gateways which expose the Messages API in front of
other models (OpenCode Zen, LiteLLM's ``/v1/messages`` route, Bedrock's Mantle
endpoint).  The benchmark's internal message and tool shapes stay
OpenAI-flavoured — this adapter translates in both directions so scenarios,
evaluators, and reports are format-agnostic.

Translation notes that are not obvious from the shapes alone:

* System messages become the top-level ``system`` string; ``messages`` only
  accepts the ``user`` and ``assistant`` roles.
* Tool calls are ``tool_use`` content blocks on the assistant turn.  Tool
  results go back as ``tool_result`` blocks on a ``user`` turn, and parallel
  results merge into one turn: the API requires every ``tool_use`` to be
  answered in the very next message.
* An assistant turn that carried thinking and a tool call must replay its
  thinking blocks verbatim (including the signature) on the next request, or
  the API rejects the history.  They round-trip through
  ``ChatCompletionResult.message_extra_content`` under ``{"anthropic": ...}``.
* ``response_format`` maps to ``output_config.format``.  The API rejects the
  numeric and string constraints JSON Schema allows, so they are stripped the
  same way the official SDKs strip them client-side.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from tool_eval_bench.adapters.http_retry import RetryingHTTPAdapter
from tool_eval_bench.adapters.wire_format import anthropic_messages_url
from tool_eval_bench.domain.adapters import (
    RETRYABLE_STATUS_CODES,
    BackendAdapter,
    ChatCompletionResult,
    ProviderToolCall,
)
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS, ChatMessage
from tool_eval_bench.utils.openai_compat import sampling_retry_payload
from tool_eval_bench.utils.urls import redact_url as _redact_url

logger = logging.getLogger(__name__)

# Every request must name an API version; this is the only released one.
ANTHROPIC_VERSION = "2023-06-01"

# Key under ``extra_content`` where an assistant turn's thinking blocks travel
# between requests.
_EXTRA_CONTENT_KEY = "anthropic"
_THINKING_BLOCKS_KEY = "thinking_blocks"

_STOP_REASONS: dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}

# Schema keywords ``output_config.format`` rejects.  The SDKs drop them before
# sending and validate client-side; the benchmark's evaluators do the latter.
_UNSUPPORTED_FORMAT_KEYS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)

# OpenAI sampling knobs with a direct Messages API equivalent.
_REQUEST_ALIASES: dict[str, str] = {
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop_sequences",
    "stop_sequences": "stop_sequences",
}

# Request-level keys a caller may set through --backend-kwargs; passed through
# untouched so Anthropic-only features stay reachable.
_PASSTHROUGH_REQUEST_KEYS: frozenset[str] = frozenset(
    {"thinking", "output_config", "metadata", "service_tier", "speed", "fallbacks"}
)

# Stream ``error`` events arrive on an HTTP 200, so the status the runner would
# have seen on a non-streamed request is reconstructed from the error type.
_STREAM_ERROR_STATUS: dict[str, int] = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "overloaded_error": 529,
}


def _finish_reason(value: Any) -> str | None:
    """Map ``stop_reason`` onto the OpenAI vocabulary the runner reads."""
    if not value:
        return None
    reason = str(value)
    return _STOP_REASONS.get(reason, reason)


def _sse_payload(line: str) -> str | None:
    """Return the payload from an SSE data line, tolerating a missing space."""
    if not line.startswith("data:"):
        return None
    payload = line[5:]
    return payload[1:] if payload.startswith(" ") else payload


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Return tool-call arguments as an object, degrading to empty on bad JSON."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _thinking_blocks(extra_content: Any) -> list[dict[str, Any]]:
    """Pull replayable thinking blocks out of a message's ``extra_content``."""
    if not isinstance(extra_content, dict):
        return []
    container = extra_content.get(_EXTRA_CONTENT_KEY)
    if not isinstance(container, dict):
        return []
    blocks = container.get(_THINKING_BLOCKS_KEY)
    return [b for b in blocks if isinstance(b, dict)] if isinstance(blocks, list) else []


def _to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool definitions into Messages API tool entries."""
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        func = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(func, dict) or not func.get("name"):
            continue
        entry: dict[str, Any] = {"name": func["name"]}
        if func.get("description"):
            entry["description"] = func["description"]
        parameters = func.get("parameters")
        # ``input_schema`` is required, even for a tool that takes no arguments.
        entry["input_schema"] = (
            parameters
            if isinstance(parameters, dict) and parameters
            else {"type": "object", "properties": {}}
        )
        converted.append(entry)
    return converted or None


def _to_tool_choice(
    tool_choice: str | dict[str, Any] | None,
    parallel_tool_calls: bool | None,
) -> dict[str, Any] | None:
    """Map an OpenAI ``tool_choice`` (plus the parallel flag) onto the API's."""
    choice: dict[str, Any] | None
    if tool_choice is None:
        choice = None
    elif isinstance(tool_choice, str):
        mode = tool_choice.lower()
        if mode in ("auto", "none"):
            choice = {"type": mode}
        elif mode in ("required", "any"):
            choice = {"type": "any"}
        else:
            choice = None
    else:
        func = tool_choice.get("function") if isinstance(tool_choice, dict) else None
        name = func.get("name") if isinstance(func, dict) else None
        choice = {"type": "tool", "name": name} if name else None
    if parallel_tool_calls is False:
        choice = choice or {"type": "auto"}
        if choice["type"] != "none":
            choice["disable_parallel_tool_use"] = True
    return choice


def _tool_result_content(content: Any) -> str:
    """Tool results are sent as text; the API accepts a plain string."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content)


def _to_anthropic_messages(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], str | None]:
    """Split OpenAI-style messages into ``messages`` plus the ``system`` prompt."""
    converted: list[dict[str, Any]] = []
    system_texts: list[str] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        # Consecutive same-role turns (parallel tool results) merge into one.
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].extend(blocks)
            return
        converted.append({"role": role, "content": blocks})

    for message in messages:
        role = message.get("role")
        raw_content = message.get("content")
        content = raw_content if isinstance(raw_content, str) else ""

        if role == "system":
            if content:
                system_texts.append(content)
            continue

        if role == "user":
            append("user", [{"type": "text", "text": content}] if content else [])
            continue

        if role == "assistant":
            # Thinking blocks must come first and be byte-identical to what the
            # API returned, signature included.
            blocks: list[dict[str, Any]] = list(_thinking_blocks(message.get("extra_content")))
            if content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                func = call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or f"toolu_{len(blocks) + 1}",
                        "name": func.get("name") or "unknown_tool",
                        "input": _parse_arguments(func.get("arguments")),
                    }
                )
            append("assistant", blocks)
            continue

        if role in ("tool", "function"):
            result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id") or "",
                "content": _tool_result_content(raw_content),
            }
            append("user", [result_block])
            continue

        if content:
            append("user", [{"type": "text", "text": content}])

    return converted, "\n\n".join(system_texts) or None


def _strip_unsupported_format_keys(schema: Any) -> Any:
    """Drop the JSON Schema constraints ``output_config.format`` rejects."""
    if isinstance(schema, list):
        return [_strip_unsupported_format_keys(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    return {
        key: _strip_unsupported_format_keys(value)
        for key, value in schema.items()
        if key not in _UNSUPPORTED_FORMAT_KEYS
    }


def _to_output_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map an OpenAI ``response_format`` onto ``output_config.format``.

    Only ``json_schema`` has an equivalent.  ``json_object`` carries no schema
    to constrain against, so it is left to the prompt.
    """
    if not response_format or response_format.get("type") != "json_schema":
        return None
    wrapper = response_format.get("json_schema")
    schema = wrapper.get("schema") if isinstance(wrapper, dict) else None
    if not isinstance(schema, dict):
        return None
    return {"type": "json_schema", "schema": _strip_unsupported_format_keys(schema)}


def _apply_extra_params(
    payload: dict[str, Any],
    headers: dict[str, str],
    extra_params: dict[str, Any] | None,
) -> None:
    """Fold ``--backend-kwargs`` into the request, translating OpenAI names."""
    for key, value in (extra_params or {}).items():
        if key in _PASSTHROUGH_REQUEST_KEYS:
            if key == "output_config" and isinstance(value, dict):
                payload.setdefault("output_config", {}).update(value)
            else:
                payload[key] = value
        elif key in _REQUEST_ALIASES:
            payload[_REQUEST_ALIASES[key]] = [value] if isinstance(value, str) else value
        elif key in ("max_tokens", "max_completion_tokens"):
            payload["max_tokens"] = value
        elif key == "temperature":
            payload["temperature"] = value
        elif key == "chat_template_kwargs" and isinstance(value, dict):
            # --no-think: the OpenAI-side switch maps onto disabled thinking.
            # Whether the model accepts that is the model's business.
            if value.get("enable_thinking") is False:
                payload["thinking"] = {"type": "disabled"}
        elif key == "betas" and isinstance(value, (list, tuple)):
            headers["anthropic-beta"] = ",".join(str(beta) for beta in value)
        else:
            logger.debug("Dropping option %r: no Messages API equivalent", key)


def _parse_content_blocks(
    blocks: list[Any],
) -> tuple[list[str], list[str], list[ProviderToolCall], list[dict[str, Any]]]:
    """Split content blocks into text, reasoning, tool calls, and replayable thinking."""
    texts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[ProviderToolCall] = []
    thinking_blocks: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
        elif kind == "tool_use":
            tool_calls.append(
                ProviderToolCall(
                    id=block.get("id") or f"toolu_{len(tool_calls) + 1}",
                    name=block.get("name") or "unknown_tool",
                    arguments_str=json.dumps(block.get("input") or {}),
                )
            )
        elif kind == "thinking":
            thought = block.get("thinking")
            if isinstance(thought, str) and thought:
                thoughts.append(thought)
            thinking_blocks.append(block)
        elif kind == "redacted_thinking":
            thinking_blocks.append(block)
    return texts, thoughts, tool_calls, thinking_blocks


def _message_extra_content(thinking_blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not thinking_blocks:
        return None
    return {_EXTRA_CONTENT_KEY: {_THINKING_BLOCKS_KEY: thinking_blocks}}


class AnthropicAdapter(RetryingHTTPAdapter, BackendAdapter):
    """Adapter for the Anthropic Messages API.

    Shares the OpenAI adapter's retry, rate-limit, and connection handling; only
    the wire format differs.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # (url, model) pairs that rejected sampling parameters, so the retry
        # happens once per endpoint rather than once per request.
        self._sampling_rejected_endpoints: set[tuple[str, str]] = set()

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
        converted, system = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": converted,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system

        declarations = _to_anthropic_tools(tools)
        if declarations:
            payload["tools"] = declarations
            choice = _to_tool_choice(tool_choice, parallel_tool_calls)
            if choice:
                payload["tool_choice"] = choice

        output_format = _to_output_format(response_format)
        if output_format:
            payload["output_config"] = {"format": output_format}

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if api_key:
            # Anthropic reads x-api-key; most Messages-compatible gateways read
            # one or the other, so send both.
            headers["x-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"

        payload["temperature"] = temperature
        _apply_extra_params(payload, headers, extra_params)
        url = anthropic_messages_url(base_url)
        if (url, model) in self._sampling_rejected_endpoints:
            for key in ("temperature", "top_p", "top_k"):
                payload.pop(key, None)

        if stream:
            payload["stream"] = True

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
            if self._retry_without_sampling(
                payload, url, exc.response.status_code, exc.response.text
            ):
                return await self._non_stream_request(client, url, payload, headers, timeout)
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
        """Stream SSE events, measuring TTFT on the first generated delta.

        Content blocks are keyed by ``index``: text and thinking accumulate
        their deltas, and a ``tool_use`` block collects ``partial_json``
        fragments that only parse once ``content_block_stop`` arrives.
        """
        started = time.perf_counter()
        ttft_ms: float | None = None
        # index -> the block under construction, in the API's own shape
        blocks: dict[int, dict[str, Any]] = {}
        partial_json: dict[int, list[str]] = {}
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        stream_error: dict[str, Any] | None = None

        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=timeout
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                body = (await exc.response.aread()).decode("utf-8", errors="replace")
                if self._retry_without_sampling(payload, url, exc.response.status_code, body):
                    return await self._stream_request(client, url, payload, headers, timeout)
                graceful = self._graceful_error(exc, url, elapsed_ms, body)
                if graceful is None:
                    raise
                return graceful

            # A gateway may ignore ``stream`` and answer with a plain JSON
            # message on HTTP 200.  Parse it rather than dropping the body.
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
                    event = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue

                kind = event.get("type")
                if kind == "message_start":
                    usage.update((event.get("message") or {}).get("usage") or {})
                elif kind == "content_block_start":
                    index = int(event.get("index", len(blocks)))
                    block = dict(event.get("content_block") or {})
                    blocks[index] = block
                    if block.get("type") == "tool_use":
                        partial_json[index] = []
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - started) * 1000
                elif kind == "content_block_delta":
                    index = int(event.get("index", 0))
                    block = blocks.setdefault(index, {"type": "text", "text": ""})
                    delta = event.get("delta") or {}
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        block["text"] = block.get("text", "") + str(delta.get("text", ""))
                    elif delta_type == "thinking_delta":
                        block["thinking"] = block.get("thinking", "") + str(
                            delta.get("thinking", "")
                        )
                    elif delta_type == "signature_delta":
                        block["signature"] = str(delta.get("signature", ""))
                    elif delta_type == "input_json_delta":
                        partial_json.setdefault(index, []).append(
                            str(delta.get("partial_json", ""))
                        )
                    else:
                        continue
                    # Thinking is generated output even though it is not
                    # visible answer text.
                    if ttft_ms is None and delta_type != "signature_delta":
                        ttft_ms = (time.perf_counter() - started) * 1000
                elif kind == "content_block_stop":
                    index = int(event.get("index", 0))
                    fragments = partial_json.pop(index, None)
                    if fragments is not None and index in blocks:
                        blocks[index]["input"] = _parse_arguments("".join(fragments))
                elif kind == "message_delta":
                    usage.update(event.get("usage") or {})
                    stop = (event.get("delta") or {}).get("stop_reason")
                    if stop:
                        finish_reason = _finish_reason(stop)
                elif kind == "error":
                    stream_error = event.get("error") or {}
                    break

        elapsed_ms = (time.perf_counter() - started) * 1000
        # A tool_use block whose stop event never arrived still has its
        # fragments; parse what there is rather than sending an empty input.
        for index, fragments in partial_json.items():
            if index in blocks:
                blocks[index]["input"] = _parse_arguments("".join(fragments))

        ordered = [blocks[index] for index in sorted(blocks)]
        texts, thoughts, tool_calls, thinking_blocks = _parse_content_blocks(ordered)

        if stream_error is not None:
            error_type = str(stream_error.get("type") or "api_error")
            status = _STREAM_ERROR_STATUS.get(error_type, 500)
            message = str(stream_error.get("message") or "")[:200].strip()
            logger.warning(
                "Messages API stream error %s for %s: %s", error_type, _redact_url(url), message
            )
            return ChatCompletionResult(
                content=f"[stream error {error_type}] {message}",
                tool_calls=[],
                raw_response={},
                elapsed_ms=elapsed_ms,
                ttft_ms=ttft_ms,
                transport_error_status=status,
                transport_error_is_infrastructure=status >= 500,
            )

        return ChatCompletionResult(
            content="".join(texts),
            tool_calls=tool_calls,
            raw_response={},
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            reasoning="".join(thoughts) or None,
            message_extra_content=_message_extra_content(thinking_blocks),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            finish_reason=finish_reason,
        )

    def _retry_without_sampling(self, payload: dict, url: str, status: int, body: str) -> bool:
        """Strip rejected sampling knobs in place; True when a retry is due."""
        retry_payload = sampling_retry_payload(payload, status, body)
        if retry_payload is None:
            return False
        payload.clear()
        payload.update(retry_payload)
        self._sampling_rejected_endpoints.add((url, str(payload.get("model", ""))))
        logger.info("Endpoint rejected sampling parameters; retrying without them")
        return True

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
        on.  5xx, 529 (overloaded), and rate-limit statuses are infrastructure
        and must be retried.
        """
        status = exc.response.status_code
        if status >= 500 or status in RETRYABLE_STATUS_CODES:
            return None
        snippet = body[:200].strip()
        logger.warning("Messages API returned %d for %s: %s", status, _redact_url(url), snippet)
        return ChatCompletionResult(
            content=f"[server error {status}] {snippet}",
            tool_calls=[],
            raw_response={},
            elapsed_ms=elapsed_ms,
            transport_error_status=status,
        )

    @staticmethod
    def _parse_response(data: dict, elapsed_ms: float) -> ChatCompletionResult:
        content_blocks = data.get("content") or []
        texts, thoughts, tool_calls, thinking_blocks = _parse_content_blocks(
            content_blocks if isinstance(content_blocks, list) else []
        )
        stop_reason = data.get("stop_reason")

        content = "".join(texts)
        if not content and not tool_calls and stop_reason == "refusal":
            # A classifier refusal returns HTTP 200 with no answer; name it
            # rather than grading an empty turn.
            details = data.get("stop_details") or {}
            category = details.get("category") if isinstance(details, dict) else None
            content = f"[no content: refusal{f' ({category})' if category else ''}]"

        usage = data.get("usage") or {}
        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_response=data,
            elapsed_ms=elapsed_ms,
            reasoning="".join(thoughts) or None,
            message_extra_content=_message_extra_content(thinking_blocks),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            finish_reason=_finish_reason(stop_reason),
        )
