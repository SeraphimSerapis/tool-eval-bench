"""OpenAI-compatible adapter for any /v1/chat/completions endpoint.

Works with vLLM, LiteLLM, llama.cpp, and any other server exposing
the OpenAI chat completions API with tool-calling support.
When stream=True, uses SSE to measure time-to-first-token (TTFT).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from tool_eval_bench.adapters.http_retry import (
    DEFAULT_MAX_RATE_LIMIT_RETRIES,
    DEFAULT_MAX_RETRIES,
    PreEventLines,
    RateLimitCoordinator,
    RateLimitObserver,
    RateLimitStatus,
    RetryingHTTPAdapter,
    SseDataBuffer,
    _backoff_delay,
    _rate_limit_delay,
    _retry_after_seconds,
    stream_error_is_infrastructure,
)
from tool_eval_bench.domain.adapters import (
    RETRYABLE_STATUS_CODES,
    BackendAdapter,
    ChatCompletionResult,
    ProviderToolCall,
)
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS, ChatMessage
from tool_eval_bench.utils.openai_compat import completion_stream_error, max_tokens_retry_payload
from tool_eval_bench.utils.urls import chat_completions_url as _chat_completions_url
from tool_eval_bench.utils.urls import redact_url as _redact_url

logger = logging.getLogger(__name__)


def _rejection_log_level(payload: dict[str, Any], status: int) -> int:
    """Quiet the rejection the orchestrator answers by retrying a weaker response_format."""
    if "response_format" in payload and status in (400, 422):
        return logging.DEBUG
    return logging.WARNING


def _is_pre_inference_backend_error(body: str) -> bool:
    """Return whether a 4xx body names a known failure before inference."""
    return "failed to initialize samplers" in body.lower()


# Re-exported for callers and tests that reach for the retry policy through the
# adapter module it used to live in.
__all__ = [
    "DEFAULT_MAX_RATE_LIMIT_RETRIES",
    "DEFAULT_MAX_RETRIES",
    "OpenAICompatibleAdapter",
    "RateLimitCoordinator",
    "RateLimitObserver",
    "RateLimitStatus",
    "_backoff_delay",
    "_rate_limit_delay",
    "_retry_after_seconds",
]


def _repair_streamed_tool_args(s: str) -> str:
    """Repair truncated JSON from streamed tool-call arguments.

    When a server uses ``--stream-interval > 1``, tool-call argument tokens
    are batched into larger SSE chunks.  In some cases the server's own
    tool-call parser may not detect the closing brace within a batch,
    causing the accumulated arguments string to be missing its final ``}``
    or have unbalanced quotes.

    This function applies minimal repairs:
      1. Close unterminated strings (odd number of unescaped quotes).
      2. Close unbalanced braces and brackets.

    If the string is already valid JSON, it is returned unchanged.
    """
    if not s or not s.strip():
        return "{}"
    try:
        json.loads(s)
        return s  # already valid
    except json.JSONDecodeError:
        pass

    repaired = s
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in s:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                logger.warning("Could not repair streamed tool-call arguments: %.80s", s)
                return s
            stack.pop()

    # Close an unfinished escape before closing the JSON string.  Otherwise a
    # trailing backslash would escape the quote we append and remain invalid.
    if in_string:
        if escaped:
            repaired += "\\"
        repaired += '"'
    # Stack order matters for nested objects and arrays.  Closing by aggregate
    # brace counts can produce invalid JSON such as ``{"x": [1}}]``.
    repaired += "".join(reversed(stack))

    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return repaired
        logger.warning("Could not repair streamed tool-call arguments into an object: %.80s", s)
        return s
    except json.JSONDecodeError:
        logger.warning("Could not repair streamed tool-call arguments: %.80s", s)
        return s  # return original; let downstream parsing handle it


def _parse_sse_error_field(value: str) -> object:
    """The payload of a non-standard ``error:`` SSE field: JSON when it parses, else text."""
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _starts_new_call(slot: dict, func: dict, ids_seen: dict[str, int]) -> bool:
    """Whether a delta with an unseen id at an occupied index is a separate call.

    Only split when the occupied slot holds a real id (not a synthetic one),
    the delta carries a name or arguments, and either the delta names a
    function or the slot's arguments are already complete JSON.  A server that
    sends a fresh id on every fragment of one call therefore still merges its
    fragments, and an id-only delta never opens an empty call.
    """
    if slot["id"] not in ids_seen:
        return False
    if not func.get("name") and not func.get("arguments"):
        return False
    if func.get("name"):
        return True
    try:
        json.loads(slot["arguments"])
    except json.JSONDecodeError:
        return False
    return True


def _normalize_tool_calls(raw_calls: list[dict] | None) -> list[ProviderToolCall]:
    if not raw_calls or not isinstance(raw_calls, list):
        return []
    result = []
    for idx, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            continue
        func = call.get("function")
        if not isinstance(func, dict):
            func = {}
        args = func.get("arguments", "{}")
        if args is None:
            args_str = "{}"
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = json.dumps(args)
        extra = call.get("extra_content")
        result.append(
            ProviderToolCall(
                id=str(call.get("id") or f"tool_call_{idx + 1}"),
                name=str(func.get("name") or "unknown_tool"),
                arguments_str=args_str,
                extra_content=extra if isinstance(extra, dict) else None,
            )
        )
    return result


class OpenAICompatibleAdapter(RetryingHTTPAdapter, BackendAdapter):
    """OpenAI-compatible adapter with connection reuse.

    Works with any server exposing /v1/chat/completions (vLLM, LiteLLM,
    llama.cpp, etc.).  The underlying httpx.AsyncClient is lazily created
    on first use and reused across all requests.
    Call ``aclose()`` when done (optional — Python GC handles cleanup).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._max_completion_tokens_endpoints: set[tuple[str, str]] = set()

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
        conversation_id: str | None = None,
    ) -> ChatCompletionResult:
        url = _chat_completions_url(base_url)
        endpoint = (url, model)
        token_key = (
            "max_completion_tokens"
            if endpoint in self._max_completion_tokens_endpoints
            else "max_tokens"
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            token_key: max_tokens,
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
            if "max_completion_tokens" in extra_params:
                payload.pop("max_tokens", None)
            elif "max_tokens" in extra_params:
                payload.pop("max_completion_tokens", None)

        if stream:
            payload["stream"] = True
            # Usage is emitted by OpenAI-compatible servers in a final
            # stream chunk only when requested.  Preserve an explicit caller
            # mapping, including an intentional ``include_usage=False``.
            payload.setdefault("stream_options", {"include_usage": True})

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers = self._request_headers(headers, conversation_id)

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
            # 4xx errors (400/422) often mean the server couldn't process
            # malformed tool-call arguments in conversation history (e.g.
            # vLLM's _postprocess_messages).  Return a graceful error
            # instead of crashing the scenario.  5xx and rate-limit/gateway
            # statuses must propagate so they can be retried and, if they
            # persist, classified as infrastructure rather than model failures.
            if exc.response.status_code >= 500 or exc.response.status_code in (
                RETRYABLE_STATUS_CODES
            ):
                raise
            body_text = exc.response.text[:200].strip()
            retry_payload = max_tokens_retry_payload(
                payload, exc.response.status_code, exc.response.text
            )
            if retry_payload is not None:
                payload.clear()
                payload.update(retry_payload)
                self._max_completion_tokens_endpoints.add((url, str(payload.get("model", ""))))
                logger.info("Endpoint rejected max_tokens; retrying with max_completion_tokens")
                return await self._non_stream_request(client, url, payload, headers, timeout)
            logger.log(
                _rejection_log_level(payload, exc.response.status_code),
                "Server returned %d for %s: %s",
                exc.response.status_code,
                _redact_url(url),
                body_text,
            )
            return ChatCompletionResult(
                content=f"[server error {exc.response.status_code}] {body_text}",
                tool_calls=[],
                raw_response={},
                elapsed_ms=elapsed_ms,
                transport_error_status=exc.response.status_code,
                transport_error_is_infrastructure=_is_pre_inference_backend_error(
                    exc.response.text
                ),
            )
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"expected a JSON object, got {type(data).__name__}")
        except Exception as exc:
            logger.warning("Malformed JSON in response from %s: %s", _redact_url(url), exc)
            return ChatCompletionResult(
                content="[malformed response]",
                tool_calls=[],
                raw_response={},
                elapsed_ms=elapsed_ms,
                malformed=True,
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
        """Stream SSE response, measuring TTFT accurately."""
        started = time.perf_counter()
        ttft_ms: float | None = None
        content_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}
        tool_call_indices_by_id: dict[str, int] = {}
        # Where a server's explicit index now points after a split, so its
        # later id-less deltas follow the most recent call at that index.
        slot_for_index: dict[int, int] = {}
        next_tool_call_index = 0
        message_extra_content: dict[str, Any] | None = None
        reasoning_parts: list[str] = []
        stream_usage: dict = {}  # usage from final chunk
        finish_reason: str | None = None
        # Whether a line parsed as an SSE event, and the lines before the
        # first one, which may be a plain JSON body.
        saw_event = False
        unparsed_lines = PreEventLines()

        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=timeout
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                if exc.response.status_code >= 500 or exc.response.status_code in (
                    RETRYABLE_STATUS_CODES
                ):
                    await exc.response.aread()  # allow retry logic to inspect Retry-After
                    raise
                body_bytes = await exc.response.aread()
                full_body_text = body_bytes.decode("utf-8", errors="replace")
                body_text = full_body_text[:200].strip()
                retry_payload = max_tokens_retry_payload(
                    payload, exc.response.status_code, full_body_text
                )
                if retry_payload is not None:
                    payload.clear()
                    payload.update(retry_payload)
                    self._max_completion_tokens_endpoints.add((url, str(payload.get("model", ""))))
                    logger.info("Endpoint rejected max_tokens; retrying with max_completion_tokens")
                    return await self._stream_request(client, url, payload, headers, timeout)
                logger.log(
                    _rejection_log_level(payload, exc.response.status_code),
                    "Stream request returned %d for %s: %s",
                    exc.response.status_code,
                    _redact_url(url),
                    body_text,
                )
                return ChatCompletionResult(
                    content=f"[server error {exc.response.status_code}] {body_text}",
                    tool_calls=[],
                    raw_response={},
                    elapsed_ms=elapsed_ms,
                    transport_error_status=exc.response.status_code,
                    transport_error_is_infrastructure=_is_pre_inference_backend_error(
                        full_body_text
                    ),
                )

            # Some OpenAI-compatible endpoints ignore ``stream=true`` and
            # return an ordinary JSON completion with HTTP 200.  Treat that as
            # a valid non-streamed response instead of silently returning an
            # empty completion after iterating its JSON bytes as SSE lines.
            response_headers = getattr(response, "headers", None)
            content_type = str(
                response_headers.get("content-type", "text/event-stream")
                if response_headers is not None
                else "text/event-stream"
            )
            if "text/event-stream" not in content_type.lower():
                body = await response.aread()
                try:
                    data = json.loads(body)
                except (json.JSONDecodeError, TypeError):
                    # A few servers omit the content type while still sending
                    # SSE.  Continue through the normal line parser only when
                    # the body is not an ordinary JSON document.
                    pass
                else:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if isinstance(data, dict):
                        return self._parse_response(data, elapsed_ms)
                    logger.warning(
                        "Malformed non-SSE JSON from %s: expected object, got %s",
                        _redact_url(url),
                        type(data).__name__,
                    )
                    return ChatCompletionResult(
                        content="[malformed response]",
                        tool_calls=[],
                        raw_response={},
                        elapsed_ms=elapsed_ms,
                        malformed=True,
                    )

            sse_buffer = SseDataBuffer()
            async for raw_line in response.aiter_lines():
                line = raw_line.rstrip("\r")
                if not saw_event:
                    unparsed_lines.add(line)
                if line.startswith("error:"):
                    # llama-server builds before ggml-org/llama.cpp#16109
                    # (September 2025) report a mid-stream failure in a
                    # non-standard ``error:`` field instead of ``data:``.
                    sse_buffer.reset()
                    chunk: dict[str, Any] = {"error": _parse_sse_error_field(line[6:])}
                    saw_event = True
                else:
                    if not line.startswith("data:"):
                        if line == "" or line.startswith(("event:", "id:", "retry:")):
                            sse_buffer.reset()
                        continue
                    payload_str = line[5:]
                    if payload_str.startswith(" "):
                        payload_str = payload_str[1:]
                    if payload_str.strip() == "[DONE]":
                        sse_buffer.reset()
                        saw_event = True
                        break

                    parsed_chunk = sse_buffer.push_data(payload_str)
                    if not isinstance(parsed_chunk, dict):
                        continue
                    chunk = parsed_chunk
                    saw_event = True

                error = completion_stream_error(chunk)
                if error is not None:
                    status, message = error
                    details = chunk.get("error")
                    context_overflow = (
                        isinstance(details, dict)
                        and details.get("code") == "context_length_exceeded"
                    )
                    # Discard partial calls; repairing a failed stream could
                    # turn unfinished arguments into an executable tool call.
                    return ChatCompletionResult(
                        content=f"[server error {status}] {message}",
                        raw_response=chunk,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        ttft_ms=ttft_ms,
                        transport_error_status=status,
                        transport_error_is_infrastructure=stream_error_is_infrastructure(
                            status, context_overflow=context_overflow
                        ),
                    )

                # Capture usage from final chunk (vLLM/OpenAI include this)
                usage_chunk = chunk.get("usage")
                if isinstance(usage_chunk, dict) and usage_chunk:
                    stream_usage = usage_chunk

                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    continue

                if choices[0].get("finish_reason"):
                    finish_reason = str(choices[0]["finish_reason"])
                raw_delta = choices[0].get("delta")
                delta = raw_delta if isinstance(raw_delta, dict) else {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")

                # Measure TTFT on the first generated reasoning, content, or
                # tool-call delta.  Thinking tokens precede visible output on
                # reasoning models such as Ornith.
                if ttft_ms is None and (
                    reasoning or delta.get("content") or delta.get("tool_calls")
                ):
                    ttft_ms = (time.perf_counter() - started) * 1000

                # Accumulate content
                if delta.get("content"):
                    content_parts.append(str(delta["content"]))

                # Accumulate reasoning
                if reasoning:
                    reasoning_parts.append(str(reasoning))

                # Accumulate tool calls
                raw_tc_deltas = delta.get("tool_calls")
                for tc_delta in raw_tc_deltas if isinstance(raw_tc_deltas, list) else []:
                    if not isinstance(tc_delta, dict):
                        continue
                    call_id = tc_delta.get("id")
                    explicit_idx = tc_delta.get("index")
                    raw_func = tc_delta.get("function")
                    func = raw_func if isinstance(raw_func, dict) else {}
                    if explicit_idx is not None:
                        idx = slot_for_index.get(int(explicit_idx), int(explicit_idx))
                        slot = tool_calls_map.get(idx)
                        if call_id and slot is not None and slot["id"] != call_id:
                            # Ollama's legacy tool parser tags every parallel
                            # call ``index: 0`` (ollama/ollama#15497).  Route a
                            # known id back to its slot, and give a new call
                            # its own slot instead of appending its arguments
                            # to the previous call's.
                            if call_id in tool_call_indices_by_id:
                                idx = tool_call_indices_by_id[call_id]
                                slot_for_index[int(explicit_idx)] = idx
                            elif _starts_new_call(slot, func, tool_call_indices_by_id):
                                while next_tool_call_index in tool_calls_map:
                                    next_tool_call_index += 1
                                idx = next_tool_call_index
                                next_tool_call_index += 1
                                slot_for_index[int(explicit_idx)] = idx
                    elif call_id and call_id in tool_call_indices_by_id:
                        # Gemini's OpenAI-compatible stream identifies parallel
                        # calls by id but omits OpenAI's numeric index.
                        idx = tool_call_indices_by_id[call_id]
                    elif call_id:
                        while next_tool_call_index in tool_calls_map:
                            next_tool_call_index += 1
                        idx = next_tool_call_index
                        tool_call_indices_by_id[call_id] = idx
                        next_tool_call_index += 1
                    elif len(tool_calls_map) == 1:
                        # A continuation delta from a provider that omits both
                        # id and index can still be associated unambiguously.
                        idx = next(iter(tool_calls_map))
                    else:
                        while next_tool_call_index in tool_calls_map:
                            next_tool_call_index += 1
                        idx = next_tool_call_index
                        next_tool_call_index += 1

                    if call_id:
                        tool_call_indices_by_id[call_id] = idx
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": call_id or f"tool_call_{idx + 1}",
                            "name": "",
                            "arguments": "",
                            "extra_content": None,
                        }
                    if func.get("name"):
                        tool_calls_map[idx]["name"] = str(func["name"])
                    arg_fragment = func.get("arguments")
                    if arg_fragment:
                        tool_calls_map[idx]["arguments"] += (
                            arg_fragment
                            if isinstance(arg_fragment, str)
                            else json.dumps(arg_fragment)
                        )
                    if call_id:
                        tool_calls_map[idx]["id"] = str(call_id)
                    if isinstance(tc_delta.get("extra_content"), dict):
                        tool_calls_map[idx]["extra_content"] = tc_delta["extra_content"]

                if isinstance(delta.get("extra_content"), dict):
                    message_extra_content = delta["extra_content"]

        elapsed_ms = (time.perf_counter() - started) * 1000
        if not saw_event:
            # A server that ignored stream=true can still label a plain JSON
            # completion as an event stream. Anything else with no SSE event,
            # an empty body, a comment-only stream, or a proxy's HTML page,
            # would be graded as an empty answer, so flag it the way the
            # non-streamed path does.
            data = unparsed_lines.json()
            if isinstance(data, dict):
                return self._parse_response(data, elapsed_ms)
            logger.warning("Malformed stream from %s: no SSE data event", _redact_url(url))
            return ChatCompletionResult(
                content="[malformed response]",
                tool_calls=[],
                raw_response={},
                elapsed_ms=elapsed_ms,
                malformed=True,
            )
        content = "".join(content_parts)
        reasoning_str = "".join(reasoning_parts) or None

        # Build tool calls
        tool_calls: list[ProviderToolCall] = []
        for idx in sorted(tool_calls_map.keys()):
            tc = tool_calls_map[idx]
            # Repair truncated JSON arguments.  When the server uses
            # --stream-interval > 1, tool-call arguments may arrive in
            # larger batches where the closing brace is split across
            # chunks or missing entirely (vLLM issue with batched token
            # streaming).  Apply the same repair logic used for the
            # round-trip message to ensure arguments are parseable.
            # A ``length`` stop is real truncation, not a batching artefact:
            # keep the arguments malformed so the trace shows the cut-off call
            # instead of a plausible one with half its values.
            raw_args = tc["arguments"]
            repaired_args = (
                raw_args if finish_reason == "length" else _repair_streamed_tool_args(raw_args)
            )
            if repaired_args != raw_args:
                logger.debug(
                    "Repaired streamed tool-call arguments for index %d: %.80s → %.80s",
                    idx,
                    raw_args,
                    repaired_args,
                )
            tool_calls.append(
                ProviderToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments_str=repaired_args,
                    extra_content=tc["extra_content"],
                )
            )

        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_response={},
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            reasoning=reasoning_str,
            message_extra_content=message_extra_content,
            prompt_tokens=stream_usage.get("prompt_tokens"),
            completion_tokens=stream_usage.get("completion_tokens"),
            finish_reason=finish_reason,
        )

    @staticmethod
    def _parse_response(data: dict, elapsed_ms: float) -> ChatCompletionResult:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            error = completion_stream_error(data)
            if error is not None:
                status, err_message = error
                details = data.get("error")
                context_overflow = (
                    isinstance(details, dict) and details.get("code") == "context_length_exceeded"
                )
                return ChatCompletionResult(
                    content=f"[server error {status}] {err_message}",
                    raw_response=data,
                    elapsed_ms=elapsed_ms,
                    transport_error_status=status,
                    transport_error_is_infrastructure=stream_error_is_infrastructure(
                        status, context_overflow=context_overflow
                    ),
                )

        message: dict[str, Any] = {}
        finish_reason: str | None = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            raw_message = choices[0].get("message")
            if isinstance(raw_message, dict):
                message = raw_message
            finish_reason = choices[0].get("finish_reason") or None

        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
        elif not isinstance(content, str):
            content = str(content)

        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        reasoning = message.get("reasoning_content") or message.get("reasoning")

        # Extract token usage (most OpenAI-compatible servers include this)
        raw_usage = data.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        extra_content = message.get("extra_content")

        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_response=data,
            elapsed_ms=elapsed_ms,
            reasoning=str(reasoning) if reasoning is not None else None,
            message_extra_content=extra_content if isinstance(extra_content, dict) else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=str(finish_reason) if finish_reason else None,
        )
