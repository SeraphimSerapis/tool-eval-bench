"""Wire/argument parsing, error injection, failure classification, and trace formatting."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import random
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import httpx

from tool_eval_bench.domain.adapters import (
    RETRYABLE_STATUS_CODES,
    ChatCompletionResult,
    ProviderToolCall,
)
from tool_eval_bench.domain.loop_detection import (
    LoopDetectionResult,
    detect_degenerate_loop,
)
from tool_eval_bench.domain.models import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ChatMessage,
)
from tool_eval_bench.domain.scenarios import (
    FailureKind,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.domain.tools import (
    BENCHMARK_REFERENCE_DATE,
    BENCHMARK_REFERENCE_DAY,
    SYSTEM_PROMPT,
)
from tool_eval_bench.evals.helpers import has_explicit_tool_error, matching_tool_results

logger = logging.getLogger(__name__)

MAX_TURNS = 8
# Separate cap for turns where every tool call received a harness-injected
# --error-rate failure so simulated outages do not exhaust a scenario's normal
# turn budget while still keeping retries strictly finite.
_MAX_INJECTED_ERROR_TURNS = 12
DEFAULT_TIMEOUT_SECONDS = DEFAULT_REQUEST_TIMEOUT_SECONDS

# Prefixes for side-effecting / state-mutating tools across universal and custom
# Hard Mode toolsets. Dependency bypasses on mutating tools record a safety
# violation; read-only tools (get_*, list_*, search_*, read_*, lookup_*,
# calculator, etc.) fail normally without claiming a safety violation.
_MUTATING_TOOL_PREFIXES: tuple[str, ...] = (
    "add_",
    "annotate_",
    "apply_",
    "archive_",
    "book_",
    "cancel_",
    "copy_",
    "create_",
    "delete_",
    "deploy_",
    "disable_",
    "draft_",
    "file_",
    "grant_",
    "issue_",
    "modify_",
    "notify_",
    "patch_",
    "pay_",
    "post_",
    "process_",
    "publish_",
    "purge_",
    "put_",
    "refund_",
    "reissue_",
    "release_",
    "remove_",
    "request_",
    "reserve_",
    "resolve_",
    "restart_",
    "restore_",
    "revoke_",
    "rollback_",
    "rotate_",
    "scale_",
    "schedule_",
    "send_",
    "set_",
    "submit_",
    "switch_",
    "transfer_",
    "update_",
    "upload_",
    "write_",
)


def _is_mutating_tool(name: str) -> bool:
    """Return whether ``name`` represents a state-mutating tool."""
    return name.startswith(_MUTATING_TOOL_PREFIXES)


def _producer_result_observed(state: ScenarioState, call: ToolCallRecord) -> bool:
    """Whether ``call`` produced an observed outcome a dependent consumer may rely on.

    Harness-injected failures and explicit tool errors (e.g. ``{"error": ...}``)
    supply no data. Structured non-retryable domain failures without an ``error``
    key (such as TC-89's ``PAYEE_ACCOUNT_CLOSED`` triggering compensation) still
    count as observed outcomes.
    """
    if call.injected:
        return False
    if not has_explicit_tool_error(state, call):
        return True
    results = matching_tool_results(state, call)
    return bool(results) and all(
        isinstance(r.result, dict)
        and "error" not in r.result
        and r.result.get("retryable") is False
        for r in results
    )


class _HarnessExecutionError(Exception):
    """Internal error raised when a scenario handler, checkpoint, or serializer fails."""

    def __init__(self, stage: str, cause: BaseException, tb_str: str) -> None:
        super().__init__(f"{stage} error: {cause}")
        self.stage = stage
        self.cause = cause
        self.tb_str = tb_str


def _reject_non_finite_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant not allowed: {token}")


def _parse_tool_arguments(arguments_str: str) -> dict[str, Any]:
    """Parse tool-call arguments as a strict JSON object, rejecting NaN/Infinity."""
    try:
        parsed = json.loads(arguments_str, parse_constant=_reject_non_finite_constant)
    except Exception:
        logger.debug("Failed to parse strict tool call arguments: %.80s", arguments_str)
        return {}
    return parsed if isinstance(parsed, dict) else {}


# Error injection types (Claw-Eval methodology: 429/500/timeout)
_INJECTED_ERRORS = [
    {"error": "Rate limit exceeded. Please retry after 2 seconds.", "status": 429},
    {"error": "Internal server error. The service is temporarily unavailable.", "status": 500},
    {"error": "Request timed out. The service did not respond in time.", "status": 503},
]


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def _classify_runtime_error(exc: BaseException) -> str:
    """Map a runtime exception to a failure-kind label."""
    if isinstance(exc, asyncio.TimeoutError):
        return FailureKind.TIMEOUT
    if isinstance(exc, httpx.TimeoutException):
        return FailureKind.TIMEOUT
    if isinstance(exc, httpx.NetworkError | httpx.ProtocolError | httpx.ProxyError):
        return FailureKind.CONNECTION_ERROR
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        # 429 survives the adapter's retry budget only when the endpoint is
        # persistently saturated — an infrastructure condition, not a model one.
        if status >= 500 or status in RETRYABLE_STATUS_CODES:
            return FailureKind.SERVER_ERROR
        # 4xx errors (400/422) indicate the server rejected the request —
        # typically malformed tool-call arguments generated by the model.
        # The adapter catches 4xx gracefully and returns an error result,
        # so this path is rarely hit, but classifying correctly is safer
        # than falling through to MODEL_CRASH.
        return FailureKind.WRONG_ARGS
    return FailureKind.MODEL_CRASH


def _classify_evaluation_failure(
    state: ScenarioState,
    evaluation: ScenarioEvaluation,
) -> str | None:
    """Heuristic failure-kind label from the scenario state and evaluation.

    Evaluators may optionally set ``evaluation.failure_kind``; when they do not,
    we fall back to simple state-based heuristics so reports still surface a
    coarse "why" for every failure.
    """
    if evaluation.failure_kind is not None:
        return evaluation.failure_kind
    if evaluation.status != ScenarioStatus.FAIL:
        return None
    # Simple heuristics
    if not state.tool_calls:
        return FailureKind.MISSING_STEP
    summary_lower = (evaluation.summary or "").lower()
    if "forbidden" in summary_lower or "should not" in summary_lower:
        return FailureKind.FORBIDDEN_ACTION
    if "wrong tool" in summary_lower or "unexpected tool" in summary_lower:
        return FailureKind.WRONG_TOOL
    if "argument" in summary_lower or "parameter" in summary_lower:
        return FailureKind.WRONG_ARGS
    return FailureKind.WRONG_ARGS


def _is_visible_answer(content: str | None) -> bool:
    """Whether a turn's content is something a user would see as an answer.

    The Gemini adapter stands in "[no content: REASON]" for an empty candidate,
    so that placeholder counts as no answer, like empty or whitespace text.
    """
    if not content or not content.strip():
        return False
    return not content.startswith("[no content:")


def _detect_turn_degenerate_loop(
    result: ChatCompletionResult,
    *,
    extra_texts: Sequence[tuple[str, str]] = (),
) -> tuple[str, LoopDetectionResult] | None:
    """Inspect reasoning, content, and any extra text for a degenerate repetition loop."""
    candidates: list[tuple[str, str | None]] = [
        ("reasoning", result.reasoning),
        ("content", result.content),
        *extra_texts,
    ]
    for channel, text in candidates:
        detected = detect_degenerate_loop(text)
        if detected.detected:
            return channel, detected
    return None


def _scenario_seed_offset(scenario_id: str) -> int:
    """Return a stable per-scenario offset for seeded error injection."""
    return int.from_bytes(hashlib.sha256(scenario_id.encode()).digest()[:4], "big")


def _parses_as_json(text: str) -> bool:
    try:
        json.loads(text)
    except (ValueError, RecursionError):
        # RecursionError: deeply nested input exhausts the decoder's stack.
        return False
    return True


def _draw_injected_error(
    error_rate: float,
    rng: random.Random | None = None,
) -> dict[str, Any] | None:
    """Decide whether the next tool call fails with a simulated error.

    Returns the simulated error, or ``None`` when the call should run. The
    caller draws before running the mock handler: a simulated 429/500/503
    means the request never took effect, so the handler must not commit a
    side effect the model was told failed. The error distribution follows
    Claw-Eval: ~33% each of 429, 500, timeout.

    Args:
        rng: Optional seeded Random instance for reproducibility.
             Falls back to process-global random if None.
    """
    _rng = rng or random
    if error_rate <= 0 or _rng.random() >= error_rate:
        return None
    return dict(_rng.choice(_INJECTED_ERRORS))


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _initial_messages(
    user_message: str,
    *,
    reference_date: str | None = None,
    reference_day: str | None = None,
    context_pressure_messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    ref_date = reference_date or BENCHMARK_REFERENCE_DATE
    ref_day = reference_day or BENCHMARK_REFERENCE_DAY
    # An explicit override replaces the built-in "helpful assistant" persona;
    # the benchmark reference-date line is always kept because relative-time
    # scenarios depend on it.
    base_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
    msgs: list[ChatMessage] = [
        {
            "role": "system",
            "content": (
                f"{base_prompt}\n\n"
                f"Benchmark context: today is {ref_date} ({ref_day}). "
                "Use this date for any relative time request."
            ),
        },
    ]
    # Inject context pressure filler turns (if any) between system prompt
    # and the real user message to simulate a busy conversation history.
    # Every scenario gets the same filler, so a server with prefix caching
    # prefills it once per run instead of once per scenario. A cached prefix
    # yields the same KV as a cold one; only latency differs, and the timeout
    # is scaled to the fill. Copied so a scenario cannot mutate the shared list.
    if context_pressure_messages:
        msgs.extend(copy.deepcopy(context_pressure_messages))
    msgs.append({"role": "user", "content": user_message})
    return msgs


def _repair_json_str(s: str) -> str:
    """Best-effort repair of truncated JSON argument strings.

    Some models (especially under constrained generation or aggressive
    max_tokens) emit tool-call arguments with unterminated strings or
    missing closing brackets.  vLLM's ``_postprocess_messages`` does
    ``json.loads()`` on the arguments when they come back in the
    conversation history and crashes with a 400 if they're malformed.

    This function applies minimal fixes so the arguments survive a
    round-trip through the server.  If the string is already valid JSON,
    it is returned unchanged.
    """
    if not s or not s.strip():
        return "{}"
    try:
        parsed = json.loads(s, parse_constant=_reject_non_finite_constant)
        if isinstance(parsed, dict):
            return s  # already valid JSON object
        logger.warning("Tool-call arguments were not a JSON object: %s", s[:120])
        return "{}"
    except (json.JSONDecodeError, ValueError):
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
                logger.warning("Could not repair malformed tool-call arguments: %s", s[:120])
                return "{}"
            stack.pop()

    if in_string:
        if escaped:
            repaired += "\\"
        repaired += '"'
    repaired += "".join(reversed(stack))

    try:
        parsed = json.loads(repaired, parse_constant=_reject_non_finite_constant)
        if isinstance(parsed, dict):
            return repaired
        logger.warning("Repaired tool-call arguments were not a JSON object: %s", s[:120])
        return "{}"
    except (json.JSONDecodeError, ValueError):
        # Last resort: return empty object so the server doesn't crash
        logger.warning("Could not repair malformed tool-call arguments: %s", s[:120])
        return "{}"


def _ensure_unique_tool_call_ids(
    tool_calls: Sequence[ProviderToolCall], seen_ids: set[str], turn: int
) -> list[ProviderToolCall]:
    """Return tool calls with non-empty IDs that are unique across the conversation."""
    normalized: list[ProviderToolCall] = []
    for idx, tc in enumerate(tool_calls, start=1):
        candidate = str(tc.id).strip() if tc.id is not None else ""
        if not candidate:
            candidate = f"tool_call_t{turn}_{idx}"
        if candidate in seen_ids:
            suffix = 2
            while f"{candidate}_{suffix}" in seen_ids:
                suffix += 1
            candidate = f"{candidate}_{suffix}"
        seen_ids.add(candidate)
        normalized.append(replace(tc, id=candidate) if candidate != tc.id else tc)
    return normalized


def _format_calls_made(tool_calls: Sequence[ToolCallRecord]) -> list[str]:
    return [
        f"{tc.name}({', '.join(f'{k}={v}' for k, v in tc.arguments.items())})" for tc in tool_calls
    ]


def _assistant_message(
    result: ChatCompletionResult, *, preserve_reasoning: bool = False
) -> ChatMessage:
    msg: ChatMessage = {"role": "assistant", "content": result.content}
    # DeepSeek requires reasoning_content for every assistant message in a user
    # turn that called tools. Ordinary no-tool turns must not add it back.
    if result.reasoning is not None and (result.tool_calls or preserve_reasoning):
        msg["reasoning_content"] = result.reasoning
    if result.message_extra_content is not None:
        msg["extra_content"] = result.message_extra_content
    if result.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": _repair_json_str(tc.arguments_str),
                },
                **({"extra_content": tc.extra_content} if tc.extra_content is not None else {}),
            }
            for tc in result.tool_calls
        ]
    return msg


def _tool_result_message(call_id: str, name: str, result: Any) -> ChatMessage:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result, allow_nan=False) if not isinstance(result, str) else result,
    }


# ---------------------------------------------------------------------------
# Trace formatting
# ---------------------------------------------------------------------------


# Consecutive tool-call turns that must be identical (same calls, same
# arguments, same results) before the loop guard stops the scenario. Legitimate
# polling (TC-61) repeats a call once and then sees a different result, so it
# never reaches this count; a model stuck re-issuing the same request does.
_LOOP_GUARD_TURNS = 3


def _turn_signature(tool_calls: Sequence[ProviderToolCall], results: Sequence[Any]) -> str:
    """Order-independent fingerprint of one turn's tool calls and their results."""
    pairs = sorted(
        (call.name, call.arguments_str, json.dumps(result, sort_keys=True, default=str))
        for call, result in zip(tool_calls, results, strict=True)
    )
    return json.dumps(pairs)


def _format_trace(
    model: str,
    scenario: ScenarioDefinition,
    evaluation: dict[str, Any],
    trace_lines: list[str],
) -> str:
    lines = [
        f"model={model}",
        f"scenario={scenario.id} {scenario.title}",
        f"prompt={scenario.user_message}",
        "",
        *trace_lines,
        "",
        f"verdict={evaluation['status']}",
        f"summary={evaluation['summary']}",
    ]
    if evaluation.get("note"):
        lines.append(f"note={evaluation['note']}")
    return "\n".join(line for line in lines if line is not None)
