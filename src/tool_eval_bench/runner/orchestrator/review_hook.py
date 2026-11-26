"""Turn-review integration and revision validation helpers for the orchestrator."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.domain.models import ChatMessage
from tool_eval_bench.domain.wire_hygiene import detect_unparsed_tool_call
from tool_eval_bench.runner.orchestrator.wire import _is_visible_answer, _parses_as_json
from tool_eval_bench.runner.review import (
    ReviewStats,
    TurnReviewer,
    finish_reason_label,
    revision_request,
    storable_text,
    storable_value,
)


def _review_payload(stats: ReviewStats | None) -> dict[str, Any] | None:
    return stats.to_dict() if stats is not None else None


async def _review_turn(
    reviewer: TurnReviewer,
    draft: ChatCompletionResult,
    *,
    messages: list[ChatMessage],
    tools: list[dict[str, Any]] | None,
    revise: Callable[[ChatMessage], Awaitable[ChatCompletionResult]],
    stats: ReviewStats,
    trace_lines: list[str],
    turn: int,
    omitted_after_system: int,
    conversation_id: str,
    tool_choice: str | dict[str, Any] | None = None,
    detect_unparsed: bool = True,
) -> tuple[ChatCompletionResult, list[ChatCompletionResult]]:
    """Review ``draft`` and let the benchmarked model revise it on feedback.

    Returns the turn to act on and every revision the model produced, used or
    not, so the caller can count their tokens. A reviewer error, a revision
    request the endpoint rejects, or a revision with nothing to act on leaves
    the current draft in place: review must never turn a working turn into a
    failed one. Replaced drafts appear only in the trace, never in
    ``messages`` or the scenario state.
    """
    from tool_eval_bench.runner import orchestrator as _orchestrator_pkg

    revisions: list[ChatCompletionResult] = []
    for round_no in range(1, reviewer.max_rounds + 1):
        if not reviewer.applies(
            has_tool_calls=bool(draft.tool_calls), has_answer=_is_visible_answer(draft.content)
        ):
            break
        verdict = await reviewer.review(
            messages=messages,
            tools=tools,
            draft=draft,
            omitted_after_system=omitted_after_system,
            conversation_id=conversation_id,
            tool_choice=tool_choice,
        )
        stats.record(verdict)
        tag = f"{turn}.{round_no}"
        trace_lines.append(f"review_{tag}={verdict.trace_text()}")
        if verdict.verdict != "revise":
            break
        try:
            revised = await revise(revision_request(draft, verdict.feedback))
        except Exception as exc:  # noqa: BLE001 - keep the draft, as for a reviewer error
            stats.revision_errors += 1
            # The type only: exception text can quote a server's reply.
            trace_lines.append(f"revision_error_{tag}={type(exc).__name__}; kept the draft")
            break
        if revised.malformed or revised.transport_error_status is not None:
            stats.revision_errors += 1
            reason = (
                "malformed response"
                if revised.malformed
                else f"HTTP {revised.transport_error_status}"
            )
            trace_lines.append(f"revision_error_{tag}={reason}; kept the draft")
            break
        revisions.append(revised)
        # The revision becomes model output in the trace and the report, both
        # written as UTF-8; a lone surrogate from a JSON escape would break them.
        revised = _storable_revision(revised)
        try:
            # A revision is judged by the same markup rule as the turn path,
            # which follows the scenario's choice, not a wire override.
            detect = tools if detect_unparsed else None
            unusable = _orchestrator_pkg._unusable_revision(revised, detect)
        except Exception as exc:  # noqa: BLE001 - a revision we cannot inspect is not used
            unusable = f"revision could not be inspected ({type(exc).__name__})"
        if unusable is not None:
            stats.revision_errors += 1
            trace_lines.append(f"revision_error_{tag}={unusable}; kept the draft")
            break
        trace_lines.append(f"rejected_draft_{tag}={draft.content or '[tool_calls_only]'}")
        for call in draft.tool_calls:
            trace_lines.append(f"rejected_tool_call_{tag}={call.name} {call.arguments_str}")
        stats.revised += 1
        draft = revised
    return draft, revisions


def _wire_tool_choice(
    extra: dict[str, Any], choice: str | dict[str, Any], wire_format: str = "openai"
) -> str | dict[str, Any]:
    """The tool choice a request carries once ``--backend-kwargs`` overrides apply.

    Each adapter honours a different override: the OpenAI one sends
    ``tool_choice`` from the extra params over its own, the Gemini one sends
    ``toolConfig``, and the Messages API adapter drops both. A malformed
    override leaves the scenario's own choice in place.
    """
    if wire_format == "openai":
        overridden = extra.get("tool_choice")
        if isinstance(overridden, str | dict):
            return overridden
    elif wire_format == "gemini":
        native = extra.get("toolConfig")
        config = native.get("functionCallingConfig") if isinstance(native, dict) else None
        mode = config.get("mode") if isinstance(config, dict) else None
        if isinstance(mode, str):
            return {"NONE": "none", "ANY": "required", "AUTO": "auto"}.get(mode.upper(), choice)
    return choice


def _storable_revision(result: ChatCompletionResult) -> ChatCompletionResult:
    """``result`` with every text field UTF-8 can encode.

    Content, reasoning, and tool calls all reach the trace, tool arguments are
    measured as UTF-8, and IDs and provider metadata are replayed in the next
    request; an unencodable character in any of them would fail a scenario
    whose draft worked.
    """
    return replace(
        result,
        content=storable_text(result.content or ""),
        reasoning=storable_text(result.reasoning) if result.reasoning is not None else None,
        message_extra_content=storable_value(result.message_extra_content),
        tool_calls=[
            replace(
                call,
                id=storable_text(str(call.id)) if call.id is not None else call.id,
                name=storable_text(call.name),
                arguments_str=storable_text(call.arguments_str),
                extra_content=storable_value(call.extra_content),
            )
            for call in result.tool_calls
        ],
    )


def _is_cut_off_tool_call(result: ChatCompletionResult) -> bool:
    """Whether ``result`` stopped at ``max_tokens`` mid-way through tool arguments."""
    return result.finish_reason == "length" and any(
        not _parses_as_json(call.arguments_str) for call in result.tool_calls
    )


def _unusable_revision(
    result: ChatCompletionResult, tools: list[dict[str, Any]] | None
) -> str | None:
    """Why a revision gives the harness nothing to act on, or None when it does.

    A revision that stops at the token ceiling with no answer, cuts off its
    tool arguments mid-generation, or emits tool-call markup the server did
    not parse would turn a draft that worked into a turn the harness tags as
    truncated, empty, or unparsed. Those failures say nothing about the draft,
    so the draft stays.
    """
    if result.tool_calls:
        if _is_cut_off_tool_call(result):
            return "revision cut off its tool-call arguments at the token ceiling"
        return None
    if tools and detect_unparsed_tool_call(result.content, tools) is not None:
        return "revision emitted tool-call markup the server did not parse"
    if _is_visible_answer(result.content):
        return None
    return (
        "revision had no answer and no tool call "
        f"(finish_reason={finish_reason_label(result.finish_reason)})"
    )
