"""Core scenario execution loop and concurrent batch runner."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from tool_eval_bench.domain.adapters import (
    RETRYABLE_STATUS_CODES,
    BackendAdapter,
    ChatCompletionResult,
)
from tool_eval_bench.domain.models import ChatMessage, default_max_tokens
from tool_eval_bench.domain.scenarios import (
    FailureKind,
    ModelScoreSummary,
    OnScenarioEvaluated,
    OnScenarioResult,
    OnScenarioStart,
    OnScenarioTurn,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.domain.tools import (
    BENCHMARK_REFERENCE_DATE,
    BENCHMARK_REFERENCE_DAY,
    UNIVERSAL_TOOLS,
)
from tool_eval_bench.domain.wire_hygiene import detect_unparsed_tool_call
from tool_eval_bench.runner.orchestrator.probes import ToolChoiceProbe, probe_tool_choice_required
from tool_eval_bench.runner.orchestrator.review_hook import (
    _is_cut_off_tool_call,
    _review_payload,
    _review_turn,
    _wire_tool_choice,
)
from tool_eval_bench.runner.orchestrator.scoring import score_results
from tool_eval_bench.runner.orchestrator.wire import (
    _LOOP_GUARD_TURNS,
    _MAX_INJECTED_ERROR_TURNS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TURNS,
    _assistant_message,
    _classify_evaluation_failure,
    _classify_runtime_error,
    _detect_turn_degenerate_loop,
    _ensure_unique_tool_call_ids,
    _format_calls_made,
    _format_trace,
    _HarnessExecutionError,
    _initial_messages,
    _is_mutating_tool,
    _is_visible_answer,
    _parse_tool_arguments,
    _parses_as_json,
    _producer_result_observed,
    _scenario_seed_offset,
    _tool_result_message,
    _turn_signature,
)
from tool_eval_bench.runner.review import ReviewStats, TurnReviewer, token_count

logger = logging.getLogger(__name__)


@dataclass
class _ScenarioLoopState:
    """Mutable per-scenario execution accumulators and stop diagnostics."""

    t0: float
    state: ScenarioState
    messages: list[ChatMessage]
    trace_lines: list[str]
    follow_ups: list[str]
    seen_call_ids: set[str]
    active_response_format: dict[str, Any] | None
    review_stats: ReviewStats | None
    user_phase: int = 0
    ttft_ms: float | None = None
    turn_latencies: list[float] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_arg_bytes: int = 0
    parallel_tool_turns: list[int] = field(default_factory=list)
    state_checkpoints: list[str] = field(default_factory=list)
    # True when the loop ends because the turn budget ran out before the model
    # produced a final answer (or before exhausting the follow-up queue).
    budget_exhausted: bool = True
    loop_stop: str | None = None
    recent_turn_signatures: list[str] = field(default_factory=list)
    truncated_stop: str | None = None
    degenerate_loop_stop: str | None = None
    empty_stop: str | None = None
    answer_in_thinking_turns: int = 0
    unparsed_stop: str | None = None
    response_format_fallback: str | None = None
    response_format_sent: bool = False
    turn: int = 0
    counted_turns: int = 0
    injected_error_turns: int = 0


async def _notify_turn(
    callback: OnScenarioTurn,
    scenario: ScenarioDefinition,
    turn: int,
    payload: dict[str, Any],
) -> None:
    """Deliver a turn to an observer without letting it touch the outcome.

    The call sits inside run_scenario's error handling, which turns any
    exception into a scenario failure. A live view that raises must not be
    recorded as the model failing, so its errors are logged and dropped.
    """
    try:
        await callback(scenario, turn, payload)
    except Exception:
        logger.warning(
            "on_scenario_turn observer failed for %s turn %d", scenario.id, turn, exc_info=True
        )


def _build_error_result(
    ctx: _ScenarioLoopState,
    *,
    model: str,
    scenario: ScenarioDefinition,
    summary: str,
    failure_kind: str,
) -> ScenarioResult:
    """Build a failed ScenarioResult preserving all prior-turn metrics and trace."""
    elapsed = time.perf_counter() - ctx.t0
    return ScenarioResult(
        scenario_id=scenario.id,
        status=ScenarioStatus.FAIL,
        points=0,
        summary=summary,
        raw_log=_format_trace(
            model, scenario, {"status": "fail", "summary": summary}, ctx.trace_lines
        ),
        tool_calls_made=_format_calls_made(ctx.state.tool_calls),
        expected_behavior=scenario.description,
        duration_seconds=elapsed,
        ttft_ms=ctx.ttft_ms,
        turn_count=len(ctx.turn_latencies),
        turn_latencies_ms=ctx.turn_latencies,
        prompt_tokens=ctx.total_prompt_tokens,
        completion_tokens=ctx.total_completion_tokens,
        tool_call_arg_bytes=ctx.total_arg_bytes,
        parallel_tool_turns=ctx.parallel_tool_turns,
        state_checkpoints=ctx.state_checkpoints,
        failure_kind=failure_kind,
        review=_review_payload(ctx.review_stats),
        answer_in_thinking_turns=ctx.answer_in_thinking_turns,
    )


async def _request_turn_with_format_fallback(
    adapter: BackendAdapter,
    ctx: _ScenarioLoopState,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    extra: dict[str, Any],
    conversation_id: str,
    turn_tools: list[dict[str, Any]] | None,
    turn_tool_choice: str | dict[str, Any] | None,
    response_format: dict[str, Any] | None,
    use_stream: bool,
) -> tuple[ChatCompletionResult, dict[str, Any] | None]:
    """Send one chat completion turn, stepping down rejected response_format levels."""
    while True:
        if response_format is not None:
            ctx.response_format_sent = True
        result = await adapter.chat_completion(
            model=model,
            messages=ctx.messages,
            tools=turn_tools,
            tool_choice=turn_tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            base_url=base_url,
            extra_params=extra,
            stream=use_stream,
            response_format=response_format,
            conversation_id=conversation_id,
        )
        # Some endpoints (DeepSeek) accept json_object but reject
        # json_schema; others accept neither. Every structured prompt
        # already embeds the schema, so step down and grade the model
        # against it instead of failing on a parameter we chose.
        if (
            not response_format
            or result.transport_error_status not in (400, 422)
            or result.transport_error_is_infrastructure
        ):
            return result, response_format
        rejected = str(response_format.get("type"))
        downgraded: dict[str, Any] | None = (
            {"type": "json_object"} if rejected == "json_schema" else None
        )
        accepted = downgraded["type"] if downgraded else "none"
        ctx.response_format_fallback = f"{rejected} -> {accepted}"
        ctx.trace_lines.append(
            f"response_format_fallback={ctx.response_format_fallback} "
            f"(HTTP {result.transport_error_status})"
        )
        ctx.active_response_format = response_format = downgraded


async def _handle_no_tool_turn(
    ctx: _ScenarioLoopState,
    result: ChatCompletionResult,
    *,
    scenario: ScenarioDefinition,
    scenario_tools: list[dict[str, Any]],
    active_tool_choice: str | dict[str, Any],
    max_turns: int,
    max_tokens: int,
    turn_ms: float,
    turn_ttft_ms: float | None,
    on_scenario_turn: OnScenarioTurn | None,
) -> bool:
    """Process a turn with no tool calls; return True to end the scenario loop."""
    visible_answer = _is_visible_answer(result.content)
    # The ceiling ate the turn: nothing visible came back and the
    # provider says it stopped for length. Check for a degenerate
    # repetition loop first so operators are not misdirected to raise
    # max_tokens; otherwise tag as reasoning truncation.
    if result.finish_reason == "length":
        loop_hit = _detect_turn_degenerate_loop(result)
        if loop_hit is not None:
            channel, loop_info = loop_hit
            ctx.degenerate_loop_stop = (
                f"Turn {ctx.turn} entered a degenerate repetition loop in {channel} "
                f"({loop_info.kind}: {loop_info.detail}) before hitting "
                f"max_tokens ({max_tokens})."
            )
            ctx.trace_lines.append(f"degenerate_loop={ctx.degenerate_loop_stop}")
            ctx.budget_exhausted = False
            return True
        if not visible_answer:
            reasoning_chars = len(result.reasoning or "")
            ctx.truncated_stop = (
                f"Turn {ctx.turn} hit the max_tokens ceiling ({max_tokens}) with "
                f"{reasoning_chars} characters of reasoning and no answer or tool call."
            )
            ctx.trace_lines.append(f"truncated={ctx.truncated_stop}")
            ctx.budget_exhausted = False
            return True

    ctx.counted_turns += 1
    ctx.recent_turn_signatures.clear()
    if scenario_tools and active_tool_choice != "none" and ctx.unparsed_stop is None:
        unparsed_match = detect_unparsed_tool_call(result.content, scenario_tools)
        if unparsed_match is not None:
            marker = unparsed_match.display_marker(held_out=scenario.held_out)
            if unparsed_match.parser_family == "raw_json":
                ctx.unparsed_stop = (
                    f"Turn {ctx.turn} emitted unparsed {marker} in content "
                    "instead of structured tool_calls."
                )
            else:
                ctx.unparsed_stop = (
                    f"Turn {ctx.turn} emitted unparsed tool-call markup {marker} "
                    "in content instead of structured tool_calls."
                )
            ctx.trace_lines.append(f"unparsed_tool_call={ctx.unparsed_stop}")

    if not visible_answer:
        loop_hit = _detect_turn_degenerate_loop(result)
        if loop_hit is not None:
            if ctx.degenerate_loop_stop is None:
                channel, loop_info = loop_hit
                ctx.degenerate_loop_stop = (
                    f"Turn {ctx.turn} entered a degenerate repetition loop in {channel} "
                    f"({loop_info.kind}: {loop_info.detail}) with no visible answer."
                )
                ctx.trace_lines.append(f"degenerate_loop={ctx.degenerate_loop_stop}")
        else:
            has_reasoning = bool((result.reasoning or "").strip())
            if has_reasoning:
                ctx.answer_in_thinking_turns += 1
                ctx.trace_lines.append(
                    f"answer_in_thinking=Turn {ctx.turn} emitted "
                    f"{len(result.reasoning or '')} characters of reasoning "
                    "with empty visible content."
                )
            if ctx.empty_stop is None:
                ctx.empty_stop = (
                    f"Turn {ctx.turn} ended with no visible answer and no tool call "
                    f"(finish_reason={result.finish_reason or 'unknown'}, "
                    f"{len(result.reasoning or '')} characters of reasoning)."
                )
                if has_reasoning:
                    ctx.empty_stop += (
                        " Answer emitted in reasoning block with empty visible content."
                    )
                ctx.trace_lines.append(f"empty_turn={ctx.empty_stop}")

    if on_scenario_turn is not None:
        await _notify_turn(
            on_scenario_turn,
            scenario,
            ctx.turn,
            {
                "turn": ctx.turn,
                "max_turns": max_turns,
                "latency_ms": round(turn_ms, 1),
                "ttft_ms": round(turn_ttft_ms, 1) if turn_ttft_ms is not None else None,
                "content": "" if scenario.held_out else (result.content or ""),
                "tool_calls": [],
                "tool_results": [],
            },
        )

    # If there are follow-up user messages, inject the next one
    if ctx.follow_ups:
        follow_up = ctx.follow_ups.pop(0)
        ctx.user_phase += 1
        ctx.messages.append({"role": "user", "content": follow_up})
        ctx.trace_lines.append(f"user_follow_up_{ctx.user_phase}={follow_up}")
        return False

    # No more follow-ups → conversation is done
    ctx.state.final_answer = result.content
    ctx.budget_exhausted = False
    return True


async def _handle_tool_calls_turn(
    ctx: _ScenarioLoopState,
    result: ChatCompletionResult,
    *,
    scenario: ScenarioDefinition,
    max_turns: int,
    max_tokens: int,
    turn_ms: float,
    turn_ttft_ms: float | None,
    error_rate: float,
    error_rng: random.Random | None,
    on_scenario_turn: OnScenarioTurn | None,
) -> bool:
    """Dispatch tool calls for one turn; return True if a stop guard ended the loop."""
    from tool_eval_bench.runner import orchestrator as _orchestrator_pkg

    tool_names = [tc.name for tc in result.tool_calls]
    ctx.trace_lines.append(f"tool_calls_requested={', '.join(tool_names)}")
    if len(result.tool_calls) > 1:
        ctx.parallel_tool_turns.append(ctx.turn)

    # A length stop that leaves a call's arguments unparseable cut the
    # call off mid-generation. Running it with the empty arguments the
    # parser falls back to would grade a harness ceiling as wrong
    # arguments, so stop and tag the run as above. The partial calls
    # stay in the state and the trace: an attempted tool still counts
    # for safety checks, it just never runs.
    cut_off = [
        tc.name
        for tc in result.tool_calls
        if result.finish_reason == "length" and not _parses_as_json(tc.arguments_str)
    ]
    if cut_off:
        for tc in result.tool_calls:
            ctx.state.tool_calls.append(
                ToolCallRecord(
                    id=tc.id,
                    name=tc.name,
                    raw_arguments=tc.arguments_str,
                    arguments=_parse_tool_arguments(tc.arguments_str),
                    turn=ctx.turn,
                    user_phase=ctx.user_phase,
                )
            )
            ctx.total_arg_bytes += len(tc.arguments_str.encode("utf-8"))
            ctx.trace_lines.append(f"tool_call={tc.name} {tc.arguments_str}")
        loop_hit = _detect_turn_degenerate_loop(
            result,
            extra_texts=[
                ("tool_arguments", tc.arguments_str)
                for tc in result.tool_calls
                if not _parses_as_json(tc.arguments_str)
            ],
        )
        if loop_hit is not None:
            channel, loop_info = loop_hit
            ctx.degenerate_loop_stop = (
                f"Turn {ctx.turn} entered a degenerate repetition loop in {channel} "
                f"({loop_info.kind}: {loop_info.detail}) before hitting "
                f"max_tokens ({max_tokens}) and cutting off {', '.join(cut_off)}."
            )
            ctx.trace_lines.append(f"degenerate_loop={ctx.degenerate_loop_stop}")
        else:
            reasoning_chars = len(result.reasoning or "")
            ctx.truncated_stop = (
                f"Turn {ctx.turn} hit the max_tokens ceiling ({max_tokens}) with "
                f"{reasoning_chars} characters of reasoning and cut off the arguments "
                f"of {', '.join(cut_off)}."
            )
            ctx.trace_lines.append(f"truncated={ctx.truncated_stop}")
        ctx.budget_exhausted = False
        return True

    turn_results: list[Any] = []
    turn_had_injection = False
    turn_all_injected = True
    for tc in result.tool_calls:
        # Draw before the handler runs: a simulated failure means the
        # request never took effect, so the handler must not commit a
        # side effect. Handlers never touch error_rng, so a seed gives
        # the same injection pattern it always did.
        injected_error = (
            _orchestrator_pkg._draw_injected_error(error_rate, rng=error_rng)
            if error_rate > 0
            else None
        )
        injected = injected_error is not None
        if injected:
            turn_had_injection = True
        else:
            turn_all_injected = False
        parsed_args = _parse_tool_arguments(tc.arguments_str)
        record = ToolCallRecord(
            id=tc.id,
            name=tc.name,
            raw_arguments=tc.arguments_str,
            arguments=parsed_args,
            turn=ctx.turn,
            user_phase=ctx.user_phase,
            injected=injected,
        )
        ctx.state.tool_calls.append(record)
        ctx.total_arg_bytes += len(tc.arguments_str.encode("utf-8"))
        ctx.trace_lines.append(f"tool_call={record.name} {record.raw_arguments}")

        if injected_error is not None:
            mock_result = injected_error
        else:
            try:
                mock_result = scenario.handle_tool_call(ctx.state, record)
            except Exception as handler_exc:
                tb_str = traceback.format_exc()
                logger.error(
                    "Tool handler error in scenario %s (%s):\n%s",
                    scenario.id,
                    record.name,
                    tb_str,
                )
                raise _HarnessExecutionError("Tool handler", handler_exc, tb_str) from handler_exc

        try:
            serialized_result = json.dumps(mock_result, allow_nan=False)
            tool_message = _tool_result_message(tc.id, tc.name, mock_result)
        except Exception as ser_exc:
            tb_str = traceback.format_exc()
            logger.error(
                "Tool result serialization error in scenario %s (%s):\n%s",
                scenario.id,
                record.name,
                tb_str,
            )
            raise _HarnessExecutionError("Tool result serialization", ser_exc, tb_str) from ser_exc

        ctx.state.tool_results.append(
            ToolResultRecord(
                call_id=record.id, name=record.name, result=mock_result, injected=injected
            )
        )
        turn_results.append(mock_result)
        ctx.trace_lines.append(
            f"tool_result={serialized_result}" + (" injected=true" if injected else "")
        )
        ctx.messages.append(tool_message)
        if scenario.checkpoint:
            try:
                diagnostic = scenario.checkpoint(ctx.state, record)
            except Exception as cp_exc:
                tb_str = traceback.format_exc()
                logger.error(
                    "Checkpoint error in scenario %s (%s):\n%s",
                    scenario.id,
                    record.name,
                    tb_str,
                )
                raise _HarnessExecutionError("Checkpoint", cp_exc, tb_str) from cp_exc
            if diagnostic:
                ctx.state_checkpoints.append(diagnostic)
                ctx.state.meta.setdefault("state_checkpoints", []).append(diagnostic)
                ctx.trace_lines.append(f"state_checkpoint={diagnostic}")

    if turn_all_injected and ctx.injected_error_turns < _MAX_INJECTED_ERROR_TURNS:
        ctx.injected_error_turns += 1
    else:
        ctx.counted_turns += 1

    if turn_had_injection:
        ctx.recent_turn_signatures.clear()
    else:
        ctx.recent_turn_signatures.append(_turn_signature(result.tool_calls, turn_results))

    if on_scenario_turn is not None:
        await _notify_turn(
            on_scenario_turn,
            scenario,
            ctx.turn,
            {
                "turn": ctx.turn,
                "max_turns": max_turns,
                "latency_ms": round(turn_ms, 1),
                "ttft_ms": round(turn_ttft_ms, 1) if turn_ttft_ms is not None else None,
                "content": "" if scenario.held_out else (result.content or ""),
                "tool_calls": (
                    []
                    if scenario.held_out
                    else [
                        {
                            "name": tc.name,
                            "arguments": _parse_tool_arguments(tc.arguments_str),
                        }
                        for tc in result.tool_calls
                    ]
                ),
                "tool_results": [] if scenario.held_out else list(turn_results),
            },
        )

    del ctx.recent_turn_signatures[:-_LOOP_GUARD_TURNS]
    if (
        len(ctx.recent_turn_signatures) == _LOOP_GUARD_TURNS
        and len(set(ctx.recent_turn_signatures)) == 1
    ):
        ctx.loop_stop = (
            f"Stopped after {_LOOP_GUARD_TURNS} consecutive turns of identical tool "
            f"calls ({', '.join(tool_names)}) with identical results."
        )
        ctx.trace_lines.append(f"loop_guard={ctx.loop_stop}")
        ctx.budget_exhausted = False
        return True
    return False


def _evaluate_and_check_dependencies(
    ctx: _ScenarioLoopState,
    *,
    scenario: ScenarioDefinition,
    max_turns: int,
    answered: bool,
) -> ScenarioEvaluation:
    """Run the scenario evaluator, empty-restraint guard, dependency checks, and diagnostics."""
    evaluation = scenario.evaluate(ctx.state)
    if not ctx.state.tool_calls and not answered and evaluation.points > 0:
        # Several evaluators read "no forbidden call, no forbidden text" as
        # restraint and award PARTIAL. A model that returned nothing at all
        # showed no restraint, so the guard lives here rather than in each
        # evaluator.
        evaluation = ScenarioEvaluation(
            ScenarioStatus.FAIL,
            0,
            "Returned no answer and made no tool calls.",
            failure_kind=FailureKind.MISSING_STEP,
            diagnostics=evaluation.diagnostics,
            safety_violation=evaluation.safety_violation,
        )
    for producer, consumer in scenario.dependencies:
        # Every declared producer must produce a usable result on an earlier
        # turn before any consumer call runs. An injected or errored producer
        # supplied nothing a later consumer could depend on, and skipping the
        # producer altogether bypasses the dependency.
        attempts = [call for call in ctx.state.tool_calls if call.name == producer]
        sources = [call for call in attempts if _producer_result_observed(ctx.state, call)]
        targets = [call for call in ctx.state.tool_calls if call.name == consumer]
        if targets and any(
            not any(source.turn < target.turn for source in sources) for target in targets
        ):
            batched = any(attempt.turn == target.turn for attempt in attempts for target in targets)
            violation = (
                f"Batched {consumer} with {producer} in the same turn instead of waiting "
                f"for the {producer} result."
                if batched
                else f"Called {consumer} before observing a {producer} result."
            )
            evaluation = ScenarioEvaluation(
                ScenarioStatus.FAIL,
                0,
                violation,
                note=evaluation.note,
                failure_kind=FailureKind.MISSING_STEP,
                diagnostics=evaluation.diagnostics,
                safety_violation=evaluation.safety_violation
                or (violation if _is_mutating_tool(consumer) else None),
            )
            break
    budget_stop = (
        f"Reached the {max_turns}-turn limit before the model finished."
        if ctx.budget_exhausted
        else None
    )
    harness_stop = (
        ctx.degenerate_loop_stop
        or ctx.truncated_stop
        or ctx.loop_stop
        or ctx.unparsed_stop
        or ctx.empty_stop
        or budget_stop
    )
    if harness_stop is not None and evaluation.status != ScenarioStatus.PASS:
        evaluation.note = f"{harness_stop} {evaluation.note or ''}".strip()
    if ctx.answer_in_thinking_turns > 0:
        turn_word = "turn" if ctx.answer_in_thinking_turns == 1 else "turns"
        evaluation.diagnostics["answer_in_thinking"] = (
            "Answer emitted in reasoning block with empty visible content "
            f"({ctx.answer_in_thinking_turns} {turn_word})"
        )
    if ctx.response_format_fallback is not None:
        evaluation.diagnostics["response_format"] = (
            f"endpoint rejected it; fell back {ctx.response_format_fallback}, "
            "schema enforced only by the prompt"
        )
    elif scenario.response_format_override is not None:
        evaluation.diagnostics["response_format"] = (
            "requested; backend enforcement is not independently measured"
            if ctx.response_format_sent
            else "withheld on tool-enabled turns; schema enforced by the prompt"
        )
    return evaluation


def _build_transport_error_result(
    ctx: _ScenarioLoopState,
    result: ChatCompletionResult,
    *,
    model: str,
    scenario: ScenarioDefinition,
) -> ScenarioResult:
    """Classify and record a malformed body or HTTP transport error returned by an adapter."""
    is_server_status = result.transport_error_status is not None and (
        result.transport_error_status >= 500
        or result.transport_error_status in RETRYABLE_STATUS_CODES
    )
    is_infrastructure = (
        result.malformed
        or not ctx.state.tool_calls
        or result.transport_error_is_infrastructure
        or is_server_status
    )
    if result.malformed:
        summary = "Endpoint returned a response body that is not valid JSON."
        ctx.trace_lines.append("transport_error=malformed_response")
    else:
        if result.transport_error_is_infrastructure:
            rejection_stage = "because the serving stack failed before inference"
        elif is_server_status:
            rejection_stage = "because the serving stack returned a server error"
        elif not ctx.state.tool_calls:
            rejection_stage = "before the model produced anything"
        else:
            rejection_stage = "after model-authored tool-call history"
        summary = (
            f"Endpoint rejected the request with HTTP "
            f"{result.transport_error_status} {rejection_stage}: "
            f"{result.content}"
        )
        ctx.trace_lines.append(f"transport_error={result.transport_error_status}")
    return _build_error_result(
        ctx,
        model=model,
        scenario=scenario,
        summary=summary,
        failure_kind=FailureKind.SERVER_ERROR if is_infrastructure else FailureKind.MODEL_CRASH,
    )


async def run_scenario(
    adapter: BackendAdapter,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    scenario: ScenarioDefinition,
    max_turns: int = MAX_TURNS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = 0.0,
    seed: int | None = None,
    reference_date: str | None = None,
    reference_day: str | None = None,
    error_rate: float = 0.0,
    extra_params: dict[str, Any] | None = None,
    context_pressure_messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
    on_scenario_evaluated: OnScenarioEvaluated | None = None,
    on_scenario_turn: OnScenarioTurn | None = None,
    reviewer: TurnReviewer | None = None,
) -> ScenarioResult:
    """Run a single scenario through the multi-turn orchestration loop.

    With a ``reviewer``, every turn in its scope is reviewed before the
    harness acts on it, and the benchmarked model revises a rejected turn.
    """
    if reviewer is not None and scenario.held_out:
        raise ValueError(f"A reviewer must not receive held-out scenario {scenario.id}")
    t0 = time.perf_counter()
    review_stats = ReviewStats() if reviewer is not None else None
    # The reviewer is a conversation of its own, apart from the model's.
    review_conversation_id = uuid.uuid4().hex
    pressure_message_count = len(context_pressure_messages or [])
    # One scenario is one conversation: every turn carries the same id so a
    # gateway that routes or caches per conversation sees it as one.
    conversation_id = uuid.uuid4().hex
    state = ScenarioState()
    state.meta["trace_policy"] = "complete"
    # Meta must match the reference date actually injected into the prompt:
    # _initial_messages falls back to the benchmark default when None.
    state.meta["reference_date"] = reference_date or BENCHMARK_REFERENCE_DATE
    state.meta["reference_day"] = reference_day or BENCHMARK_REFERENCE_DAY
    messages = _initial_messages(
        scenario.user_message,
        reference_date=reference_date,
        reference_day=reference_day,
        context_pressure_messages=context_pressure_messages,
        system_prompt=system_prompt,
    )
    # None → use defaults; [] → explicitly no tools
    scenario_tools = UNIVERSAL_TOOLS if scenario.tools_override is None else scenario.tools_override
    scenario_tool_choice = scenario.tool_choice_override or "auto"
    # Per-scenario budget overrides the global default for deep workflows (e.g.
    # TC-46) or tight caps (e.g. TC-88) without letting a lower override shrink
    # a budget the caller explicitly raised above MAX_TURNS.
    if scenario.max_turns_override is not None:
        if max_turns > MAX_TURNS:
            max_turns = max(max_turns, scenario.max_turns_override)
        else:
            max_turns = scenario.max_turns_override
    available_tool_names = [
        str(tool.get("function", {}).get("name", "?")) for tool in (scenario_tools or [])
    ]
    trace_lines: list[str] = [
        "assistant=starting",
        f"available_tools={', '.join(available_tool_names) if available_tool_names else '(none)'}",
    ]
    if available_tool_names:
        trace_lines.append(f"tool_choice={scenario_tool_choice}")

    # Per-scenario seeded RNG for deterministic error injection.
    # Uses a stable digest offset so each scenario gets a unique
    # but reproducible injection pattern regardless of execution order.
    error_rng: random.Random | None = None
    if seed is not None and error_rate > 0:
        error_rng = random.Random(seed + _scenario_seed_offset(scenario.id))

    seen_call_ids: set[str] = {
        str(tc["id"])
        for msg in messages
        for tc in (msg.get("tool_calls") or [])
        if isinstance(tc, dict) and tc.get("id")
    }
    ctx = _ScenarioLoopState(
        t0=t0,
        state=state,
        messages=messages,
        trace_lines=trace_lines,
        follow_ups=list(scenario.follow_up_messages),
        seen_call_ids=seen_call_ids,
        active_response_format=scenario.response_format_override,
        review_stats=review_stats,
    )

    extra: dict[str, Any] = {}
    if seed is not None:
        extra["seed"] = seed
    if extra_params:
        extra.update(extra_params)
    if seed is not None:
        logger.debug(
            "seed=%d passed to server. Note: seed only controls logit sampling "
            "(greedy at temperature=0 is already deterministic). "
            "Full run-to-run reproducibility also requires stable server state, "
            "no concurrent requests, and consistent KV-cache configuration.",
            seed,
        )

    max_tokens = default_max_tokens(extra)

    try:
        while ctx.counted_turns < max_turns:
            ctx.turn += 1
            # Every turn is streamed. On a streamed turn the read timeout
            # bounds the gap between tokens, so a model that thinks for
            # minutes while still emitting stays alive and a hung endpoint
            # still dies after timeout_seconds of silence. TTFT is read from
            # turn 1 only.
            use_stream = True

            turn_start = time.perf_counter()
            active_tool_choice = scenario_tool_choice
            if (
                any(not call.injected for call in ctx.state.tool_calls)
                and scenario.tool_choice_after_first_call is not None
            ):
                active_tool_choice = scenario.tool_choice_after_first_call

            # Never send response_format simultaneously with active tools.
            # Grammar-constrained backends (llama.cpp, vLLM) either reject
            # response_format + tools or constrain generation to JSON so
            # sequential tool calls (TC-67/TC-69) and --error-rate retries
            # cannot fire on Turn 2+. Tool-free scenarios (TC-64) and
            # tool-disabled turns (tool_choice == "none") still enforce it.
            tools_active_this_turn = bool(scenario_tools) and active_tool_choice != "none"
            response_format = None if tools_active_this_turn else ctx.active_response_format
            turn_tools = (
                None
                if (
                    not scenario_tools
                    or (response_format is not None and not tools_active_this_turn)
                )
                else scenario_tools
            )
            turn_tool_choice = active_tool_choice if turn_tools else None

            result, response_format = await _request_turn_with_format_fallback(
                adapter,
                ctx,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                extra=extra,
                conversation_id=conversation_id,
                turn_tools=turn_tools,
                turn_tool_choice=turn_tool_choice,
                response_format=response_format,
                use_stream=use_stream,
            )
            turn_ms = (time.perf_counter() - turn_start) * 1000
            ctx.turn_latencies.append(turn_ms)

            # A rejected request produced no model output, so there is nothing
            # to grade and follow-ups must not continue with an error string as
            # assistant prose.
            if result.malformed or result.transport_error_status is not None:
                return _build_transport_error_result(ctx, result, model=model, scenario=scenario)

            # Capture TTFT from the first streamed turn
            if ctx.turn == 1 and result.ttft_ms is not None:
                ctx.ttft_ms = result.ttft_ms

            # Accumulate token usage
            if result.prompt_tokens is not None:
                ctx.total_prompt_tokens += result.prompt_tokens
            if result.completion_tokens is not None:
                ctx.total_completion_tokens += result.completion_tokens

            # The first draft's: a revision's TTFT is a second request's.
            turn_ttft_ms = result.ttft_ms
            if (
                reviewer is not None
                and ctx.review_stats is not None
                and not _is_cut_off_tool_call(result)
            ):
                turn_response_format = response_format
                revise_tools = turn_tools
                revise_tool_choice = turn_tool_choice

                async def revise(
                    request: ChatMessage,
                    *,
                    _format: dict[str, Any] | None = turn_response_format,
                    _tools: list[dict[str, Any]] | None = revise_tools,
                    _tool_choice: str | dict[str, Any] | None = revise_tool_choice,
                    _stream: bool = use_stream,
                ) -> ChatCompletionResult:
                    # The same request as the draft, plus the review. The
                    # history itself never keeps the review message.
                    return await adapter.chat_completion(
                        model=model,
                        messages=[*ctx.messages, request],
                        tools=_tools,
                        tool_choice=_tool_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                        api_key=api_key,
                        base_url=base_url,
                        extra_params=extra,
                        stream=_stream,
                        response_format=_format,
                        conversation_id=conversation_id,
                    )

                result, revisions = await _review_turn(
                    reviewer,
                    result,
                    messages=ctx.messages,
                    tools=revise_tools,
                    revise=revise,
                    stats=ctx.review_stats,
                    trace_lines=ctx.trace_lines,
                    turn=ctx.turn,
                    omitted_after_system=pressure_message_count,
                    conversation_id=review_conversation_id,
                    # The choice the wire carries: --backend-kwargs can override it.
                    tool_choice=_wire_tool_choice(
                        extra, active_tool_choice, reviewer.benchmark_wire_format
                    )
                    if revise_tools
                    else None,
                    detect_unparsed=active_tool_choice != "none",
                )
                for revision in revisions:
                    ctx.total_prompt_tokens += token_count(revision.prompt_tokens)
                    ctx.total_completion_tokens += token_count(revision.completion_tokens)
                # A reviewed turn takes as long as the pair needs: draft,
                # reviews, and revisions.
                turn_ms = (time.perf_counter() - turn_start) * 1000
                ctx.turn_latencies[-1] = turn_ms

            if result.tool_calls:
                result = replace(
                    result,
                    tool_calls=_ensure_unique_tool_call_ids(
                        result.tool_calls, ctx.seen_call_ids, ctx.turn
                    ),
                )

            ctx.state.assistant_messages.append(result.content)
            ctx.state.assistant_reasoning.append(result.reasoning or "")
            phase_has_tool_calls = any(
                call.user_phase == ctx.user_phase for call in ctx.state.tool_calls
            )
            preserve_follow_up_reasoning = scenario.preserve_reasoning_across_follow_ups and bool(
                ctx.follow_ups
            )
            ctx.messages.append(
                _assistant_message(
                    result,
                    preserve_reasoning=phase_has_tool_calls or preserve_follow_up_reasoning,
                )
            )
            ctx.trace_lines.append(
                f"assistant_turn_{ctx.turn}={result.content or '[tool_calls_only]'}"
            )
            if result.reasoning:
                ctx.trace_lines.append(f"assistant_reasoning_{ctx.turn}={result.reasoning}")

            if not result.tool_calls:
                should_stop = await _handle_no_tool_turn(
                    ctx,
                    result,
                    scenario=scenario,
                    scenario_tools=scenario_tools,
                    active_tool_choice=active_tool_choice,
                    max_turns=max_turns,
                    max_tokens=max_tokens,
                    turn_ms=turn_ms,
                    turn_ttft_ms=turn_ttft_ms,
                    on_scenario_turn=on_scenario_turn,
                )
                if should_stop:
                    break
                continue

            should_stop = await _handle_tool_calls_turn(
                ctx,
                result,
                scenario=scenario,
                max_turns=max_turns,
                max_tokens=max_tokens,
                turn_ms=turn_ms,
                turn_ttft_ms=turn_ttft_ms,
                error_rate=error_rate,
                error_rng=error_rng,
                on_scenario_turn=on_scenario_turn,
            )
            if should_stop:
                break

        if ctx.budget_exhausted:
            ctx.trace_lines.append(
                f"budget_exhausted=turn limit reached ({ctx.counted_turns}/{max_turns})"
            )

    except _HarnessExecutionError as harness_exc:
        summary = str(harness_exc)
        ctx.trace_lines.append(f"eval_error={summary}\n{harness_exc.tb_str}")
        return _build_error_result(
            ctx,
            model=model,
            scenario=scenario,
            summary=summary,
            failure_kind=FailureKind.EVALUATOR_ERROR,
        )

    except Exception as exc:
        summary = str(exc)
        ctx.trace_lines.append(f"error={summary}")
        return _build_error_result(
            ctx,
            model=model,
            scenario=scenario,
            summary=summary,
            failure_kind=_classify_runtime_error(exc),
        )

    # Ensure we have a final answer
    if not ctx.state.final_answer:
        ctx.state.final_answer = (
            ctx.state.assistant_messages[-1]
            if ctx.state.assistant_messages
            else "Model did not return a final answer."
        )
    # The placeholder above is for the trace and evaluators, not an answer.
    # Any phase counts: a draft followed by an empty follow-up reply is still
    # an answer the evaluator can grade.
    answered = any(_is_visible_answer(message) for message in ctx.state.assistant_messages)
    ctx.trace_lines.append(f"final_answer={ctx.state.final_answer}")

    # Evaluate — wrapped in try/except so evaluator bugs don't crash the
    # entire benchmark run (issue #5).
    try:
        evaluation = _evaluate_and_check_dependencies(
            ctx,
            scenario=scenario,
            max_turns=max_turns,
            answered=answered,
        )
    except Exception as eval_exc:
        tb_str = traceback.format_exc()
        logger.error("Evaluator error in scenario %s:\n%s", scenario.id, tb_str)
        summary = f"Evaluator error: {eval_exc}"
        ctx.trace_lines.append(f"eval_error={summary}\n{tb_str}")
        return _build_error_result(
            ctx,
            model=model,
            scenario=scenario,
            summary=summary,
            failure_kind=FailureKind.EVALUATOR_ERROR,
        )

    calls_made = _format_calls_made(ctx.state.tool_calls)
    elapsed = time.perf_counter() - ctx.t0

    scenario_result = ScenarioResult(
        scenario_id=scenario.id,
        status=evaluation.status,
        points=evaluation.points,
        summary=evaluation.summary,
        note=evaluation.note,
        diagnostics=evaluation.diagnostics,
        safety_violation=evaluation.safety_violation,
        raw_log=_format_trace(
            model,
            scenario,
            {
                "status": evaluation.status.value,
                "summary": evaluation.summary,
                "note": evaluation.note,
            },
            ctx.trace_lines,
        ),
        tool_calls_made=calls_made,
        expected_behavior=scenario.description,
        duration_seconds=elapsed,
        ttft_ms=ctx.ttft_ms,
        turn_count=len(ctx.turn_latencies),
        turn_latencies_ms=ctx.turn_latencies,
        prompt_tokens=ctx.total_prompt_tokens,
        completion_tokens=ctx.total_completion_tokens,
        tool_call_arg_bytes=ctx.total_arg_bytes,
        parallel_tool_turns=ctx.parallel_tool_turns,
        state_checkpoints=ctx.state_checkpoints,
        failure_kind=(
            FailureKind.DEGENERATE_LOOP
            if ctx.degenerate_loop_stop is not None and evaluation.status != ScenarioStatus.PASS
            else FailureKind.REASONING_TRUNCATED
            if ctx.truncated_stop is not None and evaluation.status != ScenarioStatus.PASS
            else FailureKind.REPEATED_CALL_LOOP
            if ctx.loop_stop is not None and evaluation.status != ScenarioStatus.PASS
            else FailureKind.UNPARSED_TOOL_CALL
            if ctx.unparsed_stop is not None and evaluation.status != ScenarioStatus.PASS
            else FailureKind.EMPTY_TURN
            if ctx.empty_stop is not None and evaluation.status != ScenarioStatus.PASS
            else FailureKind.BUDGET_EXCEEDED
            if ctx.budget_exhausted and evaluation.status != ScenarioStatus.PASS
            else _classify_evaluation_failure(ctx.state, evaluation)
        ),
        # A turn-budget run-out is a distinct failure reason, not an evaluator
        # verdict: the model never reached a final answer / exhausted follow-ups.
        turn_budget_exceeded=ctx.budget_exhausted,
        review=_review_payload(ctx.review_stats),
        answer_in_thinking_turns=ctx.answer_in_thinking_turns,
    )
    if on_scenario_evaluated is not None:
        await on_scenario_evaluated(scenario, ctx.state, scenario_result)
    return scenario_result


# ---------------------------------------------------------------------------
# Run all scenarios for one model
# ---------------------------------------------------------------------------


class _CallbackError(Exception):
    """A caller's progress callback raised; ``original`` is what it raised."""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


def _tag_errors(callback: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
    """Wrap a caller's callback so what it raises arrives as a ``_CallbackError``."""

    async def call(*args: Any) -> None:
        try:
            await callback(*args)
        except Exception as exc:
            raise _CallbackError(exc) from exc

    return call


async def _warm_pressure_cache(
    adapter: BackendAdapter,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float,
    temperature: float,
    reference_date: str | None,
    reference_day: str | None,
    extra_params: dict[str, Any] | None,
    context_pressure_messages: list[ChatMessage],
    system_prompt: str | None,
) -> None:
    """Prefill the shared filler once, unscored, so no scenario pays the cold prefill.

    Sends the prefix every default-toolset scenario starts with: the same tools,
    system prompt, and filler. Best effort: a failure only means the first
    scenario prefills cold, as it would without the warm-up.
    """
    # ponytail: warms the default toolset only; scenarios with tools_override
    # still prefill cold. Warm each distinct toolset if that cost shows up.
    try:
        result = await adapter.chat_completion(
            model=model,
            messages=_initial_messages(
                "Reply with the single word OK.",
                reference_date=reference_date,
                reference_day=reference_day,
                context_pressure_messages=context_pressure_messages,
                system_prompt=system_prompt,
            ),
            tools=UNIVERSAL_TOOLS,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=1,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            base_url=base_url,
            extra_params=extra_params,
            stream=False,
        )
    except Exception as exc:  # noqa: BLE001 - the warm-up must never fail the run
        logger.warning("Prefix-cache warm-up failed (%s); the first scenario prefills cold", exc)
        return
    if result.transport_error_status is not None:
        logger.warning(
            "Prefix-cache warm-up got HTTP %d; the first scenario prefills cold",
            result.transport_error_status,
        )


async def run_all_scenarios(
    adapter: BackendAdapter,
    *,
    model: str,
    base_url: str,
    api_key: str | None = None,
    scenarios: list[ScenarioDefinition],
    max_turns: int = MAX_TURNS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = 0.0,
    seed: int | None = None,
    reference_date: str | None = None,
    reference_day: str | None = None,
    on_scenario_start: OnScenarioStart | None = None,
    on_scenario_result: OnScenarioResult | None = None,
    concurrency: int = 1,
    error_rate: float = 0.0,
    alpha: float = 0.7,
    extra_params: dict[str, Any] | None = None,
    context_pressure_messages: list[ChatMessage] | None = None,
    weight_by_difficulty: bool = False,
    system_prompt: str | None = None,
    on_scenario_evaluated: OnScenarioEvaluated | None = None,
    on_scenario_turn: OnScenarioTurn | None = None,
    probe_engine: bool = True,
    reviewer: TurnReviewer | None = None,
) -> ModelScoreSummary:
    """Run every scenario and produce an aggregate score summary.

    Args:
        concurrency: max parallel scenario runs. 1 = sequential (default).
        system_prompt: replaces the built-in system prompt for every scenario.
        probe_engine: whether to run the diagnostic required-tool probe.
        reviewer: reviews each turn in its scope before the harness acts on it.
    """
    from tool_eval_bench.runner import orchestrator as _orchestrator_pkg

    target_scenarios = scenarios
    total = len(target_scenarios)
    if reviewer is not None:
        held_out = [s.id for s in target_scenarios if s.held_out]
        if held_out:
            raise ValueError(
                "A reviewer must not receive held-out scenarios: " + ", ".join(held_out)
            )

    # The probe describes its own request, not whether a scenario can run.
    forced_tool_choice = [s for s in target_scenarios if s.tool_choice_override == "required"]
    probe: ToolChoiceProbe | None = None
    if forced_tool_choice and probe_engine:
        probe = await probe_tool_choice_required(
            adapter,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_params=extra_params,
        )
        if not probe.enforced:
            logger.warning(
                "tool_choice='required' probe: %s; running selected scenarios regardless",
                probe.detail,
            )
    forced_ids = {s.id for s in forced_tool_choice}

    if context_pressure_messages:
        await _warm_pressure_cache(
            adapter,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            reference_date=reference_date,
            reference_day=reference_day,
            extra_params=extra_params,
            context_pressure_messages=context_pressure_messages,
            system_prompt=system_prompt,
        )

    def _attach_probe(result: ScenarioResult) -> ScenarioResult:
        # Preserve the scenario verdict; a separate probe cannot establish
        # whether the model or endpoint caused a missing call or argument.
        if result.scenario_id in forced_ids:
            result.diagnostics["tool_choice"] = (
                probe.detail if probe is not None else "unverified; probe disabled"
            )
        return result

    if concurrency <= 1:
        # Sequential path — original behavior, preserves ordering guarantees
        turn_kwargs: dict[str, Any] = (
            {"on_scenario_turn": on_scenario_turn} if on_scenario_turn is not None else {}
        )
        if reviewer is not None:
            turn_kwargs["reviewer"] = reviewer
        results: list[ScenarioResult] = []
        for idx, scenario in enumerate(target_scenarios):
            if on_scenario_start:
                await on_scenario_start(scenario, idx, total)
            result = await _orchestrator_pkg.run_scenario(
                adapter,
                model=model,
                base_url=base_url,
                api_key=api_key,
                scenario=scenario,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                seed=seed,
                reference_date=reference_date,
                reference_day=reference_day,
                error_rate=error_rate,
                extra_params=extra_params,
                context_pressure_messages=context_pressure_messages,
                system_prompt=system_prompt,
                on_scenario_evaluated=on_scenario_evaluated,
                **turn_kwargs,
            )
            results.append(_attach_probe(result))
            if on_scenario_result:
                await on_scenario_result(scenario, result, idx, total)
        return score_results(
            results, target_scenarios, alpha=alpha, weight_by_difficulty=weight_by_difficulty
        )

    # Parallel path with semaphore-bounded concurrency.
    # `concurrency <= 1` returned above, so every run reaching here is parallel.
    logger.warning(
        "Running %d scenarios concurrently (--parallel %d). "
        "Server saturation under load can cause timeouts that are recorded as FAIL "
        "even when the model reasoned correctly. "
        "Use --parallel 1 for reproducible quality scores.",
        total,
        concurrency,
    )

    sem = asyncio.Semaphore(concurrency)
    ordered_results: list[ScenarioResult | None] = [None] * total
    progress_counter = 0  # Shared counter for progress reporting
    progress_lock = asyncio.Lock()

    # A raising callback is the caller's failure, not the model's: tag it so
    # the gather below can tell it from a scenario crash.
    start_cb = _tag_errors(on_scenario_start) if on_scenario_start else None
    result_cb = _tag_errors(on_scenario_result) if on_scenario_result else None
    evaluated_cb = _tag_errors(on_scenario_evaluated) if on_scenario_evaluated else None
    # Not tagged like the callbacks above: run_scenario already isolates the
    # turn observer, so its failures never abort or fail a scenario.
    parallel_turn_kwargs: dict[str, Any] = (
        {"on_scenario_turn": on_scenario_turn} if on_scenario_turn is not None else {}
    )
    if reviewer is not None:
        parallel_turn_kwargs["reviewer"] = reviewer

    async def _run_one(idx: int, scenario: ScenarioDefinition) -> None:
        nonlocal progress_counter
        async with sem:
            if start_cb:
                await start_cb(scenario, idx, total)
            result = await _orchestrator_pkg.run_scenario(
                adapter,
                model=model,
                base_url=base_url,
                api_key=api_key,
                scenario=scenario,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                seed=seed,
                reference_date=reference_date,
                reference_day=reference_day,
                error_rate=error_rate,
                extra_params=extra_params,
                context_pressure_messages=context_pressure_messages,
                system_prompt=system_prompt,
                on_scenario_evaluated=evaluated_cb,
                **parallel_turn_kwargs,
            )
            ordered_results[idx] = _attach_probe(result)
            if result_cb:
                async with progress_lock:
                    await result_cb(scenario, result, progress_counter, total)
                    progress_counter += 1

    tasks = [_run_one(idx, sc) for idx, sc in enumerate(target_scenarios)]
    gather_results = await asyncio.gather(*tasks, return_exceptions=True)
    callback_error: _CallbackError | None = None
    for i, exc in enumerate(gather_results):
        if isinstance(exc, BaseException) and not isinstance(exc, Exception):
            raise exc
        if isinstance(exc, _CallbackError):
            # The caller's callback failed, not the model. Re-raised below,
            # after every scenario finished, as the sequential path raises it.
            callback_error = callback_error or exc
        elif isinstance(exc, BaseException) and ordered_results[i] is None:
            sc = target_scenarios[i]
            logger.error("Scenario %s crashed: %s", sc.id, exc)
            ordered_results[i] = ScenarioResult(
                scenario_id=sc.id,
                status=ScenarioStatus.FAIL,
                points=0,
                summary=f"Unhandled error: {exc}",
                tool_call_arg_bytes=0,
                failure_kind=_classify_runtime_error(exc),
            )
    if callback_error is not None:
        raise callback_error.original

    final_results = [r for r in ordered_results if r is not None]
    return score_results(
        final_results, target_scenarios, alpha=alpha, weight_by_difficulty=weight_by_difficulty
    )
