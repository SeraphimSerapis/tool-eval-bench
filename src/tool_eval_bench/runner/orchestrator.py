"""Multi-turn scenario orchestrator.

Runs a single scenario against a model through the multi-turn tool-call loop:
  model → tool_calls → mock results → model → … (up to max_turns)

Captures full traces for auditability.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult
from tool_eval_bench.domain.scenarios import (
    CATEGORY_LABELS,
    SAFETY_CATEGORIES,
    SAFETY_GATE_THRESHOLD,
    Category,
    CategoryScore,
    ModelScoreSummary,
    OnScenarioResult,
    OnScenarioStart,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
    compute_deployability,
    rating_for_score,
)
from tool_eval_bench.domain.tools import (
    BENCHMARK_REFERENCE_DATE,
    BENCHMARK_REFERENCE_DAY,
    SYSTEM_PROMPT,
    UNIVERSAL_TOOLS,
)

logger = logging.getLogger(__name__)

MAX_TURNS = 12
DEFAULT_TIMEOUT_SECONDS = 60.0

# Error injection types (Claw-Eval methodology: 429/500/timeout)
_INJECTED_ERRORS = [
    {"error": "Rate limit exceeded. Please retry after 2 seconds.", "status": 429},
    {"error": "Internal server error. The service is temporarily unavailable.", "status": 500},
    {"error": "Request timed out. The service did not respond in time.", "status": 503},
]


def _maybe_inject_error(result: Any, error_rate: float) -> Any:
    """Randomly replace a mock tool response with a simulated error.

    Returns the original result unchanged if no error is injected.
    The error distribution follows Claw-Eval: ~33% each of 429, 500, timeout.
    """
    if error_rate <= 0 or random.random() >= error_rate:
        return result
    return random.choice(_INJECTED_ERRORS)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _initial_messages(
    user_message: str,
    *,
    reference_date: str | None = None,
    reference_day: str | None = None,
    context_pressure_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ref_date = reference_date or BENCHMARK_REFERENCE_DATE
    ref_day = reference_day or BENCHMARK_REFERENCE_DAY
    msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"Benchmark context: today is {ref_date} ({ref_day}). "
                "Use this date for any relative time request."
            ),
        },
    ]
    # Inject context pressure filler turns (if any) between system prompt
    # and the real user message to simulate a busy conversation history.
    if context_pressure_messages:
        msgs.extend(context_pressure_messages)
    msgs.append({"role": "user", "content": user_message})
    return msgs


def _assistant_message(result: ChatCompletionResult) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments_str},
            }
            for tc in result.tool_calls
        ]
    return msg


def _tool_result_message(call_id: str, result: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result) if not isinstance(result, str) else result,
    }


# ---------------------------------------------------------------------------
# Trace formatting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core: run one scenario for one model
# ---------------------------------------------------------------------------

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
    context_pressure_messages: list[dict[str, Any]] | None = None,
) -> ScenarioResult:
    """Run a single scenario through the multi-turn orchestration loop."""
    t0 = time.monotonic()
    state = ScenarioState()
    messages = _initial_messages(
        scenario.user_message,
        reference_date=reference_date,
        reference_day=reference_day,
        context_pressure_messages=context_pressure_messages,
    )
    trace_lines: list[str] = ["assistant=starting"]

    ttft_ms: float | None = None
    turn_latencies: list[float] = []
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # Build the queue of user messages: initial + any follow-ups
    follow_ups = list(scenario.follow_up_messages)  # copy so we can pop
    user_phase = 0  # which user message we're processing

    extra: dict[str, Any] = {}
    if seed is not None:
        extra["seed"] = seed
    if extra_params:
        extra.update(extra_params)
    extra = extra or None
    if seed is not None:
        logger.debug(
            "seed=%d passed to server. Note: seed only controls logit sampling "
            "(greedy at temperature=0 is already deterministic). "
            "Full run-to-run reproducibility also requires stable server state, "
            "no concurrent requests, and consistent KV-cache configuration.",
            seed,
        )

    try:
        for turn in range(1, max_turns + 1):
            # Stream the first turn to measure TTFT accurately
            use_stream = turn == 1

            turn_start = time.perf_counter()
            scenario_tools = scenario.tools_override or UNIVERSAL_TOOLS
            scenario_tool_choice = scenario.tool_choice_override or "auto"
            result = await adapter.chat_completion(
                model=model,
                messages=messages,
                tools=scenario_tools,
                tool_choice=scenario_tool_choice,
                temperature=temperature,
                max_tokens=4096,
                timeout_seconds=timeout_seconds,
                api_key=api_key,
                base_url=base_url,
                extra_params=extra,
                stream=use_stream,
            )
            turn_ms = (time.perf_counter() - turn_start) * 1000
            turn_latencies.append(turn_ms)

            # Capture TTFT from the first streamed turn
            if turn == 1 and result.ttft_ms is not None:
                ttft_ms = result.ttft_ms

            # Accumulate token usage
            if result.prompt_tokens is not None:
                total_prompt_tokens += result.prompt_tokens
            if result.completion_tokens is not None:
                total_completion_tokens += result.completion_tokens

            state.assistant_messages.append(result.content)
            messages.append(_assistant_message(result))
            trace_lines.append(
                f"assistant_turn_{turn}={result.content or '[tool_calls_only]'}"
            )
            if result.reasoning:
                trace_lines.append(f"assistant_reasoning_{turn}={result.reasoning}")

            # No tool calls → model finished this conversational phase
            if not result.tool_calls:
                # If there are follow-up user messages, inject the next one
                if follow_ups:
                    follow_up = follow_ups.pop(0)
                    user_phase += 1
                    messages.append({"role": "user", "content": follow_up})
                    trace_lines.append(f"user_follow_up_{user_phase}={follow_up}")
                    # Continue the loop — model will respond to the follow-up
                    continue

                # No more follow-ups → conversation is done
                state.final_answer = result.content
                break

            tool_names = [tc.name for tc in result.tool_calls]
            trace_lines.append(f"tool_calls_requested={', '.join(tool_names)}")

            for tc in result.tool_calls:
                record = ToolCallRecord(
                    id=tc.id,
                    name=tc.name,
                    raw_arguments=tc.arguments_str,
                    arguments=tc.arguments,
                    turn=turn,
                )
                state.tool_calls.append(record)
                trace_lines.append(f"tool_call={record.name} {record.raw_arguments}")

                # Call the scenario's mock handler
                mock_result = scenario.handle_tool_call(state, record)

                # Error injection: randomly replace with simulated failure
                if error_rate > 0:
                    mock_result = _maybe_inject_error(mock_result, error_rate)

                state.tool_results.append(
                    ToolResultRecord(call_id=record.id, name=record.name, result=mock_result)
                )
                trace_lines.append(f"tool_result={json.dumps(mock_result)}")
                messages.append(_tool_result_message(tc.id, mock_result))

    except Exception as exc:
        elapsed = time.monotonic() - t0
        summary = str(exc)
        trace_lines.append(f"error={summary}")
        return ScenarioResult(
            scenario_id=scenario.id,
            status=ScenarioStatus.FAIL,
            points=0,
            summary=summary,
            raw_log=_format_trace(
                model, scenario, {"status": "fail", "summary": summary}, trace_lines
            ),
            duration_seconds=elapsed,
            ttft_ms=ttft_ms,
            turn_count=len(turn_latencies),
            turn_latencies_ms=turn_latencies,
        )

    # Ensure we have a final answer
    if not state.final_answer:
        state.final_answer = (
            state.assistant_messages[-1] if state.assistant_messages else "Model did not return a final answer."
        )
    trace_lines.append(f"final_answer={state.final_answer}")

    # Evaluate
    evaluation = scenario.evaluate(state)

    # Build diagnostic summary of what tools were actually called
    calls_made = [
        f"{tc.name}({', '.join(f'{k}={v}' for k, v in tc.arguments.items())})"
        for tc in state.tool_calls
    ]
    expected = scenario.description

    elapsed = time.monotonic() - t0

    return ScenarioResult(
        scenario_id=scenario.id,
        status=evaluation.status,
        points=evaluation.points,
        summary=evaluation.summary,
        note=evaluation.note,
        raw_log=_format_trace(
            model,
            scenario,
            {
                "status": evaluation.status.value,
                "summary": evaluation.summary,
                "note": evaluation.note,
            },
            trace_lines,
        ),
        tool_calls_made=calls_made,
        expected_behavior=expected,
        duration_seconds=elapsed,
        ttft_ms=ttft_ms,
        turn_count=len(turn_latencies),
        turn_latencies_ms=turn_latencies,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )


# ---------------------------------------------------------------------------
# Run all scenarios for one model
# ---------------------------------------------------------------------------

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
    context_pressure_messages: list[dict[str, Any]] | None = None,
) -> ModelScoreSummary:
    """Run every scenario and produce an aggregate score summary.

    Args:
        concurrency: max parallel scenario runs. 1 = sequential (default).
    """
    target_scenarios = scenarios
    total = len(target_scenarios)

    if concurrency <= 1:
        # Sequential path — original behavior, preserves ordering guarantees
        results: list[ScenarioResult] = []
        for idx, scenario in enumerate(target_scenarios):
            if on_scenario_start:
                await on_scenario_start(scenario, idx, total)
            result = await run_scenario(
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
            )
            results.append(result)
            if on_scenario_result:
                await on_scenario_result(scenario, result, idx, total)
        return score_results(results, target_scenarios, alpha=alpha)

    # Parallel path with semaphore-bounded concurrency
    import asyncio

    if concurrency > 1:
        logger.warning(
            "Running %d scenarios concurrently (--parallel %d). "
            "Server saturation under load can cause timeouts that are recorded as FAIL "
            "even when the model reasoned correctly. "
            "Use --parallel 1 for reproducible quality scores.",
            total, concurrency,
        )

    sem = asyncio.Semaphore(concurrency)
    ordered_results: list[ScenarioResult | None] = [None] * total
    progress_counter = 0  # Shared counter for progress reporting
    progress_lock = asyncio.Lock()

    async def _run_one(idx: int, scenario: ScenarioDefinition) -> None:
        nonlocal progress_counter
        async with sem:
            result = await run_scenario(
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
            )
            ordered_results[idx] = result
            if on_scenario_result:
                async with progress_lock:
                    await on_scenario_result(scenario, result, progress_counter, total)
                    progress_counter += 1

    tasks = [_run_one(idx, sc) for idx, sc in enumerate(target_scenarios)]
    await asyncio.gather(*tasks)

    final_results = [r for r in ordered_results if r is not None]
    return score_results(final_results, target_scenarios, alpha=alpha)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(
    results: list[ScenarioResult],
    scenarios: list[ScenarioDefinition],
    alpha: float = 0.7,
) -> ModelScoreSummary:
    """Compute category and aggregate scores from scenario results.

    Args:
        alpha: Quality/speed weight for deployability (0.0–1.0).
               Higher values weight quality more. Default: 0.7.
    """
    all_scenarios = scenarios
    scenario_map = {s.id: s for s in all_scenarios}

    category_scores: list[CategoryScore] = []
    # Count scenarios per category to handle extended packs
    cat_scenario_counts: dict[Category, int] = {}
    for s in all_scenarios:
        cat_scenario_counts[s.category] = cat_scenario_counts.get(s.category, 0) + 1

    for cat in Category:
        if cat not in cat_scenario_counts:
            continue  # skip categories with no scenarios in this run
        earned = sum(
            r.points
            for r in results
            if r.scenario_id in scenario_map
            and scenario_map[r.scenario_id].category == cat
        )
        cat_max = cat_scenario_counts[cat] * 2  # 2 points per scenario
        percent = round((earned / cat_max) * 100) if cat_max > 0 else 0
        category_scores.append(
            CategoryScore(
                category=cat,
                label=CATEGORY_LABELS[cat],
                earned=earned,
                max_points=cat_max,
                percent=percent,
            )
        )

    total_points = sum(r.points for r in results)
    max_points = len(all_scenarios) * 2

    final_score = round(
        (total_points / max_points) * 100
    ) if max_points > 0 else 0

    # Collect safety warnings for failed safety-category scenarios
    safety_warnings: list[str] = []
    for r in results:
        sc = scenario_map.get(r.scenario_id)
        if sc and sc.category in SAFETY_CATEGORIES and r.status == ScenarioStatus.FAIL:
            safety_warnings.append(f"{r.scenario_id} ({sc.title}): {r.summary}")

    # Identify worst-performing category
    worst_cat: str | None = None
    worst_pct: int | None = None
    if category_scores:
        worst = min(category_scores, key=lambda cs: cs.percent)
        worst_cat = f"{worst.category.value} {worst.label} ({worst.percent}%)"
        worst_pct = worst.percent

    # Safety gate: cap the rating if any safety category scores below threshold
    safety_capped = any(
        cs.category in SAFETY_CATEGORIES and cs.percent < SAFETY_GATE_THRESHOLD
        for cs in category_scores
    )

    # Deployability: combine quality with responsiveness from turn latencies
    all_turn_latencies = []
    for r in results:
        all_turn_latencies.extend(r.turn_latencies_ms)
    median_turn_ms: float | None = None
    if all_turn_latencies:
        sorted_lats = sorted(all_turn_latencies)
        mid = len(sorted_lats) // 2
        if len(sorted_lats) % 2 == 0:
            median_turn_ms = (sorted_lats[mid - 1] + sorted_lats[mid]) / 2
        else:
            median_turn_ms = sorted_lats[mid]

    deploy, resp, med = compute_deployability(final_score, median_turn_ms, alpha)

    # Token usage aggregate
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    token_eff: float | None = None
    if total_tokens > 0:
        token_eff = round(total_points / (total_tokens / 1000), 2)

    return ModelScoreSummary(
        scenario_results=results,
        category_scores=category_scores,
        final_score=final_score,
        total_points=total_points,
        max_points=max_points,
        rating=rating_for_score(final_score, safety_capped=safety_capped),
        safety_warnings=safety_warnings,
        worst_category=worst_cat,
        worst_category_percent=worst_pct,
        median_turn_ms=med,
        responsiveness=resp,
        deployability=deploy,
        alpha=alpha,
        total_tokens=total_tokens,
        token_efficiency=token_eff,
    )
