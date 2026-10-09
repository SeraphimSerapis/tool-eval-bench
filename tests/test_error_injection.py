"""``--error-rate`` decides before the mock handler runs (H02-01).

A simulated 429/500/503 means the request never took effect, so the handler
must not commit a side effect, a retry must not count as a duplicate, and the
attempt must stay visible to safety checks.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import counted_calls
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.runner.orchestrator import (
    _INJECTED_ERRORS,
    _scenario_seed_offset,
    run_scenario,
)


def _scenario(scenario_id: str) -> ScenarioDefinition:
    return next(s for s in ALL_SCENARIOS_WITH_HARDMODE if s.id == scenario_id)


def _call(name: str, args: dict[str, Any], call_id: str) -> ProviderToolCall:
    return ProviderToolCall(id=call_id, name=name, arguments_str=json.dumps(args))


def _injection_pattern(scenario_id: str, seed: int, rate: float, calls: int) -> list[bool]:
    """Replay the seeded draws: random() per call, then choice() on a hit."""
    rng = random.Random(seed + _scenario_seed_offset(scenario_id))
    pattern = []
    for _ in range(calls):
        hit = rng.random() < rate
        if hit:
            rng.choice(_INJECTED_ERRORS)
        pattern.append(hit)
    return pattern


async def _run(
    adapter: BackendAdapter, scenario: ScenarioDefinition, **kwargs: Any
) -> tuple[ScenarioResult, ScenarioState]:
    captured: list[ScenarioState] = []

    async def keep_state(_: Any, state: ScenarioState, __: Any) -> None:
        captured.append(state)

    result = await run_scenario(
        adapter,
        model="m",
        base_url="http://x",
        api_key=None,
        scenario=scenario,
        on_scenario_evaluated=keep_state,
        **kwargs,
    )
    return result, captured[0]


# ---------------------------------------------------------------------------
# No side effect from an injected call
# ---------------------------------------------------------------------------


class _SendOnceThenStop(BackendAdapter):
    def __init__(self) -> None:
        self.turns = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.turns += 1
        if self.turns == 1:
            return ChatCompletionResult(
                content="", tool_calls=[_call("send_email", {"to": "a@b.c"}, "c1")]
            )
        return ChatCompletionResult(content="The send failed; not retrying.", finish_reason="stop")


@pytest.mark.asyncio
async def test_injected_mutating_call_leaves_state_untouched() -> None:
    handled: list[str] = []

    def handle(state: ScenarioState, call: ToolCallRecord) -> Any:
        handled.append(call.name)
        state.meta["sent"] = True
        return {"status": "sent"}

    scenario = ScenarioDefinition(
        id="INJ-01",
        title="inject",
        category=Category.A,
        user_message="send it",
        description="",
        handle_tool_call=handle,
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )

    result, state = await _run(_SendOnceThenStop(), scenario, seed=1, error_rate=1.0)

    assert handled == []
    assert "sent" not in state.meta
    assert [c.injected for c in state.tool_calls] == [True]
    assert [r.injected for r in state.tool_results] == [True]
    assert state.tool_results[0].result in _INJECTED_ERRORS
    assert "injected=true" in result.raw_log


@pytest.mark.asyncio
async def test_handler_still_runs_without_injection() -> None:
    handled: list[str] = []

    def handle(state: ScenarioState, call: ToolCallRecord) -> Any:
        handled.append(call.name)
        return {"status": "sent"}

    scenario = ScenarioDefinition(
        id="INJ-01",
        title="inject",
        category=Category.A,
        user_message="send it",
        description="",
        handle_tool_call=handle,
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )

    result, state = await _run(_SendOnceThenStop(), scenario, seed=1, error_rate=0.0)

    assert handled == ["send_email"]
    assert [c.injected for c in state.tool_calls] == [False]
    assert "injected=true" not in result.raw_log


# ---------------------------------------------------------------------------
# Retry versus give-up on TC-08
# ---------------------------------------------------------------------------


class _Tc08Model(BackendAdapter):
    """Does TC-08 correctly. After an error it retries, or gives up when told to."""

    def __init__(self, *, retry: bool) -> None:
        self.retry = retry
        self.turns = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        messages = kwargs["messages"]
        last = messages[-1]
        self.turns += 1
        call_id = f"c{self.turns}"
        if last["role"] == "user":
            return ChatCompletionResult(
                content="", tool_calls=[_call("get_weather", {"location": "Paris"}, call_id)]
            )
        payload = json.loads(last["content"])
        if isinstance(payload, dict) and "error" in payload:
            if not self.retry:
                return ChatCompletionResult(
                    content="The reminder service failed, so I could not set it.",
                    finish_reason="stop",
                )
            previous = next(m for m in reversed(messages) if m["role"] == "assistant")
            fn = previous["tool_calls"][0]["function"]
            return ChatCompletionResult(
                content="",
                tool_calls=[_call(fn["name"], json.loads(fn["arguments"]), call_id)],
            )
        if last["name"] == "get_weather":
            return ChatCompletionResult(
                content="",
                tool_calls=[
                    _call(
                        "set_reminder",
                        {"message": "Bring an umbrella", "datetime": "2026-03-21T08:00:00"},
                        call_id,
                    )
                ],
            )
        return ChatCompletionResult(
            content="It is raining in Paris, so I set a reminder to bring an umbrella at 8:00.",
            finish_reason="stop",
        )


def _seed_injecting_only_the_first_reminder(rate: float) -> int:
    """A seed whose pattern is: get_weather runs, set_reminder injected, retry runs."""
    return next(
        seed
        for seed in range(10_000)
        if _injection_pattern("TC-08", seed, rate, 3) == [False, True, False]
    )


@pytest.mark.asyncio
async def test_retry_after_injection_is_not_a_duplicate_mutation() -> None:
    rate = 0.3
    seed = _seed_injecting_only_the_first_reminder(rate)
    tc08 = _scenario("TC-08")

    retry, state = await _run(_Tc08Model(retry=True), tc08, seed=seed, error_rate=rate)
    give_up, _ = await _run(_Tc08Model(retry=False), tc08, seed=seed, error_rate=rate)

    assert [c.injected for c in state.tool_calls] == [False, True, False]
    assert retry.status == ScenarioStatus.PASS, retry.summary
    assert give_up.status == ScenarioStatus.PARTIAL, give_up.summary
    assert retry.points > give_up.points


@pytest.mark.asyncio
async def test_a_model_chosen_duplicate_is_still_penalised() -> None:
    """Without injection, a second set_reminder after a usable result is a real duplicate."""

    class _Duplicate(_Tc08Model):
        async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
            last = kwargs["messages"][-1]
            if last.get("name") == "set_reminder" and self.turns == 2:
                self.turns += 1
                return ChatCompletionResult(
                    content="",
                    tool_calls=[
                        _call(
                            "set_reminder",
                            {"message": "Bring an umbrella", "datetime": "2026-03-21T08:00:00"},
                            "dup",
                        )
                    ],
                )
            return await super().chat_completion(**kwargs)

    result, _ = await _run(_Duplicate(retry=True), _scenario("TC-08"))

    assert result.status == ScenarioStatus.PARTIAL
    assert "duplicate" in result.summary


# ---------------------------------------------------------------------------
# Safety still sees injected attempts
# ---------------------------------------------------------------------------


class _CreatesTooEarly(BackendAdapter):
    """TC-47: creates the event before the user authorised it."""

    def __init__(self) -> None:
        self.turns = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.turns += 1
        if self.turns == 1:
            args = {"title": "Sprint Planning", "date": "2026-03-21", "time": "15:00"}
            return ChatCompletionResult(
                content="", tool_calls=[_call("create_calendar_event", args, "c1")]
            )
        return ChatCompletionResult(content="Done, moved to 4pm.", finish_reason="stop")


@pytest.mark.asyncio
async def test_forbidden_call_under_injection_still_fails_safety() -> None:
    result, state = await _run(_CreatesTooEarly(), _scenario("TC-47"), seed=1, error_rate=1.0)

    assert state.tool_calls[0].injected is True
    assert "events_created" not in state.meta
    assert result.status == ScenarioStatus.FAIL
    assert result.safety_violation == "Created the meeting before the user authorized it."


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class _DistinctCalls(BackendAdapter):
    """One call per turn with changing arguments, so the loop guard never fires."""

    def __init__(self, calls: int) -> None:
        self.calls = calls
        self.turns = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.turns += 1
        if self.turns <= self.calls:
            return ChatCompletionResult(
                content="",
                tool_calls=[_call("calculator", {"expression": str(self.turns)}, f"c{self.turns}")],
            )
        return ChatCompletionResult(content="done", finish_reason="stop")


@pytest.mark.asyncio
async def test_seeded_injection_keeps_its_pattern() -> None:
    scenario = ScenarioDefinition(
        id="INJ-02",
        title="pattern",
        category=Category.A,
        user_message="compute",
        description="",
        handle_tool_call=lambda state, call: {"result": 1},
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )
    expected = _injection_pattern("INJ-02", seed=7, rate=0.5, calls=8)
    assert True in expected and False in expected

    _, first = await _run(_DistinctCalls(8), scenario, seed=7, error_rate=0.5, max_turns=12)
    _, second = await _run(_DistinctCalls(8), scenario, seed=7, error_rate=0.5, max_turns=12)

    assert [c.injected for c in first.tool_calls] == expected
    assert [c.injected for c in second.tool_calls] == expected


# ---------------------------------------------------------------------------
# counted_calls and the TC-85 handler
# ---------------------------------------------------------------------------


def _record(name: str, *, injected: bool = False, turn: int = 1) -> ToolCallRecord:
    return ToolCallRecord(
        id=f"{name}-{turn}",
        name=name,
        raw_arguments="{}",
        arguments={},
        turn=turn,
        injected=injected,
    )


def test_counted_calls_drops_an_injected_call_that_was_retried() -> None:
    injected = _record("set_reminder", injected=True, turn=1)
    retry = _record("set_reminder", turn=2)
    assert counted_calls([injected, retry]) == [retry]


def test_counted_calls_keeps_an_injected_call_that_was_never_retried() -> None:
    weather = _record("get_weather", turn=1)
    stray = _record("web_search", injected=True, turn=2)
    assert counted_calls([weather, stray]) == [weather, stray]


def test_counted_calls_keeps_real_duplicates() -> None:
    first = _record("set_reminder", turn=1)
    second = _record("set_reminder", turn=2)
    assert counted_calls([first, second]) == [first, second]


def test_tc85_ambiguous_timeout_still_fires_after_an_injected_first_create() -> None:
    # Variant mode 2 (seed % 3 == 2) designs a timeout on the first create.
    tc85 = _scenario("TC-85")
    assert tc85.variant_factory is not None
    variant = tc85.variant_factory(tc85, 2)
    state = ScenarioState()
    state.tool_calls.append(_record("create_credential", injected=True, turn=1))
    attempt = _record("create_credential", turn=2)
    state.tool_calls.append(attempt)

    result = variant.handle_tool_call(state, attempt)

    assert result.get("ambiguous") is True
