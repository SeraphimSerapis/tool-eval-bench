"""A stack's limits must not be recorded as a model's mistakes.

Three places where the benchmark graded the serving environment and reported
the result as model quality:

- a 4xx that rejected the request before the model produced anything, which the
  adapter degrades to a soft result whose ``content`` was then scored as an
  answer;
- ``tool_choice="required"``, which an endpoint may accept and silently drop,
  leaving a scenario unable to tell an ignored instruction from one that was
  never delivered;
- TC-88's PASS tier, which needs a reasoning channel most OpenAI-compatible
  endpoints do not expose.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.scenarios import (
    Category,
    FailureKind,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.runner.orchestrator import (
    run_all_scenarios,
    run_scenario,
    supports_tool_choice_required,
)

BY_ID = {s.id: s for s in ALL_SCENARIOS_WITH_HARDMODE}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedAdapter(BackendAdapter):
    """Returns queued results, recording the tool_choice each request carried."""

    def __init__(self, results: list[ChatCompletionResult]) -> None:
        self._results = list(results)
        self.tool_choices: list[Any] = []
        self.calls = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.tool_choices.append(kwargs.get("tool_choice"))
        self.calls += 1
        if not self._results:
            return ChatCompletionResult(content="[exhausted]")
        return self._results.pop(0)


class RaisingAdapter(BackendAdapter):
    """Every request explodes, standing in for an endpoint that refuses one."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        raise self._exc


def _tool_call(name: str = "probe_ping") -> ProviderToolCall:
    return ProviderToolCall(id="tc_1", name=name, arguments_str=json.dumps({"value": "ok"}))


def _trivial_scenario(
    scenario_id: str = "TC-TEST",
    *,
    tool_choice_override: str | None = None,
) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=scenario_id,
        title="Test",
        category=Category.A,
        user_message="hello",
        description="test scenario",
        handle_tool_call=lambda state, call: {"ok": True},
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "fine"),
        difficulty=1,
        tool_choice_override=tool_choice_override,
    )


async def _run(adapter: BackendAdapter, scenario: ScenarioDefinition) -> Any:
    return await run_scenario(
        adapter,
        model="m",
        base_url="http://x",
        api_key=None,
        scenario=scenario,
        max_turns=4,
    )


# ---------------------------------------------------------------------------
# 1. A rejected request is the stack's failure, not the model's
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_first_request_is_infrastructure_not_model_failure() -> None:
    """A 400 before the model authored anything cannot be the model's fault."""
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(
                content="[server error 400] tool_choice 'required' is not supported",
                transport_error_status=400,
            )
        ]
    )
    result = await _run(adapter, _trivial_scenario())

    assert result.failure_kind == FailureKind.SERVER_ERROR
    assert result.is_infrastructure_failure
    assert "400" in result.summary


@pytest.mark.asyncio
async def test_rejected_request_never_reaches_the_evaluator() -> None:
    """The error string must not be graded as though the model wrote it."""
    seen: list[str] = []

    def _evaluate(state: ScenarioState) -> ScenarioEvaluation:
        seen.append(state.final_answer)
        return ScenarioEvaluation(ScenarioStatus.PASS, 2, "graded")

    scenario = _trivial_scenario()
    scenario.evaluate = _evaluate
    adapter = ScriptedAdapter(
        [ChatCompletionResult(content="[server error 422] bad", transport_error_status=422)]
    )

    result = await _run(adapter, scenario)

    assert seen == [], "the evaluator ran on a turn that produced no model output"
    assert result.status == ScenarioStatus.FAIL


@pytest.mark.asyncio
async def test_rejection_after_the_model_authored_a_tool_call_stays_a_model_failure() -> None:
    """A strict server refusing model-authored arguments is still the model.

    This is the case the adapter's soft result was built for, so it has to keep
    working: the 4xx is caused by what the model put in the history.
    """
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="", tool_calls=[_tool_call("anything")]),
            ChatCompletionResult(content="[server error 400] bad args", transport_error_status=400),
        ]
    )
    scenario = _trivial_scenario()
    scenario.evaluate = lambda state: ScenarioEvaluation(ScenarioStatus.FAIL, 0, "model failed")

    result = await _run(adapter, scenario)

    assert result.failure_kind != FailureKind.SERVER_ERROR
    assert not result.is_infrastructure_failure


@pytest.mark.asyncio
async def test_infrastructure_rejection_drops_out_of_the_score() -> None:
    """An excluded scenario leaves the denominator, it does not score zero."""
    good = _trivial_scenario("TC-GOOD")
    bad = _trivial_scenario("TC-BAD")
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="done"),
            ChatCompletionResult(content="[server error 400] no", transport_error_status=400),
        ]
    )

    summary = await run_all_scenarios(
        adapter, model="m", base_url="http://x", scenarios=[good, bad]
    )

    assert summary.excluded_scenarios == ["TC-BAD"]
    assert summary.max_points == 2, "the excluded scenario must leave the denominator"
    assert summary.final_score == 100
    assert summary.completion_rate == 50.0


# ---------------------------------------------------------------------------
# 2. tool_choice="required" is probed, not assumed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_reports_support_when_a_tool_call_comes_back() -> None:
    adapter = ScriptedAdapter([ChatCompletionResult(content="", tool_calls=[_tool_call()])])

    assert await supports_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )
    assert adapter.tool_choices == ["required"]


@pytest.mark.asyncio
async def test_probe_reports_no_support_when_the_parameter_is_silently_dropped() -> None:
    """The failure mode nothing else can detect: accepted, then ignored."""
    adapter = ScriptedAdapter([ChatCompletionResult(content="56", tool_calls=[])])

    assert not await supports_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )


@pytest.mark.asyncio
async def test_probe_reports_no_support_when_the_endpoint_rejects_the_parameter() -> None:
    adapter = ScriptedAdapter(
        [ChatCompletionResult(content="[server error 400] no", transport_error_status=400)]
    )

    assert not await supports_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )


@pytest.mark.asyncio
async def test_probe_treats_an_exception_as_unsupported() -> None:
    adapter = RaisingAdapter(RuntimeError("connection reset"))

    assert not await supports_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )


@pytest.mark.asyncio
async def test_forced_tool_choice_scenario_is_excluded_on_an_endpoint_that_ignores_it() -> None:
    """TC-45 scored zero for answering 7x8 correctly. Now it is excluded."""
    forced = _trivial_scenario("TC-45", tool_choice_override="required")
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="56"),  # the probe, no tool call
        ]
    )

    summary = await run_all_scenarios(adapter, model="m", base_url="http://x", scenarios=[forced])

    assert summary.excluded_scenarios == ["TC-45"]
    assert summary.max_points == 0
    assert summary.completion_rate == 0.0
    assert adapter.calls == 1, "the excluded scenario must not be run"


@pytest.mark.asyncio
async def test_forced_tool_choice_scenario_runs_when_the_endpoint_enforces_it() -> None:
    forced = _trivial_scenario("TC-45", tool_choice_override="required")
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="", tool_calls=[_tool_call()]),  # probe
            ChatCompletionResult(content="56"),  # the scenario itself
        ]
    )

    summary = await run_all_scenarios(adapter, model="m", base_url="http://x", scenarios=[forced])

    assert summary.excluded_scenarios == []
    assert summary.max_points == 2
    assert adapter.tool_choices[0] == "required"


@pytest.mark.asyncio
async def test_probe_is_skipped_when_no_scenario_forces_a_tool_call() -> None:
    """An ordinary run must not pay for a capability it never uses."""
    adapter = ScriptedAdapter([ChatCompletionResult(content="done")])

    await run_all_scenarios(
        adapter, model="m", base_url="http://x", scenarios=[_trivial_scenario()]
    )

    assert adapter.calls == 1
    assert "required" not in adapter.tool_choices


@pytest.mark.asyncio
async def test_exclusion_also_applies_on_the_parallel_path() -> None:
    forced = _trivial_scenario("TC-45", tool_choice_override="required")
    other = _trivial_scenario("TC-OTHER")
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="56"),  # probe: unsupported
            ChatCompletionResult(content="done"),
        ]
    )

    summary = await run_all_scenarios(
        adapter, model="m", base_url="http://x", scenarios=[forced, other], concurrency=2
    )

    assert summary.excluded_scenarios == ["TC-45"]
    assert summary.max_points == 2


def test_only_tc45_forces_a_tool_call() -> None:
    """If another scenario adopts ``required`` it inherits the probe by design."""
    forced = [s.id for s in ALL_SCENARIOS_WITH_HARDMODE if s.tool_choice_override == "required"]
    assert forced == ["TC-45"]


# ---------------------------------------------------------------------------
# 3. TC-88's PASS tier depends on a channel the stack may not have
# ---------------------------------------------------------------------------

# Three distinct 20-digit numbers with digit sums 73, 91 and 109, where each
# value's last six digits reverse the previous value's first six.
_TC88_VALID = (
    "11111199999994000000",
    "99999999940000111111",
    "99999910000000999999",
)


def _tc88_state(reasoning: list[str]) -> ScenarioState:
    state = ScenarioState()
    state.assistant_messages = list(_TC88_VALID)
    state.final_answer = _TC88_VALID[-1]
    state.assistant_reasoning = reasoning
    return state


def test_tc88_fixture_actually_satisfies_the_scenario_constraints() -> None:
    for number, expected in zip(_TC88_VALID, (73, 91, 109), strict=True):
        assert len(number) == 20 and number[0] != "0"
        assert sum(int(digit) for digit in number) == expected
    assert _TC88_VALID[1][-6:] == _TC88_VALID[0][:6][::-1]
    assert _TC88_VALID[2][-6:] == _TC88_VALID[1][:6][::-1]


def test_tc88_passes_when_the_provider_exposes_the_plan() -> None:
    plan = f"Plan: {' '.join(_TC88_VALID)}. Sums and reversals verified."
    result = BY_ID["TC-88"].evaluate(_tc88_state([plan, "", ""]))

    assert result.status == ScenarioStatus.PASS


def test_tc88_names_the_missing_channel_instead_of_blaming_the_model() -> None:
    """Same three values, no reasoning channel: PARTIAL, and the report says why."""
    result = BY_ID["TC-88"].evaluate(_tc88_state(["", "", ""]))

    assert result.status == ScenarioStatus.PARTIAL
    assert "no reasoning channel" in result.summary
    assert "unreachable" in result.summary


def test_tc88_still_reports_a_missing_plan_when_the_channel_exists() -> None:
    """A provider that does expose reasoning is held to the original bar."""
    result = BY_ID["TC-88"].evaluate(_tc88_state(["thinking about it", "", ""]))

    assert result.status == ScenarioStatus.PARTIAL
    assert "no proof" in result.summary


def test_tc88_constraint_violations_outrank_the_channel_message() -> None:
    """A wrong answer fails on its own terms, whatever the provider exposed."""
    state = _tc88_state(["", "", ""])
    state.assistant_messages = ["1" * 20, "2" * 20, "3" * 20]
    state.final_answer = "3" * 20
    result = BY_ID["TC-88"].evaluate(state)

    assert result.status == ScenarioStatus.FAIL


def test_tc88_tool_use_still_fails_regardless_of_reasoning() -> None:
    state = _tc88_state(["", "", ""])
    state.tool_calls.append(
        ToolCallRecord(id="c", name="web_search", raw_arguments="{}", arguments={}, turn=1)
    )
    result = BY_ID["TC-88"].evaluate(state)

    assert result.status == ScenarioStatus.FAIL
