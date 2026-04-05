"""Evaluator robustness tests — edge cases and adversarial inputs.

TEST-02 from critical review: verify evaluators handle degenerate inputs
gracefully without crashing or producing incorrect scores.

Tests cover:
- Empty state (no tool calls, no final answer)
- Massive tool call lists (50+ calls)
- Malformed/empty arguments
- Unicode and special characters in answers
- Very long answers (performance concern)
"""

from __future__ import annotations

import json

import pytest

from tool_eval_bench.domain.scenarios import (
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.evals.scenarios import SCENARIOS
from tool_eval_bench.evals.scenarios_extended import EXTENDED_SCENARIOS
from tool_eval_bench.evals.scenarios_agentic import AGENTIC_SCENARIOS
from tool_eval_bench.evals.scenarios_large_toolset import LARGE_TOOLSET_SCENARIOS
from tool_eval_bench.evals.scenarios_planning import PLANNING_SCENARIOS
from tool_eval_bench.evals.scenarios_adversarial import ADVERSARIAL_SCENARIOS
from tool_eval_bench.evals.helpers import contains_refusal, asks_for_clarification

ALL = (SCENARIOS + EXTENDED_SCENARIOS + AGENTIC_SCENARIOS + LARGE_TOOLSET_SCENARIOS
       + PLANNING_SCENARIOS + ADVERSARIAL_SCENARIOS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    *,
    tool_calls: list[ToolCallRecord] | None = None,
    final_answer: str = "",
    assistant_messages: list[str] | None = None,
) -> ScenarioState:
    state = ScenarioState()
    if tool_calls:
        state.tool_calls = tool_calls
    state.final_answer = final_answer
    if assistant_messages:
        state.assistant_messages = assistant_messages
    else:
        state.assistant_messages = [final_answer] if final_answer else []
    return state


def _make_call(
    name: str = "unknown_tool",
    arguments: dict | None = None,
    turn: int = 1,
) -> ToolCallRecord:
    args = arguments or {}
    return ToolCallRecord(
        id=f"tc_{turn}",
        name=name,
        raw_arguments=json.dumps(args),
        arguments=args,
        turn=turn,
    )


# ---------------------------------------------------------------------------
# Empty state tests — every evaluator should handle gracefully
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ALL, ids=[s.id for s in ALL])
def test_empty_state_does_not_crash(scenario):
    """Every evaluator must return a valid ScenarioEvaluation even with empty state."""
    state = _make_state()
    result = scenario.evaluate(state)
    assert isinstance(result, ScenarioEvaluation)
    assert result.status in (ScenarioStatus.PASS, ScenarioStatus.PARTIAL, ScenarioStatus.FAIL)
    assert result.points in (0, 1, 2)


@pytest.mark.parametrize("scenario", ALL, ids=[s.id for s in ALL])
def test_empty_string_answer_does_not_crash(scenario):
    """Evaluator with empty final_answer and no tools should not crash."""
    state = _make_state(final_answer="")
    result = scenario.evaluate(state)
    assert isinstance(result, ScenarioEvaluation)
    assert result.points in (0, 1, 2)


# ---------------------------------------------------------------------------
# Massive tool call lists
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ALL[:10], ids=[s.id for s in ALL[:10]])
def test_50_tool_calls_does_not_crash(scenario):
    """Evaluator should handle 50 tool calls without crashing or hanging."""
    calls = [_make_call(name="calculator", arguments={"expression": "1+1"}, turn=i)
             for i in range(50)]
    state = _make_state(tool_calls=calls, final_answer="The result is 2.")
    result = scenario.evaluate(state)
    assert isinstance(result, ScenarioEvaluation)
    assert result.points in (0, 1, 2)


# ---------------------------------------------------------------------------
# Malformed arguments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ALL[:10], ids=[s.id for s in ALL[:10]])
def test_empty_arguments_dict(scenario):
    """Tool calls with completely empty arguments should not crash evaluators."""
    calls = [_make_call(name="get_weather", arguments={})]
    state = _make_state(tool_calls=calls, final_answer="Something.")
    result = scenario.evaluate(state)
    assert isinstance(result, ScenarioEvaluation)


@pytest.mark.parametrize("scenario", ALL[:10], ids=[s.id for s in ALL[:10]])
def test_none_value_arguments(scenario):
    """Tool calls where argument values are None."""
    calls = [_make_call(name="get_weather", arguments={"location": None, "units": None})]
    state = _make_state(tool_calls=calls, final_answer="Something.")
    result = scenario.evaluate(state)
    assert isinstance(result, ScenarioEvaluation)


# ---------------------------------------------------------------------------
# Unicode and special characters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ALL[:10], ids=[s.id for s in ALL[:10]])
def test_unicode_answer(scenario):
    """Evaluator should handle unicode/emoji in final answer."""
    state = _make_state(final_answer="Das Wetter ist 🌤️ in Бeрлин — температура 8°C‼️")
    result = scenario.evaluate(state)
    assert isinstance(result, ScenarioEvaluation)


@pytest.mark.parametrize("scenario", ALL[:10], ids=[s.id for s in ALL[:10]])
def test_unicode_tool_arguments(scenario):
    """Tool calls with unicode in arguments should not crash."""
    calls = [_make_call(name="get_weather", arguments={"location": "München 🇩🇪"})]
    state = _make_state(tool_calls=calls, final_answer="Weather info.")
    result = scenario.evaluate(state)
    assert isinstance(result, ScenarioEvaluation)


# ---------------------------------------------------------------------------
# Very long answers (performance)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ALL[:5], ids=[s.id for s in ALL[:5]])
def test_very_long_answer(scenario):
    """Evaluator should handle a 100KB answer without excessive runtime."""
    long_answer = "The weather in Berlin is 8°C and overcast. " * 3000  # ~130KB
    state = _make_state(final_answer=long_answer)
    result = scenario.evaluate(state)
    assert isinstance(result, ScenarioEvaluation)


# ---------------------------------------------------------------------------
# Heuristic function robustness
# ---------------------------------------------------------------------------

class TestContainsRefusal:
    """Expanded refusal pattern coverage."""

    @pytest.mark.parametrize("phrase", [
        "I cannot do that.",
        "I can't help with that.",
        "I'm sorry, I don't have that ability.",
        "Unfortunately, that functionality isn't available.",
        "I'm unable to perform this action.",
        "That's not something I can do.",
        "I'm afraid I don't have access to that.",
        "This is beyond my capabilities.",
        "That is outside the scope of my tools.",
        "I don't have the capability to delete emails.",
        "No appropriate tool exists for this operation.",
        "I regrettably must decline this request.",
        "There is no way to accomplish that.",
        "The requested functionality does not exist.",
        "That isn't part of my available toolset.",
        "I lack the necessary tools.",
    ])
    def test_detects_refusal(self, phrase):
        assert contains_refusal(phrase), f"Failed to detect refusal: {phrase!r}"

    @pytest.mark.parametrize("phrase", [
        "Here is the weather for Berlin.",
        "The temperature is 8 degrees.",
        "I found 3 results for your search.",
        "The file has been updated successfully.",
    ])
    def test_no_false_positives(self, phrase):
        assert not contains_refusal(phrase), f"False positive on: {phrase!r}"


class TestAsksForClarification:
    """Expanded clarification detection."""

    @pytest.mark.parametrize("phrase", [
        "Which report did you mean?",
        "Could you clarify which Sarah you're referring to?",
        "Please provide the recipient's email address.",
        "Did you mean the Q2 or Q3 report?",
        "Could you specify the file name?",
        "Before I proceed, could you confirm the date?",
        "I'd need to know the recipient's name first.",
        "Just to confirm — should I send this to everyone?",
        "There are multiple options available. Which would you prefer?",
        "I'm not sure which file you're referring to.",
    ])
    def test_detects_clarification(self, phrase):
        assert asks_for_clarification(phrase), f"Failed to detect clarification: {phrase!r}"

    @pytest.mark.parametrize("phrase", [
        "The weather in Berlin is 8°C.",
        "I've sent the email successfully.",
        "Here are the search results.",
    ])
    def test_no_false_positives(self, phrase):
        assert not asks_for_clarification(phrase), f"False positive on: {phrase!r}"


# ---------------------------------------------------------------------------
# Verify every evaluator returns valid summary text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ALL, ids=[s.id for s in ALL])
def test_evaluation_has_nonempty_summary(scenario):
    """Every evaluation should have a non-empty summary string."""
    state = _make_state(final_answer="Hello world.")
    result = scenario.evaluate(state)
    assert isinstance(result.summary, str)
    assert len(result.summary.strip()) > 0, f"{scenario.id} returned empty summary"
