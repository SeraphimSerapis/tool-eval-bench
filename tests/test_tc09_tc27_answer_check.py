"""Tests for TC-09 and TC-27 answer content validation (issue #22).

Verifies that evaluators correctly demote from pass → partial when
the model calls the right tools but doesn't surface the actual result
values in its final answer.
"""

from __future__ import annotations

from conftest import make_state

from tool_eval_bench.evals.scenarios.agentic.tc27 import _tc27_eval
from tool_eval_bench.evals.scenarios.core.tc09 import _tc09_eval

# ===================================================================
# TC-09: Parallel Independence — answer content checks
# ===================================================================


class TestTC09AnswerValidation:
    """TC-09 should pass only when tool results are surfaced in the answer."""

    def test_pass_with_values(self):
        """Correct tools + answer contains temperature and price → pass."""
        state = make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
                {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
            ],
            final_answer="London is 12°C and Cloudy. MSFT is trading at $412.78.",
        )
        result = _tc09_eval(state)
        assert result.status.value == "pass"
        assert result.points == 2

    def test_pass_parallel_note(self):
        """When both calls are in turn 1, the note mentions parallel."""
        state = make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
                {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
            ],
            final_answer="The weather in London is 12°C, cloudy. Microsoft (MSFT) is at $412.78.",
        )
        result = _tc09_eval(state)
        assert result.status.value == "pass"
        assert result.note is not None
        assert "same assistant turn" in result.note

    def test_partial_placeholder_answer(self):
        """Correct tools but placeholder text instead of values → partial."""
        state = make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
                {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
            ],
            final_answer=(
                "Here is the information you requested:\n"
                "- **Weather in London:** [Current conditions and temperature]\n"
                "- **MSFT Stock Price:** $[Current trading price]\n"
                "Let me know if you need any further details!"
            ),
        )
        result = _tc09_eval(state)
        assert result.status.value == "partial"
        assert result.points == 1
        assert "surface" in result.summary.lower() or "results" in result.summary.lower()

    def test_partial_vague_acknowledgment(self):
        """Correct tools but vague 'I retrieved it' answer → partial."""
        state = make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
                {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
            ],
            final_answer=(
                "I've retrieved the current weather for London and the latest "
                "stock price for MSFT. Let me know if you need the details!"
            ),
        )
        result = _tc09_eval(state)
        assert result.status.value == "partial"
        assert result.points == 1

    def test_partial_only_temp(self):
        """Only temperature present but no price → partial."""
        state = make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
                {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
            ],
            final_answer="London is currently 12°C and cloudy.",
        )
        result = _tc09_eval(state)
        assert result.status.value == "partial"
        assert result.points == 1

    def test_partial_only_price(self):
        """Only price present but no temperature → partial.

        $412.78 contains '12' as a substring but the evaluator uses
        word-boundary matching so it correctly doesn't match.
        """
        state = make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
                {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
            ],
            final_answer="MSFT is trading at $412.78 on NASDAQ.",
        )
        result = _tc09_eval(state)
        assert result.status.value == "partial"
        assert result.points == 1

    def test_partial_web_search_fallback(self):
        """web_search fallback still gets partial (existing behavior)."""
        state = make_state(
            tool_calls=[
                {"name": "web_search", "arguments": {"query": "London weather MSFT price"}},
            ],
            final_answer="London is 12°C, MSFT at $412.",
        )
        result = _tc09_eval(state)
        assert result.status.value == "partial"
        assert result.points == 1

    def test_fail_missing_tool(self):
        """Only one tool called → fail."""
        state = make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "London"}},
            ],
            final_answer="London is 12°C and cloudy.",
        )
        result = _tc09_eval(state)
        assert result.status.value == "fail"
        assert result.points == 0


# ===================================================================
# TC-27: Deduplication Awareness — answer content checks
# ===================================================================


class TestTC27AnswerValidation:
    """TC-27 should pass only when both temperature values are in the answer."""

    def test_pass_with_both_temperatures(self):
        """2 calls with correct units + answer has both temps → pass."""
        state = make_state(
            tool_calls=[
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "units": "celsius"},
                    "turn": 1,
                },
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "units": "fahrenheit"},
                    "turn": 1,
                },
            ],
            final_answer=(
                "The current weather in London:\n- Celsius: 10°C, Rainy\n- Fahrenheit: 50°F, Rainy"
            ),
        )
        result = _tc27_eval(state)
        assert result.status.value == "pass"
        assert result.points == 2

    def test_partial_placeholder_answer(self):
        """2 correct calls but no actual temps in answer → partial."""
        state = make_state(
            tool_calls=[
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "units": "celsius"},
                    "turn": 1,
                },
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "units": "fahrenheit"},
                    "turn": 1,
                },
            ],
            final_answer=(
                "I've successfully retrieved the current weather for London "
                "in both Celsius and Fahrenheit. Let me know if you'd like "
                "the exact temperature readings!"
            ),
        )
        result = _tc27_eval(state)
        assert result.status.value == "partial"
        assert result.points == 1
        assert "surface" in result.summary.lower() or "temperatures" in result.summary.lower()

    def test_partial_only_celsius(self):
        """Only Celsius value present → partial."""
        state = make_state(
            tool_calls=[
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "units": "celsius"},
                    "turn": 1,
                },
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "units": "fahrenheit"},
                    "turn": 1,
                },
            ],
            final_answer="The temperature in London is 10°C.",
        )
        result = _tc27_eval(state)
        assert result.status.value == "partial"
        assert result.points == 1

    def test_partial_wrong_units(self):
        """2 calls but both with same units → partial (existing behavior)."""
        state = make_state(
            tool_calls=[
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "units": "celsius"},
                    "turn": 1,
                },
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "units": "celsius"},
                    "turn": 1,
                },
            ],
            final_answer="London is 10°C.",
        )
        result = _tc27_eval(state)
        assert result.status.value == "partial"
        assert "distinguish" in result.summary.lower()

    def test_partial_single_call(self):
        """Only 1 call → partial (existing behavior)."""
        state = make_state(
            tool_calls=[
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "units": "celsius"},
                },
            ],
            final_answer="London is 10°C, Rainy.",
        )
        result = _tc27_eval(state)
        assert result.status.value == "partial"
        assert result.points == 1

    def test_fail_no_calls(self):
        """No get_weather calls at all → fail."""
        state = make_state(
            tool_calls=[],
            final_answer="I don't have weather data.",
        )
        result = _tc27_eval(state)
        assert result.status.value == "fail"
        assert result.points == 0
