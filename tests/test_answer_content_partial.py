"""Tests for answer-content partial-downgrade branches (issue #22).

Each test verifies that when a model calls the correct tools but produces
a placeholder/vague answer without surfacing the actual data, the evaluator
returns _partial instead of _pass.

This covers the downgrade branches added in commits 1a8f287, 415e362,
fff2930, and b7760de. The TC-09/TC-27 branches are already covered in
test_tc09_tc27_answer_check.py.
"""

from __future__ import annotations

from conftest import make_state

from tool_eval_bench.evals.scenarios.agentic.tc22 import _tc22_eval
from tool_eval_bench.evals.scenarios.agentic.tc45 import _tc45_eval
from tool_eval_bench.evals.scenarios.core.tc01 import _tc01_eval
from tool_eval_bench.evals.scenarios.core.tc02 import _tc02_eval
from tool_eval_bench.evals.scenarios.core.tc04 import _tc04_eval
from tool_eval_bench.evals.scenarios.core.tc06 import _tc06_eval
from tool_eval_bench.evals.scenarios.core.tc14 import _tc14_eval
from tool_eval_bench.evals.scenarios.core.tc15 import _tc15_eval
from tool_eval_bench.evals.scenarios.extended.tc16 import _tc16_eval
from tool_eval_bench.evals.scenarios.hardmode.tc70 import _tc70_eval
from tool_eval_bench.evals.scenarios.large_toolset.tc37 import _tc37_eval
from tool_eval_bench.evals.scenarios.large_toolset.tc40 import _tc40_eval
from tool_eval_bench.evals.scenarios.planning.tc52 import _tc52_eval
from tool_eval_bench.evals.scenarios.planning.tc61 import _tc61_eval


class TestTC01PartialDowngrade:
    """TC-01: correct tool + missing weather data → partial."""

    def test_partial_placeholder_answer(self):
        state = make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            final_answer="I checked the weather in Berlin for you.",
        )
        result = _tc01_eval(state)
        assert result.status.value == "partial"
        assert "surface" in result.summary.lower()


class TestTC02PartialDowngrade:
    """TC-02: correct tool + missing stock price → partial."""

    def test_partial_placeholder_answer(self):
        state = make_state(
            tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}],
            final_answer="I've retrieved the current stock price for Apple.",
        )
        result = _tc02_eval(state)
        assert result.status.value == "partial"
        assert "surface" in result.summary.lower() or "price" in result.summary.lower()


class TestTC04PartialDowngrade:
    """TC-04: correct tool + units=fahrenheit but missing temperature → partial."""

    def test_partial_placeholder_answer(self):
        state = make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Tokyo", "units": "fahrenheit"}}
            ],
            final_answer="I checked the weather in Tokyo for you.",
        )
        result = _tc04_eval(state)
        assert result.status.value == "partial"
        assert "surface" in result.summary.lower() or "temperature" in result.summary.lower()


class TestTC06PartialDowngrade:
    """TC-06: correct translate calls but no translations surfaced → partial."""

    def test_partial_placeholder_answer(self):
        state = make_state(
            tool_calls=[
                {
                    "name": "translate_text",
                    "arguments": {
                        "text": "Where is the nearest hospital?",
                        "source_language": "English",
                        "target_language": "Spanish",
                    },
                },
                {
                    "name": "translate_text",
                    "arguments": {
                        "text": "Where is the nearest hospital?",
                        "source_language": "English",
                        "target_language": "Japanese",
                    },
                },
            ],
            final_answer="Here are the translations you requested.",
        )
        result = _tc06_eval(state)
        assert result.status.value == "partial"
        assert "surface" in result.summary.lower() or "translation" in result.summary.lower()


class TestTC14PartialDowngrade:
    """TC-14: error acknowledged + recovery but price not surfaced → partial."""

    def test_partial_no_price(self):
        state = make_state(
            tool_calls=[
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}, "turn": 1},
                {"name": "web_search", "arguments": {"query": "AAPL stock price"}, "turn": 2},
            ],
            final_answer="The stock price service was temporarily unavailable. I searched the web instead.",
            assistant_messages=[
                "The stock price service was temporarily unavailable. I searched the web instead."
            ],
        )
        result = _tc14_eval(state)
        assert result.status.value == "partial"
        assert "price" in result.summary.lower()


class TestTC15PartialDowngrade:
    """TC-15: correct search + calculator but result not surfaced → partial."""

    def test_partial_no_result(self):
        state = make_state(
            tool_calls=[
                {"name": "web_search", "arguments": {"query": "population of iceland"}},
                {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}},
            ],
            final_answer="I calculated 2% of Iceland's population for you.",
        )
        result = _tc15_eval(state)
        assert result.status.value == "partial"
        assert "surface" in result.summary.lower() or "result" in result.summary.lower()


class TestTC16PartialDowngrade:
    """TC-16: correct tool + German response but no temperature → partial."""

    def test_partial_german_no_data(self):
        # Answer has 3+ German weather tokens (wetter, münchen, celsius, temperatur)
        # but does NOT contain temperature 14 or condition bewölkt.
        state = make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "München"}}],
            final_answer="Das Wetter in München ist kühl mit Celsius Temperatur.",
        )
        result = _tc16_eval(state)
        assert result.status.value == "partial"
        assert "temperature" in result.summary.lower()


class TestTC22PartialDowngrade:
    """TC-22: correct tool + valid JSON keys but wrong temp value → partial."""

    def test_partial_wrong_temp_value(self):
        state = make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            final_answer='{"temp": 99, "condition": "Overcast", "humidity": 82}',
        )
        result = _tc22_eval(state)
        assert result.status.value == "partial"
        assert "wrong values" in result.summary.lower()


class TestTC37PartialDowngrade:
    """TC-37: correct tool from 52 but missing weather data → partial."""

    def test_partial_placeholder_answer(self):
        state = make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            final_answer="I checked the weather in Berlin using the global weather tool.",
        )
        result = _tc37_eval(state)
        assert result.status.value == "partial"
        assert "surface" in result.summary.lower()


class TestTC40PartialDowngrade:
    """TC-40: correct tool but no order details surfaced → partial."""

    def test_partial_no_details(self):
        state = make_state(
            tool_calls=[{"name": "get_order_status", "arguments": {"order_id": "Sarah Chen"}}],
            final_answer="I've retrieved the requested information for Sarah Chen.",
        )
        result = _tc40_eval(state)
        assert result.status.value == "partial"
        assert "surface" in result.summary.lower() or "details" in result.summary.lower()


class TestTC45PartialDowngrade:
    """TC-45: calculator returns 56 but answer doesn't contain it → partial."""

    def test_partial_no_result_in_answer(self):
        state = make_state(
            tool_calls=[{"name": "calculator", "arguments": {"expression": "7 * 8"}}],
            final_answer="I used the calculator to compute your expression.",
        )
        result = _tc45_eval(state)
        assert result.status.value == "partial"
        assert "surface" in result.summary.lower() or "56" in result.summary


class TestTC52PartialDowngrade:
    """TC-52: both data sources but no actual numbers in comparison → partial."""

    def test_partial_vague_comparison(self):
        state = make_state(
            tool_calls=[
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
                {"name": "web_search", "arguments": {"query": "S&P 500 market performance"}},
            ],
            final_answer="AAPL has been compared against the market benchmark.",
        )
        result = _tc52_eval(state)
        assert result.status.value == "partial"
        assert "synthesize" in result.summary.lower()


class TestTC61PartialDowngrade:
    """TC-61: polled correctly but bare 'anomalies' without specifics → partial."""

    def test_partial_bare_anomalies(self):
        state = make_state(
            tool_calls=[
                {"name": "run_code", "arguments": {"code": "analyze_data()"}, "turn": 1},
                {"name": "run_code", "arguments": {"code": "check_status()"}, "turn": 2},
            ],
            final_answer="The analysis is complete. Some anomalies were found in the data.",
        )
        result = _tc61_eval(state)
        assert result.status.value == "partial"
        assert "result" in result.summary.lower() or "surface" in result.summary.lower()


class TestTC70PartialDowngrade:
    """TC-70: correct global tool but no weather data surfaced → partial."""

    def test_partial_placeholder_answer(self):
        state = make_state(
            tool_calls=[{"name": "get_weather_global", "arguments": {"location": "Tokyo"}}],
            final_answer="I retrieved the weather for Tokyo using the global weather service.",
        )
        result = _tc70_eval(state)
        assert result.status.value == "partial"
        assert "surface" in result.summary.lower()
