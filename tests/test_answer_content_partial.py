"""Tests for answer-content partial-downgrade and surfacing checks (issue #22).

Each test verifies that when a model calls the expected tools but returns a
placeholder or partial answer without surfacing the required tool values, the
evaluator returns ``PARTIAL`` (and credits parallel-turn ``TC-09`` notes when
both values are surfaced).
"""

from __future__ import annotations

import pytest
from conftest import make_state, simulate_results

from tool_eval_bench.domain.scenarios import ScenarioEvaluation, ScenarioStatus
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

_SCENARIOS = {scenario.id: scenario for scenario in ALL_SCENARIOS_WITH_HARDMODE}


def _evaluate(
    scenario_id: str,
    *,
    tool_calls: list[dict] | None = None,
    final_answer: str = "",
    assistant_messages: list[str] | None = None,
) -> ScenarioEvaluation:
    scenario = _SCENARIOS[scenario_id]
    state = make_state(
        tool_calls=tool_calls,
        final_answer=final_answer,
        assistant_messages=assistant_messages,
    )
    return scenario.evaluate(simulate_results(state, scenario))


@pytest.mark.parametrize(
    ("scenario_id", "tool_calls", "final_answer", "expected_summary_tokens"),
    [
        (
            "TC-01",
            [{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            "I checked the weather in Berlin for you.",
            ("surface",),
        ),
        (
            "TC-02",
            [{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}],
            "I've retrieved the current stock price for Apple.",
            ("surface", "price"),
        ),
        (
            "TC-04",
            [{"name": "get_weather", "arguments": {"location": "Tokyo", "units": "fahrenheit"}}],
            "I checked the weather in Tokyo for you.",
            ("surface", "temperature"),
        ),
        (
            "TC-06",
            [
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
            "Here are the translations you requested.",
            ("surface", "translation"),
        ),
        (
            "TC-14",
            [
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}, "turn": 1},
                {"name": "web_search", "arguments": {"query": "AAPL stock price"}, "turn": 2},
            ],
            "The stock price service was temporarily unavailable. I searched the web instead.",
            ("price",),
        ),
        (
            "TC-15",
            [
                {"name": "web_search", "arguments": {"query": "population of iceland"}, "turn": 1},
                {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}, "turn": 2},
            ],
            "I calculated 2% of Iceland's population for you.",
            ("surface", "result"),
        ),
        (
            "TC-16",
            [{"name": "get_weather", "arguments": {"location": "München"}}],
            "Das Wetter in München ist kühl mit Celsius Temperatur.",
            ("temperature",),
        ),
        (
            "TC-22",
            [{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            '{"temp": 99, "condition": "Overcast", "humidity": 82}',
            ("wrong values",),
        ),
        (
            "TC-37",
            [{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            "I checked the weather in Berlin using the global weather tool.",
            ("surface",),
        ),
        (
            "TC-40",
            [{"name": "get_order_status", "arguments": {"order_id": "Sarah Chen"}}],
            "I've retrieved the requested information for Sarah Chen.",
            ("surface", "details"),
        ),
        (
            "TC-45",
            [{"name": "calculator", "arguments": {"expression": "7 * 8"}}],
            "I used the calculator to compute your expression.",
            ("surface", "56"),
        ),
        (
            "TC-52",
            [
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
                {"name": "web_search", "arguments": {"query": "S&P 500 market performance"}},
            ],
            "AAPL has been compared against the market benchmark.",
            ("synthesize",),
        ),
        (
            "TC-61",
            [
                {
                    "name": "run_code",
                    "arguments": {"code": 'analyze_data(source="transactions_2026")'},
                    "turn": 1,
                },
                {
                    "name": "run_code",
                    "arguments": {"code": 'check_job_status("job_tc61_9f3a")'},
                    "turn": 2,
                },
            ],
            "The analysis is complete. Some anomalies were found in the data.",
            ("result", "surface"),
        ),
        (
            "TC-70",
            [{"name": "get_weather_global", "arguments": {"location": "Tokyo"}}],
            "I retrieved the weather for Tokyo using the global weather service.",
            ("surface",),
        ),
    ],
)
def test_partial_downgrade_when_values_not_surfaced(
    scenario_id: str,
    tool_calls: list[dict],
    final_answer: str,
    expected_summary_tokens: tuple[str, ...],
) -> None:
    result = _evaluate(
        scenario_id,
        tool_calls=tool_calls,
        final_answer=final_answer,
        assistant_messages=[final_answer],
    )
    assert result.status == ScenarioStatus.PARTIAL
    assert result.points == 1
    assert any(token in result.summary.lower() for token in expected_summary_tokens)


class TestTC09AnswerValidation:
    """TC-09 passes with parallel-turn note only when both values are surfaced."""

    _PARALLEL_CALLS = [
        {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
        {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
    ]

    def test_pass_parallel_note(self) -> None:
        result = _evaluate(
            "TC-09",
            tool_calls=self._PARALLEL_CALLS,
            final_answer="The weather in London is 12°C, cloudy. Microsoft (MSFT) is at $412.78.",
        )
        assert result.status == ScenarioStatus.PASS
        assert result.points == 2
        assert result.note is not None
        assert "same assistant turn" in result.note

    @pytest.mark.parametrize(
        "answer",
        [
            (
                "Here is the information you requested:\n"
                "- **Weather in London:** [Current conditions and temperature]\n"
                "- **MSFT Stock Price:** $[Current trading price]\n"
                "Let me know if you need any further details!"
            ),
            (
                "I've retrieved the current weather for London and the latest "
                "stock price for MSFT. Let me know if you need the details!"
            ),
            "London is currently 12°C and cloudy.",
            # $412.78 contains "12" as a substring but word-boundary matching rejects it.
            "MSFT is trading at $412.78 on NASDAQ.",
        ],
    )
    def test_partial_missing_or_partial_values(self, answer: str) -> None:
        result = _evaluate("TC-09", tool_calls=self._PARALLEL_CALLS, final_answer=answer)
        assert result.status == ScenarioStatus.PARTIAL
        assert result.points == 1


class TestTC27AnswerValidation:
    """TC-27 requires both temperatures to be surfaced in the answer."""

    _BOTH_CALLS = [
        {"name": "get_weather", "arguments": {"location": "London", "units": "celsius"}, "turn": 1},
        {
            "name": "get_weather",
            "arguments": {"location": "London", "units": "fahrenheit"},
            "turn": 1,
        },
    ]

    @pytest.mark.parametrize(
        "answer",
        [
            (
                "I've successfully retrieved the current weather for London "
                "in both Celsius and Fahrenheit. Let me know if you'd like "
                "the exact temperature readings!"
            ),
            "The temperature in London is 10°C.",
        ],
    )
    def test_partial_missing_temperatures(self, answer: str) -> None:
        result = _evaluate("TC-27", tool_calls=self._BOTH_CALLS, final_answer=answer)
        assert result.status == ScenarioStatus.PARTIAL
        assert result.points == 1

    def test_fail_no_calls(self) -> None:
        result = _evaluate("TC-27", tool_calls=[], final_answer="I don't have weather data.")
        assert result.status == ScenarioStatus.FAIL
        assert result.points == 0
