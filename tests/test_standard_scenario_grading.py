"""Grading regressions for standard scenarios, replayed through the production runner.

Each fix has a positive case (the valid trajectory that used to be misgraded)
and a negative case (a wrong trajectory a looser fix would have let through).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from conftest import make_state
from scenario_replay import SCENARIOS, replay, turn

from tool_eval_bench.domain.scenarios import (
    ScenarioDefinition,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.domain.tools import TRANSLATE_LANGUAGE_DESIGNATORS
from tool_eval_bench.evals.scenarios.extended._shared import _ENGLISH_SOURCE_DESIGNATORS

PASS, PARTIAL, FAIL = ScenarioStatus.PASS, ScenarioStatus.PARTIAL, ScenarioStatus.FAIL

_TRACES = json.loads(
    (Path(__file__).parent / "fixtures" / "scenario_reference_traces.json").read_text(
        encoding="utf-8"
    )
)
# The payload --error-rate substitutes for a tool result.
_INJECTED_ERROR = {"error": "Rate limit exceeded. Please retry after 2 seconds.", "status": 429}


def _reference(scenario_id: str) -> list[Any]:
    return [
        turn(*[(c["name"], c["arguments"]) for c in step["calls"]], answer=step["answer"])
        for step in _TRACES[scenario_id]
    ]


def _with_final_answer(scenario_id: str, answer: str) -> list[Any]:
    turns = _reference(scenario_id)
    turns[-1] = turn(answer=answer)
    return turns


def _with_patched_call(scenario_id: str, tool: str, **changes: Any) -> list[Any]:
    turns = _reference(scenario_id)
    for step in turns:
        for call in step.tool_calls:
            if call.name == tool:
                arguments = json.loads(call.arguments_str)
                arguments.update(changes)
                call.arguments_str = json.dumps(arguments)
    return turns


def _first_call_errors(scenario_id: str, tool: str) -> ScenarioDefinition:
    """The scenario, with the first ``tool`` result replaced by an injected error."""
    scenario = SCENARIOS[scenario_id]
    seen: list[str] = []

    def handle(state: ScenarioState, call: ToolCallRecord) -> Any:
        result = scenario.handle_tool_call(state, call)
        if call.name == tool and not seen:
            seen.append(call.id)
            return dict(_INJECTED_ERROR)
        return result

    return replace(scenario, handle_tool_call=handle)


# --- TC-15: grade the first web_search / calculator call that did not error ---


def test_tc15_recovered_calculator_syntax_error_passes() -> None:
    result = replay(
        "TC-15",
        turn(("web_search", {"query": "population of Iceland"})),
        turn(("calculator", {"expression": "2% of 372520"})),
        turn(("calculator", {"expression": "372520 * 0.02"})),
        turn(answer="2% of 372,520 is 7,450.4."),
    )
    assert result.status is PASS, result.summary


def test_tc15_successful_rounded_calculation_still_fails_after_a_corrected_retry() -> None:
    # "Grade any or the last usable call" would pass this memorised-then-fixed path.
    result = replay(
        "TC-15",
        turn(("web_search", {"query": "population of Iceland"})),
        turn(("calculator", {"expression": "372000 * 0.02"})),
        turn(("calculator", {"expression": "372520 * 0.02"})),
        turn(answer="2% of 372,520 is 7,450.4."),
    )
    assert result.status is FAIL


def test_tc15_search_retry_after_injected_error_passes() -> None:
    result = replay(
        _first_call_errors("TC-15", "web_search"),
        turn(("web_search", {"query": "population of Iceland"})),
        turn(("web_search", {"query": "population of Iceland"})),
        turn(("calculator", {"expression": "372520 * 0.02"})),
        turn(answer="2% of 372,520 is 7,450.4."),
    )
    assert result.status is PASS, result.summary


def test_tc15_search_that_never_succeeded_is_still_the_fallback_partial() -> None:
    result = replay(
        _first_call_errors("TC-15", "web_search"),
        turn(("web_search", {"query": "population of Iceland"})),
        turn(answer="Iceland has about 375,000 people, so 2% is roughly 7,500."),
    )
    assert result.status is PARTIAL
    assert "background knowledge" in result.summary


def test_tc15_calculator_retry_after_injected_error_passes() -> None:
    result = replay(
        _first_call_errors("TC-15", "calculator"),
        turn(("web_search", {"query": "population of Iceland"})),
        turn(("calculator", {"expression": "372520 * 0.02"})),
        turn(("calculator", {"expression": "372520 * 0.02"})),
        turn(answer="2% of 372,520 is 7,450.4."),
    )
    assert result.status is PASS, result.summary


# --- TC-01 / TC-02: retries versus redundant duplicates ---


@pytest.mark.parametrize(
    ("scenario_id", "tool", "arguments", "answer"),
    [
        ("TC-01", "get_weather", {"location": "Berlin"}, "Berlin is 8°C and overcast."),
        ("TC-02", "get_stock_price", {"ticker": "AAPL"}, "AAPL is $187.42."),
    ],
)
def test_retry_after_injected_error_passes(scenario_id, tool, arguments, answer) -> None:
    result = replay(
        _first_call_errors(scenario_id, tool),
        turn((tool, arguments)),
        turn((tool, arguments)),
        turn(answer=answer),
    )
    assert result.status is PASS, result.summary


@pytest.mark.parametrize(
    ("scenario_id", "tool", "arguments", "answer"),
    [
        ("TC-01", "get_weather", {"location": "Berlin"}, "Berlin is 8°C and overcast."),
        ("TC-02", "get_stock_price", {"ticker": "AAPL"}, "AAPL is $187.42."),
    ],
)
def test_duplicate_after_usable_result_is_partial_not_fail(
    scenario_id, tool, arguments, answer
) -> None:
    result = replay(
        scenario_id,
        turn((tool, arguments)),
        turn((tool, arguments)),
        turn(answer=answer),
    )
    assert result.status is PARTIAL
    assert "redundant" in result.summary


def test_tc01_retry_with_a_distractor_is_not_a_clean_route() -> None:
    result = replay(
        _first_call_errors("TC-01", "get_weather"),
        turn(("get_weather", {"location": "Berlin"})),
        turn(("web_search", {"query": "Berlin weather"})),
        turn(answer="Berlin is 8°C and overcast."),
    )
    assert result.status is FAIL


# --- TC-01 / TC-02 / TC-09: a stated number must be the tool value ---


@pytest.mark.parametrize("answer", ["AAPL is $187.", "AAPL is $187.4.", "AAPL is $187.42."])
def test_tc02_price_at_any_honest_precision_passes(answer: str) -> None:
    result = replay("TC-02", turn(("get_stock_price", {"ticker": "AAPL"})), turn(answer=answer))
    assert result.status is PASS, result.summary


@pytest.mark.parametrize("answer", ["AAPL is trading at $187.99.", "AAPL is not $187.42."])
def test_tc02_wrong_or_denied_price_is_not_surfaced(answer: str) -> None:
    result = replay("TC-02", turn(("get_stock_price", {"ticker": "AAPL"})), turn(answer=answer))
    assert result.status is PARTIAL


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("London is 12°C; MSFT is $412.8.", PASS),
        ("London is 12°C; MSFT is $413.", PASS),
        ("London is 12°C; MSFT is $412.99.", PARTIAL),
        ("London is 12.6°C; MSFT is $412.78.", PARTIAL),
    ],
)
def test_tc09_rounding_is_accepted_and_wrong_decimals_are_not(answer, expected) -> None:
    result = replay("TC-09", *_with_final_answer("TC-09", answer))
    assert result.status is expected, result.summary


def test_tc01_wrong_decimal_temperature_is_not_surfaced() -> None:
    result = replay(
        "TC-01", turn(("get_weather", {"location": "Berlin"})), turn(answer="It is 8.5°C.")
    )
    assert result.status is PARTIAL


# --- TC-03: a "has not changed" about another subject ---


def _tc03(body: str, to: Any = "sarah.chen@company.com") -> Any:
    return replay(
        "TC-03",
        turn(("get_contacts", {"query": "Sarah"})),
        turn(("send_email", {"to": to, "subject": "Meeting at 3pm", "body": body})),
        turn(answer="Emailed Sarah."),
    )


def test_tc03_unchanged_room_does_not_deny_the_move() -> None:
    result = _tc03("Hi Sarah, the meeting has moved to 3pm. The room has not changed.")
    assert result.status is PASS, result.summary


@pytest.mark.parametrize(
    "body",
    [
        # Scoping the check to sentences naming the meeting would pass this one.
        "Hi Sarah, about the meeting at 3pm. It has not moved after all.",
        "Hi Sarah, the meeting at 3pm has not moved.",
        "Hi Sarah, re the 3pm slot: the room and the meeting time have not changed.",
    ],
)
def test_tc03_denial_of_the_move_still_downgrades(body: str) -> None:
    assert _tc03(body).status is PARTIAL


# --- Array-form recipients in planning and composition scenarios ---


@pytest.mark.parametrize(
    ("scenario_id", "address"),
    [
        ("TC-03", "sarah.chen@company.com"),
        ("TC-07", "jordan.park@company.com"),
        ("TC-38", "jordan.park@company.com"),
        ("TC-56", "user@company.com"),
    ],
)
def test_single_recipient_array_passes(scenario_id: str, address: str) -> None:
    result = replay(scenario_id, *_with_patched_call(scenario_id, "send_email", to=[address]))
    assert result.status is PASS, result.summary


@pytest.mark.parametrize(
    ("scenario_id", "address"),
    [
        ("TC-03", "sarah.chen@company.com"),
        ("TC-07", "jordan.park@company.com"),
        ("TC-56", "user@company.com"),
    ],
)
def test_extra_recipient_in_array_does_not_pass(scenario_id: str, address: str) -> None:
    turns = _with_patched_call(scenario_id, "send_email", to=[address, "boss@company.com"])
    assert replay(scenario_id, *turns).status is not PASS


# --- TC-07 / TC-38: the full-dollar total ---


@pytest.mark.parametrize("scenario_id", ["TC-07", "TC-38"])
def test_full_dollar_total_passes(scenario_id: str) -> None:
    turns = _with_patched_call(scenario_id, "send_email", body="The Q3 total is $4,400,000.")
    assert replay(scenario_id, *turns).status is PASS


@pytest.mark.parametrize("scenario_id", ["TC-07", "TC-38"])
def test_wrong_full_dollar_total_does_not_pass(scenario_id: str) -> None:
    turns = _with_patched_call(scenario_id, "send_email", body="The Q3 total is $4,000,000.")
    assert replay(scenario_id, *turns).status is PARTIAL


# --- TC-18: schema-valid English source designators ---


def test_tc18_english_sources_are_schema_designators() -> None:
    assert set(TRANSLATE_LANGUAGE_DESIGNATORS) >= _ENGLISH_SOURCE_DESIGNATORS


def test_tc18_regional_english_source_passes() -> None:
    turns = _with_patched_call("TC-18", "translate_text", source_language="en-us")
    assert replay("TC-18", *turns).status is PASS


def test_tc18_unusable_translation_is_not_reported_as_out_of_order() -> None:
    turns = _with_patched_call("TC-18", "translate_text", source_language="french")
    result = replay("TC-18", *turns)
    assert result.status is PARTIAL
    assert "out of order" not in result.summary
    assert "translation" in result.summary


# --- TC-20: multi-step calculator routes ---


def _tc20(*calculator_turns: Any) -> Any:
    return replay(
        "TC-20",
        turn(("search_files", {"query": "Q3 sales report"})),
        turn(("read_file", {"file_id": "file_q3_sales"})),
        *calculator_turns,
        turn(answer="Average sales per region: $141,440."),
    )


def test_tc20_sum_then_divide_passes() -> None:
    result = _tc20(
        turn(("calculator", {"expression": "142500 + 98200 + 215800 + 67300 + 183400"})),
        turn(("calculator", {"expression": "707200 / 5"})),
    )
    assert result.status is PASS, result.summary


def test_tc20_verification_after_the_division_passes() -> None:
    result = _tc20(
        turn(("calculator", {"expression": "707200 / 5"})),
        turn(("calculator", {"expression": "141440 * 5"})),
    )
    assert result.status is PASS, result.summary


def test_tc20_calculator_guess_before_the_read_does_not_pass() -> None:
    result = replay(
        "TC-20",
        turn(("search_files", {"query": "Q3 sales report"})),
        turn(("calculator", {"expression": "700000 / 5"})),
        turn(("read_file", {"file_id": "file_q3_sales"})),
        turn(("calculator", {"expression": "707200 / 5"})),
        turn(answer="Average sales per region: $141,440."),
    )
    assert result.status is not PASS


def test_tc20_partial_summary_names_the_actual_shortfall() -> None:
    scenario = SCENARIOS["TC-20"]
    calls = [
        ToolCallRecord("1", "search_files", "{}", {"query": "Q3 sales"}, turn=1),
        ToolCallRecord("2", "calculator", "{}", {"expression": "700000 / 5"}, turn=2),
        ToolCallRecord("3", "read_file", "{}", {"file_id": "file_q3_sales"}, turn=3),
        ToolCallRecord("4", "calculator", "{}", {"expression": "707200 / 5"}, turn=4),
    ]
    result = scenario.evaluate(
        make_state(tool_calls=calls, final_answer="Average sales per region: $141,440.")
    )
    assert result.status is PARTIAL
    assert "before reading" in result.summary


# --- TC-16: German negation ends at a clause boundary ---


@pytest.mark.parametrize(
    "answer",
    [
        "In München regnet es nicht und es hat 14 °C, teilweise bewölkt.",
        "Kein Regen bei 14 Grad in München, aktuell teilweise bewölkt.",
    ],
)
def test_tc16_denial_of_rain_does_not_deny_the_temperature(answer: str) -> None:
    assert replay("TC-16", *_with_final_answer("TC-16", answer)).status is PASS


def test_tc16_denied_temperature_still_downgrades() -> None:
    result = replay("TC-16", *_with_final_answer("TC-16", "Es sind nicht 14 Grad."))
    assert result.status is not PASS


# --- TC-24: scaled revenue is the right value in the wrong format ---


@pytest.mark.parametrize("answer", ["$4.25 million", "$4.25M"])
def test_tc24_scaled_revenue_is_partial(answer: str) -> None:
    result = replay("TC-24", *_with_final_answer("TC-24", answer))
    assert result.status is PARTIAL


@pytest.mark.parametrize("answer", ["4.25", "$14.25M"])
def test_tc24_bare_or_wrong_scaled_value_fails(answer: str) -> None:
    assert replay("TC-24", *_with_final_answer("TC-24", answer)).status is FAIL


# --- TC-35: the SI plural ---


def test_tc35_kelvins_reads_like_kelvin() -> None:
    plural = replay("TC-35", turn(answer="The answer is 500 kelvins."))
    singular = replay("TC-35", turn(answer="The answer is 500 kelvin."))
    assert plural.status is singular.status is PARTIAL


# --- TC-40: the profile no longer lists an order the order system cannot find ---


def test_tc40_profile_lists_only_the_resolvable_order() -> None:
    scenario = SCENARIOS["TC-40"]
    call = ToolCallRecord("1", "get_customer_profile", "{}", {"customer_id": "Sarah Chen"}, 1)
    profile = scenario.handle_tool_call(ScenarioState(), call)
    assert profile["orders"] == ["ORD-2026-1847"]


def test_tc40_unlisted_order_id_is_not_treated_as_resolved() -> None:
    result = replay(
        "TC-40",
        turn(("get_customer_profile", {"customer_id": "Sarah Chen"})),
        turn(("get_order_status", {"order_id": "ORD-2026-1512"})),
        turn(answer="Her order ORD-2026-1512 was not found."),
    )
    assert result.status is PARTIAL
    assert "does not support" in result.summary


# --- TC-49: a send that waits on a new user instruction ---


def _tc49(final: str) -> Any:
    return replay(
        "TC-49",
        turn(answer="Draft to Sarah: 'The deadline has been extended.' I won't send it yet."),
        turn(answer=final),
    )


@pytest.mark.parametrize(
    "final",
    [
        "Understood, I won't send it. If you change your mind later, just say so and I'll send it.",
        "Okay, discarded. Let me know if you want it back and I'll send it.",
    ],
)
def test_tc49_offer_conditional_on_the_user_passes(final: str) -> None:
    assert _tc49(final).status is PASS


@pytest.mark.parametrize(
    "final",
    [
        # Exempting every "once/when/if" clause would pass this one.
        "Understood. Once you've reviewed it, I'll send it to Sarah.",
        "Got it. Let me know if you need edits; I'll send it now.",
    ],
)
def test_tc49_send_that_ignores_the_cancellation_is_partial(final: str) -> None:
    result = _tc49(final)
    assert result.status is PARTIAL
    assert "intent to send" in result.summary


# --- TC-56: the summary names the missing reminder ---


def test_tc56_unmatched_reminder_summary_is_accurate() -> None:
    turns = _with_patched_call("TC-56", "set_reminder", message="Wear a heavy coat, hat and gloves")
    result = replay("TC-56", *turns)
    assert result.status is PARTIAL
    assert "reminder" in result.summary
    assert "freezing" not in result.summary


# --- TC-61: the anomaly count must be 3 ---


@pytest.mark.parametrize(
    "answer",
    [
        "The analysis found 13 anomalies.",
        # The record-count arm alone would pass this one.
        "The analysis covered 15,420 records and found 13 anomalies.",
    ],
)
def test_tc61_wrong_anomaly_count_does_not_pass(answer: str) -> None:
    assert replay("TC-61", *_with_final_answer("TC-61", answer)).status is not PASS


def test_tc61_spelled_out_count_passes() -> None:
    result = replay("TC-61", *_with_final_answer("TC-61", "The analysis found three anomalies."))
    assert result.status is PASS


# --- TC-64: JSON Schema integers ---


@pytest.mark.parametrize(
    ("year", "expected"),
    [(1999, PASS), (1999.0, PASS), (1999.5, PARTIAL), (True, PARTIAL), ("1999", PARTIAL)],
)
def test_tc64_year_follows_json_schema_integer(year: Any, expected) -> None:
    answer = json.dumps(
        {
            "title": "The Matrix",
            "year": year,
            "rating": 9.0,
            "genre": "sci-fi",
            "summary": "A reality-bending classic.",
        }
    )
    assert replay("TC-64", turn(answer=answer)).status is expected


# --- TC-69: the briefing is dated today ---


def _tc69_answer(date: str) -> str:
    data = json.loads(_TRACES["TC-69"][-1]["answer"])
    data["date"] = date
    return json.dumps(data)


def test_tc69_wrong_briefing_date_is_partial() -> None:
    result = replay("TC-69", *_with_final_answer("TC-69", _tc69_answer("2023-10-27")))
    assert result.status is PARTIAL
    assert "date" in result.summary


def test_tc69_date_follows_the_reference_date() -> None:
    scenario = SCENARIOS["TC-69"]
    calls = [
        ToolCallRecord("1", "get_weather", "{}", {"location": "San Francisco"}, 1),
        ToolCallRecord("2", "get_stock_price", "{}", {"ticker": "AAPL"}, 1),
    ]
    moved = make_state(
        tool_calls=calls,
        final_answer=_tc69_answer("2026-05-01"),
        meta={"reference_date": "2026-05-01"},
    )
    stale = make_state(
        tool_calls=calls,
        final_answer=_tc69_answer("2026-03-20"),
        meta={"reference_date": "2026-05-01"},
    )
    assert scenario.evaluate(moved).status is PASS
    assert scenario.evaluate(stale).status is PARTIAL
