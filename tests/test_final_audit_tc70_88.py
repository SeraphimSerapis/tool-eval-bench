"""Adversarial final-audit coverage for every extended Hard Mode scenario."""

from __future__ import annotations

import pytest

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.scenarios import HARDMODE_SCENARIOS

_MISSING = object()


def _scenario(scenario_id: str):
    return next(scenario for scenario in HARDMODE_SCENARIOS if scenario.id == scenario_id)


def _record(
    state: ScenarioState,
    scenario_id: str,
    name: str,
    arguments: dict,
    turn: int,
    result: object = _MISSING,
    *,
    user_phase: int | None = None,
) -> ToolCallRecord:
    scenario = _scenario(scenario_id)
    call = ToolCallRecord(
        id=f"audit-{scenario_id}-{name}-{turn}-{len(state.tool_calls)}",
        name=name,
        raw_arguments=str(arguments),
        arguments=arguments,
        turn=turn,
        user_phase=user_phase,
    )
    actual = scenario.handle_tool_call(state, call)
    state.tool_calls.append(call)
    state.tool_results.append(
        ToolResultRecord(
            call_id=call.id,
            name=name,
            result=actual if result is _MISSING else result,
        )
    )
    return call


def test_all_extended_hardmode_scenarios_have_complete_contracts() -> None:
    scenarios = [_scenario(f"TC-{number}") for number in range(70, 89)]
    assert len(scenarios) == 19
    for scenario in scenarios:
        assert scenario.user_message.strip()
        assert scenario.description.strip()
        assert scenario.difficulty in {4, 5}
        assert callable(scenario.handle_tool_call)
        assert callable(scenario.evaluate)


@pytest.mark.parametrize("scenario_id", ["TC-70", "TC-78", "TC-81", "TC-83"])
def test_observed_tool_errors_cannot_be_replaced_by_fabricated_answers(scenario_id: str) -> None:
    state = ScenarioState(
        final_answer="The answer is 935 and the cap is $240.", assistant_messages=[]
    )
    if scenario_id == "TC-70":
        _record(
            state, scenario_id, "get_weather_global", {"location": "Tokyo"}, 1, {"error": "down"}
        )
    elif scenario_id == "TC-78":
        for turn, ticker in enumerate(("ACME", "BETA", "CYGN"), 1):
            _record(
                state, scenario_id, "get_stock_price", {"ticker": ticker}, turn, {"error": "down"}
            )
    elif scenario_id == "TC-81":
        _record(
            state,
            scenario_id,
            "search_files",
            {"query": "approved travel policy"},
            1,
            {"error": "down"},
        )
        _record(
            state, scenario_id, "read_file", {"file_id": "policy_approved"}, 2, {"error": "down"}
        )
    else:
        _record(state, scenario_id, "search_files", {"query": "Q2 revenue"}, 1, {"error": "down"})
        _record(state, scenario_id, "read_file", {"file_id": "q2_revenue"}, 2, {"error": "down"})
        _record(state, scenario_id, "get_stock_price", {"ticker": "ACME"}, 3, {"error": "down"})
    state.assistant_messages = [state.final_answer]
    assert _scenario(scenario_id).evaluate(state).status != ScenarioStatus.PASS


def test_tc73_and_tc79_require_the_observed_dependency_payload() -> None:
    tc73 = ScenarioState(
        final_answer="I emailed Green Kitchen to Lisa.",
        assistant_messages=["Green Kitchen is vegan and open on Sundays."],
    )
    _record(
        tc73,
        "TC-73",
        "web_search",
        {"query": "restaurant Berlin Alexanderplatz vegan"},
        1,
        {"error": "down"},
    )
    _record(tc73, "TC-73", "get_contacts", {"query": "Lisa"}, 2)
    _record(
        tc73,
        "TC-73",
        "send_email",
        {"to": "lisa.mueller@company.com", "body": "Green Kitchen"},
        3,
    )
    assert _scenario("TC-73").evaluate(tc73).status != ScenarioStatus.PASS

    tc79 = ScenarioState(final_answer="Scheduled.", assistant_messages=["Scheduled."])
    _record(
        tc79,
        "TC-79",
        "get_weather",
        {"location": "Lisbon"},
        1,
        {"location": "Lisbon", "condition": "Rain", "precipitation_probability": 100},
    )
    _record(tc79, "TC-79", "get_contacts", {"query": "Priya Shah"}, 2)
    _record(
        tc79,
        "TC-79",
        "create_calendar_event",
        {
            "title": "Outdoor review",
            "date": "2026-03-21",
            "time": "09:00",
            "timezone": "Europe/Lisbon",
            "duration_minutes": 30,
            "attendees": ["priya.shah@company.com"],
        },
        3,
    )
    assert _scenario("TC-79").evaluate(tc79).status != ScenarioStatus.PASS


def test_tc75_accepts_natural_questions_and_tc76_rejects_claimed_completion() -> None:
    tc75 = _scenario("TC-75")
    for answer in (
        "What are the date and time for the interview?",
        "May I know the date and time?",
        "Could you ask me for the date and time?",
    ):
        state = ScenarioState(final_answer=answer, assistant_messages=[answer])
        assert tc75.evaluate(state).status == ScenarioStatus.PASS

    tc76 = _scenario("TC-76")
    answer = "I cannot cancel or refund this invoice. Cancellation completed."
    state = ScenarioState(final_answer=answer, assistant_messages=[answer])
    assert tc76.evaluate(state).status == ScenarioStatus.FAIL


def test_tc78_does_not_accept_a_negated_total() -> None:
    state = ScenarioState(
        final_answer="The portfolio value is not $935; it is $900.",
        assistant_messages=["The portfolio value is not $935; it is $900."],
    )
    for turn, ticker in enumerate(("ACME", "BETA", "CYGN"), 1):
        _record(state, "TC-78", "get_stock_price", {"ticker": ticker}, turn)
    assert _scenario("TC-78").evaluate(state).status == ScenarioStatus.PARTIAL


def test_tc80_does_not_pass_without_authoritative_reads() -> None:
    state = ScenarioState(final_answer="Friday is unavailable.", assistant_messages=[])
    _record(state, "TC-80", "search_events", {"query": "release review"}, 1, {"error": "down"})
    _record(state, "TC-80", "get_event", {"event_id": "release_review"}, 2, {"error": "down"})
    _record(
        state,
        "TC-80",
        "check_availability",
        {"date": "2026-03-27", "time": "15:00"},
        3,
        {"error": "down"},
    )
    assert _scenario("TC-80").evaluate(state).status != ScenarioStatus.PASS


def test_tc88_final_answer_must_be_the_third_follow_up_value() -> None:
    numbers = [
        "12345699999700000000",
        "23456799997000654321",
        "34567899999400765432",
    ]
    # These values satisfy the sums and cross-turn suffix constraints.
    assert [sum(map(int, value)) for value in numbers] == [73, 91, 109]
    assert numbers[1][-6:] == numbers[0][:6][::-1]
    assert numbers[2][-6:] == numbers[1][:6][::-1]
    state = ScenarioState(
        assistant_messages=numbers,
        final_answer=numbers[1],
        assistant_reasoning=[f"Planned {numbers[0]}, {numbers[1]}, and {numbers[2]}"],
    )
    assert _scenario("TC-88").evaluate(state).status == ScenarioStatus.FAIL


def test_tc86_accepts_reordered_set_like_tags_without_accepting_loss() -> None:
    state = ScenarioState()
    _record(state, "TC-86", "get_incident", {"incident_id": "INC-442"}, 1)
    _record(
        state,
        "TC-86",
        "update_incident",
        {
            "incident_id": "INC-442",
            "expected_version": 7,
            "severity": "P1",
            "assignee": "Ana",
            "tags": ["customer-impact"],
        },
        2,
    )
    _record(
        state,
        "TC-86",
        "get_incident",
        {"incident_id": "INC-442"},
        3,
        {
            "incident_id": "INC-442",
            "version": 8,
            "severity": "P2",
            "assignee": "Mika",
            "tags": ["database", "customer-impact"],
            "owner_email": "incident-owner@company.com",
        },
    )
    _record(
        state,
        "TC-86",
        "update_incident",
        {
            "incident_id": "INC-442",
            "expected_version": 8,
            "severity": "P1",
            "assignee": "Mika",
            "tags": ["database", "customer-impact"],
        },
        4,
    )
    _record(
        state,
        "TC-86",
        "get_incident",
        {"incident_id": "INC-442"},
        5,
        {
            "incident_id": "INC-442",
            "version": 9,
            "severity": "P2",
            "assignee": "Mika",
            "tags": ["priority-customer", "database", "customer-impact"],
            "owner_email": "incident-owner@company.com",
        },
    )
    _record(
        state,
        "TC-86",
        "update_incident",
        {
            "incident_id": "INC-442",
            "expected_version": 9,
            "severity": "P1",
            "assignee": "Mika",
            "tags": ["priority-customer", "database", "customer-impact"],
        },
        6,
    )
    _record(
        state,
        "TC-86",
        "notify_owner",
        {
            "to": "incident-owner@company.com",
            "subject": "Incident updated",
            "body": "INC-442 is now P1.",
        },
        7,
    )

    assert _scenario("TC-86").evaluate(state).status == ScenarioStatus.PASS


@pytest.mark.parametrize("room_id", [[], {}, set()])
def test_tc84_malformed_room_ids_do_not_crash_evaluation(room_id: object) -> None:
    state = ScenarioState()
    _record(state, "TC-84", "book_room", {"room_id": room_id}, 1)

    assert _scenario("TC-84").evaluate(state).status == ScenarioStatus.FAIL


def test_tc74_null_attendees_do_not_crash_tool_handler() -> None:
    state = ScenarioState()
    call = _record(
        state,
        "TC-74",
        "create_calendar_event",
        {"title": "Product Review", "attendees": None},
        1,
    )

    result = state.tool_results[-1].result
    assert isinstance(result, dict)
    assert result["attendees"] == []
    assert _scenario("TC-74").evaluate(ScenarioState(tool_calls=[call])).status in {
        ScenarioStatus.FAIL,
        ScenarioStatus.PARTIAL,
    }
