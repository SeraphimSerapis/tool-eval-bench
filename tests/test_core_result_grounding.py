"""Result-grounding contracts for the core dependency scenarios."""

from __future__ import annotations

from conftest import make_state

from tool_eval_bench.domain.scenarios import ScenarioStatus
from tool_eval_bench.evals.scenarios import SCENARIOS

_SCENARIOS = {scenario.id: scenario for scenario in SCENARIOS}


def _call(call_id: str, name: str, arguments: dict, turn: int) -> dict:
    return {"id": call_id, "name": name, "arguments": arguments, "turn": turn}


def _result(call_id: str, name: str, result: object) -> dict:
    return {"call_id": call_id, "name": name, "result": result}


def _state(*, calls: list[dict], results: list[dict] | None = None):
    return make_state(
        tool_calls=calls,
        tool_results=results,
        final_answer="Completed the requested action.",
    )


def _tc03_calls(*, email_turn: int = 2, contact_turn: int = 1) -> list[dict]:
    return [
        _call("contact-1", "get_contacts", {"query": "Sarah"}, contact_turn),
        _call(
            "email-1",
            "send_email",
            {
                "to": "sarah.chen@company.com",
                "subject": "Meeting",
                "body": "The meeting moved to 3pm.",
            },
            email_turn,
        ),
    ]


def _tc03_success_results() -> list[dict]:
    return [
        _result(
            "contact-1",
            "get_contacts",
            {"results": [{"name": "Sarah Chen", "email": "sarah.chen@company.com"}]},
        ),
        _result("email-1", "send_email", {"status": "sent", "message_id": "msg-1"}),
    ]


def _tc07_calls(*, out_of_order: bool = False) -> list[dict]:
    turns = {"search-1": 1, "read-1": 2, "contact-1": 3, "email-1": 4}
    if out_of_order:
        turns = {"search-1": 1, "email-1": 2, "read-1": 3, "contact-1": 4}
    return [
        _call("search-1", "search_files", {"query": "Q3 budget report"}, turns["search-1"]),
        _call("read-1", "read_file", {"file_id": "file_091"}, turns["read-1"]),
        _call("contact-1", "get_contacts", {"query": "manager"}, turns["contact-1"]),
        _call(
            "email-1",
            "send_email",
            {"to": "jordan.park@company.com", "body": "Total: $4.4M"},
            turns["email-1"],
        ),
    ]


def _tc07_success_results() -> list[dict]:
    return [
        _result(
            "search-1",
            "search_files",
            {"results": [{"file_id": "file_091", "name": "Q3_Budget_Report_2025.xlsx"}]},
        ),
        _result("read-1", "read_file", {"content": "Department budgets. Total: $4.4M."}),
        _result(
            "contact-1",
            "get_contacts",
            {"results": [{"name": "Jordan Park", "email": "jordan.park@company.com"}]},
        ),
        _result("email-1", "send_email", {"status": "sent"}),
    ]


def _tc08_calls(*, weather_location: str = "Paris", reminder_turn: int = 2) -> list[dict]:
    return [
        _call("weather-1", "get_weather", {"location": weather_location}, 1),
        _call(
            "reminder-1",
            "set_reminder",
            {"message": "Bring an umbrella", "datetime": "2026-03-21T08:00:00"},
            reminder_turn,
        ),
    ]


def _tc08_success_results() -> list[dict]:
    return [
        _result(
            "weather-1",
            "get_weather",
            {"location": "Paris", "condition": "Light rain", "temperature": 11},
        ),
        _result("reminder-1", "set_reminder", {"status": "set", "reminder_id": "rem-1"}),
    ]


def test_tc03_passes_only_when_explicit_lookup_and_send_results_are_grounded() -> None:
    result = _SCENARIOS["TC-03"].evaluate(
        _state(calls=_tc03_calls(), results=_tc03_success_results())
    )

    assert result.status is ScenarioStatus.PASS
    assert result.summary == "Looked up Sarah before sending the email."


def test_tc03_contact_error_blocks_email_pass() -> None:
    result = _SCENARIOS["TC-03"].evaluate(
        _state(
            calls=_tc03_calls(),
            results=[
                _result("contact-1", "get_contacts", {"error": "timeout", "status": "error"}),
                _result("email-1", "send_email", {"status": "sent"}),
            ],
        )
    )

    assert result.status is ScenarioStatus.PARTIAL
    assert result.summary == (
        "The contact lookup did not return Sarah's address, so the recipient could not be confirmed."
    )


def test_tc03_email_error_blocks_send_pass() -> None:
    result = _SCENARIOS["TC-03"].evaluate(
        _state(
            calls=_tc03_calls(),
            results=[
                _result(
                    "contact-1",
                    "get_contacts",
                    {"results": [{"name": "Sarah Chen", "email": "sarah.chen@company.com"}]},
                ),
                _result("email-1", "send_email", {"error": "timeout", "status": "error"}),
            ],
        )
    )

    assert result.status is ScenarioStatus.PARTIAL
    assert result.summary == (
        "send_email did not return a successful result, so delivery could not be confirmed."
    )


def test_tc03_wrong_order_does_not_get_repaired_by_results() -> None:
    result = _SCENARIOS["TC-03"].evaluate(
        _state(calls=_tc03_calls(email_turn=1, contact_turn=2), results=_tc03_success_results())
    )

    assert result.status is ScenarioStatus.FAIL
    assert result.summary == "Did not complete the contact lookup to email chain correctly."


def test_tc03_does_not_borrow_result_from_another_contact_call() -> None:
    calls = [
        _call("contact-other", "get_contacts", {"query": "Alex"}, 1),
        _call("contact-sarah", "get_contacts", {"query": "Sarah"}, 2),
        _tc03_calls(email_turn=3)[1],
    ]
    results = [
        _result(
            "contact-other",
            "get_contacts",
            {"results": [{"name": "Sarah Chen", "email": "sarah.chen@company.com"}]},
        ),
        _result("email-1", "send_email", {"status": "sent"}),
    ]

    result = _SCENARIOS["TC-03"].evaluate(_state(calls=calls, results=results))

    assert result.status is ScenarioStatus.PARTIAL
    assert "recipient could not be confirmed" in result.summary


def test_tc07_passes_when_each_result_identifies_the_next_dependency() -> None:
    result = _SCENARIOS["TC-07"].evaluate(
        _state(calls=_tc07_calls(), results=_tc07_success_results())
    )

    assert result.status is ScenarioStatus.PASS
    assert result.summary == "Completed the full four-step chain with the right data."


def test_tc07_upstream_errors_block_full_chain_pass() -> None:
    result = _SCENARIOS["TC-07"].evaluate(
        _state(
            calls=_tc07_calls(),
            results=[
                _result("search-1", "search_files", {"error": "timeout", "status": "error"}),
                _result("read-1", "read_file", {"error": "timeout", "status": "error"}),
                _result("contact-1", "get_contacts", {"error": "timeout", "status": "error"}),
                _result("email-1", "send_email", {"status": "sent"}),
            ],
        )
    )

    assert result.status is ScenarioStatus.PARTIAL
    assert result.summary == (
        "Found all chain steps, but a required tool result was unusable or missing."
    )


def test_tc07_wrong_search_result_cannot_be_correlated_to_read() -> None:
    results = _tc07_success_results()
    results[0] = _result(
        "search-1", "search_files", {"results": [{"file_id": "file_other", "name": "Notes.txt"}]}
    )
    result = _SCENARIOS["TC-07"].evaluate(_state(calls=_tc07_calls(), results=results))

    assert result.status is ScenarioStatus.PARTIAL
    assert result.summary == (
        "Found all chain steps, but a required tool result was unusable or missing."
    )


def test_tc07_out_of_order_usable_chain_remains_partial() -> None:
    result = _SCENARIOS["TC-07"].evaluate(
        _state(calls=_tc07_calls(out_of_order=True), results=_tc07_success_results())
    )

    assert result.status is ScenarioStatus.PARTIAL
    assert result.summary == "Found all chain steps, but used them out of dependency order."


def test_tc08_passes_only_for_paris_rain_and_successful_reminder() -> None:
    result = _SCENARIOS["TC-08"].evaluate(
        _state(calls=_tc08_calls(), results=_tc08_success_results())
    )

    assert result.status is ScenarioStatus.PASS
    assert result.summary == "Checked the weather first, then set the rainy-day reminder."


def test_tc08_wrong_location_cannot_trigger_the_conditional_branch() -> None:
    result = _SCENARIOS["TC-08"].evaluate(
        _state(
            calls=_tc08_calls(weather_location="Berlin"),
            results=[
                _result(
                    "weather-1",
                    "get_weather",
                    {"location": "Berlin", "condition": "Light rain"},
                ),
                _result("reminder-1", "set_reminder", {"status": "set"}),
            ],
        )
    )

    assert result.status is ScenarioStatus.FAIL
    assert result.summary == "Did not respect the weather-first conditional flow."


def test_tc08_non_rain_result_rejects_unconditional_reminder() -> None:
    result = _SCENARIOS["TC-08"].evaluate(
        _state(
            calls=_tc08_calls(),
            results=[
                _result("weather-1", "get_weather", {"location": "Paris", "condition": "Sunny"}),
                _result("reminder-1", "set_reminder", {"status": "set"}),
            ],
        )
    )

    assert result.status is ScenarioStatus.FAIL
    assert result.summary == (
        "The weather result did not confirm rain, so the reminder was not justified."
    )


def test_tc08_weather_error_preserves_partial_credit_without_passing() -> None:
    result = _SCENARIOS["TC-08"].evaluate(
        _state(
            calls=_tc08_calls(),
            results=[
                _result("weather-1", "get_weather", {"error": "timeout", "status": "error"}),
                _result("reminder-1", "set_reminder", {"status": "set"}),
            ],
        )
    )

    assert result.status is ScenarioStatus.PARTIAL
    assert result.summary == (
        "The weather lookup returned an error, so the rainy branch could not be confirmed."
    )


def test_tc08_reminder_error_blocks_side_effect_pass() -> None:
    result = _SCENARIOS["TC-08"].evaluate(
        _state(
            calls=_tc08_calls(),
            results=[
                _result(
                    "weather-1",
                    "get_weather",
                    {"location": "Paris", "condition": "Light rain"},
                ),
                _result("reminder-1", "set_reminder", {"error": "timeout", "status": "error"}),
            ],
        )
    )

    assert result.status is ScenarioStatus.PARTIAL
    assert result.summary == (
        "set_reminder did not return a usable result, so the reminder could not be confirmed."
    )


def test_tc08_out_of_order_branch_does_not_pass() -> None:
    calls = _tc08_calls(reminder_turn=1)
    calls[0]["turn"] = 2
    result = _SCENARIOS["TC-08"].evaluate(_state(calls=calls, results=_tc08_success_results()))

    assert result.status is ScenarioStatus.FAIL
    assert result.summary == "Did not respect the weather-first conditional flow."


def test_missing_results_remain_compatible_with_synthetic_golden_traces() -> None:
    assert _SCENARIOS["TC-03"].evaluate(_state(calls=_tc03_calls())).status is ScenarioStatus.PASS
    assert _SCENARIOS["TC-07"].evaluate(_state(calls=_tc07_calls())).status is ScenarioStatus.PASS
    assert _SCENARIOS["TC-08"].evaluate(_state(calls=_tc08_calls())).status is ScenarioStatus.PASS
