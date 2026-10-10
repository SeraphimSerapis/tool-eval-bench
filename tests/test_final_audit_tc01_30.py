"""Adversarial final-audit contracts for the first thirty scenarios.

The golden traces cover ordinary behavior.  These cases target the failure
modes that can otherwise look successful in a report: fabricated values,
unusable mutation results, wrong dependency provenance, negated arguments, and
duplicate or unrelated actions.
"""

from __future__ import annotations

import pytest
from conftest import make_state

from tool_eval_bench.domain.scenarios import ScenarioStatus
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

_SCENARIOS = {scenario.id: scenario for scenario in ALL_SCENARIOS_WITH_HARDMODE}


def _evaluate(scenario_id: str, **kwargs):
    return _SCENARIOS[scenario_id].evaluate(make_state(**kwargs))


def test_tc01_does_not_accept_fabricated_weather_after_wrong_result() -> None:
    result = _evaluate(
        "TC-01",
        tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
        tool_results=[
            {
                "name": "get_weather",
                "call_id": "call_0",
                "result": {"location": "Berlin", "temperature": -4, "condition": "Snow"},
            }
        ],
        final_answer="Berlin is 8°C and overcast.",
    )
    assert result.status is ScenarioStatus.FAIL


def test_tc02_does_not_accept_fabricated_stock_price() -> None:
    result = _evaluate(
        "TC-02",
        tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}],
        tool_results=[
            {
                "name": "get_stock_price",
                "call_id": "call_0",
                "result": {"ticker": "AAPL", "price": 1.0},
            }
        ],
        final_answer="AAPL is trading at $187.42.",
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc03_negated_email_body_is_not_a_completed_action() -> None:
    result = _evaluate(
        "TC-03",
        tool_calls=[
            {"name": "get_contacts", "arguments": {"query": "Sarah"}, "turn": 1},
            {
                "name": "send_email",
                "arguments": {
                    "to": "sarah.chen@company.com",
                    "subject": "Meeting",
                    "body": "The meeting has not moved to 3pm.",
                },
                "turn": 2,
            },
        ],
    )
    assert result.status is not ScenarioStatus.PASS


def test_tc04_extra_tool_call_cannot_hide_an_unnecessary_action() -> None:
    result = _evaluate(
        "TC-04",
        tool_calls=[
            {
                "name": "get_weather",
                "arguments": {"location": "Tokyo", "units": "fahrenheit"},
            },
            {"name": "calculator", "arguments": {"expression": "1 + 1"}},
        ],
        final_answer="Tokyo is 64°F.",
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc05_failed_calendar_mutation_cannot_score_pass() -> None:
    result = _evaluate(
        "TC-05",
        tool_calls=[
            {
                "name": "create_calendar_event",
                "arguments": {
                    "title": "Team Standup",
                    "date": "2026-03-23",
                    "time": "09:30",
                    "duration_minutes": 30,
                    "attendees": ["Alex", "Jamie"],
                },
            }
        ],
        tool_results=[
            {
                "name": "create_calendar_event",
                "call_id": "call_0",
                "result": {"error": "calendar unavailable"},
            }
        ],
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc06_accepts_a_successful_retry_for_one_language() -> None:
    result = _evaluate(
        "TC-06",
        tool_calls=[
            {
                "name": "translate_text",
                "arguments": {
                    "text": "Where is the nearest hospital?",
                    "source_language": "English",
                    "target_language": "Spanish",
                },
                "turn": 1,
            },
            {
                "name": "translate_text",
                "arguments": {
                    "text": "Where is the nearest hospital?",
                    "source_language": "English",
                    "target_language": "Spanish",
                },
                "turn": 2,
            },
            {
                "name": "translate_text",
                "arguments": {
                    "text": "Where is the nearest hospital?",
                    "source_language": "English",
                    "target_language": "Japanese",
                },
                "turn": 3,
            },
        ],
        tool_results=[
            {
                "name": "translate_text",
                "call_id": "call_0",
                "result": {"error": "temporary failure"},
            },
            {
                "name": "translate_text",
                "call_id": "call_1",
                "result": {"translated": "¿Dónde está el hospital más cercano?"},
            },
            {
                "name": "translate_text",
                "call_id": "call_2",
                "result": {"translated": "最寄りの病院はどこですか？"},
            },
        ],
        final_answer="Spanish: ¿Dónde está el hospital más cercano? Japanese: 最寄りの病院はどこですか？",
    )
    assert result.status is ScenarioStatus.PASS


def test_tc07_requires_correlated_results_before_email_side_effect() -> None:
    result = _evaluate(
        "TC-07",
        tool_calls=[
            {"name": "search_files", "arguments": {"query": "Q3 budget"}, "turn": 1},
            {"name": "read_file", "arguments": {"file_id": "file_091"}, "turn": 2},
            {"name": "get_contacts", "arguments": {"query": "manager"}, "turn": 3},
            {
                "name": "send_email",
                "arguments": {
                    "to": "jordan.park@company.com",
                    "subject": "Budget",
                    "body": "Total is $4.4M",
                },
                "turn": 4,
            },
        ],
        tool_results=[
            {"name": "search_files", "call_id": "call_0", "result": {"results": []}},
            {"name": "read_file", "call_id": "call_1", "result": {"content": "unrelated"}},
            {
                "name": "get_contacts",
                "call_id": "call_2",
                "result": {
                    "results": [{"name": "Jordan Park", "email": "jordan.park@company.com"}]
                },
            },
            {"name": "send_email", "call_id": "call_3", "result": {"status": "sent"}},
        ],
        final_answer="Email sent.",
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc08_does_not_follow_rainy_branch_without_rain_evidence() -> None:
    result = _evaluate(
        "TC-08",
        tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Paris"}, "turn": 1},
            {
                "name": "set_reminder",
                "arguments": {"message": "Bring an umbrella", "datetime": "2026-03-21T08:00:00"},
                "turn": 2,
            },
        ],
        tool_results=[
            {
                "name": "get_weather",
                "call_id": "call_0",
                "result": {"location": "Paris", "condition": "Sunny"},
            },
            {"name": "set_reminder", "call_id": "call_1", "result": {"status": "set"}},
        ],
    )
    assert result.status is ScenarioStatus.FAIL


def test_tc09_requires_actual_values_from_both_tools() -> None:
    result = _evaluate(
        "TC-09",
        tool_calls=[
            {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
            {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
        ],
        tool_results=[
            {
                "name": "get_weather",
                "call_id": "call_0",
                "result": {"location": "London", "temperature": -1},
            },
            {
                "name": "get_stock_price",
                "call_id": "call_1",
                "result": {"ticker": "MSFT", "price": 1.0},
            },
        ],
        final_answer="London is 12°C and MSFT is $412.78.",
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc10_unrelated_1945_mention_is_not_a_history_answer() -> None:
    result = _evaluate("TC-10", final_answer="I have 1945 unread messages.")
    assert result.status is ScenarioStatus.FAIL


def test_tc13_explicit_missing_retry_result_is_not_success() -> None:
    result = _evaluate(
        "TC-13",
        tool_calls=[
            {"name": "search_files", "arguments": {"query": "Johnson proposal"}, "turn": 1},
            {"name": "search_files", "arguments": {"query": "Johnson project"}, "turn": 2},
        ],
        tool_results=[{"name": "search_files", "call_id": "call_0", "result": {"results": []}}],
        final_answer="I could not find it.",
    )
    assert result.status is ScenarioStatus.FAIL


def test_tc14_requires_a_real_aapl_fallback_after_the_error() -> None:
    result = _evaluate(
        "TC-14",
        tool_calls=[
            {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}, "turn": 1},
            {"name": "web_search", "arguments": {"query": "weather"}, "turn": 2},
        ],
        tool_results=[
            {
                "name": "get_stock_price",
                "call_id": "call_0",
                "result": {"error": "rate limit"},
            },
            {
                "name": "web_search",
                "call_id": "call_1",
                "result": {"results": [{"snippet": "London weather"}]},
            },
        ],
        assistant_messages=["The stock service was unavailable. I searched the web. AAPL is $187."],
        final_answer="The stock service was unavailable. I searched the web. AAPL is $187.",
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc15_requires_the_population_to_come_from_search() -> None:
    result = _evaluate(
        "TC-15",
        tool_calls=[
            {"name": "web_search", "arguments": {"query": "population of Iceland"}, "turn": 1},
            {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}, "turn": 2},
        ],
        tool_results=[
            {"name": "web_search", "call_id": "call_0", "result": {"results": []}},
            {"name": "calculator", "call_id": "call_1", "result": {"result": 7450.4}},
        ],
        final_answer="2% of Iceland is 7,450.4.",
    )
    assert result.status is ScenarioStatus.FAIL


def test_tc16_does_not_present_weather_from_a_failed_call() -> None:
    result = _evaluate(
        "TC-16",
        tool_calls=[{"name": "get_weather", "arguments": {"location": "München"}}],
        tool_results=[{"name": "get_weather", "call_id": "call_0", "result": {"error": "down"}}],
        final_answer="Das Wetter in München ist 14 Grad und bewölkt.",
    )
    assert result.status is ScenarioStatus.FAIL


def test_tc17_failed_calendar_mutation_is_not_success() -> None:
    result = _evaluate(
        "TC-17",
        tool_calls=[
            {
                "name": "create_calendar_event",
                "arguments": {
                    "title": "Team Standup",
                    "date": "2026-03-24",
                    "time": "14:00",
                    "timezone": "Europe/Berlin",
                },
            }
        ],
        tool_results=[
            {
                "name": "create_calendar_event",
                "call_id": "call_0",
                "result": {"error": "calendar unavailable"},
            }
        ],
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc18_reversed_translation_and_email_workflow_is_partial() -> None:
    result = _evaluate(
        "TC-18",
        tool_calls=[
            {
                "name": "send_email",
                "arguments": {
                    "to": "hans.mueller@firma.de",
                    "subject": "Meeting",
                    "body": "Der Termin wurde verschoben.",
                },
                "turn": 1,
            },
            {
                "name": "translate_text",
                "arguments": {
                    "text": "The meeting has moved.",
                    "source_language": "English",
                    "target_language": "German",
                },
                "turn": 2,
            },
        ],
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc19_negated_labels_do_not_count_as_classifications() -> None:
    result = _evaluate(
        "TC-19",
        final_answer=(
            "1. not code_help\n2. not scheduling\n3. not billing\n4. not devops\n5. research"
        ),
    )
    assert result.status is not ScenarioStatus.PASS


def test_tc20_requires_the_report_content_before_calculating() -> None:
    result = _evaluate(
        "TC-20",
        tool_calls=[
            {"name": "search_files", "arguments": {"query": "Q3 sales"}, "turn": 1},
            {"name": "read_file", "arguments": {"file_id": "file_q3_sales"}, "turn": 2},
        ],
        tool_results=[
            {"name": "search_files", "call_id": "call_0", "result": {"results": []}},
            {"name": "read_file", "call_id": "call_1", "result": {"content": "unrelated"}},
        ],
        final_answer="The average is $141,440.",
    )
    assert result.status is ScenarioStatus.PARTIAL


_TC21_ISSUES = "ISSUES:\nemail: R1\nage: R2\nphone: R3\nstart_date: R4\namount: R5"


def test_tc21_answer_key_matches_the_stated_rules() -> None:
    # The grader's answer key is only as good as the payload: check every field
    # against the rules as the prompt states them, independently of the grader.
    import datetime
    import re as _re

    from tool_eval_bench.evals.scenarios.extended.tc21 import _INVALID, _PAYLOAD

    def breaks(field: str, value: object) -> int | None:
        if field.endswith("email"):
            return None if _re.fullmatch(r"[^@]+@[^@.]+(\.[^@.]+)+", str(value)) else 1
        if field.endswith("age"):
            return None if isinstance(value, int) and 0 <= value <= 150 else 2
        if field.endswith("phone"):
            return None if len(_re.sub(r"[\s.()-]", "", str(value))) == 10 else 3
        if field.endswith("date"):
            try:
                datetime.date.fromisoformat(str(value))
            except ValueError:
                return 4
            return None
        return None if isinstance(value, (int, float)) and value >= 0 else 5

    assert {f: r for f, v in _PAYLOAD.items() if (r := breaks(f, v))} == _INVALID
    assert all(str(v) in _SCENARIOS["TC-21"].user_message for v in _PAYLOAD.values())


def test_tc21_credits_correct_diagnoses_whatever_their_wording() -> None:
    # Reported phrasings that the old vocabulary grader scored 4, 3, and 2 out
    # of 5 although each named all five errors: wording above the issue list
    # no longer matters.
    answer = (
        "1. `email` `john@.com` lacks a valid domain label.\n"
        "2. `age` 151 is outside the allowed range of 0 to 150.\n"
        "3. `phone` `555-12` has only `5` digits.\n"
        "4. `start_date` 2023-02-29: 2023 is not a leap year.\n"
        "5. `amount` −50 is less than 0.\n\n" + _TC21_ISSUES
    )
    result = _evaluate("TC-21", final_answer=answer)
    assert result.status is ScenarioStatus.PASS


def test_tc21_wrong_rule_earns_nothing() -> None:
    answer = _TC21_ISSUES.replace("email: R1", "email: R3")
    result = _evaluate("TC-21", final_answer=answer)
    assert result.status is ScenarioStatus.PARTIAL
    assert "4/5" in result.summary and "wrong rule for email" in result.summary


def test_tc21_conflicting_rules_for_one_field_earn_nothing() -> None:
    result = _evaluate("TC-21", final_answer=_TC21_ISSUES + "\nemail: R2")
    assert result.status is ScenarioStatus.PARTIAL
    assert "wrong rule for email" in result.summary


def test_tc21_a_false_flag_costs_the_pass() -> None:
    result = _evaluate("TC-21", final_answer=_TC21_ISSUES + "\nguardian_age: R2")
    assert result.status is ScenarioStatus.PARTIAL
    assert "flagged valid guardian_age" in result.summary


def test_tc21_flagging_every_field_fails() -> None:
    answer = _TC21_ISSUES + (
        "\nalt_email: R1\nguardian_age: R2\noffice_phone: R3\nrenewal_date: R4\ndiscount: R5"
    )
    result = _evaluate("TC-21", final_answer=answer)
    assert result.status is ScenarioStatus.FAIL


def test_tc21_reads_only_the_last_issues_section() -> None:
    # Reasoning above the section may cite rules for valid fields, and a draft
    # section can be corrected by a later one.
    answer = (
        "guardian_age: 150 satisfies R2. office_phone: R3 holds after stripping.\n"
        "ISSUES:\nemail: R1\nrenewal_date: R4\n\n"
        "Correction, 2024 is a leap year.\n\n" + _TC21_ISSUES
    )
    result = _evaluate("TC-21", final_answer=answer)
    assert result.status is ScenarioStatus.PASS


def test_tc21_prose_without_rule_ids_flags_nothing() -> None:
    result = _evaluate(
        "TC-21",
        final_answer=(
            "The email is invalid, the age is too high, the phone is too short, the start "
            "date does not exist, and the amount is negative."
        ),
    )
    assert result.status is ScenarioStatus.FAIL
    assert "0/5" in result.summary


@pytest.mark.parametrize(
    ("line", "flagged"),
    [
        ("- **email** (`john@.com`): R1", {"email": {1}}),
        ("| 1 | `alt_email` | `ops@mail.example.org` | R1 |", {"alt_email": {1}}),
        ("Guardian age: R2", {"guardian_age": {2}}),
        ("StartDate breaks rule 4", {"start_date": {4}}),
        ("start-date — R4 (renewal_date is fine)", {"start_date": {4}}),
        ("R5: amount", {"amount": {5}}),
        ("amount: R50", {}),
        ("email: invalid domain", {}),
    ],
)
def test_tc21_reads_common_issue_line_shapes(line: str, flagged: dict[str, set[int]]) -> None:
    from tool_eval_bench.evals.scenarios.extended.tc21 import _flagged_rules

    assert _flagged_rules(line) == flagged


def test_tc22_wrong_recorded_weather_cannot_support_canonical_json() -> None:
    result = _evaluate(
        "TC-22",
        tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
        tool_results=[
            {
                "name": "get_weather",
                "call_id": "call_0",
                "result": {"temperature": -4, "condition": "Snow", "humidity": 70},
            }
        ],
        final_answer='{"temp": 7, "condition": "Overcast", "humidity": 82}',
    )
    assert result.status is not ScenarioStatus.PASS


def test_tc24_parallel_read_cannot_depend_on_search_result() -> None:
    result = _evaluate(
        "TC-24",
        tool_calls=[
            {"name": "search_files", "arguments": {"query": "Q3 report"}, "turn": 1},
            {"name": "read_file", "arguments": {"file_id": "file_q3_report"}, "turn": 1},
        ],
        final_answer="$4,250,000",
    )
    assert result.status is not ScenarioStatus.PASS


def test_tc24_short_prose_still_violates_number_only_contract() -> None:
    result = _evaluate(
        "TC-24",
        tool_calls=[
            {"name": "search_files", "arguments": {"query": "Q3 report"}, "turn": 1},
            {"name": "read_file", "arguments": {"file_id": "file_q3_report"}, "turn": 2},
        ],
        final_answer="Revenue: $4,250,000",
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc25_same_turn_reminder_is_not_grounded_in_weather() -> None:
    result = _evaluate(
        "TC-25",
        tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Berlin"}, "turn": 1},
            {"name": "set_reminder", "arguments": {"message": "Bring a coat"}, "turn": 1},
        ],
    )
    assert result.status is ScenarioStatus.FAIL


def test_tc25_failed_reminder_cannot_score_pass() -> None:
    result = _evaluate(
        "TC-25",
        tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Berlin"}, "turn": 1},
            {"name": "set_reminder", "arguments": {"message": "Bring a coat"}, "turn": 2},
        ],
        tool_results=[
            {
                "name": "get_weather",
                "call_id": "call_0",
                "result": {"temperature": 5, "condition": "Clear"},
            },
            {
                "name": "set_reminder",
                "call_id": "call_1",
                "result": {"error": "reminder service unavailable"},
            },
        ],
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc26_wrong_event_cannot_pass_on_honest_attendee_answer() -> None:
    result = _evaluate(
        "TC-26",
        tool_calls=[
            {
                "name": "create_calendar_event",
                "arguments": {
                    "title": "Sales Review",
                    "date": "2026-03-21",
                    "time": "15:00",
                },
            }
        ],
        final_answer="No attendees were specified.",
    )
    assert result.status is ScenarioStatus.FAIL


def test_tc27_fabricated_temperatures_are_not_perfect() -> None:
    result = _evaluate(
        "TC-27",
        tool_calls=[
            {
                "name": "get_weather",
                "arguments": {"location": "London", "units": "celsius"},
            },
            {
                "name": "get_weather",
                "arguments": {"location": "London", "units": "fahrenheit"},
            },
        ],
        tool_results=[
            {
                "name": "get_weather",
                "call_id": "call_0",
                "result": {"units": "celsius", "temperature": -2},
            },
            {
                "name": "get_weather",
                "call_id": "call_1",
                "result": {"units": "fahrenheit", "temperature": 28},
            },
        ],
        final_answer="London is 10°C and 50°F.",
    )
    assert result.status is not ScenarioStatus.PASS


def test_tc28_failed_read_cannot_support_a_guessed_fix() -> None:
    result = _evaluate(
        "TC-28",
        tool_calls=[{"name": "read_file", "arguments": {"file_id": "config_yaml"}}],
        tool_results=[
            {
                "name": "read_file",
                "call_id": "call_0",
                "result": {"error": "file unavailable"},
            }
        ],
        final_answer="Replace localhsot with localhost.",
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc29_unrelated_tool_prevents_a_perfect_score() -> None:
    result = _evaluate(
        "TC-29",
        tool_calls=[{"name": "send_email", "arguments": {"to": "someone@example.com"}}],
        final_answer="It produces [0, 1, 4, 9, 16].",
    )
    assert result.status is not ScenarioStatus.PASS


def test_tc30_same_turn_branch_is_not_grounded_in_calculation() -> None:
    result = _evaluate(
        "TC-30",
        tool_calls=[
            {"name": "run_code", "arguments": {"code": "print(2+2)"}, "turn": 1},
            {"name": "run_code", "arguments": {"code": "print('correct')"}, "turn": 1},
        ],
    )
    assert result.status is ScenarioStatus.FAIL


def test_tc30_branch_only_skips_required_calculation() -> None:
    result = _evaluate(
        "TC-30",
        tool_calls=[{"name": "run_code", "arguments": {"code": "print('correct')"}}],
    )
    assert result.status is ScenarioStatus.PARTIAL
