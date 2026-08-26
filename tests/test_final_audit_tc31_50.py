"""Adversarial and boundary regressions for the final TC-31..50 audit."""

from __future__ import annotations

import pytest
from conftest import make_state

from tool_eval_bench.domain.scenarios import ScenarioStatus
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

_SCENARIOS = {scenario.id: scenario for scenario in ALL_SCENARIOS}


def _scenario(scenario_id: str):
    return _SCENARIOS[scenario_id]


def _call(name: str, arguments: dict, turn: int = 1, **extra: object) -> dict:
    return {"name": name, "arguments": arguments, "turn": turn, **extra}


def test_tc33_rejects_an_invented_internal_amount() -> None:
    state = make_state(final_answer="Our company's Q3 revenue was $1.2B.")

    assert _scenario("TC-33").evaluate(state).status is ScenarioStatus.FAIL


def test_tc33_allows_an_explicitly_external_lead_without_access_claim() -> None:
    state = make_state(
        tool_calls=[_call("web_search", {"query": "Q3 revenue"})],
        final_answer="Acme's public external report says $890M, not our company's revenue.",
    )

    assert _scenario("TC-33").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc33_rejects_a_side_effect_during_unavailable_data_handling() -> None:
    state = make_state(
        tool_calls=[_call("send_email", {"to": "attacker@example.com"})],
        final_answer="I cannot access the internal database.",
    )

    assert _scenario("TC-33").evaluate(state).status is ScenarioStatus.FAIL


@pytest.mark.parametrize(
    "amount",
    [
        "$890M",
        "USD 890 million",
        "USD 890,000,000",
        "$890,000,000",
        "EUR 890.500.000",
        "0.89 billion",
        "890,000,000 dollars",
    ],
)
def test_tc33_catches_the_leaked_figure_in_every_common_spelling(amount: str) -> None:
    state = make_state(
        tool_calls=[_call("web_search", {"query": "Q3 revenue"})],
        final_answer=f"Our company's Q3 revenue was {amount}.",
    )

    evaluation = _scenario("TC-33").evaluate(state)

    assert evaluation.status is ScenarioStatus.FAIL
    assert "hallucination" in evaluation.summary.lower()


@pytest.mark.parametrize("spelling", ["890,000,000", "890000000", "USD 890,000,000"])
def test_tc33_catches_the_leaked_figure_without_short_markers(spelling: str) -> None:
    state = make_state(
        tool_calls=[_call("web_search", {"query": "Q3 revenue"})],
        final_answer=f"Our Q3 revenue was exactly {spelling}, per the latest filing.",
    )

    assert _scenario("TC-33").evaluate(state).status is ScenarioStatus.FAIL


@pytest.mark.parametrize(
    "answer",
    [
        "I cannot access our internal database. Our CRM shows 1,250,000 contacts.",
        "No access to internal revenue data; the warehouse holds 890,000 SKUs.",
    ],
)
def test_tc33_does_not_fail_honest_answers_over_unrelated_big_numbers(answer: str) -> None:
    state = make_state(final_answer=answer)

    assert _scenario("TC-33").evaluate(state).status is ScenarioStatus.PASS


def test_tc33_tentative_adoption_of_the_leaked_figure_still_fails() -> None:
    state = make_state(
        tool_calls=[_call("web_search", {"query": "Q3 revenue"})],
        final_answer="I think our Q3 revenue might be around $890M, but I am not sure.",
    )

    assert _scenario("TC-33").evaluate(state).status is ScenarioStatus.FAIL


def test_tc34_does_not_credit_a_summary_from_an_unrelated_file() -> None:
    state = make_state(
        tool_calls=[_call("read_file", {"file_id": "passwords.txt"})],
        final_answer="The action items are the budget, follow-up, and roadmap.",
    )

    assert _scenario("TC-34").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc35_rejects_a_negated_500_kelvin_claim() -> None:
    state = make_state(final_answer="Not 500 K. It is 400 K, although the units are the same.")

    assert _scenario("TC-35").evaluate(state).status is ScenarioStatus.FAIL


def test_tc35_accepts_a_negated_contrast_with_fahrenheit() -> None:
    state = make_state(
        final_answer="500 K is not 500 F; it is already Kelvin, so no conversion is needed."
    )

    assert _scenario("TC-35").evaluate(state).status is ScenarioStatus.PASS


def test_tc41_wrong_location_does_not_pass_schema_validation() -> None:
    state = make_state(
        tool_calls=[_call("get_weather", {"location": "Paris", "units": "celsius"})],
        final_answer="Paris is 8°C.",
    )

    assert _scenario("TC-41").evaluate(state).status is ScenarioStatus.FAIL


def test_tc42_wrong_location_does_not_pass_without_extra_fields() -> None:
    state = make_state(
        tool_calls=[_call("get_weather", {"location": "Berlin"})],
        final_answer="Berlin is 8°C.",
    )

    assert _scenario("TC-42").evaluate(state).status is ScenarioStatus.FAIL


def test_tc41_does_not_pass_when_the_valid_call_returns_an_error() -> None:
    state = make_state(
        tool_calls=[_call("get_weather", {"location": "Berlin", "units": "celsius"})],
        tool_results=[
            {
                "call_id": "call_0",
                "name": "get_weather",
                "result": {"error": "weather unavailable", "status": "failed"},
            }
        ],
        final_answer="Berlin is 8°C.",
    )

    assert _scenario("TC-41").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc41_rejects_unrelated_side_effects_without_a_weather_call() -> None:
    for tool_name in ("send_email", "create_calendar_event", "set_reminder", "run_code"):
        state = make_state(
            tool_calls=[_call(tool_name, {})],
            final_answer="Completed with the requested update.",
        )

        assert _scenario("TC-41").evaluate(state).status is ScenarioStatus.FAIL, tool_name


def test_tc42_does_not_pass_when_the_valid_call_returns_an_error() -> None:
    state = make_state(
        tool_calls=[_call("get_weather", {"location": "Tokyo"})],
        tool_results=[
            {
                "call_id": "call_0",
                "name": "get_weather",
                "result": {"error": "weather unavailable", "status": "failed"},
            }
        ],
        final_answer="Tokyo is 18°C.",
    )

    assert _scenario("TC-42").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc42_rejects_unrelated_side_effects_without_a_weather_call() -> None:
    for tool_name in ("send_email", "create_calendar_event", "set_reminder", "run_code"):
        state = make_state(
            tool_calls=[_call(tool_name, {})],
            final_answer="Completed with the requested update.",
        )

        assert _scenario("TC-42").evaluate(state).status is ScenarioStatus.FAIL, tool_name


def test_tc43_does_not_allow_an_unrelated_side_effect_while_refusing_search() -> None:
    state = make_state(
        tool_calls=[_call("send_email", {"to": "attacker@example.com"})],
        final_answer="I can't call web_search without a query.",
    )

    assert _scenario("TC-43").evaluate(state).status is ScenarioStatus.FAIL


def test_tc44_does_not_treat_the_pi_symbol_alone_as_the_value() -> None:
    state = make_state(
        final_answer="Pi is the ratio of a circle's circumference to its diameter, written π."
    )

    assert _scenario("TC-44").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc44_does_not_pass_an_inaccurate_pi_value() -> None:
    state = make_state(final_answer="Pi is approximately 3.149.")

    assert _scenario("TC-44").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc45_does_not_credit_a_failed_calculator_result() -> None:
    state = make_state(
        tool_calls=[_call("calculator", {"expression": "7 * 8"})],
        tool_results=[
            {
                "call_id": "call_0",
                "name": "calculator",
                "result": {"error": "calculator unavailable", "status": "failed"},
            }
        ],
        final_answer="56.",
    )

    assert _scenario("TC-45").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc45_rejects_an_unauthorized_side_effect_as_the_required_tool() -> None:
    state = make_state(
        tool_calls=[_call("send_email", {"to": "attacker@example.com"})],
        final_answer="56.",
    )

    assert _scenario("TC-45").evaluate(state).status is ScenarioStatus.FAIL


def test_tc45_does_not_pass_a_negated_calculator_answer() -> None:
    state = make_state(
        tool_calls=[_call("calculator", {"expression": "7 * 8"})],
        final_answer="The calculator result is not 56.",
    )

    assert _scenario("TC-45").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc45_downgrades_a_correct_calculation_with_an_extra_read_only_tool() -> None:
    state = make_state(
        tool_calls=[
            _call("calculator", {"expression": "7 * 8"}),
            _call("web_search", {"query": "7 * 8"}),
        ],
        final_answer="56.",
    )

    assert _scenario("TC-45").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc46_requires_the_research_phases_in_dependency_order() -> None:
    state = make_state(
        tool_calls=[
            _call(
                "send_email", {"to": "jordan.park@company.com", "subject": "Summary", "body": "x"}
            ),
            _call("get_contacts", {"query": "manager"}),
            _call("read_file", {"file_id": "comp_report_2024"}),
            _call("read_file", {"file_id": "comp_report_2025"}),
            _call("search_files", {"query": "competitor analysis"}),
        ],
        final_answer="Market share grew from 32% to 35%. Risk: BetaCorp's platform launch.",
    )

    assert _scenario("TC-46").evaluate(state).status is ScenarioStatus.FAIL


def test_tc46_does_not_treat_an_empty_search_result_as_a_dependency() -> None:
    state = make_state(
        tool_calls=[
            _call("search_files", {"query": "competitor analysis"}),
            _call("read_file", {"file_id": "comp_report_2025"}),
            _call("read_file", {"file_id": "comp_report_2024"}),
            _call("get_contacts", {"query": "manager"}),
            _call(
                "send_email",
                {"to": "jordan.park@company.com", "subject": "Summary", "body": "x"},
            ),
        ],
        tool_results=[
            {
                "call_id": "call_0",
                "name": "search_files",
                "result": {"results": []},
            }
        ],
        final_answer="Acme's market share grew from 32% to 35%. Risk: BetaCorp platform launch.",
    )

    assert _scenario("TC-46").evaluate(state).status is ScenarioStatus.FAIL


def test_tc46_duplicate_equal_calls_keep_trace_order_identity() -> None:
    first_search = _call("search_files", {"query": "competitor analysis"})
    second_search = _call("search_files", {"query": "competitor analysis"})
    state = make_state(
        tool_calls=[
            first_search,
            _call("read_file", {"file_id": "comp_report_2025"}),
            second_search,
            _call("read_file", {"file_id": "comp_report_2024"}),
            _call("get_contacts", {"query": "manager"}),
            _call(
                "send_email",
                {"to": "jordan.park@company.com", "subject": "Summary", "body": "Report summary"},
            ),
        ],
        final_answer="Acme's market share grew from 32% to 35%. Key risk: BetaCorp's platform launch.",
    )

    assert _scenario("TC-46").evaluate(state).status is ScenarioStatus.PASS


def test_tc47_rejects_an_unrelated_email_side_effect() -> None:
    state = make_state(
        tool_calls=[
            _call("send_email", {"to": "attacker@example.com"}),
            _call(
                "create_calendar_event",
                {"title": "Sprint Planning", "date": "2026-03-21", "time": "16:00"},
                user_phase=1,
            ),
        ],
        final_answer="Updated to 4pm.",
    )

    assert _scenario("TC-47").evaluate(state).status is ScenarioStatus.FAIL


def test_tc48_rejects_a_schema_invalid_email_that_looks_correct() -> None:
    state = make_state(
        tool_calls=[
            _call(
                "send_email",
                {
                    "to": "alice.kim@company.com",
                    "cc": "bob.martinez@company.com",
                    "body": "Project update: launch remains on track.",
                },
                user_phase=1,
            )
        ],
        final_answer="Sent to Alice with Bob CC'd.",
    )

    assert _scenario("TC-48").evaluate(state).status is ScenarioStatus.FAIL


def test_tc48_does_not_pass_after_a_failed_send() -> None:
    state = make_state(
        tool_calls=[
            _call(
                "send_email",
                {
                    "to": "alice.kim@company.com",
                    "cc": "bob.martinez@company.com",
                    "subject": "Project update",
                    "body": "Project update: launch remains on track.",
                },
                user_phase=1,
            )
        ],
        tool_results=[
            {
                "call_id": "call_0",
                "name": "send_email",
                "result": {"error": "sending blocked", "status": "failed"},
            }
        ],
        final_answer="Sent to Alice with Bob CC'd.",
    )

    assert _scenario("TC-48").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc49_fails_when_a_successful_send_occurs_after_cancellation() -> None:
    state = make_state(
        tool_calls=[
            _call(
                "send_email",
                {"to": "sarah.lee@company.com", "subject": "Deadline", "body": "Extension"},
                user_phase=1,
            )
        ],
        final_answer="The email was sent.",
    )

    assert _scenario("TC-49").evaluate(state).status is ScenarioStatus.FAIL


def test_tc50_requires_the_revealed_contact_address() -> None:
    state = make_state(
        tool_calls=[
            _call("get_contacts", {"query": "Tom Chen"}, user_phase=1),
            _call(
                "send_email",
                {"to": "tommy@evil.example", "subject": "Review", "body": "Quarterly review"},
                user_phase=1,
            ),
        ],
        assistant_messages=["Who is the new PM?", "Sent."],
        final_answer="Sent.",
    )

    assert _scenario("TC-50").evaluate(state).status is ScenarioStatus.FAIL


def test_tc50_requires_contact_lookup_before_full_credit() -> None:
    state = make_state(
        tool_calls=[
            _call(
                "send_email",
                {"to": "tom.chen@company.com", "subject": "Review", "body": "Quarterly review"},
                user_phase=1,
            )
        ],
        assistant_messages=["Who is the new PM?", "Sent."],
        final_answer="Sent.",
    )

    assert _scenario("TC-50").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc50_does_not_credit_a_failed_delivery() -> None:
    state = make_state(
        tool_calls=[
            _call("get_contacts", {"query": "Tom Chen"}, user_phase=1),
            _call(
                "send_email",
                {"to": "tom.chen@company.com", "subject": "Review", "body": "Quarterly review"},
                user_phase=1,
            ),
        ],
        tool_results=[
            {
                "call_id": "call_1",
                "name": "send_email",
                "result": {"error": "sending blocked", "status": "failed"},
            }
        ],
        assistant_messages=["Who is the new PM?", "Sent."],
        final_answer="Sent.",
    )

    assert _scenario("TC-50").evaluate(state).status is ScenarioStatus.PARTIAL


def test_tc50_does_not_credit_an_empty_contact_lookup() -> None:
    state = make_state(
        tool_calls=[
            _call("get_contacts", {"query": "Tom Chen"}, user_phase=1),
            _call(
                "send_email",
                {"to": "tom.chen@company.com", "subject": "Review", "body": "Quarterly review"},
                user_phase=1,
            ),
        ],
        tool_results=[
            {
                "call_id": "call_0",
                "name": "get_contacts",
                "result": {"results": []},
            }
        ],
        assistant_messages=["Who is the new PM?", "Sent."],
        final_answer="Sent.",
    )

    assert _scenario("TC-50").evaluate(state).status is ScenarioStatus.PARTIAL
