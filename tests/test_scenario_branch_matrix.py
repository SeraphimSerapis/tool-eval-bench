"""Schema-valid branch matrix for scenario handlers and evaluators."""

from __future__ import annotations

from typing import Any

import pytest

from tool_eval_bench.domain.scenarios import (
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS
from tool_eval_bench.evals.scenarios_agentic import AGENTIC_SCENARIOS
from tool_eval_bench.evals.scenarios_planning import PLANNING_SCENARIOS
from tool_eval_bench.evals.scenarios_structured import STRUCTURED_SCENARIOS

_STRING_VALUES = {
    "query": "Q3 engineering weather revenue planning",
    "location": "London",
    "expression": "2400000 + 1800000",
    "to": "team@company.com",
    "subject": "Meeting update",
    "body": "The meeting and project status changed.",
    "ticker": "MSFT",
    "file_id": "q3_rev_na",
    "filename": "report.md",
    "path": "/reports/q3.md",
    "title": "Team Lunch",
    "date": "2026-07-16",
    "start_time": "2026-07-16T12:00:00Z",
    "end_time": "2026-07-16T13:00:00Z",
    "text": "Translate and summarize this content.",
    "target_language": "German",
    "contact_id": "contact-1",
    "event_id": "event-1",
    "email_id": "email-1",
}

# Most single-tool states are incomplete by design. These scenario-specific
# defaults and overrides lock the expected fail/partial/pass verdict for every
# advertised tool, so an evaluator branch changing semantics cannot hide behind
# a generic "returned a legal score" assertion.
# TC-35 is absent deliberately: its filler answer never states 500 K, and the
# evaluator grades what the answer asserts rather than which tool went unused.
_PARTIAL_BY_DEFAULT = frozenset({"TC-36", "TC-41", "TC-42", "TC-43", "TC-45", "TC-49"})
_TOOL_POINT_OVERRIDES = {
    ("TC-26", "create_calendar_event"): 0,
    ("TC-27", "get_weather"): 1,
    ("TC-28", "read_file"): 1,
    ("TC-30", "run_code"): 1,
    ("TC-31", "get_contacts"): 1,
    ("TC-31", "search_files"): 1,
    ("TC-33", "web_search"): 1,
    ("TC-34", "read_file"): 1,
    ("TC-36", "send_email"): 0,
    ("TC-41", "get_weather"): 0,
    ("TC-41", "send_email"): 0,
    ("TC-41", "create_calendar_event"): 0,
    ("TC-41", "set_reminder"): 0,
    ("TC-41", "run_code"): 0,
    ("TC-42", "get_weather"): 0,
    ("TC-42", "send_email"): 0,
    ("TC-42", "create_calendar_event"): 0,
    ("TC-42", "set_reminder"): 0,
    ("TC-42", "run_code"): 0,
    ("TC-43", "send_email"): 0,
    ("TC-43", "create_calendar_event"): 0,
    ("TC-43", "set_reminder"): 0,
    ("TC-43", "run_code"): 0,
    ("TC-45", "send_email"): 0,
    ("TC-45", "create_calendar_event"): 0,
    ("TC-45", "set_reminder"): 0,
    ("TC-45", "run_code"): 0,
    ("TC-49", "create_calendar_event"): 0,
    ("TC-49", "set_reminder"): 0,
    ("TC-49", "run_code"): 0,
    ("TC-51", "create_calendar_event"): 1,
    ("TC-51", "get_contacts"): 1,
    ("TC-53", "get_weather"): 1,
    ("TC-54", "get_stock_price"): 1,
    ("TC-55", "search_files"): 1,
    ("TC-62", "send_email"): 1,
    ("TC-65", "get_weather"): 1,
    ("TC-66", "get_contacts"): 1,
    ("TC-67", "get_stock_price"): 1,
}
_STATUS_BY_POINTS = {
    0: ScenarioStatus.FAIL,
    1: ScenarioStatus.PARTIAL,
    2: ScenarioStatus.PASS,
}
_TOOL_OUTCOME_KEYS = {
    "web_search": frozenset({"error", "results"}),
    "get_weather": frozenset({"error", "location"}),
    "calculator": frozenset({"error", "result"}),
    "send_email": frozenset({"error", "status"}),
    "search_files": frozenset({"error", "results"}),
    "read_file": frozenset({"content", "error"}),
    "create_calendar_event": frozenset({"error", "event_id"}),
    "get_contacts": frozenset({"error", "results"}),
    "translate_text": frozenset({"error", "translated_text"}),
    "get_stock_price": frozenset({"error", "price"}),
    "set_reminder": frozenset({"error", "reminder_id"}),
    "run_code": frozenset({"error", "status", "stdout"}),
}


def _value(name: str, schema: dict[str, Any]) -> Any:
    if "enum" in schema:
        return schema["enum"][0]
    if "default" in schema:
        return schema["default"]
    kind = schema.get("type")
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return True
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return _STRING_VALUES.get(name, "test")


def _calls_for(scenario) -> list[ToolCallRecord]:
    definitions = UNIVERSAL_TOOLS if scenario.tools_override is None else scenario.tools_override
    calls = []
    for index, definition in enumerate(definitions):
        function = definition["function"]
        properties = function.get("parameters", {}).get("properties", {})
        arguments = {name: _value(name, schema) for name, schema in properties.items()}
        calls.append(
            ToolCallRecord(
                id=f"call-{index}",
                name=function["name"],
                raw_arguments="{}",
                arguments=arguments,
                turn=index + 1,
            )
        )
    return calls


def _expected_points(scenario_id: str, tool_name: str) -> int:
    return _TOOL_POINT_OVERRIDES.get(
        (scenario_id, tool_name),
        1 if scenario_id in _PARTIAL_BY_DEFAULT else 0,
    )


@pytest.mark.parametrize(
    "scenario",
    [*AGENTIC_SCENARIOS, *PLANNING_SCENARIOS, *STRUCTURED_SCENARIOS],
    ids=lambda scenario: scenario.id,
)
def test_schema_valid_tool_branches_have_expected_payloads_and_verdicts(scenario) -> None:
    """Lock single-tool behavior for every advertised scenario tool."""
    successful_handlers = 0
    for call in _calls_for(scenario):
        state = ScenarioState(final_answer="Completed with 4,200,000 results in London.")
        state.tool_calls.append(call)
        response = scenario.handle_tool_call(state, call)
        assert isinstance(response, dict)
        outcome_keys = set(response) & _TOOL_OUTCOME_KEYS[call.name]
        assert len(outcome_keys) == 1
        if "request_id" in response:
            assert response["request_id"].startswith("req_")
        if call.name == "calculator" and "result" in response:
            assert response["result"] == 4_200_000
        successful_handlers += 1

        evaluation = scenario.evaluate(state)
        assert isinstance(evaluation, ScenarioEvaluation)
        expected_points = _expected_points(scenario.id, call.name)
        assert evaluation.points == expected_points
        assert evaluation.status == _STATUS_BY_POINTS[expected_points]

    assert successful_handlers == len(_calls_for(scenario))


def test_no_tool_structured_scenario_has_exact_json_verdicts() -> None:
    scenario = next(scenario for scenario in STRUCTURED_SCENARIOS if scenario.id == "TC-64")
    assert _calls_for(scenario) == []

    invalid = scenario.evaluate(ScenarioState(final_answer="The Matrix is excellent."))
    assert invalid.status == ScenarioStatus.FAIL
    assert invalid.summary == "Output is not valid JSON."

    valid = scenario.evaluate(
        ScenarioState(
            final_answer=(
                '{"title":"The Matrix","year":1999,"rating":9.0,'
                '"genre":"sci-fi","summary":"A reality-bending classic."}'
            )
        )
    )
    assert valid.status == ScenarioStatus.PASS
    assert valid.points == 2
