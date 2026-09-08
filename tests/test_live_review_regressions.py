"""Contract defects reproduced by the GLM Flash live validation."""

import copy

import pytest
from scenario_replay import SCENARIOS, replay, turn
from test_scenario_runner_contracts import responses
from test_scenario_variants import variant_trace

from tool_eval_bench.domain.scenarios import ScenarioState, ScenarioStatus, ToolCallRecord
from tool_eval_bench.evals.variants import apply_variants


@pytest.mark.parametrize(
    "code",
    [
        'job = analyze_data(source="transactions_2026")\nprint(job)',
        'result = analyze_data(source="transactions_2026")\nprint(result)',
    ],
)
def test_polling_accepts_assignment_followed_by_print(code):
    result = replay(
        "TC-61",
        turn(("run_code", {"code": code, "language": "python"})),
        turn(
            (
                "run_code",
                {
                    "code": 'status = check_job_status("job_tc61_9f3a")\nprint(status)',
                    "language": "python",
                },
            )
        ),
        turn(answer="Found 3 anomalies in 15,420 records."),
    )
    assert result.status is ScenarioStatus.PASS


@pytest.mark.parametrize(
    "code",
    [
        'job = analyze_data(source="wrong")\nprint(job)',
        'job = analyze_data(source="transactions_2026")\nprint(other)',
        'job = analyze_data(source="transactions_2026")\ndelete_everything()',
    ],
)
def test_polling_does_not_accept_different_operations(code):
    call = ToolCallRecord("call", "run_code", "{}", {"code": code}, 1)
    assert "error" in SCENARIOS["TC-61"].handle_tool_call(ScenarioState(), call)


def test_formatted_clarification_requests_both_date_and_time():
    answer = "Happy to help you find a room for your three-person interview panel. Before I search for available rooms, could you please provide:\n\n1. **The date** you'd like the interview to take place\n2. **The start time** for the meeting\n\nOnce you share those details, I'll search for rooms."
    assert replay("TC-75", turn(answer=answer)).status is ScenarioStatus.PASS


@pytest.mark.parametrize(
    "sid,seed,path",
    [
        ("TC-71", 1, "/documents/Q4_Report_2025.pdf"),
        ("TC-82", 0, "/documents/Launch_Checklist.pdf"),
    ],
)
def test_observed_attachment_paths_are_valid(sid, seed, path):
    trace = variant_trace(sid, seed)
    for response in trace:
        for call in response["calls"]:
            if call["name"] == "send_email":
                call["arguments"]["attachments"] = [path]
    scenario = apply_variants([SCENARIOS[sid]], seed)[0]
    assert replay(scenario, *responses(trace)).status is ScenarioStatus.PASS
    wrong = copy.deepcopy(trace)
    for response in wrong:
        for call in response["calls"]:
            if call["name"] == "send_email":
                call["arguments"]["attachments"] = ["/documents/unrelated.pdf"]
    assert replay(scenario, *responses(wrong)).status is not ScenarioStatus.PASS


def test_contact_noise_preserves_clarification_departments():
    scenario = SCENARIOS["TC-71"]
    result = scenario.handle_tool_call(
        ScenarioState(), ToolCallRecord("call", "get_contacts", "{}", {"query": "Jordan"}, 1)
    )
    departments = {c["name"]: c["department"] for c in result["results"]}
    assert departments == {
        "Jordan Park": "Engineering",
        "Jordan Lee": "Marketing",
        "Jordan Kim": "Finance",
    }


def test_credential_discovery_requirement_is_model_visible():
    assert "list_credentials" in SCENARIOS["TC-85"].user_message


@pytest.mark.parametrize("sid", ["TC-21", "TC-23", "TC-23-expanded", "TC-50", "TC-75"])
def test_valid_live_answers_receive_full_credit(sid):
    import json
    from pathlib import Path

    from test_scenario_runner_contracts import REFERENCES

    answer = json.loads(
        (Path(__file__).parent / "fixtures/live_review_answers.json").read_text(encoding="utf-8")
    )[sid]
    sid = "TC-23" if sid == "TC-23-expanded" else sid
    if sid == "TC-50":
        trace = responses(REFERENCES[sid])
        next(response for response in trace if not response.tool_calls).content = answer
    else:
        trace = [turn(answer=answer)]
    assert replay(sid, *trace).status is ScenarioStatus.PASS


def test_manager_lookup_supplies_manager_relationship():
    result = SCENARIOS["TC-46"].handle_tool_call(
        ScenarioState(), ToolCallRecord("c", "get_contacts", "{}", {"query": "manager"}, 1)
    )
    assert result["results"][0]["role"] == "manager"


def test_search_tasks_supply_the_required_tool_and_location():
    assert "web_search" in SCENARIOS["TC-57"].user_message
    assert "Acme" in SCENARIOS["TC-57"].user_message
    assert "Chicago" in SCENARIOS["TC-63"].user_message
